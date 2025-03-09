from abc import ABC
from dataclasses import dataclass

from keras import Layer, Input
from keras.src.layers import Embedding, Dense, MultiHeadAttention
from keras.src.regularizers import OrthogonalRegularizer

from main.embedding import Word2VecWrapper
from main.abae.embedding import AspectEmbedding
from main.abae.layer import Attention, Weight, Average, WeightedAspectEmbedding, MaxMargin
from main.model import KerasModelGenerator


@dataclass
class ABAEGeneratorConfig:
    max_seq_len: int = 80
    negative_sample_size: int = 20
    embedding_size: int = 100
    aspect_size: int = 14


class BaseABAE(KerasModelGenerator, ABC):
    def __init__(self, config: ABAEGeneratorConfig, emb_model: Word2VecWrapper, aspect_model: AspectEmbedding):
        self.c: ABAEGeneratorConfig = config
        self.emb_model: Word2VecWrapper = emb_model
        self.aspect_model: AspectEmbedding = aspect_model
        # The shape of the negative input layer
        self.negative_shape = (self.c.negative_sample_size, self.c.max_seq_len)


class ABAE(BaseABAE):
    def get_input_output_layers(self, model, is_train: bool) -> tuple:
        if is_train:
            # To evaluate the contrastive max margin loss we only require the loss calculated value.
            return model.inputs, model.outputs[0]
        # Inference model so we care to watch the attention and the sentence aspect. (Classification).
        return model.inputs[0], [model.get_layer('attention').output, model.get_layer('sentence_aspect').output]

    def make_layers(self) -> tuple[list[Layer], list[Layer]]:
        sentence_input = Input(shape=(self.c.max_seq_len,), name='positive', dtype='int32')
        n_sentences_input = Input(shape=self.negative_shape, name='negative', dtype='int32')

        emb_size = self.c.embedding_size

        embedding_layer = Embedding(
            input_dim=self.emb_model.actual_vocab_size(), output_dim=emb_size,
            weights=self.emb_model.weights(), trainable=False, name='word_embedding', mask_zero=True
        )

        s_emb = embedding_layer(sentence_input)
        att = Attention(name="attention")(s_emb)
        # Sentence embeddings weighted based on the attention mechanism
        w_s_emb = Weight(name="weight")([s_emb, att])

        # Average negative embeddings
        avg_n_emb = Average()(embedding_layer(n_sentences_input))
        aspect_size = self.c.aspect_size

        aspect_pred = Dense(aspect_size, activation='softmax', name='sentence_aspect')(w_s_emb)
        w_aspect_emb_layer = WeightedAspectEmbedding(
            emb_size, self.aspect_model.weights(), OrthogonalRegularizer()
        )

        r_w_emb = w_aspect_emb_layer(aspect_pred)
        output = MaxMargin(name="max_margin")([w_s_emb, avg_n_emb, r_w_emb])
        return [sentence_input, n_sentences_input], [output, att, aspect_pred]


class SelfAttentionABAE(BaseABAE):
    def get_input_output_layers(self, model, is_train: bool) -> tuple:
        if is_train:
            # To evaluate the contrastive max margin loss we only require the loss calculated value.
            return model.inputs, model.outputs[0]
        # Inference model so we care to watch the attention and the sentence aspect. (Classification).
        return model.inputs[0], [model.get_layer('attention').output, model.get_layer('sentence_aspect').output]

    # todo vedi se riesci a farlo andare
    def make_layers(self) -> tuple[list[Layer], list[Layer]]:
        sentence_input = Input(shape=(self.c.max_seq_len,), name='positive', dtype='int32')
        n_sentences_input = Input(shape=self.negative_shape, name='negative', dtype='int32')

        emb_size = self.c.embedding_size

        embedding_layer = Embedding(
            self.emb_model.actual_vocab_size(), emb_size,
            weights=self.emb_model.weights(), trainable=False, name='word_embedding', mask_zero=True
        )

        s_emb = embedding_layer(sentence_input)
        # Todo: ovviamente le forme cambiano devo fare comunque weighted sum!
        w_s_emb, att = MultiHeadAttention(num_heads=1, key_dim=emb_size, name="attention")(
            query=s_emb, key=s_emb, value=s_emb, return_attention_scores=True
        )

        avg_n_emb = Average()(embedding_layer(n_sentences_input))
        aspect_size = self.c.aspect_size

        aspect_pred = Dense(aspect_size, activation='softmax', name='sentence_aspect')(w_s_emb)
        w_aspect_emb_layer = WeightedAspectEmbedding(
            emb_size, self.aspect_model.weights(), OrthogonalRegularizer(factor=0.1)
        )

        r_w_emb = w_aspect_emb_layer(aspect_pred)
        output = MaxMargin(name="max_margin")([w_s_emb, avg_n_emb, r_w_emb])
        return [sentence_input, n_sentences_input], [output, att, aspect_pred]
