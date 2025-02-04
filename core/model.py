from abc import abstractmethod

import keras
from keras import Input
from keras.src.layers import Embedding
from keras.src.regularizers import OrthogonalRegularizer

import core.embeddings
import core.layer as layer
from core.utils import max_margin_loss

## TODO: Qualcosa non va. (54% acc, mi sarei aspettato di piu)
class ModelGenerator:
    @abstractmethod
    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        pass

    @abstractmethod
    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None) -> keras.Model:
        pass


class ABAEGenerator(ModelGenerator):
    def __init__(self, max_seq_length: int, negative_length: int, embeddings_model: core.embeddings.WordEmbedding,
                 aspect_embeddings_model: core.embeddings.AspectEmbedding):
        self.max_seq_length = max_seq_length
        self.negative_length = negative_length

        self.word_embeddings = embeddings_model
        self.aspect_embeddings = aspect_embeddings_model

    def __training_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        sentence_input_layer = Input(shape=(self.max_seq_length,), name='positive', dtype='int32')

        negative_input_shape = (self.negative_length, self.max_seq_length)
        n_sentences_input_layer = Input(shape=negative_input_shape, name='negative', dtype='int32')
        # todo vedi se il caricamento di questi pesi é corretto
        word_embeddings_layer = Embedding(
            self.word_embeddings.actual_vocab_size(), self.word_embeddings.embedding_size,
            weights=self.word_embeddings.weights(), trainable=False, name="word_embeddings", mask_zero=True
        )

        s_embedding = word_embeddings_layer(sentence_input_layer)  # Sentence Embedding
        attention_w = core.layer.SelfAttention(name="attention")(s_embedding)  # Attention weights

        # Weighted sentence embeddings
        weighted_s_emb = core.layer.WeightLayer(name="weight_sentences")([attention_w, s_embedding])

        # Negative instance representation
        n_embedding = word_embeddings_layer(n_sentences_input_layer)  # Negative Sentences Embedding
        mean_n_emb = layer.Mean()(n_embedding)  # (64, 10, 1017, 128) -> (64, 128)

        aspect_size = self.aspect_embeddings.aspect_size
        predicted_aspect = keras.layers.Dense(aspect_size, activation='softmax', name="sentence_aspect")(weighted_s_emb)
        # todo che sia il w_regularization il problema?
        aspect_embeddings_layer = layer.AspectEmbeddings(
            weights=self.aspect_embeddings.weights(), embedding_size=self.word_embeddings.embedding_size,
            w_regularization=OrthogonalRegularizer(factor=0.1)  # Pass factor?
        )

        reconstruction_emb = aspect_embeddings_layer(predicted_aspect)
        output = layer.MaxMargin(name="max_margin")([weighted_s_emb, mean_n_emb, reconstruction_emb])
        # Model outputs: [Loss, AttentionWeights, AspectProbability]
        return [sentence_input_layer, n_sentences_input_layer], [output, attention_w, predicted_aspect]


    @staticmethod
    def restore_training_model(custom_objects: dict, existing_model_path: str):
        try:
            template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

            model = keras.Model(inputs=template_model.inputs, outputs=template_model.outputs[0])
            # Transfer properties of the template model.
            model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)
            return model

        except Exception as error:
            # We keep going and simply generate a new model if we fail in finding the one in the path provided
            print(error)
            return None  # No model could be restored.

    def generate_training_model(self, custom_objects: dict, existing_model_path: str = None):
        if existing_model_path is not None:
            existing_model = ABAEGenerator.restore_training_model(custom_objects, existing_model_path)
            if existing_model is not None:
                # The model could be loaded with success
                return existing_model

        inputs, outputs = self.__training_layers()
        # The model is not compiled. Compilation is delegated
        return keras.Model(inputs=inputs, outputs=outputs[0])

    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None) -> keras.Model:
        if existing_model_path is None:
            raise FileNotFoundError("Cannot load inference model from {fs} as it is missing")

        # custom_objects = {'max_margin_loss': max_margin_loss}
        template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

        # inputs = [template_model.get_layer('positive').inputs]
        outputs = [template_model.get_layer('attention').output, template_model.get_layer('sentence_aspect').output]
        # If the previously stored model was a training model I have to build the correct new output shape.

        model = keras.Model(inputs=template_model.inputs[0], outputs=outputs)
        # TODO: Vedi se questo tipo di chiamata va bene
        model.compile(optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics)

        return model
