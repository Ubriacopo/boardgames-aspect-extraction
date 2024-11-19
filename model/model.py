from abc import abstractmethod
import layer

import keras

import model.embeddings


class ModelGenerator:
    @abstractmethod
    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        pass

    def make_model(self):
        inputs, outputs = self.make_layers()
        return keras.Model(inputs=inputs, outputs=outputs)


class CustomEmbeddingsModelGenerator(ModelGenerator):
    def __init__(self, input_shape: tuple, vocabulary: list):
        super(CustomEmbeddingsModelGenerator, self).__init__(input_shape)
        self.vocabulary = vocabulary

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        pass

# todo: fn load existing model, generate training model and generate eval model
class ABAEGenerator(ModelGenerator):
    def __init__(self, max_seq_length: int, negative_length: int, embeddings_model: model.embeddings.Embedding,
                 aspect_embeddings_model: model.embeddings.AspectEmbedding):
        self.max_seq_length = max_seq_length
        self.negative_length = negative_length

        self.emb_model = embeddings_model
        self.aspect_emb_model = aspect_embeddings_model

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        # Probabilemnte sbaglio nell'unboxing dei valori
        positive_input_shape = (self.max_seq_length,)
        negative_input_shape = (self.negative_length, self.max_seq_length)

        pos_input_layer = keras.layers.Input(shape=positive_input_shape, name='positive', dtype='int32')
        neg_input_layer = keras.layers.Input(shape=negative_input_shape, name='negative', dtype='int32')

        emb_layer = self.emb_model.build_embedding_layer(layer_name="word_embedding")

        embeddings = emb_layer(pos_input_layer)
        average = layer.Average()(embeddings)  # (64, 1017, 128) -> (64, 128) Avg of the embeddings

        negative_embeddings = emb_layer(neg_input_layer)
        neg_average = layer.Average()(negative_embeddings)  # (64, 10, 1017, 128) -> (64, 128)

        att_weights = layer.Attention(name='att_weights')([embeddings, average])
        weighted_positive = layer.WeightedSum()([embeddings, att_weights])

        aspect_size = self.aspect_emb_model.aspect_size
        dense_layer = keras.layers.Dense(units=aspect_size, activation='softmax')(weighted_positive)
        aspect_embeddings = self.aspect_emb_model.build_embedding_layer("aspect_embedding")(dense_layer)

        output = layer.MaxMargin(name="max_margin")([weighted_positive, neg_average, aspect_embeddings])
        # Model outputs: [Loss, AttentionWeights, AspectProbability]
        return [pos_input_layer, neg_input_layer], [output, att_weights, dense_layer]
