from abc import abstractmethod
import layer

import keras


class ModelGenerator:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

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


class ABAEModelGenerator(ModelGenerator):
    def __init__(self, input_shape: tuple, vocabulary: list, embedding_size: int, aspect_size: int):
        super(ABAEModelGenerator, self).__init__(input_shape=input_shape)
        self.vocabulary = vocabulary

        self.aspect_size = aspect_size
        self.embedding_size = embedding_size

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        vocabulary_size = len(self.vocabulary)

        input_layer = keras.layers.Input(shape=self.input_shape, name='input')
        # todo -------------------------
        embedding_layer = keras.layers.Embedding(vocabulary_size, self.embedding_size, mask_zero=True, name='embedding')
        # todo load parameters of embedding model
        # todo -------------------------



        embeddings = embedding_layer(input_layer)
        avg = layer.MaskedAverage()(embeddings)

        # todo: On code of paper it was inverse Attention call but impl was custom.
        #       Check that they behave the same
        attention_weights = keras.layers.Attention()([avg, embeddings])
        weighted_positive = layer.WeightedSumLayer()(embeddings, attention_weights)

        # Negative representation for negative feedback
        negative_input_layer = keras.layers.Input(shape=self.input_shape, name='negative_input')
        negative_embeddings = embedding_layer(negative_input_layer)
        avg_neg = layer.MaskedAverage()(list(negative_embeddings))

        # Sentence reconstruction
        dense_layer = keras.layers.Dense(units=self.aspect_size, activation='softmax')
        # todo finish (load parameters for embedding model.
        aspect_embeddings = keras.layers.Embedding(input_dim=self.aspect_size, output_dim=self.embedding_size,
                                                   embeddings_regularizer=None)(dense_layer)
        output = layer.MaxMargin()([weighted_positive, avg_neg, aspect_embeddings])
        return [input_layer, negative_input_layer], output
