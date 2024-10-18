from abc import abstractmethod

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
    def __init__(self, input_shape: tuple, vocabulary: list):
        super(ABAEModelGenerator, self).__init__(input_shape=input_shape)
        self.vocabulary = vocabulary

    # todo make. Or pass an embedding model generator
    def make_embedding_layer(self) -> keras.layers.Embedding:
        pass

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        vocabulary_size = len(self.vocabulary)
        # mask_zero=True
        embedding_layer = self.make_embedding_layer()

        input_layer = keras.layers.Input(shape=self.input_shape, name='input')

        embeddings = embedding_layer(input_layer)
        avg = keras.layers.Average()(list(embeddings))

        attention_weights = keras.layers.Attention()([embeddings, avg])
        # todo contiunua


        # Negative representation for negative feedback
        negative_input_layer = keras.layers.Input(shape=self.input_shape, name='negative_input')
        negative_embeddings = embedding_layer(negative_input_layer)
        avg_neg = keras.layers.Average()(list(negative_embeddings))
