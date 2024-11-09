from abc import abstractmethod
import layer

import keras

import model.embeddings


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
    def __init__(self, input_shape: tuple, embeddings_model: model.embeddings.Embedding,
                 aspect_embeddings_model: model.embeddings.AspectEmbedding):
        """
        @param input_shape: Input shape of the model
        @param embeddings_model: Object that has the vocabulary, matrices to generate embeddings etc.
        """
        super(ABAEModelGenerator, self).__init__(input_shape=input_shape)

        self.embeddings_model = embeddings_model
        self.aspect_embeddings_model = aspect_embeddings_model

    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        """
        We also initialize the aspect embedding matrix T with the centroids of clusters resulting from running k-means
        on word embeddings. Other parameters are initialized randomly. ~ Rudan
            -> Can we just randomly initialize them?
        @return:
        """
        input_layer = keras.layers.Input(shape=self.input_shape, name='input', dtype='int32')
        embedding_layer = self.embeddings_model.build_embedding_layer(layer_name="word_embedding")

        embeddings = embedding_layer(input_layer)
        avg = layer.MaskedAverage()(embeddings)
        # https://stackoverflow.com/questions/70034327/understanding-key-dim-and-num-heads-in-tf-keras-layers-multiheadattention
        # todo: On code of paper it was inverse Attention call but impl was custom. Check that they behave the same.
        attention_weights = keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(
            query=avg, key=embeddings, value=embeddings
        )

        # attention_weights = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)([avg, embeddings])
        weighted_positive = layer.WeightedSumLayer()(embeddings, attention_weights)

        # Negative representation for negative feedback
        negative_input_layer = keras.layers.Input(shape=self.input_shape, name='negative_input', dtype='int32')
        negative_embeddings = embedding_layer(negative_input_layer)

        avg_neg = layer.MaskedAverage()(negative_embeddings)

        # Sentence reconstruction
        aspect_size = self.aspect_embeddings_model.aspect_size
        dense_layer = keras.layers.Dense(units=aspect_size, activation='softmax')(weighted_positive)
        aspect_embeddings = self.aspect_embeddings_model.build_embedding_layer("aspect_embedding")(dense_layer)

        output = layer.MaxMargin()([weighted_positive, avg_neg, aspect_embeddings])
        return [input_layer, negative_input_layer], output
