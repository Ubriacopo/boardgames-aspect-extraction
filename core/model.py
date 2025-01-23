from abc import abstractmethod

import keras
from keras import Input, Layer
from keras.src.layers import Embedding, Dense
from keras import ops as K
import core.embeddings
import core.layer as layer
from core.utils import max_margin_loss


class ModelGenerator:
    @abstractmethod
    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        pass

    @abstractmethod
    def generate_model(self, existing_model_path: str = None, is_train: bool = True) -> keras.Model:
        pass


class ABAEGenerator(ModelGenerator):
    def __init__(self, max_seq_length: int, negative_length: int, embeddings_model: core.embeddings.Embedding,
                 aspect_embeddings_model: core.embeddings.AspectEmbedding):
        self.max_seq_length = max_seq_length
        self.negative_length = negative_length

        self.emb_model = embeddings_model
        self.aspect_emb_model = aspect_embeddings_model

    # todo rename to train layers
    def make_layers(self) -> tuple[list[keras.Layer], list[keras.Layer]]:
        positive_input_shape = (self.max_seq_length,)  # 512
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

    def make_evaluation_layer(self) -> keras.Model:
        pass  # todo

    def generate_training_model(self, existing_model_path: str = None):
        if existing_model_path is not None:
            try:
                custom_objects = {'max_margin_loss': max_margin_loss}
                template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

                model = keras.Model(inputs=template_model.inputs, outputs=template_model.outputs[0])
                # Transfer properties of the template model.
                model.compile(
                    optimizer=template_model.optimizer, loss=template_model.loss, metrics=template_model.metrics
                )

                return model

            except Exception as error:
                # We keep going and simply generate a new model if we fail in finding the one in the path provided
                print(error)

        inputs, outputs = self.make_layers()
        return keras.Model(inputs=inputs, outputs=outputs[0])

    # todo non mi servira piu visto che i due modelli vengono costruiti diversamente probabilmente
    def generate_model(self, existing_model_path: str = None, is_train: bool = True) -> keras.Model:
        if is_train:  # Give the training model on demand.
            return self.generate_training_model(existing_model_path=existing_model_path)

        if existing_model_path is None:
            raise FileNotFoundError("Cannot load inference model from fs as it is missing")
        # todo delegare a train parte di questo?
        custom_objects = {'max_margin_loss': max_margin_loss}
        template_model = keras.models.load_model(existing_model_path, custom_objects=custom_objects)

        outputs = template_model.outputs

        # If the previously stored model was a training model I have to build the correct new output shape.
        if len(template_model.outputs) == 1:
            outputs = [template_model.outputs[0], template_model.layers[3].output, template_model.layers[6].output]

        model = keras.Model(inputs=template_model.inputs, outputs=outputs)
        model.compile(
            optimizer=template_model.optimizer, loss=template_model.loss, metrics={'max_margin': max_margin_loss}
        )

        return model


class SelfAttention(keras.layers.Layer):
    def __init__(self):
        self.supports_masking = True
        super(SelfAttention, self).__init__()

        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[0][-1], input_shape[1][-1]))

    def call(self, embeddings):
        mean_embeddings = K.mean(embeddings, (-1,)).unsqueeze(2)
        # (wv, wv) x (b, wv, 1) -> (b, wv, 1)
        p1 = K.matmul(self.w, mean_embeddings)
        # (b, maxlen, wv) x (b, wv, 1) -> (b, maxlen, 1)
        p2 = K.matmul(embeddings, p1).squeeze(2)
        return keras.activations.softmax(p2, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]


class AspectEmbeddings(keras.layers.Layer):
    def __init__(self, embedding_size: int, weights=None, w_regularization=None):
        self.supports_masking = True

        self.embedding_size = embedding_size
        self.W_regularization = w_regularization
        self.initial_weights = weights

        super(AspectEmbeddings, self).__init__()

        self.w = None

    def build(self, input_shape):
        w_shape = (input_shape[1], self.embedding_size)
        w_name = self.name + '_W'
        self.w = self.add_weight(name=w_name, shape=w_shape, initializer="uniform", regularizer=self.W_regularization)

        # Use the weights generated as a starting point if provided
        if self.initial_weights is not None:
            self.set_weights([self.initial_weights])
        super(AspectEmbeddings, self).build(input_shape)

    def call(self, x, mask=None):
        return K.matmul(self.w, x.unsqueeze(2)).squeeze()

    def get_config(self):
        config = super(AspectEmbeddings, self).get_config()
        config["embedding_size"] = self.embedding_size
        return config

    @classmethod
    def from_config(cls, config):
        embedding_size = config["embedding_size"]
        del config["embedding_size"]
        return cls(embedding_size, **config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_size


class WeightLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        attention_weights, w_embeddings = x[0], x[1]
        return K.matmul(attention_weights.unsqueezee(1), w_embeddings).squeeze()

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]


class MaxMarginsLoss(keras.layers.Layer):
    def __init__(self, aspect_size: int, aspect_weights, ortho_reg: float = 0.1, **kwargs):
        super(MaxMarginsLoss, self).__init__(**kwargs)
        self.ortho_reg = ortho_reg
        self.aspect_size = aspect_size

        self.aspect_weights = aspect_weights

    def call(self, input_tensor, mask=None):
        positive_emb = input_tensor[0]
        reconstruction_emb = input_tensor[1]
        averaged_negative_emb = input_tensor[2]

        positive_dot_products = K.matmul(positive_emb.unsqueeze(1), reconstruction_emb.unsqueeze(2)).squeeze()
        negative_dot_products = K.matmul(averaged_negative_emb, reconstruction_emb.unsqueeze(2)).squeeze()

        reconstruction_triplet_loss = K.sum(1 - positive_dot_products.unsqueeze(1) + negative_dot_products, dim=1)

        max_margin_l = K.max(reconstruction_triplet_loss, K.zeros_like(reconstruction_triplet_loss)).unsqueeze(dim=-1)
        reg_term = K.norm(K.matmul(self.aspect_weights.t(), self.aspect_weights) - K.eye(self.aspect_size))
        return self.ortho_reg * reg_term + max_margin_l

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1


class ABAE:
    def __init__(self, max_sequence_length: int, negative_samples_length: int,
                 aspect_embeddings_model: core.embeddings.AspectEmbedding,
                 word_emb_model: core.embeddings.WordEmbedding, aspect_size: int, ortho_reg: float = 0.1):
        self.max_seq_length = max_sequence_length
        self.negative_samples_length = negative_samples_length

        self.aspect_size = aspect_size

        self.ortho_reg = ortho_reg

        self.word_embeddings = word_emb_model
        self.aspect_embeddings = aspect_embeddings_model

    def __make_layers(self, weights):
        ## Input Layers
        pos_input_layer = Input(shape=(self.max_seq_length,), name="pos_input", dtype="int32")
        # Contrastive samples. Used to evaluate the loss.
        negative_shape = (self.negative_samples_length, self.max_seq_length)
        neg_input_layer = Input(shape=negative_shape, name="neg_input", dtype="int32")

        ## Word Embeddings of the inputs -> w_e and n_e
        # We won't delegate the construction of keras elements to others.
        # word_embeddings_layer = self.word_embeddings.build_embedding_layer("word_embedding")
        word_embeddings_layer = Embedding(
            input_dim=self.word_embeddings.actual_vocab_size(), output_dim=self.word_embeddings.embedding_size,
            weights=self.word_embeddings.weights(), trainable=False, name="word_embeddings", mask_zero=True
        )
        w_embeddings = word_embeddings_layer(pos_input_layer)
        # TODO Vedi se qui sbaglio essendo shape diversa
        neg_w_embeddings = word_embeddings_layer(neg_input_layer)

        # Calculate self attention on the embeddings.
        attention_weights = SelfAttention()(w_embeddings)
        # Weight our embeddings based on attention mechanism
        weighted_embeddings = WeightLayer()([attention_weights, w_embeddings])
        # Dense layer -> Chooses which aspect is the most fitting for sample
        aspect_weight = Dense(units=self.aspect_size, activation="softmax")(weighted_embeddings)

        # Reconstruct input via embeddings. (Decoder)
        aspect_embeddings_layer = AspectEmbeddings(weights=weights, embedding_size=self.aspect_size)
        decoded_embeddings = aspect_embeddings_layer(aspect_weight)

        # Evaluate loss (We measure how the reconstructed embedding relates to the correct element and differs from the
        # contrastive ones that we introduced)
        loss_layer = MaxMarginsLoss(
            aspect_size=self.aspect_size, ortho_reg=self.ortho_reg, aspect_weights=aspect_embeddings_layer.w
        )

        return [pos_input_layer, neg_input_layer], loss_layer([w_embeddings, decoded_embeddings, neg_w_embeddings])

    def __make_inference_layers(self, weights):
        pos_input_layer = Input(shape=(self.max_seq_length,), name="pos_input", dtype="int32")

        word_embeddings_layer = self.word_embeddings.build_embedding_layer("word_embedding")
        w_embeddings = word_embeddings_layer(pos_input_layer)

        attention_weights = SelfAttention()(w_embeddings)
        weighted_embeddings = K.matmul(attention_weights.unsqueezee(1), w_embeddings).squeeze()

        aspect_weight = Dense(units=self.aspect_size, activation="softmax")(weighted_embeddings)
        aspect_embeddings_layer = AspectEmbeddings(weights=weights, embedding_size=self.aspect_size)

        decoded_embeddings = aspect_embeddings_layer(aspect_weight)

        return pos_input_layer, decoded_embeddings

    def make_trainable_model(self, existing_model_path: str = None) -> keras.Model:
        inputs, outputs = self.__make_layers(self.aspect_embeddings.weights())
        return keras.Model(inputs=inputs, outputs=outputs)

    def make_inference_model(self) -> keras.Model:
        # todo finire
        inputs, outputs = self.__make_inference_layers(self.aspect_embeddings.weights())
        return keras.Model(inputs=inputs, outputs=outputs)
