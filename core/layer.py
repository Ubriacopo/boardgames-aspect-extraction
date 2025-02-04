import torch
from keras import Layer, Variable
from keras import backend as B
from keras import ops as K
from keras.src.layers import Embedding


class SelfAttention(Layer):
    def __init__(self, bias: bool = False, **kwargs):
        self.supports_masking = True
        super(SelfAttention, self).__init__(**kwargs)

        self.bias = bias
        # Weights
        self.b: Variable | None = None
        self.w: Variable | None = None

        self.steps: int | None = None

    def build(self, input_shape):
        self.steps = input_shape[1]

        self.w = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[-1], input_shape[-1]))
        self.b = self.add_weight(name='{}_b'.format(self.name), shape=(1,), initializer="zero") if self.bias else None
        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, embeddings, mask=None):
        # Calculate the mean of the embeddings.
        # mask.shape = (b, max_len) -> term.shape = (b, max_len, 1)
        term = K.expand_dims(K.cast(mask, B.floatx()), axis=-1)
        # (b, max_len, wv) x (b, max_len, 1) -> (b, max_len, wv) §broadcasting§ -> (b, wv)
        mean_embeddings = K.sum(embeddings * term, axis=-2) / K.sum(term, axis=-2)

        # p1.shape = (wv, wv) x (wv, b) -> (b, wv) (Already transposed in notation)
        p1 = K.matmul(self.w, mean_embeddings.T).T
        p1 = K.repeat(K.expand_dims(p1, axis=-2), self.steps, axis=1)

        p_sum = K.sum(embeddings * p1, axis=-1)
        p_sum = p_sum + K.repeat(self.b, self.steps, axis=0) if self.bias else p_sum

        a = K.exp(K.tanh(p_sum)) * K.cast(mask, B.floatx()) if mask is not None else K.exp(K.tanh(p_sum))
        # errore qui! il b.epsionl non al denominatore! BINGO! ECCCO  CAUSA DEI MIEI MALI (UNO DI TNATI)
        # old : res = a / K.sum(a, axis=-1, keepdims=True) + B.epsilon()
        res = a / (K.sum(a, axis=-1, keepdims=True) + B.epsilon())

        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


class WeightedAspectEmb(Layer):

    def __init__(self, embedding_size: int, weights=None, w_regularization=None, dropout=0., **kwargs):
        self.w = None
        self.dropout = dropout

        self.embedding_size = embedding_size
        self.W_regularization = w_regularization

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True

        self.initial_weights = weights
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        w_shape = (input_shape[1], self.embedding_size)
        w_name = self.name + '_W'
        self.w = self.add_weight(name=w_name, shape=w_shape, initializer="uniform", regularizer=self.W_regularization)

        # Use the weights generated as a starting point if provided
        if self.initial_weights is not None:
            self.set_weights([self.initial_weights])

        super(WeightedAspectEmb, self).build(input_shape)

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        return K.dot(x, self.w)

    def get_config(self):
        config = super(WeightedAspectEmb, self).get_config()
        config["embedding_size"] = self.embedding_size
        return config

    @classmethod
    def from_config(cls, config):
        embedding_size = config["embedding_size"]
        del config["embedding_size"]
        return cls(embedding_size, **config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_size


class Mean(Layer):
    def __init__(self, **kwargs):
        """
        todo: doc what it does.
        @param kwargs: Parameters to pass to super
        """
        self.supports_masking = True  # We require masking.
        super(Mean, self).__init__(**kwargs)

    def call(self, x, mask=None):
        mask = K.expand_dims(K.cast(mask, B.floatx()), axis=-1)
        mean = K.sum(x * mask, axis=-2) / K.sum(mask, axis=-2)
        return mean

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]


class MaxMargin(Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        """
        @param input_tensor: Of the shape (positive, negative, aspect) embeddings (a list).
        @param mask: Masking. Won't be used therefore we do not support it in this layer.
        @return: The MaxMargin loss given on the positive and negative embeddings projection on the aspects
        """
        sentence_embedding = torch.nn.functional.normalize(input_tensor[0], p=2, dim=-1)
        negative_embeddings = torch.nn.functional.normalize(input_tensor[1], p=2, dim=-1)
        reconstruction_embedding = torch.nn.functional.normalize(input_tensor[2], p=2, dim=-1)

        repeat = negative_embeddings.shape[1]
        pos = (reconstruction_embedding * sentence_embedding).sum(dim=-1, keepdim=True).repeat(1, repeat)
        neg = (negative_embeddings * reconstruction_embedding.unsqueeze(-2).repeat(1, repeat, 1)).sum(dim=-1)

        res = K.cast(K.maximum(0., (1. - pos + neg)).sum(dim=-1), dtype=B.floatx())
        # The reason I get high values (like 6) is that it's not scaled on the number
        # of negative samples. The more I have the harder the task! This means that if I run
        # with a total of 5 negative samples per sentence the loss will be significantly lower
        return res

    def compute_mask(self, input_tensor, mask=None):
        return None

    # todo vedi se serve
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1


class AspectEmbeddings(Layer):
    def __init__(self, embedding_size: int, weights=None, w_regularization=None, **kwargs):
        self.supports_masking = True

        self.embedding_size = embedding_size
        self.W_regularization = w_regularization
        self.initial_weights = weights

        super(AspectEmbeddings, self).__init__(**kwargs)

        self.w = None

    def build(self, input_shape):
        # w_shape = (self.embedding_size, input_shape[0]) # (aspect, wv)
        w_shape = (input_shape[1], self.embedding_size)  # (wv, aspect)
        w_name = self.name + '_W'
        self.w = self.add_weight(name=w_name, shape=w_shape, initializer="uniform", regularizer=self.W_regularization)

        # Use the weights generated as a starting point if provided
        if self.initial_weights is not None:
            self.set_weights([self.initial_weights])

        super(AspectEmbeddings, self).build(input_shape)

    def call(self, x, mask=None):
        # (batch, wv, aspect_size) x (batch, wv, 1) -> (wv, 1)
        return K.dot(x, self.w)

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
        return input_shape[1], self.embedding_size


class WeightLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        attention_weights, w_embeddings = x[0], x[1]
        # (batch, 1, max_length) x (batch, max_length, wv) -> 1 x wv
        return (w_embeddings * attention_weights.unsqueeze(-1)).sum(1)

    def compute_mask(self, inputs, previous_mask):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][-1]
