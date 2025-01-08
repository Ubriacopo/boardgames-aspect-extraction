import torch
import math
from keras import Layer
from keras import backend as B
from keras import constraints
from keras import initializers
from keras import ops as K
from keras import regularizers


class Attention(Layer):

    def __init__(self, bias: bool = True, **kwargs):
        """
        Keras Layer that implements a Content Attention mechanism. Supports Masking.
        @param bias:
        @param kwargs:
        """
        self.supports_masking = True

        self.b = None
        self.w = None
        self.steps = None

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Check that the input is good for us
        assert list == type(input_shape)
        assert len(input_shape) == 2
        self.steps = input_shape[0][1]

        self.w = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[0][-1], input_shape[1][-1]))
        self.b = self.add_weight(name='{}_b'.format(self.name), shape=(1,), initializer="zero") if self.bias else None
        super(Attention, self).build(input_shape)

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x, y = input_tensor
        mask = mask[0]

        y = K.transpose(K.dot(self.w, K.transpose(y)))
        y = K.repeat(K.expand_dims(y, axis=-2), self.steps, axis=1)

        eij = K.sum(x * y, axis=-1)

        if self.bias:
            # Add bias term if it not None (By default it is true)
            b = K.repeat(self.b, self.steps, axis=0)
            eij += b

        eij = K.tanh(eij)
        a = K.exp(eij) * K.cast(mask, B.floatx()) if mask is not None else K.exp(eij)
        return a / K.cast(K.sum(a, axis=1, keepdims=True) + B.epsilon(), B.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        x, a = input_tensor  # Unpack the inputs
        return K.sum(x * K.expand_dims(a, axis=-1), axis=1)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]


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
        input_dim, output_dim = config["input_dim"], config["output_dim"]
        del config["input_dim"], config["output_dim"]
        return cls(input_dim, output_dim, **config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_size


class Average(Layer):
    def __init__(self, **kwargs):
        """
        todo: doc what it does.
        @param kwargs: Parameters to pass to super
        """
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is None:
            # x.shape[-1] is always > 0 therefore this cannot ever be NaN.
            return torch.nan_to_num(K.sum(x, axis=-2)) / x.shape[-1]

        expanded_float_mask = K.expand_dims(K.cast(mask, B.floatx()), axis=-1)
        return torch.nan_to_num(K.sum(x * expanded_float_mask, axis=-2) / K.sum(expanded_float_mask, axis=-2), nan=0)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]


# todo turn into loss? Cant beacuse of the shape of the output!
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
        negative_sample = torch.nn.functional.normalize(input_tensor[1], p=2, dim=-1)
        reconstruction_embedding = torch.nn.functional.normalize(input_tensor[2], p=2, dim=-1)

        # Sometimes "nan". I suppose there might be a problem with Epsilon taken from backend!
        # sentence_embedding = K.normalize(input_tensor[0], order=2, axis=-1)
        # negative_sample = K.normalize(input_tensor[1], order=2, axis=-1)
        # reconstruction_embedding = K.normalize(input_tensor[2], order=2, axis=-1)

        positive = K.sum(sentence_embedding * reconstruction_embedding, axis=-1, keepdims=True)
        # We repeat for all the generated entries of the negative sample
        positive = K.repeat(positive, negative_sample.shape[1], axis=-1)

        reconstruction_embedding = K.expand_dims(reconstruction_embedding, axis=-2)
        reconstruction_embedding = K.repeat(reconstruction_embedding, negative_sample.shape[1], axis=1)

        negative = K.sum(negative_sample * reconstruction_embedding, axis=-1)
        return K.cast(K.sum(K.maximum(0., (1. - positive + negative)), axis=-1, keepdims=True), B.floatx())

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
