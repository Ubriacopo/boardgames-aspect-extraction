import torch

from keras import Layer
from keras import backend as B
from keras import constraints
from keras import initializers
from keras import ops as K
from keras import regularizers

# todo fix size passed in wrong place warning
# todo get rid of unused regularizers
class Attention(Layer):

    def __init__(self, W_regularizer=None, b_regularizer=None, W_constraint=None,
                 b_constraint=None, bias=True, **kwargs):
        """
        TODO: Studia bene
        Keras Layer that implements a Content Attention mechanism.
        Supports Masking.

        @param W_regularizer:
        @param b_regularizer:
        @param W_constraint:
        @param b_constraint:
        @param bias:
        @param kwargs:
        """
        self.b = None
        self.W = None
        self.steps = None

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularization = regularizers.get(W_regularizer)
        self.b_regularization = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert list == type(input_shape)
        assert len(input_shape) == 2

        self.steps = input_shape[0][1]

        self.W = self.add_weight(
            shape=(input_shape[0][-1], input_shape[1][-1]), initializer=self.init,
            name='{}_W'.format(self.name), regularizer=self.W_regularization, constraint=self.W_constraint
        )

        self.b = self.add_weight(
            shape=(1,), initializer='zero', name='{}_b'.format(self.name),
            regularizer=self.b_regularization, constraint=self.b_constraint
        ) if self.bias else None

        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x, y = input_tensor

        mask = mask[0]

        y = K.transpose(K.dot(self.W, K.transpose(y)))
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

    def __init__(self, input_dim, output_dim, init='uniform', input_length=None, W_regularization=None,
                 activity_regularization=None, W_constraint=None, weights=None, dropout=0., **kwargs):

        self.W = None

        # todo check warning of init. Input dim should be only given at runtime from call?
        self.input_dim = input_dim
        self.embedding_size = output_dim

        self.init = initializers.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularization = regularizers.get(W_regularization)
        self.activity_regularization = regularizers.get(activity_regularization)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.input_dim, self.embedding_size), initializer=self.init,
            name='{}_W'.format(self.name), regularizer=self.W_regularization, constraint=self.W_constraint
        )

        # Use the weights generated as a starting point if provided
        if self.initial_weights is not None:
            self.set_weights([self.initial_weights])

        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_size

    def get_config(self):
        config = super(WeightedAspectEmb, self).get_config()
        config["input_dim"] = self.input_dim
        config["output_dim"] = self.embedding_size
        return config

    @classmethod
    def from_config(cls, config):
        input_dim, output_dim = config["input_dim"], config["output_dim"]
        del config["input_dim"], config["output_dim"]
        return cls(input_dim, output_dim, **config)


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
        positive_e = torch.nn.functional.normalize(input_tensor[0], p=2, dim=-1)
        negative_e = torch.nn.functional.normalize(input_tensor[1], p=2, dim=-1)
        aspect_e = torch.nn.functional.normalize(input_tensor[2], p=2, dim=-1)

        positive = K.sum(positive_e * aspect_e, axis=-1, keepdims=True)
        # We repeat for all the generated entries of the negative sample
        positive = K.repeat(positive, negative_e.shape[1], axis=1)

        expanded_aspect_e = K.expand_dims(aspect_e, axis=-2)
        expanded_aspect_e = K.repeat(expanded_aspect_e, negative_e.shape[1], axis=1)

        negative = K.sum(negative_e * expanded_aspect_e, axis=-1)

        return K.cast(K.sum(K.maximum(0., (1. - positive + negative)), axis=-1, keepdims=True), B.floatx())

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
