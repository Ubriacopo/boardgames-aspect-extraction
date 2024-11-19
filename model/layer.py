"""
todo sistema sto file
class MaskedAverage(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedAverage, self).__init__(**kwargs)

    def call(self, input_data, mask=None):
        if mask is None:
            # We sum along the first axis as in our context we have rows of embeddings (words)
            return K.sum(input_data, axis=-2)  # todo what axis?

        mask = K.cast(mask, B.floatx())
        mask = K.expand_dims(mask, axis=-1)
        # In case the mask is none we ignore it
        return K.sum(input_data * mask, axis=-2) / K.sum(mask, axis=-2)


# Class Definition
class WeightedSumLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        
Makes
the
weighted
sum
of
rows
with a given row weight


@param


kwargs: If
needed
for layer(Naming)
    
    super(WeightedSumLayer, self).__init__(**kwargs)
    self.supports_masking = True

def call(self, inputs, weight, mask=None):
    return K.sum(inputs * weight, axis=1)

def compute_output_shape(self, input_shape):
    # TODO: Controlla se None davanti per Batch size
    return None, input_shape[1], input_shape[-1]


# Todo finish to check
class MaxMargin(keras.layers.Layer):
def __init__(self, **kwargs):
    super(MaxMargin, self).__init__(**kwargs)
    self.supports_masking = True

def call(self, input_tensor, mask=None):
    # todo: Vedi cosa sbaglia visto che ogni elemento del batch is same
    # Max Margin as:
    # loss = max(0, m + d(e_asp, e_pos) - d(e_asp, e_neg))
    e_pos, e_neg, e_asp = input_tensor

    # To make it more readable

    # Normalize the values (?) (L2 Normalization)
    # e_pos / keras.regularizers.L2(l2=eps*100000)(e_pos) #  For same result as done by hand
    # e_pos = e_pos / K.cast(eps + K.sqrt(K.sum(K.square(e_pos), axis=-1, keepdims=True)), float_x)
    e_pos = torch.nn.functional.normalize(e_pos, p=2, dim=-1)
    e_neg = torch.nn.functional.normalize(e_neg, p=2, dim=-1)  # IS NAN-NAN-NAN todo
    e_asp = torch.nn.functional.normalize(e_asp, p=2, dim=-1)
    # keras.layers.LayerNormalization(axis=-1)(e_asp)
    # todo qui c'è un errore grosso come una casa
    e_asp = K.expand_dims(e_asp, axis=-2)

    e_pos_dist = K.sum(e_pos * e_asp, axis=-1, keepdims=True)
    e_neg_dist = K.sum(e_neg * e_asp, axis=-1)

    maximum = K.maximum(0, (1. - e_pos_dist + e_neg_dist))
    return K.cast(K.sum(maximum, axis=-1, keepdims=True), B.floatx())

def compute_output_shape(self, input_shape):
    # Ah boh indovina la shape corretta
    return input_shape[0][0], 1  # Might be correct But bohh


class AspectEmbedding(keras.layers.Layer):
def __init__(self, embedding_size: int, weight_matrix, **kwargs):
    self.embedding_matrix = None
    self.supports_masking = True

    self.embedding_size = embedding_size
    self.weights_matrix = weight_matrix

    # Initialize some parameters to make layer working
    super(AspectEmbedding, self).__init__(**kwargs)

def build(self, input_shape):
    self.embedding_matrix = self.add_weight(
        (input_shape[-1], self.embedding_size), name='aspect_embedding_matrix',
        regularizer=keras.regularizers.OrthogonalRegularizer(factor=0.1, mode="rows"),
        initializer=lambda x, dtype: self.weights_matrix  # TODO Check if OK. Fai initialiozer
    )

def call(self, x, mask=None):
    return K.dot(x, self.embedding_matrix)

def compute_output_shape(self, input_shape):
    # Ah boh indovina la shape corretta
    return input_shape[0], self.embedding_size
"""
from keras import ops as K
from keras import backend as B
from keras import constraints
from keras import initializers
from keras import regularizers
from keras import Layer


class Attention(Layer):

    def __init__(self,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True, **kwargs):
        """
            Keras Layer that implements an Content Attention mechanism.
            Supports Masking.
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert type(input_shape) == list
        assert len(input_shape) == 2

        self.steps = input_shape[0][1]

        self.W = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]

        y = K.transpose(K.dot(self.W, K.transpose(y)))
        y = K.expand_dims(y, axis=-2)
        y = K.repeat(y, self.steps, axis=1)
        eij = K.sum(x * y, axis=-1)

        if self.bias:
            b = K.repeat(self.b, self.steps, axis=0)
            eij += b

        eij = K.tanh(eij)

        a = K.exp(eij) * K.cast(mask, B.floatx()) if mask is not None else K.exp(eij)
        return a / K.cast(K.sum(a, axis=1, keepdims=True) + B.epsilon(), B.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1]


class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        a = input_tensor[1]

        expanded_a = K.expand_dims(a, axis=-1)
        weighted_input = x * expanded_a

        return K.sum(weighted_input, axis=1)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]


class WeightedAspectEmb(Layer):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights([self.initial_weights])
        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = super(WeightedAspectEmb, self).get_config()
        config["input_dim"] = self.input_dim
        config["output_dim"] = self.output_dim
        return config

    @classmethod
    def from_config(cls, config):
        input_dim, output_dim = config["input_dim"], config["output_dim"]
        del config["input_dim"], config["output_dim"]
        return cls(input_dim, output_dim, **config)


class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is None:
            return torch.nan_to_num(K.sum(x, axis=-2)) / x.shape[-1]

        float_mask = K.cast(mask, B.floatx())
        expanded_float_mask = K.expand_dims(float_mask, axis=-1)
        # TODO Fix the generation but in general not known sentences should maybe be filtered somehow
        #           but for sure not cause the model to break

        avg = torch.nan_to_num(K.sum(x * expanded_float_mask, axis=-2) / K.sum(expanded_float_mask, axis=-2), nan=0)
        return avg

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]


import torch


class MaxMargin(Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0]
        z_n = input_tensor[1]
        r_s = input_tensor[2]

        nz_s = torch.nn.functional.normalize(z_s, p=2, dim=-1)
        nz_n = torch.nn.functional.normalize(z_n, p=2, dim=-1)
        nr_s = torch.nn.functional.normalize(r_s, p=2, dim=-1)

        steps = z_n.shape[1]

        pos = K.sum(nz_s * nr_s, axis=-1, keepdims=True)
        pos = K.repeat(pos, steps, axis=1)

        enr_s = K.expand_dims(nr_s, axis=-2)
        enr_s = K.repeat(enr_s, steps, axis=1)

        neg = K.sum(nz_n * enr_s, axis=-1)

        loss = K.cast(K.sum(K.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), B.floatx())
        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], 1
