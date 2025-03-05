import torch.nn.functional
from keras import Layer, Variable

from keras import ops as K
from keras import backend as B


class Attention(Layer):
    """
    It is a self attention mechanism but not the same as the one faced in transformers.
    """

    def __init__(self, use_bias: bool = True, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

        self.w: Variable | None = None
        self.b = self.add_weight(name='{}_b'.format(self.name), shape=(1,), initializer="zero") if use_bias else None

        self.steps: int | None = None

    def build(self, input_shape):
        self.steps = input_shape[1]

        self.w = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[-1], input_shape[-1]))
        super(Attention, self).build(input_shape)

    def call(self, embeddings, mask=None):
        term = K.expand_dims(K.cast(mask, B.floatx()), axis=-1)
        mean_embeddings = (embeddings * term).sum(axis=-2) / term.sum(axis=-2)

        p = K.dot(self.w, mean_embeddings.T).T
        p = K.repeat(K.expand_dims(p, axis=-2), self.steps, axis=1)

        eij = (embeddings * p).sum(axis=-1)

        if self.b is not None:
            # Add the Bias term
            eij += K.repeat(self.b, self.steps, axis=0)

        act_eij = K.tanh(eij)

        a = K.exp(act_eij) if mask is None else K.exp(act_eij) * K.cast(mask, B.floatx())
        res = a / (a.sum(axis=1, keepdim=True) + B.epsilon())
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]

    def compute_mask(self, input_tensor, mask=None):
        return None


class WeightedAspectEmbedding(Layer):
    def __init__(self, embedding_size: int, weights=None, w_reg=None, **kwargs):
        self.embedding_size = embedding_size
        self.w: Variable | None = None
        self.initial_weights = weights
        self.w_reg = w_reg
        super(WeightedAspectEmbedding, self).__init__()

    def build(self, input_shape):
        w_shape = (input_shape[1], self.embedding_size)
        w_name = self.name + '_W'
        self.w = self.add_weight(name=w_name, shape=w_shape, initializer="uniform", regularizer=self.w_reg)

        self.initial_weights is not None and self.set_weights([self.initial_weights])
        super(WeightedAspectEmbedding, self).build(input_shape)

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        return K.dot(x, self.w)

    def get_config(self):
        config = super(WeightedAspectEmbedding, self).get_config()
        config["embedding_size"] = self.embedding_size  # Store new parameter in serialization
        return config

    @classmethod
    def from_config(cls, config):
        embedding_size = config["embedding_size"]
        del config["embedding_size"]
        return cls(embedding_size, **config)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_size


class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        mask = K.expand_dims(K.cast(mask, B.floatx()), axis=-1)
        return (x * mask).sum(axis=-2) / mask.sum(axis=-2)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + input_shape[-1:]


class MaxMargin(Layer):
    def call(self, input_t, mask=None):
        # The sentence embeddings, weighted. (wv is word vector size es. [200] emb_size)
        # (b, wv)
        s_embeddings = torch.nn.functional.normalize(input_t[0], dim=-1)
        # Negative embeddings (it is a sample of embeddings) averaged
        # (b, steps, wv)
        n_embeddings = torch.nn.functional.normalize(input_t[1], dim=-1)
        # Reconstruction embeddings
        # (b, wv)
        r_embeddings = torch.nn.functional.normalize(input_t[2], dim=-1)

        # How many negative samples I have
        steps = n_embeddings.shape[1]

        # (b, steps)
        pos = K.repeat((s_embeddings * r_embeddings).sum(-1, keepdim=True), steps, axis=-1)
        # (b, steps)
        neg = (n_embeddings * K.repeat(r_embeddings.unsqueeze(-2), steps, axis=-2)).sum(-1)

        loss = K.cast(K.maximum(0., (1. - pos + neg)).sum(1), dtype=B.floatx())
        return loss


class Weight(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Weight, self).__init__(**kwargs)

    def call(self, x, mask=None):
        vector, weights = x[0], x[1]
        return (vector * weights.unsqueeze(-1)).sum(1)

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]
