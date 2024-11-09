import keras


class MaskedAverage(keras.layers.Layer):
    def __init__(self, **kwargs):
        """

        @param kwargs: Args to pass to the base layer if any.
        """
        super(MaskedAverage, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, input_data, mask=None):
        if mask is None:
            # We sum along the first axis as in our context we have rows of embeddings (words)
            return keras.ops.sum(input_data, axis=-2) / input_data.shape[1]  # todo what axis?

        mask = keras.ops.cast(mask, keras.backend.floatx())
        mask = keras.ops.expand_dims(mask, axis=-1)
        # In case the mask is none we ignore it
        return keras.ops.sum(input_data * mask, axis=-2) / keras.ops.sum(mask, axis=-2)


# Class Definition
class WeightedSumLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Makes the weighted sum of rows with a given row weight
        @param kwargs: If needed for layer (Naming)
        """
        super(WeightedSumLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, input_data: list, mask=None):
        x, w = input_data
        w = keras.ops.expand_dims(w, axis=-1)

        return keras.ops.sum(x * w, axis=1)

    def compute_output_shape(self, input_shape):
        # TODO: Controlla se None davanti per Batch size
        return None, input_shape[1], input_shape[-1]


# Todo finish to check
class MaxMargin(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0]
        z_n = input_tensor[1]
        r_s = input_tensor[2]

        z_s = z_s / keras.ops.cast(keras.backend.epsilon() + keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(z_s), axis=-1, keepdims=True)), keras.backend.floatx())
        z_n = z_n / keras.ops.cast(keras.backend.epsilon() + keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(z_n), axis=-1, keepdims=True)), keras.backend.floatx())
        r_s = r_s / keras.ops.cast(keras.backend.epsilon() + keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(r_s), axis=-1, keepdims=True)), keras.backend.floatx())

        steps = z_n.shape[1]

        pos = keras.ops.sum(z_s * r_s, axis=-1, keepdims=True)
        pos = keras.ops.repeat(pos, steps, axis=-1)
        r_s = keras.ops.expand_dims(r_s, axis=-2)
        r_s = keras.ops.repeat(r_s, steps, axis=1)
        neg = keras.ops.sum(z_n * r_s, axis=-1)

        loss = keras.ops.cast(keras.ops.sum(keras.ops.maximum(0, (1. - pos + neg)), axis=-1, keepdims=True),
                              keras.backend.floatx())
        return loss

    def compute_output_shape(self, input_shape):
        # Ah boh indovina la shape corretta
        return None, input_shape[2][2] # Might be correct But bohh
