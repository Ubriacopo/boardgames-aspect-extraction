import keras


class MaskedAverage(keras.layers.Layer):
    def     __init__(self, **kwargs):
        """

        @param kwargs: Args to pass to the base layer if any.
        """
        super(MaskedAverage, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, input_data, mask=None):
        if mask is None:
            # We sum along the first axis as in our context we have rows of embeddings (words)
            return keras.ops.sum(input_data, axis=-2) / input_data.shape[0]

        mask = keras.ops.cast(mask, keras.backend.floatx())
        mask = keras.ops.expand_dims(mask, axis=-1)
        # In case the mask is none we ignore it
        return keras.ops.sum(input_data * mask, axis=-2) / keras.ops.sum(mask, axis=-2)
