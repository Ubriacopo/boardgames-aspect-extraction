import keras


class Model:

    def load_embedding_model(self):
        pass

    def make_layers(self, negative_size: int, max_length: int) -> tuple[keras.Layer, keras.Layer]:
        sentence_input = keras.Input(shape=(max_length,), dtype='int32', name='sentence_input')
        negative_input = keras.Input(shape=(negative_size, max_length), dtype='int32', name='neg_input')
        # todo:
        word_embeddings = keras.layers.Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

        return sentence_input, negative_input
