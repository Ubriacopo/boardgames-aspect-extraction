from abc import ABC
from pathlib import Path

import gensim.models
import keras


# We construct a vector representation zs for each input sentence s in the first step. In general, we want the vector
# representation to capture the most  relevant information in regard to the aspect (topic) of the sentence
class Embedding(ABC):

    def __init__(self, vocab_size: int, embedding_size: int, target_file: str, corpus_file: str):
        self.target_file: str = target_file  # File in which model is stored
        self.vocab_size: int = vocab_size
        self.embedding_size: int = embedding_size
        self.corpus_file: str = corpus_file

    def get_weights(self):
        pass

    def load_model(self, override: bool = False):
        pass

    def load_corpus(self, corpus_file: str) -> list:
        pass


class MyWord2Vec(keras.Model):
    """
    A simple implementation to better explore how Word2Vec models are built.
    """
    def __init__(self, vocab_size: int, embedding_size: int, target_file: str):
        super(MyWord2Vec, self).__init__()  # <-Extends model class
        self.target_embedding = keras.layers.Embedding(vocab_size, embedding_size, name='w2v_embedding')
        self.context_embedding = keras.layers.Embedding(vocab_size, embedding_size)

    def call(self, pair, *args, **kwargs):
        target, context = pair

        word_embeddings = self.target_embedding(target)
        context_embeddings = self.context_embedding(context)

        dots = keras.layers.Dot(axes=(3, 2))([context_embeddings, word_embeddings])

        return keras.layers.Flatten()(dots)


# https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
class WordEmbedding(Embedding):
    model: gensim.models.Word2Vec  # Currently loaded model

    def get_weights(self):
        if self.model is None:
            self.load_model(override=False)
        # We do the train and update our reference to model.
        return self.model.wv.vectors

    def get_embedding_layer(self) -> keras.layers.Layer:
        return keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_size,
            weights=self.get_weights(), trainable=False
        )

    def load_model(self, override: bool = False):
        # Load the already created model if we don't want to override it
        if not override and Path(self.target_file).exists():
            self.model = gensim.models.Word2Vec.load(Path(self.target_file))

        # https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
        self.model = gensim.models.Word2Vec(
            sentences=self.load_corpus(self.corpus_file), vector_size=self.embedding_size, min_count=20, workers=8
        )

        self.model.save(self.target_file)

    def load_corpus(self, corpus_file: str) -> list:
        # Remove stopwords. todo: service for this.
        # https://stackoverflow.com/questions/3182268/nltk-and-language-detection To detect if not English
        # https://pypi.org/project/langdetect/ could also be a valid alternative
        pass
