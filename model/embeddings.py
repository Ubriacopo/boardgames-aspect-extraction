from abc import ABC
from pathlib import Path
import spacy
import swifter

import gensim.models
import keras
import pandas as pd


# We construct a vector representation zs for each input sentence s in the first step. In general, we want the vector
# representation to capture the most  relevant information in regard to the aspect (topic) of the sentence
class Embedding(ABC):
    def __init__(self, max_vocab_size: int, embedding_size: int,
                 target_file: str, corpus_file: str, min_word_count: int = 4):
        self.target_file: str = target_file  # File in which model is stored
        self.max_vocab_size: int = max_vocab_size
        self.embedding_size: int = embedding_size
        self.corpus_file: str = corpus_file
        self.min_word_count: int = min_word_count

    def get_weights(self):
        pass

    def load_model(self, override: bool = False):
        pass

    def load_corpus(self, corpus_file: str, nlp) -> list:
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


# todo restructure
# https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
class WordEmbedding(Embedding):
    model: gensim.models.Word2Vec  # Currently loaded model

    def get_weights(self):
        if self.model is None:
            self.load_model(override=False)
        # We do the train and update our reference to model.
        return self.model.wv.vectors

    def build_embedding_layer(self) -> keras.layers.Layer:
        actual_vocab_size = len(self.model.wv.key_to_index)
        return keras.layers.Embedding(
            input_dim=actual_vocab_size, output_dim=self.embedding_size, weights=self.get_weights(), trainable=False
        )

    def load_model(self, override: bool = False):
        # https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial
        nlp = spacy.blank("en")  # To avoid any kind of overhead. We are not basically splitting
        # as the text was already pre-processed.

        # Load the already created model if we don't want to override it
        if not override and Path(self.target_file).exists():
            self.model = gensim.models.Word2Vec.load(str(Path(self.target_file)))
        else:  # Generate a new model if none exists
            # All that is required is that the input yields one sentence (list of utf8 words) after another
            # https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
            self.model = gensim.models.Word2Vec(
                sentences=self.load_corpus(self.corpus_file, nlp), vector_size=self.embedding_size,
                min_count=self.min_word_count, workers=8, max_vocab_size=self.max_vocab_size
            )

            self.model.save(self.target_file)

    def load_corpus(self, corpus_file: str, nlp) -> list:
        corpus = pd.read_csv(corpus_file, names=["comments"])["comments"]
        lines = corpus.swifter.apply(lambda x: self.__try_tokenization(x, nlp)).dropna()
        return [[tokenized.text for tokenized in line] for line in lines]

    @staticmethod
    def __try_tokenization(text: str, nlp):
        try:
            return nlp(text)
        except:
            print(f"{text} is not convertable")
            return None
