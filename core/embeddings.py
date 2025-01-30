import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import gensim.models
import numpy as np
from sklearn.cluster import KMeans


class Embedding(ABC):
    """
    We construct a vector representation zs for each input sentence s in the first step. In general, we want the vector
    representation to capture the most  relevant information in regard to the aspect (topic) of the sentence
    """

    def __init__(self, embedding_size: int, target_path: str, name: str):
        self.target_path: str = target_path  # File in which core is stored
        self.name: str = name
        self.embedding_size: int = embedding_size

        self.file_path = f"{self.target_path}/{self.name}.model"

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def generate(self, fit_data, persist: bool = True, load_existing: bool = False):
        """
        @param fit_data:
        @param persist
        @param load_existing:If true and there is a generated target element already -> It gets loaded
        @return:
        """
        pass

    @abstractmethod
    def vocabulary(self):
        """
        Returns the vocabulary. If not generated we expect an empty list.
        @return: the vocabulary if it was generated
        """
        pass

    def get_embedding_size(self):
        """
        To easier access interesting data of our class.
        @return: int giving the output size of an embedding of a word.
        """
        return self.embedding_size


class WordEmbedding(Embedding):
    def __init__(self, embedding_size: int, target_path: str, name: str = "embeddings",
                 max_vocab_size: int = None, min_word_count: int = 3):
        """
        As a good reference look at: https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
        @param max_vocab_size:
        @param embedding_size:
        @param min_word_count:
        """
        super(WordEmbedding, self).__init__(embedding_size, target_path, name)
        self.model: gensim.models.Word2Vec | None = None  # None loaded by default.

        self.max_vocab_size: int = max_vocab_size
        self.min_word_count: int = min_word_count

    def generate(self, corpus: list, persist: bool = True, sg: bool = True, load_existing: bool = False):

        if load_existing and Path(self.file_path).exists():
            print(f"Loading the existing found model as requested in path {self.file_path}")
            self.model = gensim.models.Word2Vec.load(str(Path(self.file_path)))
            return

        print("Creating new model")
        self.model = gensim.models.Word2Vec(
            sentences=corpus, vector_size=self.embedding_size,
            min_count=self.min_word_count, max_vocab_size=self.max_vocab_size, sg=sg
        )

        # Manual mapping of the pad token. It replaces the one at 0 position
        # We have to handle this process because keras can only handle zero padding by its default implementation.
        vec_values = np.zeros(self.embedding_size)
        index = self.model.wv.add_vector("<PAD>", vec_values)
        zero_token = self.model.wv.index_to_key[0]  # The most frequent element but we need that index!

        # Swap the tokens + vectors
        self.model.wv.index_to_key[0] = "<PAD>"
        self.model.wv.index_to_key[index] = zero_token
        self.model.wv.key_to_index["<PAD>"] = 0
        self.model.wv.key_to_index[zero_token] = index
        self.model.wv.vectors[index] = self.model.wv.vectors[0]
        self.model.wv.vectors[0] = vec_values

        if persist:
            self.model.save(self.file_path)

    def weights(self):
        if self.model is None:
            raise 'You must first generate the model to get its vocabulary.'
        return self.model.wv.vectors

    def actual_vocab_size(self) -> int:
        return len(self.model.wv.key_to_index)

    def vocabulary(self) -> dict:
        """
        If the core cannot be built we throw an error.
        @return: object with keys-value pairs for text-int encoding
        """
        if self.model is None:
            raise 'You must first generate the model to get its vocabulary.'
        return self.model.wv.key_to_index


class AspectEmbedding(Embedding):
    def __init__(self, aspect_size: int, embedding_size: int, target_path: str, name: str = "aspect-embeddings"):
        super(AspectEmbedding, self).__init__(embedding_size, target_path, name)
        self.model: KMeans | None = None
        self.aspect_size = aspect_size

    def generate(self, embedding_weights, persist: bool = True, load_existing: bool = False):
        if load_existing and Path(self.file_path).exists():
            print(f"Loading the existing found model as requested in path {self.file_path}")
            self.model = pickle.load(open(self.file_path, "rb"))
            return

        print("Creating new model")
        self.model = KMeans(n_clusters=self.aspect_size, verbose=False)
        self.model.fit(embedding_weights)  # todo controlla passato corretto

        if persist:
            pickle.dump(self.model, open(self.file_path, "wb"))

    def weights(self):
        if self.model is None:
            raise 'You must first generate the model to get its weights.'

        aspect_m = self.model.cluster_centers_ / np.linalg.norm(self.model.cluster_centers_, axis=-1, keepdims=True)
        return aspect_m.astype("float32")

    def vocabulary(self):
        """
        Our aspects have label (yet associated). We could opt for the most representative word, yet for that we have to
        calculate it or infer by hand the meaning, for now it simply is a number (its index).
        @return: The built vocabulary
        """
        # These aspects have to be inferred manually.
        return {value: value for value in range(self.aspect_size)}
