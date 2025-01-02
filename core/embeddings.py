import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import swifter
import gensim.models
import keras
from sklearn.cluster import KMeans
from keras import ops as K
import core.layer


class Embedding(ABC):
    """
    We construct a vector representation zs for each input sentence s in the first step. In general, we want the vector
    representation to capture the most  relevant information in regard to the aspect (topic) of the sentence
    """

    def __init__(self, embedding_size: int, target_path: str, name: str):
        self.target_path: str = target_path  # File in which core is stored
        self.name = name
        self.embedding_size: int = embedding_size

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        pass

    @abstractmethod
    def load(self):
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

    def get_embedding_size(self):
        """
        To easier access interesting data of our class.
        @return: int giving the output size of an embedding of a word.
        """
        return self.embedding_size

    @abstractmethod
    def vocabulary(self):
        """
        Returns the vocabulary. If not generated we expect an empty list.
        @return: the vocabulary if it was generated
        """
        pass


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

    def load(self):
        file_path = f"{self.target_path}/{self.name}.keras"
        if not Path(file_path).exists():
            raise f"To read a model the model has to exist. File: {file_path} does not exist."
        self.model = gensim.models.Word2Vec.load(str(Path(file_path)))

    def generate(self, corpus: list, persist: bool = True, sg: bool = True, load_existing: bool = False):
        if load_existing:
            try:
                return self.load()

            except Exception as e:
                print(e)

        self.model = gensim.models.Word2Vec(
            sentences=corpus, vector_size=self.embedding_size, workers=8,
            min_count=self.min_word_count, max_vocab_size=self.max_vocab_size, sg=sg
        )

        # Persist
        if persist:
            self.model.save(f"{self.target_path}/{self.name}.keras")

    def weights(self):
        if self.model is None:
            self.load()  # Try to load the model if it exists!

        return self.model.wv.vectors

    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        if self.model is None:
            self.load()  # Try to load the model if it exists!

        actual_vocab_size = len(self.model.wv.key_to_index)

        return keras.layers.Embedding(
            input_dim=actual_vocab_size, output_dim=self.embedding_size,
            weights=self.weights(), trainable=False, name=layer_name, mask_zero=True
        )

    def vocabulary(self) -> dict:
        """
        If the core cannot be built we throw an error.
        @return: object with keys-value pairs for text-int encoding
        """
        if self.model is None:
            self.load()  # Try to load the model if it exists!

        return self.model.wv.key_to_index


class AspectEmbedding(Embedding):
    def __init__(self, aspect_size: int, embedding_size: int, target_path: str, name: str = "aspect-embeddings"):
        super(AspectEmbedding, self).__init__(embedding_size, target_path, name)
        self.model: KMeans | None = None
        self.aspect_size = aspect_size

    def load(self):
        file_path = f"{self.target_path}/{self.name}.model"
        if not Path(file_path).exists():
            raise f"To read a model the model has to exist. File: {file_path} does not exist."
        self.model = pickle.load(open(file_path, "rb"))

    def generate(self, embedding_weights, persist: bool = True, load_existing: bool = False):

        if load_existing:
            try:
                return self.load()

            except Exception as e:
                print(e)

        self.model = KMeans(n_clusters=self.aspect_size)
        self.model.fit(embedding_weights)

        if persist:
            pickle.dump(self.model, open(f"{self.target_path}/{self.name}.model", "wb"))

    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        return core.layer.WeightedAspectEmb(input_dim=self.aspect_size, output_dim=self.embedding_size, weights=self.weights())

    def weights(self):
        if self.model is None:
            self.load()

        # Default value for L2 regularize
        regularize = keras.regularizers.L2(l2=0.01)
        return regularize(self.model.cluster_centers_) * K.convert_to_tensor(self.model.cluster_centers_)

    def vocabulary(self):
        """
        Our aspects have label (yet associated). We could opt for the most representative word, yet for that we have to
        calculate it or infer by hand the meaning, for now it simply is a number (its index).
        @return: The built vocabulary
        """
        vocabulary = dict()

        for i in range(self.aspect_size):
            vocabulary[i] = i

        return vocabulary
