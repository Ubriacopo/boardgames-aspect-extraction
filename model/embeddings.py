import pickle
from abc import ABC, abstractmethod
from pathlib import Path
import swifter
import gensim.models
import keras
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from keras import ops as K


# Todo move somewhere else
class LoadCorpusUtility:
    def __init__(self):
        """
        This utility considers the corpus as already pre-processed
        """

        # https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial
        # We are basically splitting only as the text was already pre-processed.

        self.nlp = spacy.blank("en")

    def _try_tokenization(self, text: str):
        try:
            return self.nlp(text)
        except Exception as exception:
            # todo: Use logger
            print(exception)  # Show the real exception
            print(f"Given text: '{text}' was not convertable")

    def load_corpus(self, corpus_file: str) -> list:
        corpus = pd.read_csv(corpus_file, names=["comments"])["comments"]
        lines = corpus.swifter.apply(lambda x: self._try_tokenization(x)).dropna()
        return [[tokenized.text for tokenized in line] for line in lines]


# We construct a vector representation zs for each input sentence s in the first step. In general, we want the vector
# representation to capture the most  relevant information in regard to the aspect (topic) of the sentence
class Embedding(ABC):
    def __init__(self, embedding_size: int, target_model_file: str):
        self.target_model_file: str = target_model_file  # File in which model is stored
        self.embedding_size: int = embedding_size

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        pass

    @abstractmethod
    def load_model(self, override: bool = False):
        pass

    def get_embedding_size(self):
        """
        To easier access interesting data of our class.
        @return: int giving the output size of an embedding of a word.
        """
        return self.embedding_size

    @abstractmethod
    def get_vocab(self):
        """
        Returns the vocabulary. If not generated we expect an empty list.
        @return: the vocabulary if it was generated
        """
        pass


class WordEmbedding(Embedding):

    def __init__(self, corpus_loader_utility, max_vocab_size: int, embedding_size: int,
                 target_model_file: str, corpus_file: str, min_word_count: int = 4):
        """
        As a good reference look at: https://github.com/piskvorky/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
        @param corpus_loader_utility:
        @param max_vocab_size:
        @param embedding_size:
        @param target_model_file:
        @param corpus_file:
        @param min_word_count:
        """
        super(WordEmbedding, self).__init__(embedding_size, target_model_file)
        self.corpus_loader_utility = corpus_loader_utility
        self.model: gensim.models.Word2Vec | None = None  # None loaded

        self.max_vocab_size: int = max_vocab_size
        self.corpus_file: str = corpus_file
        self.min_word_count: int = min_word_count

    def get_weights(self):
        if self.model is None:
            self.load_model(override=False)
        # We do the train and update our reference to model.
        return self.model.wv.vectors

    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        actual_vocab_size = len(self.model.wv.key_to_index)
        return keras.layers.Embedding(
            input_dim=actual_vocab_size, output_dim=self.embedding_size,
            # Should my embeddings be trainable?  Good question! todo: Think about it
            weights=self.get_weights(), trainable=True, name=layer_name, mask_zero=True
        )

    def load_model(self, override: bool = False):
        # Load the already created model if we don't want to override it
        if not override and Path(self.target_model_file).exists():
            self.model = gensim.models.Word2Vec.load(str(Path(self.target_model_file)))
        else:  # Generate a new model if none exists
            # All that is required is that the input yields one sentence (list of utf8 words) after another
            # https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
            self.model = gensim.models.Word2Vec(
                sentences=self.corpus_loader_utility.load_corpus(self.corpus_file), vector_size=self.embedding_size,
                min_count=self.min_word_count, workers=8, max_vocab_size=self.max_vocab_size
            )

            self.model.save(self.target_model_file)

    def get_vocab(self) -> object:
        """
        If the model cannot be built we throw an error.
        @return: object with keys-value pairs for text-int encoding
        """
        if self.model is None:
            # I have to load the model
            self.load_model(override=False)

        return self.model.wv.key_to_index


class AspectEmbedding(Embedding):
    model: KMeans

    def __init__(self, aspect_size: int, embedding_size: int, target_model_file: str, base_embeddings: Embedding):
        super(AspectEmbedding, self).__init__(embedding_size, target_model_file)
        self.aspect_size = aspect_size
        self.base_embeddings = base_embeddings

    def build_embedding_layer(self, layer_name: str) -> keras.layers.Layer:
        return keras.layers.Embedding(
            input_dim=self.aspect_size, output_dim=self.embedding_size, weights=self.get_weights(),
            embeddings_regularizer=keras.regularizers.OrthogonalRegularizer(factor=0.1, mode="rows")
        )

    """
    We also initialize the aspect embedding matrix T with the centroids of clusters resulting from running k-means
    on word embeddings. Other parameters are initialized randomly. ~ Rudan
    
    So do we now.
    """

    def load_model(self, override: bool = False):
        if not override and Path(self.target_model_file).exists():
            self.model = pickle.load(open(self.target_model_file, "rb"))
        else:  # Generate the model as it does not exist
            self.model = KMeans(n_clusters=self.aspect_size)
            self.model.fit(self.base_embeddings.get_weights())
            # Persist the model in our disk
            pickle.dump(self.model, open(self.target_model_file, "wb"))

    def get_weights(self):
        if self.model is None:
            self.load_model(override=False)

        # Default value for L2 regularize
        regularize = keras.regularizers.L2(l2=0.01)
        return regularize(self.model.cluster_centers_) * K.convert_to_tensor(self.model.cluster_centers_)

    def get_vocab(self):
        pass  # Not used as it has   None (?). Actually we should have ! (The main aspects)
