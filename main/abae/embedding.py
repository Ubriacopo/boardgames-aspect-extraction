import pickle
from pathlib import Path

import gensim
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


class WordEmbedding:
    # Generates a new Word2Vec embedding model
    def __init__(self, emb_size: int, min_word_count: int, max_vocab_size: int = None):
        self.model: Word2Vec | None = None
        self.embedding_size: int = emb_size
        self.min_word_count: int = min_word_count
        self.max_vocab_size: int = max_vocab_size

    def load_existing(self, file_path: str):
        self.model = gensim.models.Word2Vec.load(str(Path(file_path)))

    def persist(self, file_path: str):
        self.model.save(file_path)

    def generate(self, corpus: list, sg: bool = False):
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size,
                              min_count=self.min_word_count, max_vocab_size=self.max_vocab_size, sg=sg)

        # Add padding token at spot 0 by re-organizing based on counts.
        wv = self.model.wv
        wv.add_vector("<PAD>", np.zeros(self.embedding_size))
        wv.set_vecattr("<PAD>", "count", wv.get_vecattr(wv.index_to_key[0], "count") + 1)

        # Unknown words are mapped to default 0 vector.
        wv.add_vector("<UNK>", np.zeros(self.embedding_size))
        wv.sort_by_descending_frequency()

        return self.model

    def weights(self):
        if self.model is None:
            raise 'You must first generate the model to get its vocabulary.'
        return self.model.wv.vectors

    def actual_vocab_size(self) -> int:
        if self.model is None:
            raise 'You must first generate the model to get its actual vocabulary size.'
        return len(self.model.wv.key_to_index)

    def vocabulary(self) -> dict:
        if self.model is None:
            raise 'You must first generate the model to get its vocabulary.'
        return self.model.wv.key_to_index


class AspectEmbedding:
    def __init__(self, aspect_size: int, emb_size: int):
        self.aspect_size: int = aspect_size
        self.embedding_size: int = emb_size
        # We initialize the aspect weights on the centroids of the Kmeans learning algorithm
        self.model: KMeans | None = None

    def load_existing(self, file_path: str):
        print(f"Loading the existing found model as requested in path {file_path}")
        self.model = pickle.load(open(file_path, "rb"))
        return self.model

    def persist(self, file_path: str):
        pickle.dump(self.model, open(file_path, "wb"))

    def generate(self, embedding_weights):
        print("Creating new model")
        self.model = KMeans(n_clusters=self.aspect_size, verbose=False)
        self.model.fit(embedding_weights)

        return self.model

    def weights(self):
        if self.model is None:
            raise 'You must first generate the model to get its weights.'

        aspect_m = self.model.cluster_centers_ / np.linalg.norm(self.model.cluster_centers_, axis=-1, keepdims=True)
        return aspect_m.astype(np.float32)

    def vocabulary(self) -> dict:
        """
        Our aspects have label (yet associated). We could opt for the most representative word, yet for that we have to
        calculate it or infer by hand the meaning, for now it simply is a number (its index).
        @return: The built vocabulary
        """
        return {value: value for value in range(self.aspect_size)}
