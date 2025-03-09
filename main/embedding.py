from __future__ import annotations
from pathlib import Path

import gensim
import numpy as np
from gensim.models import Word2Vec

from main.utils import CorpusLoaderUtility


class Word2VecWrapper:
    def __init__(self, emb_size: int, min_word_count: int, max_vocab_size: int = None):
        """
        @param emb_size: Dimensionality of the latent dimensions of the embeddings
        @param min_word_count: How often the word has to be counted in the ds in order to compute the embeddings of it
        @param max_vocab_size: Maximum number of words in the vocabulary of the word2vec model
        """
        self.model: Word2Vec | None = None
        self.embedding_size: int = emb_size
        self.min_word_count: int = min_word_count
        self.max_vocab_size: int = max_vocab_size

    @classmethod
    def from_scratch(cls, corpus_path: str, folder_path: str, emb_size: int = 100, min_word_count: int = 5,
                     max_vocab_size: int = None, file_name: str = "word2vec", persist: bool = True) -> Word2VecWrapper:

        instance = cls(emb_size=emb_size, min_word_count=min_word_count, max_vocab_size=max_vocab_size)
        instance.generate(corpus=CorpusLoaderUtility(column_name="comments").load(corpus_path))

        if persist:
            Path.mkdir(Path(folder_path), exist_ok=True, parents=True)
            instance.persist(f"{folder_path}/{file_name}.model")

        return instance

    @classmethod
    def from_existing(cls, file_path: str) -> Word2VecWrapper:
        model = gensim.models.Word2Vec.load(str(Path(file_path)))
        instance = cls(model.vector_size, model.min_count)

        instance.model = model
        return instance

    def persist(self, file_path: str):
        self.model.save(file_path)

    def generate(self, corpus: list, sg: bool = False):

        self.model is not None and print("A new model is being created. The old one will be discarded!")
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size, epochs=25,
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
