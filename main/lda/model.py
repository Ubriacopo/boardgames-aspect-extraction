from dataclasses import dataclass

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from main.lda.dataset import LdaDataset


@dataclass
class LdaGeneratorConfig:
    corpus_file_path: str
    topics: int = 14
    random_state: int = 42
    chunk_size: int = 1000
    passes: int = 10
    alpha: float | str = 'symmetric'
    eta: float = 0.01


class LdaModelGenerator:
    def __init__(self, config: LdaGeneratorConfig, stop_words: list[str] = None):
        self.c: LdaGeneratorConfig = config
        self.stop_words: list[str] = stop_words if stop_words is not None else []

    def make_model(self, existing_path: str = None) -> tuple[LdaModel, Dictionary]:
        if existing_path is not None:
            return LdaModel.load(existing_path)

        lda_dataset = LdaDataset(pd.read_csv(self.c.corpus_file_path), self.stop_words)

        return (
            # The LdaModel
            LdaModel(corpus=lda_dataset.dataset, id2word=lda_dataset.dict, num_topics=self.c.topics,
                     alpha=self.c.alpha, eta=self.c.eta, passes=self.c.passes,
                     random_state=self.c.random_state, chunksize=self.c.chunk_size),
            # The dictionary mapping num to word
            lda_dataset.dict
        )
