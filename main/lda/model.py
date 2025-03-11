from dataclasses import dataclass

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from pandas import DataFrame

from main.lda.dataset import LdaDataset


@dataclass
class LdaGeneratorConfig:
    topics: int = 14
    random_state: int = 42
    chunk_size: int = 1000
    passes: int = 10
    alpha: float | str = 'symmetric'
    eta: float = 0.01

    def from_dict(self, dictionary: dict):
        if 'topics' in dictionary:
            self.topics = dictionary['topics']

        if 'random_state' in dictionary:
            self.random_state = dictionary['random_state']

        return self


class LdaModelGenerator:
    def __init__(self, config: LdaGeneratorConfig, stop_words: list[str] = None):
        self.c: LdaGeneratorConfig = config
        self.stop_words: list[str] = stop_words if stop_words is not None else []

    def make_model(self, corpus: str | DataFrame, existing_path: str = None) -> tuple[LdaModel, Dictionary]:
        if existing_path is not None:
            return LdaModel.load(existing_path)

        ds = LdaDataset(corpus, self.stop_words)

        return (
            # The LdaModel
            LdaMulticore(corpus=ds.dataset, id2word=ds.dict, num_topics=self.c.topics,
                         alpha=self.c.alpha, eta=self.c.eta, passes=self.c.passes, workers=14,
                         random_state=self.c.random_state, chunksize=self.c.chunk_size),

            # The dictionary mapping num to word
            ds.dict
        )
