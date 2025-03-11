from abc import abstractmethod

import pandas as pd
import swifter
from pandas import DataFrame


class DataLoaderUtility:
    @abstractmethod
    def load(self, file_path: str) -> list:
        pass


class CorpusLoaderUtility(DataLoaderUtility):
    def __init__(self, column_name: str | None = "comments"):
        # The referenced object we seek should have this col name
        self.column_name = column_name

    def load(self, corpus: str | DataFrame) -> list:
        # We suppose that this utility is called on pre-processed data that has
        # just to be split on spaces.
        if type(corpus) is str:
            corpus = pd.read_csv(corpus)
        return corpus[self.column_name].swifter.apply(lambda x: x.split()).tolist()
