from abc import abstractmethod

import pandas as pd
import swifter


class DataLoaderUtility:
    @abstractmethod
    def load(self, file_path: str) -> list:
        pass


class CorpusLoaderUtility(DataLoaderUtility):
    def __init__(self, column_name: str | None = "comments"):
        # The referenced object we seek should have this col name
        self.column_name = column_name

    def load(self, file_path: str) -> list:
        # We suppose that this utility is called on pre-processed data that has
        # just to be split on spaces.
        corpus = pd.read_csv(file_path)[self.column_name]
        return corpus.swifter.apply(lambda x: x.split()).tolist()


class CorpusLoaderFromConllUtility(DataLoaderUtility):
    pass  # todo we have conll preprocessed.
