from abc import abstractmethod
from keras import ops as K
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


def max_margin_loss(y_true, y_pred):
    """
    The max margin loss function is used to train the model.
    It is a hinge loss function that is used to train the model to maximize the
    margin between the correct class and the other classes.

    @param y_true: The true labels.
    @param y_pred: The predicted labels.
    @return: The loss value.
    """
    return K.mean(y_pred, axis=-1)


def zero_loss(y_true, y_pred):
    return K.convert_to_tensor([0])