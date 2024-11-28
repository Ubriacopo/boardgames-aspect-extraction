import logging
import sys
import swifter
from abc import abstractmethod

from keras import ops as K
import spacy
import pandas as pd

from core.pre_processing import PreProcessingService

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class LoadDataUtility:
    @abstractmethod
    def load_data(self, data_file_path: str) -> list:
        pass


class LoadCorpusAndProcessUtility(LoadDataUtility):
    def __init__(self, pre_processing_service: PreProcessingService):
        self.pre_processing_service = pre_processing_service

    def load_data(self, data_file_path: str) -> list:
        reference_dataframe = pd.read_csv(data_file_path)
        lines = reference_dataframe["comments"].swifter.apply(self.pre_processing_service.pre_process).dropna()
        return [[pre_processed.text for pre_processed in line] for line in lines]


class LoadCorpusUtility(LoadDataUtility):
    def __init__(self, custom_language_model=None):
        """
        This utility considers the corpus as already pre-processed by default. A different language model
        can be passed to apply a more complex pipeline.
        It is specialized on our corpus file and structure.

        @param custom_language_model: An optional custom language model to apply to the corpus.
        """
        # We are basically splitting only as the text was already pre-processed.
        self.nlp = spacy.blank("en") if custom_language_model is None else custom_language_model

    def _try_tokenization(self, text: str):
        try:
            return self.nlp(text)
        except Exception as exception:
            logging.error(exception)  # Show the real exception
            logging.warning(f"Given text: '{text}' was not convertable")

    def load_data(self, data_file_path: str) -> list:
        corpus = pd.read_csv(data_file_path, names=["comments"])["comments"]
        lines = corpus.swifter.apply(lambda x: self._try_tokenization(x)).dropna()
        return [[tokenized.text for tokenized in line] for line in lines]


def max_margin_loss(y_true, y_pred):
    """
    The max margin loss function is used to train the model.
    It is a hinge loss function that is used to train the model to maximize the margin between the correct class
    and the other classes.

    @param y_true: The true labels.
    @param y_pred: The predicted labels.
    @return: The loss value.
    """
    return K.mean(y_pred, axis=-1)
