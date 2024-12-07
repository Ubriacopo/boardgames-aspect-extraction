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
    def __init__(self, custom_language_model=None, min_word_count=2, column_name: str = "comments"):
        """
        This utility considers the corpus as already pre-processed by default. A different language model
        can be passed to apply a more complex pipeline.
        It is specialized on our corpus file and structure.

        @param custom_language_model: An optional custom language model to apply to the corpus.
        @param min_word_count: The minimum word count to consider a word as valid. (Default at least 3)
        """
        # We are basically splitting only as the text was already pre-processed.
        self.nlp = spacy.blank("en") if custom_language_model is None else custom_language_model
        self.min_word_count = min_word_count

        # What is the column name that contains the data.
        self.column_name = column_name

    def _try_tokenization(self, text: str, word_count: dict):
        try:
            return self.nlp(text)

        except Exception as exception:
            logging.error(exception)  # Show the real exception
            logging.warning(f"Given text: '{text}' was not convertable")

    def load_data(self, data_file_path: str) -> list:
        corpus = pd.read_csv(data_file_path)[self.column_name]

        word_count = dict()
        lines = corpus.swifter.apply(lambda x: self._try_tokenization(x, word_count)).dropna()

        for line in lines:
            for token in line:
                # At first match initialize to 1 (we counted one) then increase for any other match.
                word_count[token.text] = 1 if token.text not in word_count else word_count[token.text] + 1

        # Words that don't satisfy min count are just dropped.
        lines = lines.swifter.apply(
            lambda x: [t if word_count[str(t)] > self.min_word_count else "<UNK>" for t in x]
        )

        return [[str(tokenized) for tokenized in line] for line in lines]


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
