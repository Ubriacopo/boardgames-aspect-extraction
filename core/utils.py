import logging
import sys
from keras import ops as K
import spacy
import pandas as pd

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class LoadCorpusUtility:
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

    def load_corpus(self, corpus_file: str) -> list:
        corpus = pd.read_csv(corpus_file, names=["comments"])["comments"]
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
