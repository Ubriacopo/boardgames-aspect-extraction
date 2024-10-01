import logging
import os
import sys

import spacy
from langdetect import detect
from spacy.lang.en import stop_words

import pandas as pd
from pandas import DataFrame, Series

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class PreProcessingService:
    stop_words = stop_words.STOP_WORDS

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")

    @staticmethod
    def detect_language(text: str | None):
        """
        Detects the language of the text and if it is not english it gets removed from corpus.
        Thought as a filter cb.

        @param text: The text to be processed. Can be none, nothing will happen to text.
        @return: None if the lang code is not english else the text itself.
        """
        if text is None:
            return None  # We have to stop to avoid exception

        detected_language = detect(text)

        if detected_language != "en":
            logging.debug(f"{detected_language} For current string: {text} ")

        return text if detected_language == "en" else None

    def _make_text_lemmas(self, entry: str | None) -> list[str] | None:
        if entry is None:
            return None  # We have to stop to avoid exception

        text_tokens = self.nlp(entry)
        logging.info(f"We split the text in the lemmas: {text_tokens}")
        return [token.lemma_ for token in text_tokens if str(token) not in self.stop_words]

    def pre_process(self, entry: str) -> str | None:
        try:
            entry = PreProcessingService.detect_language(entry)
            entry = self._make_text_lemmas(entry)

            # We return None if the text is to be ignored
            return None if entry is None else "".join([f"{e} " for e in entry])
        except Exception as e:
            logging.error(e)
            # Something went wrong when processing the text
            logging.info(f"We had a problem processing the text {entry}")
            return None


def pre_process_corpus(resource_file_path: str = "", target_file_path: str = "", overwrite: bool = False):
    if os.path.exists(resource_file_path) and not overwrite:
        logging.info("Procedure aborted as we are not allowed to override and the file already exists")

    reference_dataframe = pd.read_csv(resource_file_path)
    ps = PreProcessingService()
    reference_dataframe["comments"] = reference_dataframe["comments"].apply(ps.pre_process)
