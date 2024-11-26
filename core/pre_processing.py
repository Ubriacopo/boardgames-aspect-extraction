from swifter import swifter

import logging
import os
import sys
import re
from spacy.tokens.token import Token

import pandas as pd
import spacy
from fast_langdetect import detect

from core.utils import LoadDataUtility

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class PreProcessingService(LoadDataUtility):
    """
    It can be used as LoadDataUtility, but it won't be persisted. I should think well how to restructure this.
    Avoid it doing too much. I might pass it to a load corpus utility instead of inheriting from it. (Better idea) todo.
    """
    def __init__(self, extensive_logging: bool = False):
        # We use a small model as it is faster.
        self.nlp = spacy.load("en_core_web_sm")
        self.extensive_logging = extensive_logging

        # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
        self.clean_tags_regex = r"(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]"
        # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
        self.keep_tag_content_regex = r"\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]"

        self.clean_tags_pattern_matcher = re.compile(self.clean_tags_regex)
        self.clean_tags_keep_content_matcher = re.compile(self.keep_tag_content_regex)

    def detect_language(self, text: str | None):
        """
        Detects the language of the text and if it is not english it gets removed from corpus.
        Thought as a filter cb.

        @param text: The text to be processed. Can be none, nothing will happen to text.
        @return: None if the lang code is not english else the text itself.
        """
        if text is None:
            return None  # We have to stop to avoid exception

        detected_language = detect(text)['lang']
        self.extensive_logging and detected_language != "en" and logging.debug(f"{detected_language} with text: {text}")
        return text if detected_language == "en" else None

    def clean_text(self, entry: str):
        # Remove those noise tokens we recognized in the text.
        text = self.clean_tags_pattern_matcher.sub("", entry).strip()

        # We keep the content of the tags (Are game references)
        text = self.clean_tags_keep_content_matcher.sub(r"\3", text)
        return text

    @staticmethod
    def is_invalid_token(t: Token) -> bool:
        return t.is_punct or t.is_currency or t.like_email or t.like_url or t.is_stop or t.is_space

    def _make_text_lemmas(self, entry: str | None) -> list[str] | None:
        if entry is None:
            return None  # We have to stop to avoid exception

        text_tokens = self.nlp(entry.lower())
        # self.extensive_logging and logging.info(f"We split the text in the lemmas: {text_tokens}")
        return [token.lemma_ for token in text_tokens if not PreProcessingService.is_invalid_token(token)]

    @staticmethod
    def remove_short_text(entry: list[str] | None) -> list[str] | None:
        return entry if entry is not None and len(entry) > 3 else None

    def pre_process(self, entry: str) -> str | None:
        try:

            entry = self.clean_text(entry)
            entry = self.detect_language(entry)
            entry = self._make_text_lemmas(entry)

            # We return None if the text is to be ignored
            entry = PreProcessingService.remove_short_text(entry)
            return None if entry is None else " ".join([f"{e.strip()}" for e in entry])

        except Exception as e:
            logging.error(e)
            # Something went wrong when processing the text, so we simply skip it
            logging.info(f"We had a problem processing the text {entry}")
            return None

    def load_data(self, data_file_path: str) -> list:
        reference_dataframe = pd.read_csv(data_file_path)
        lines = reference_dataframe["comments"].swifter.apply(self.pre_process).dropna()
        return [[pre_processed.text for pre_processed in line] for line in lines]


def pre_process_corpus(resource_file_path: str = "./data/corpus.csv",
                       target_file_path: str = "./data/corpus.preprocessed.og.csv", overwrite: bool = False):
    """

    @param resource_file_path:
    @param target_file_path:
    @param overwrite:
    @return:
    """
    if os.path.exists(target_file_path) and not overwrite:
        logging.info("Procedure aborted as we are not allowed to override and the file already exists")
        return  # Abort the process.

    if not os.path.exists(resource_file_path):
        logging.info("Procedure aborted as the source file is missing")
        return  # Abort the process.

    reference_dataframe = pd.read_csv(resource_file_path)
    ps = PreProcessingService()
    save_frame = pd.DataFrame()

    save_frame["comments"] = reference_dataframe["comments"].swifter.apply(ps.pre_process)
    save_frame = save_frame.dropna()
    save_frame.to_csv(target_file_path, mode="w", header=False, index=False)
