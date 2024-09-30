import logging
from abc import ABC
from typing import Final

import nltk
from nltk.corpus import stopwords
from langdetect import detect


class PreProcessStep(ABC):
    def _run(self, text: str | None) -> str | None:
        pass

    def evaluate(self, text: str | None) -> str | None:
        return text if text is None else self._run(text)


class RemoveIfNotEnglish(PreProcessStep):
    english_vocab: Final[str] = set(w.lower() for w in nltk.corpus.words.words())

    def _run(self, text: str | None) -> str | None:
        detected_language = detect(text)
        logging.debug(f"{detected_language} For current string: {text} ")
        return text if detect(text) == "en" else None
        # Tokenize text.
        # nltk.download('words')
        # nltk.download('punkt_tab')

        # TODO: Meh questo metodo deve essere studiato bene altrimenti mi conviene usare qualche libreria esistente.
        # tokenized_text = set(nltk.word_tokenize(text))
        # The unique missing words.
        # undefined_words_in_english_vocab = tokenized_text.difference(self.english_vocab)
        # Decide metric to decide if sentence is actually english enough
        # return text if len(undefined_words_in_english_vocab) / len(tokenized_text) < 0.3 else None


def remove_if_not_english(entry: str, **kwargs) -> str | None:
    if entry is None:
        return entry


class PreProcessingService:
    def __init__(self):
        self.pre_processing_functions = [
            remove_if_not_english,
        ]

    def pre_process(self, entry: str) -> str | None:
        for func in self.pre_processing_functions:
            entry = func(entry)

        pass
