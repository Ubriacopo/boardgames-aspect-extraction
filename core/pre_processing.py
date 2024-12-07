from abc import abstractmethod
from functools import reduce
from pathlib import Path

from spacy.matcher.matcher import Matcher
from spacy.matcher.phrasematcher import PhraseMatcher
from swifter import swifter

import logging
import os
import sys
import re
from spacy.tokens.token import Token

import pandas as pd
import spacy
from fast_langdetect import detect

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class ProcessingRule:
    @abstractmethod
    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        pass


class KickstarterRemovalRule(ProcessingRule):
    def __init__(self, min_sentence_word_length: int = 15):
        self.word = "kickstarter"
        self.lemma_word = "kickstart"
        self.min_sentence_word_length = min_sentence_word_length

    def process(self, e: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if e is None:
            return None  # We have to stop to avoid exception

        if type(e) is list and any(self.lemma_word in s.lower() for s in e) and len(e) < self.min_sentence_word_length:
            return None

        # Splitting text like this might be dumb but fast enough for us.
        if type(e) is str and len(e.split(' ')) < self.min_sentence_word_length and self.word in e.lower():
            return None

        return e


class CleanTextRule(ProcessingRule):
    def __init__(self, regex: str, replacement: str = ""):
        self.regex = regex
        self.replacement = replacement

        self.regex_pattern = re.compile(regex)

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception

        # We won't apply the regex to an already divided text
        if type(entry) is list:
            return entry

        # Remove those noise tokens we recognized in the text.
        return self.regex_pattern.sub(self.replacement, entry).strip()


class ShortTextFilterRule(ProcessingRule):
    def __init__(self, min_words_in_sentence: int = 4):
        self.min_words_in_sentence = min_words_in_sentence

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None

        if type(entry) is list and len(entry) < self.min_words_in_sentence:
            return None

        if type(entry) is str and len(entry.split(" ")) < self.min_words_in_sentence:
            return None
        return entry


class FilterLanguageRule(ProcessingRule):
    def __init__(self, lang_codes: list[str] = ["en"]):
        self.lang_codes = lang_codes

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        """
        Detects the language of the text and if it is not english it gets removed from corpus.

        @param extensive_logging:
        @param entry: The text to be processed. Can be none, nothing will happen to text.
        @return: None if the lang code is not english else the text itself.
        """
        if entry is None:
            return None  # We have to stop to avoid exception

        text = entry if type(entry) is str else str(" ".join(entry))

        detected_language = detect(text)['lang']

        is_valid_language = detected_language in self.lang_codes
        extensive_logging and not is_valid_language and logging.debug(f"{detected_language} with text: {text}")
        return entry if is_valid_language else None


class ListToTextRegenerationRule(ProcessingRule):
    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception
        return type(entry) is list and " ".join([f"{e.strip()}" for e in entry]) or entry.strip()


class LemmatizeTextRule(ProcessingRule):
    def __init__(self, nlp: spacy.language.Language | None = None):
        self.nlp = spacy.load("en_core_web_md") if nlp is None else nlp

    def is_invalid_token(self, t: Token) -> bool:
        return t.is_punct or t.is_currency or t.like_email or t.like_url or t.is_stop or t.is_space

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception

        if entry is list:
            # todo look at shape of this list. It should be a list of strings.
            return [self.process(e) for e in entry]

        text_tokens = self.nlp(entry.lower())
        return [token.lemma_ for token in text_tokens if not self.is_invalid_token(token)]


# todo another rule just to replace the GAME_NAME that is current to the game we are processing? This would
# require to pass a game id too (our interface doesn't allow it for now)
# todo too many games have common names that are hard to distinguish between terms and boardgames
# either I match on POS or no can do
class LemmatizeTextWithoutGameNamesRule(LemmatizeTextRule):
    def __init__(self, game_names: list[str], nlp: spacy.language.Language | None = None):
        super().__init__(nlp)
        lower_case_game_names = [name.lower() for name in game_names]
        # Now, I know what you are thinking. This is not the best way to do this.
        # Indeed, we could have used a custom pipeline component to do this BUT considering the fact that I expect
        # few games to be referenced in other texts, I think this is a good trade-off.
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        print(f"Generating game names tokenized representationL: ({len(game_names)})")
        self.matcher.add("GAME_NAME", [self.nlp(n) for n in lower_case_game_names])
        print("Done generating... cb ready for use!")

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception

        if entry is list:
            return [self.process(e) for e in entry]

        text_tokens = self.nlp(entry)
        game_name_matches = self.matcher(text_tokens)
        spans = [text_tokens[start:end] for match_id, start, end in game_name_matches]

        with text_tokens.retokenize() as retokenizer:
            for span in spacy.util.filter_spans(spans):
                retokenizer.merge(span, attrs={"LEMMA": "GAME_NAME"})

        return [token.lemma_ for token in text_tokens if not self.is_invalid_token(token)]


class PreProcessingService:
    """
    It can be used as LoadDataUtility, but it won't be persisted. I should think well how to restructure this.
    Avoid it doing too much. I might pass it to a load corpus utility instead of inheriting from it. (Better idea) todo.
    """

    @staticmethod
    def default_pipeline(target_path: str, extensive_logging: bool = False):
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                FilterLanguageRule(),
                LemmatizeTextRule(),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            extensive_logging
        )

    @staticmethod
    def game_name_less_pipeline(game_names: list[str], target_path: str, extensive_logging: bool = False):
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                FilterLanguageRule(),
                LemmatizeTextWithoutGameNamesRule(game_names),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            extensive_logging
        )

    @staticmethod
    def kickstarter_filter_pipeline(target_path: str, extensive_logging: bool = False):
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                KickstarterRemovalRule(),
                FilterLanguageRule(),
                LemmatizeTextRule(),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            extensive_logging
        )

    @staticmethod
    def kickstarter_filter_pipeline_without_game_names(
            game_names: list[str], target_path: str, extensive_logging: bool = False
    ):
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                KickstarterRemovalRule(),
                FilterLanguageRule(),
                LemmatizeTextWithoutGameNamesRule(game_names),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            extensive_logging
        )

    def __init__(self, pipeline: list[ProcessingRule], target_path: str, extensive_logging: bool = False):
        # Kickstarter is often reference as many people pledge their games from there.
        # Is this useless information? Should I ignore those reviews entirely?
        # self.nlp.Defaults.stop_words.add("kickstarter")
        self.pipeline = pipeline
        self.extensive_logging = extensive_logging

        self.target_path = target_path
        Path(self.target_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def remove_short_text(entry: list[str] | None) -> list[str] | None:
        return entry if entry is not None and len(entry) > 3 else None

    def pre_process(self, entry: str) -> str | None:
        try:
            return reduce(lambda t, rule: rule.process(t, self.extensive_logging), self.pipeline, entry)

        except Exception as e:

            logging.error(e)
            # Something went wrong when processing the text, so we simply skip it
            logging.info(f"We had a problem processing the text {entry}")
            return None

    def pre_process_corpus(self, resource_file_path: str, name: str, override: bool = False):
        if os.path.exists(f"{self.target_path}/{name}.csv") and not override:
            logging.info("Procedure aborted as we are not allowed to override and the file already exists")
            return  # Abort the process.

        if not os.path.exists(resource_file_path):
            logging.info("Procedure aborted as the source file is missing")
            return  # Abort the process.

        df = pd.read_csv(resource_file_path)

        df["original_text"] = df["comments"]
        df["comments"] = df["comments"].swifter.apply(self.pre_process)

        df = df.dropna()
        df.to_csv(f"{self.target_path}/{name}.processed.csv", mode="w", header=True, index=False)


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
    ps = PreProcessingService.default_pipeline()
    save_frame = pd.DataFrame()

    save_frame["game_id"] = reference_dataframe["game_id"]
    save_frame["original_text"] = reference_dataframe["comments"]
    save_frame["comments"] = reference_dataframe["comments"].swifter.apply(ps.pre_process)

    save_frame = save_frame.dropna()
    save_frame.to_csv(target_file_path, mode="w", header=True, index=False)
