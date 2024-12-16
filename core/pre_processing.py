import multiprocessing
from abc import abstractmethod
from functools import reduce
from pathlib import Path
import itertools
from date_spacy import find_dates
from pandas import Series, DataFrame
from spacy.matcher.matcher import Matcher
from spacy.matcher.phrasematcher import PhraseMatcher
from spacy.tokens.doc import Doc
from swifter import swifter

import logging
import os
import sys
import re
from spacy.tokens.token import Token

import pandas as pd
import spacy
from fast_langdetect import detect

from core.dataset_sampler import ConsumingDatasetSampler

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


class ProcessingRule:
    """
    "Interface" on which we base processing steps.
    """

    @abstractmethod
    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        pass


class CleanTextRule(ProcessingRule):
    def __init__(self, regex: str, replacement: str = ""):
        """
        Applies regex as processing step.
        @param regex: The matching pattern to apply
        @param replacement: What to put instead of the match pattern.
        """
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
        # We expect the input to be a list but just in case we handle str case as well.
        return type(entry) is list and " ".join([f"{e.strip()}" for e in entry]) or entry.strip()


class LemmatizeTextRule(ProcessingRule):
    def __init__(self, nlp: spacy.language.Language | None = None):
        self.nlp = spacy.load("en_core_web_md") if nlp is None else nlp

    def is_invalid_token(self, t: Token) -> bool:
        return t.is_punct or t.is_currency or t.like_email or t.like_url or t.is_stop or t.is_space

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception
        # todo test at least one of theese calls.
        if entry is list:
            return list(itertools.chain(*[self.process(e) for e in entry]))

        text_tokens = self.nlp(entry.lower())
        return [token.lemma_ for token in text_tokens if not self.is_invalid_token(token)]


class DateFilterTextRule(ProcessingRule):
    def __init__(self):
        self.nlp = spacy.blank("en")
        # We require spacy to recognize if the dates.
        # I had to separate it from the lemmatizer as using this pipe would break the trained md (Dunno why).
        self.nlp.add_pipe("find_dates")

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception

        if entry is list:
            return list(itertools.chain(*[self.process(e) for e in entry]))

        for e in self.nlp(entry).ents:
            entry = entry.replace(e.text, "<DATE>")

        return entry


class MatcherReplacementRuleOnLemma:
    def __init__(self, matcher: Matcher | PhraseMatcher, replacement_token: str):
        self.matcher: Matcher | PhraseMatcher = matcher
        self.replacement_token: str = replacement_token

    def __call__(self, tokens) -> Doc:
        matches = self.matcher(tokens)
        spans = [tokens[start:end] for match_id, start, end in matches]

        with tokens.retokenize() as retokenizer:
            for span in spacy.util.filter_spans(spans):
                retokenizer.merge(span, attrs={"LEMMA": self.replacement_token})

        return tokens


class GameNamesMatcherReplacementRule(MatcherReplacementRuleOnLemma):
    def __init__(self, vocab, game_names: list):
        matcher = PhraseMatcher(vocab)
        matcher.add("<GAME_NAME>", game_names)
        super().__init__(matcher, "<GAME_NAME>")


class DateMatcherReplacementRule(MatcherReplacementRuleOnLemma):
    def __init__(self, vocab):
        matcher = Matcher(vocab)
        matcher.add("<DATE>", [[{"ENT_TYPE": "DATE"}, {"OP": "?"}]])
        super().__init__(matcher, "<DATE>")


class LemmatizeTextWithMatcherRules(LemmatizeTextRule):
    def __init__(self, nlp: spacy.language.Language | None = None, rules: list[MatcherReplacementRuleOnLemma] = None):
        super().__init__(nlp)
        self.rules = [] if rules is None else rules

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None  # We have to stop to avoid exception

        if entry is list:
            return list(itertools.chain(*[self.process(e) for e in entry]))

        tokens = self.nlp(entry)
        for rule in self.rules:
            tokens = rule(tokens)
        return [token.lemma_ for token in tokens if not self.is_invalid_token(token)]


class PreProcessingService:
    """
    It can be used as LoadDataUtility, but it won't be persisted. I should think well how to restructure this.
    Avoid it doing too much. I might pass it to a load corpus utility instead of inheriting from it. (Better idea) todo.
    """

    @staticmethod
    def full_pipeline(game_names: list, target_path: str, extensive_logging: bool = False):
        nlp = spacy.load('en_core_web_md')
        return PreProcessingService(
            [
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                CleanTextRule(r'(?i)\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                CleanTextRule("(?i)f::o::r::e::v::e::r::blank::k::e::e::p::e::r"),
                FilterLanguageRule(),
                LemmatizeTextWithMatcherRules(rules=[
                    GameNamesMatcherReplacementRule(nlp.vocab, game_names),
                    DateMatcherReplacementRule(nlp.vocab),
                ]),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            "full_pipeline",
            extensive_logging
        )

    @staticmethod
    def default_pipeline(target_path: str, extensive_logging: bool = False):
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\].*?\[/\1\]'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'(?i)\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                CleanTextRule("(?i)f::o::r::e::v::e::r::blank::k::e::e::p::e::r"),
                FilterLanguageRule(),
                LemmatizeTextRule(),
                ShortTextFilterRule(),
                ListToTextRegenerationRule()
            ],
            target_path,
            "default_pipeline",
            extensive_logging
        )

    def __init__(self, pipeline: list[ProcessingRule], target_path: str,
                 name: str, extensive_logging: bool = False):
        self.pipeline = pipeline
        self.extensive_logging = extensive_logging

        self.name = name

        self.target_path = target_path
        Path(self.target_path).mkdir(parents=True, exist_ok=True)

    def pre_process_dataset(self, target_dataset_size: int, dataset_sampler: ConsumingDatasetSampler):
        sampler = dataset_sampler.generator()
        current_dataset = DataFrame()

        while len(current_dataset) < target_dataset_size:
            batch = next(sampler)

            batch["original_text"] = batch["comments"]

            if len(current_dataset) > 0:
                batch = (current_dataset.merge(batch, how='outer', indicator=True)
                .query("_merge == 'right_only'")[batch.columns])

            if len(batch) == 0:
                continue  # No elements to work on we pass.

            batch["comments"] = batch["comments"].swifter.apply(self.pre_process)
            batch = batch.dropna(subset="comments")
            # Add the elements

            current_dataset = pd.concat([current_dataset, batch], ignore_index=True)

        return current_dataset

    def pre_process(self, entry: str) -> str | None:
        try:
            return reduce(lambda t, rule: rule.process(t, self.extensive_logging), self.pipeline, entry)

        except Exception as e:

            logging.error(e)
            # Something went wrong when processing the text, so we simply skip it
            logging.info(f"We had a problem processing the text {entry}")
            return None

    def pre_process_corpus(self, target_size: int, dataset_sampler: ConsumingDatasetSampler,
                           name: str, override: bool = False) -> str:
        if os.path.exists(f"{self.target_path}/{name}.csv") and not override:
            print("Procedure aborted as we are not allowed to override and the file already exists")
            return f"{self.target_path}/{name}.processed.csv"  # Abort the process.

        ds = self.pre_process_dataset(target_size, dataset_sampler)
        ds.to_csv(f"{self.target_path}/{name}.preprocessed.csv", mode="w", header=True, index=False)
        return f"{self.target_path}/{name}.preprocessed.csv"
