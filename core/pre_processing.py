import multiprocessing
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import itertools

from dateparser.search import search_dates
from pandas import DataFrame
from spacy.lang.en import English
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
    @abstractmethod
    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        """
        "Interface" on which we base processing steps.
        """
        pass

    def branches(self):
        """

        @return: True if the step does branching, and we have for a single input multiple output streams.
        """
        return False

    def __call__(self, entry: str | None | list, extensive_logging: bool = False):
        return self.process(entry, extensive_logging)


class SplitSentencesRule(ProcessingRule):
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None or type(entry) is not str:
            return entry

        document = self.nlp(entry)
        return [sent.text.strip() for sent in document.sents]

    def branches(self):
        return True


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
        """

        @param min_words_in_sentence: Number of words at least required for the sentence to not be filtered out.
        """
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

        if entry is list:
            return list(itertools.chain(*[self.process(e) for e in entry]))

        text_tokens = self.nlp(entry)
        return [token.lemma_.lower() for token in text_tokens if not self.is_invalid_token(token)]


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
        matcher.add("<DATE>", [[{"ENT_TYPE": "DATE", "OP": "+"}]])
        super().__init__(matcher, "<DATE>")


class NumberMatcherReplacementRule(MatcherReplacementRuleOnLemma):
    def __init__(self, vocab):
        matcher = Matcher(vocab)
        matcher.add("<NUM>", [[{"LIKE_NUM": True}]])
        super().__init__(matcher, "<NUM>")


class PlayerCountReplacementRule(MatcherReplacementRuleOnLemma):
    def __init__(self, vocab):
        """
        To match: 2p, 2-4p 2/4p.
        Should I also match 1-2 players?
        @param vocab:
        """
        matcher = Matcher(vocab)
        matcher.add("<PLAYER_NUM>", [
            [
                {"TEXT": {"REGEX": r"^[1-9]$"}, "OP": "?"},
                {"IS_PUNCT": True, "OP": "?"},
                {"TEXT": {"REGEX": r"^[1-9]p$"}, "OP": "{1}"}
            ],
            [
                {"TEXT": {"REGEX": r"^\d{1,2}/\d{1,2}p$"}, "OP": "{1}"},
            ]
        ])

        super().__init__(matcher, "<PLAYER_NUM>")


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
            # Apply each rule on the processed tokens to replace with some better tags.
            tokens = rule(tokens)

        return [token.lemma_.lower() for token in tokens if not self.is_invalid_token(token)]


class WordNoiseRemover(ProcessingRule):
    """
    I see a problem with this approach: We lemmatized before doing the "cleanup" on words.
    It should be done before, but it might be very slow!
    """

    def __init__(self, char_sequence: str = "-+."):
        """

        @param char_sequence: Characters considered noise at the start and/or end of a word
        """
        self.char_sequence = char_sequence

    def process(self, entry: str | None | list, extensive_logging: bool = False) -> str | None | list:
        if entry is None:
            return None

        if type(entry) is list:
            for i in range(len(entry)):
                stripped = entry[i].lstrip(self.char_sequence).rstrip(self.char_sequence)
                entry[i] = stripped if len(stripped) > 0 else entry[i]

            # Return the new built sentence.
            return " ".join(entry)

        # Split the text as we expect it to be words.
        return self.process(entry.split(" "))


class PreProcessingService:
    """
    It can be used as LoadDataUtility, but it won't be persisted. I should think well how to restructure this.
    Avoid it doing too much. I might pass it to a load corpus utility instead of inheriting from it. (Better idea) todo.
    """

    @staticmethod
    def full_pipeline(game_names: list, target_path: str, extensive_logging: bool = False):
        nlp = spacy.load('en_core_web_md')
        # Do we have any custom stopwords? I considered + and - BUT they might bring meaning.
        # nlp.Defaults.stop_words |= {"<DATE_TOKEN>", "<TIME_TOKEN>"}
        return PreProcessingService(
            [
                # To remove text like: [IMG]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/IMG]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\](.*?)\[/\1\]', r'\2'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'(?i)\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                # CleanTextRule("(?i)f::o::r::e::v::e::r::blank::k::e::e::p::e::r"),
                FilterLanguageRule(),
                SplitSentencesRule(),
                ShortTextFilterRule(min_words_in_sentence=10),
                WordNoiseRemover(),
                # DateRemoverRule(), takes too long
                LemmatizeTextWithMatcherRules(rules=[
                    GameNamesMatcherReplacementRule(nlp.vocab, game_names),
                    DateMatcherReplacementRule(nlp.vocab),
                    PlayerCountReplacementRule(nlp.vocab),
                    NumberMatcherReplacementRule(nlp.vocab),
                ]),
                # The longer the sentence the higher the context information that it should provide.
                ShortTextFilterRule(min_words_in_sentence=8),
                ListToTextRegenerationRule(),
                # DateRemoverRule(), This is way too slow!
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
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'(?i)\[(?P<tag>[A-Z]+)\](.*?)\[/\1\]', r'\2'),
                # To remove text like: [game=23232]https://cf.geekdo-static.com/mbs/mb_5855_0.gif[/game]
                # Keep tag content rule (in case the tag has an inner description)
                CleanTextRule(r'(?i)\[(?P<tag>[a-z]+)(=[^\]]+)?\](.*?)\[/\1\]', r'\3'),
                # CleanTextRule("(?i)f::o::r::e::v::e::r::blank::k::e::e::p::e::r"),
                FilterLanguageRule(),
                SplitSentencesRule(),
                ShortTextFilterRule(min_words_in_sentence=3),
                WordNoiseRemover(),
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
            # For now, I only handled 1 level of nested. I should generalize.
            batch = batch.explode('comments').reset_index(drop=True)

            batch = batch.dropna(subset="comments")

            # Even if they might derive from two different reviews equal sentences are redundant.
            batch = batch.drop_duplicates(subset=["comments"])

            # Add the elements
            current_dataset = pd.concat([current_dataset, batch], ignore_index=True)

        return current_dataset

    def pre_process(self, entry: str, pipeline_start_index: int = 0) -> str | None | list:
        try:
            tmp = entry

            for i in range(pipeline_start_index, len(self.pipeline)):

                current_step_rule = self.pipeline[i]
                tmp = current_step_rule.process(tmp)

                # If the process branches we have to return a list of processed branches.
                if current_step_rule.branches() and type(tmp) == list:
                    return [self.pre_process(s, i + 1) for s in tmp]

            return tmp

            # return reduce(lambda t, rule: rule.process(t, self.extensive_logging), self.pipeline, entry)

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


@dataclass
class DatasetGeneration:
    pipeline: PreProcessingService
    target_size: int
    sampler: ConsumingDatasetSampler

    def __iter__(self):
        # For a rapid unpacking of the object
        return iter((self.pipeline, self.target_size, self.sampler))


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")
    lm = LemmatizeTextWithMatcherRules(rules=[
        DateMatcherReplacementRule(nlp.vocab),
        PlayerCountReplacementRule(nlp.vocab),
    ])

    print(lm.process("This is a 4p game"))
    print(lm.process("This is a 2-4p game"))
    print(lm.process("This is a 21/4p game"))
    print(lm.process("This is a 2-3-4p game"))
