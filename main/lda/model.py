import os

from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from pandas import DataFrame

from main.lda.config import LdaGeneratorConfig
from main.lda.dataset import LdaDataset


class LdaModelGenerator:
    def __init__(self, config: LdaGeneratorConfig, stop_words: list[str] = None):
        self.c: LdaGeneratorConfig = config
        self.stop_words: list[str] = stop_words if stop_words is not None else []

    def make_model(self, corpus: str | DataFrame, existing_path: str = None) -> LdaModel:
        try:
            if existing_path is not None:
                model = LdaModel.load(existing_path)
                return model
        except FileNotFoundError:
            print("Model not found. Making a new one.")

        ds = LdaDataset(corpus, self.stop_words)
        return LdaMulticore(
            corpus=ds.dataset, id2word=ds.dict, num_topics=self.c.topics, alpha=self.c.alpha,
            eta=self.c.eta, passes=self.c.passes, random_state=self.c.random_state
        )


# todo
class LdaClassifier:
    def __init__(self, mapped_labels: list[str], model: LdaDataset):
        self.mapped_labels = mapped_labels
        self.model = model

    def classify(self, corpus):
        return self.model.process_dataset(corpus)
