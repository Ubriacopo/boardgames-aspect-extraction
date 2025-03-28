import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore, LdaModel, CoherenceModel
from pandas import DataFrame

from main.lda.config import LdaGeneratorConfig
from main.lda.model import LdaModelGenerator


class LDAManager:
    def __init__(self, config: LdaGeneratorConfig, model_generator: LdaModelGenerator):
        self.config = config
        self.generator = model_generator

        # Where everything will be eventually stored.
        self.considered_path = f"{self.config.output_path()}/{self.config.name}.model"
        self.__model: LdaModel | None = None

    @classmethod
    def from_config(cls, config: LdaGeneratorConfig, stop_words: list[str] = None):
        return cls(config, LdaModelGenerator(config, stop_words))

    def get_model(self, corpus_path: str | DataFrame = None, load_existing: bool = False, refresh: bool = True):
        if not refresh and self.__model is not None:
            # We want to get the current model directly not a new instance
            return self.__model

        path = self.considered_path if load_existing else None
        print(f"Generating a new compiled model from {'scratch' if path is None else 'fs'}")
        self.__model = self.generator.make_model(corpus_path, path)
        self.__model.save(self.considered_path)
        return self.__model

    def evaluate(self, test_corpus: str | pd.DataFrame, topn: list = None) -> dict:
        if self.__model is None:
            raise ValueError("To evaluate you have to first instance the model")

        if type(test_corpus) == str:
            test_corpus = pd.read_csv(test_corpus)['comments'].apply(lambda x: x.split(' '))

        results = dict(coherence=[], topn=[3, 5, 10, 20] if topn is None else topn)

        dictionary = self.__model.id2word
        results['perplexity'] = self.__model.log_perplexity(test_corpus.apply(lambda x: dictionary.doc2bow(x)).tolist())

        for topn in results['topn']:
            corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in test_corpus]
            model = CoherenceModel(self.__model, corpus=corpus, dictionary=dictionary, coherence='u_mass', topn=topn)
            results['coherence'].append(model.get_coherence())

        return results
