import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from pandas import DataFrame

from main.hp_tuning import HyperparametersConfigGenerator, TuningProcedure
from main.lda.model import LdaGeneratorConfig, LdaModelGenerator


class LDATuningProcedure(TuningProcedure):
    def __init__(self, generator: HyperparametersConfigGenerator, top: list[int], folds: int = 5):
        super().__init__(generator)
        self.folds = folds
        self.top: list = top if top is not None else []
        self.results: list = []  # Where results of runs are stored with associated config

    def run(self, data: DataFrame, configurations: int):
        self.results = []
        folds = np.array_split(data, self.folds)

        # Configurations to see is max_iterations
        for i in range(configurations):
            config = next(self.generator)
            if config is None:
                print("No other configurations are available. Create a new procedure with updated confgiurations")
                break  # We cannot proceed if the generator cant generate any more elements

            i_results = dict(config=config, cv_coh={t: [] for t in self.top}, npmi_coh={t: [] for t in self.top},
                             perplexity=[])

            for k in range(self.folds):
                validation_split: DataFrame = folds[k]  # On what to compute the validation metrics
                train = pd.concat([folds[index] for index in range(len(folds)) if index != k])
                print(f"Running fold = {k}")
                model, dictionary = LdaModelGenerator(LdaGeneratorConfig.from_configuration(config)).make_model(train)
                print("Model generation over, evaluating...")

                texts = validation_split['comments'].apply(lambda x: x.split(' '))
                perplexity = model.log_perplexity(texts.apply(lambda x: dictionary.doc2bow(x)).tolist())
                i_results['perplexity'].append(perplexity)

                for top in self.top:
                    cv_coh = CoherenceModel(model, texts=texts, coherence='c_v', topn=top)
                    npmi_coh = CoherenceModel(model, texts=texts, coherence='c_npmi', topn=top)
                    i_results['cv_coh'][top].append(cv_coh.get_coherence())
                    i_results['npmi_coh'][top].append(npmi_coh.get_coherence())

            self.results.append(i_results)

        # Generated results are returned
        return self.results

    def store_results(self, file_path: str):
        if Path(file_path).is_file():
            self.results = self.results + json.load(open(file_path))
        json.dump(self.results, open(file_path, 'w'))
