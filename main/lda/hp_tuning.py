import json
from pathlib import Path
from uuid import uuid4
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from pandas import DataFrame

from main.hp_tuning import HyperparametersConfigGenerator, TuningProcedure
from main.lda.model import LdaModelGenerator
from main.lda.config import LdaGeneratorConfig


class LDATuningProcedure(TuningProcedure):
    def __init__(self, generator: HyperparametersConfigGenerator, top: list[int], folds: int = 5):
        super().__init__(generator)
        self.folds = folds
        self.top: list = top if top is not None else []
        self.results: list = []  # Where results of runs are stored with associated config

    def run(self, data: DataFrame, configurations: int, custom_stopwords: list = None):
        self.results = []
        folds = np.array_split(data, self.folds)

        # Configurations to see is max_iterations
        for i in range(configurations):
            config = next(self.generator)
            if config is None:
                print("No other configurations are available. Create a new procedure with updated confgiurations")
                break  # We cannot proceed if the generator cant generate any more elements

            i_results = dict(
                config=config, cv_coh={t: [] for t in self.top}, npmi_coh={t: [] for t in self.top}, perplexity=[]
            )
            print(f"Working on configuration: {config}")
            for k in range(self.folds):
                run_id = uuid4()
                validation_split: DataFrame = folds[k]  # On what to compute the validation metrics
                train = pd.concat([folds[index] for index in range(len(folds)) if index != k])
                print(f"Running fold = {k}")
                lda_config = LdaGeneratorConfig.from_configuration(str(run_id), config)
                model, dictionary = LdaModelGenerator(lda_config).make_model(train)
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


def make_plot_topics_selection_results(results_file_path: str, text: str) -> tuple[go.Figure, DataFrame]:
    # Refine the data so that plotting is possible
    data = pd.DataFrame(json.load(open(results_file_path)))
    data['topics'] = data['config'].map(lambda o: o['topics'])
    data['perplexity'] = data['perplexity'].map(lambda x: np.mean(x))
    for i in [3, 10, 25]:
        data[f'{i}_npmi_coh'] = data['npmi_coh'].map(lambda x: np.mean(x[str(i)]))
        data[f'{i}_cv_coh'] = data['cv_coh'].map(lambda x: np.mean(x[str(i)]))
    data = data.drop(columns=['config', 'npmi_coh', 'cv_coh'])

    # Make plot of the data
    fig = go.Figure()

    data = data.sort_values(by="topics")
    fig.add_trace(go.Scatter(x=data['topics'], y=data['3_cv_coh'], mode='lines', name='top-3'))
    fig.add_trace(go.Scatter(x=data['topics'], y=data['10_cv_coh'], mode='lines', name='top-10'))
    fig.add_trace(go.Scatter(x=data['topics'], y=data['25_cv_coh'], mode='lines', name='top-25'))

    fig.add_trace(
        go.Scatter(x=data['topics'], y=data['3_npmi_coh'], mode='lines', name='top-3', line=dict(dash='dash'))
    )
    fig.add_trace(
        go.Scatter(x=data['topics'], y=data['10_npmi_coh'], mode='lines', name='top-10', line=dict(dash='dash'))
    )
    fig.add_trace(
        go.Scatter(x=data['topics'], y=data['25_npmi_coh'], mode='lines', name='top-25', line=dict(dash='dash'))
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(title=dict(text=text), xaxis=dict(title=dict(text='Model topics K')),
                      yaxis=dict(title=dict(text='CV coherence')))
    return fig, data
