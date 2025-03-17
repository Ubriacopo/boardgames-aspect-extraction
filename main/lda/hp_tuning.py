import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame

from main.hp_tuning import HyperparametersConfigGenerator, TuningProcedure
from main.lda.config import LdaGeneratorConfig
from main.lda.model_manager import LDAManager


class LDATuningProcedure(TuningProcedure):
    def __init__(self, generator: HyperparametersConfigGenerator, top: list[int], file_path: str, folds: int = 5):
        super().__init__(generator)
        self.folds = folds
        self.top: list = top if top is not None else []

        self.file_path = file_path
        self.results: list = json.load(open(file_path)) if Path(file_path).is_file() else []

    def run(self, data: DataFrame, configurations: int, custom_stopwords: list = None):
        self.results = []
        folds = np.array_split(data, self.folds)

        # Configurations to see is max_iterations
        for i in range(configurations):
            config = next(self.generator)
            if config is None:
                print("No other configurations are available. Create a new procedure with updated confgiurations")
                break  # We cannot proceed if the generator cant generate any more elements

            config_results = dict(config=config, coherence=[], perplexity=[], coherence_type='u_mass', top=self.top)
            print(f"Working on configuration: {config}")
            for k in range(self.folds):
                run_id = uuid4()
                validation_split: DataFrame = folds[k]  # On what to compute the validation metrics
                train = pd.concat([folds[index] for index in range(len(folds)) if index != k])
                print(f"Running fold = {k}")
                lda_manager = LDAManager.from_scratch(LdaGeneratorConfig.from_configuration(str(run_id), config))

                lda_manager.get_model(train)
                print("Model generation over, evaluating...")
                validation_dataset = validation_split['comments'].apply(lambda x: x.split(' '))
                results = lda_manager.evaluate(validation_dataset, topn=self.top)
                # We run on folds so we add all the results.
                # Coherence is sorted by top so we have top order repeat for k iterations
                config_results['coherence'].append(results['coherence'])
                config_results['perplexity'].append(results['perplexity'])

            self.results.append(config_results)
            json.dump(self.results, open(self.file_path, 'w'))
        # Generated results are returned
        return self.results



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
