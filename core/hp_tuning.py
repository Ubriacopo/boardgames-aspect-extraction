from __future__ import annotations

import json
from abc import abstractmethod, ABC
from pathlib import Path
from random import Random
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.dataset import PositiveNegativeCommentGeneratorDataset
from core.evaluation import normalize, get_aspect_top_k_words, coherence_per_aspect
from core.train import AbaeModelConfiguration, AbaeModelManager

"""
    While the need to tune the parameters is a problem in real work case scenario it is very time consuming. 
    I will test out some configurations but won't dive too deep into parameter tuning and just use some good enough
    values that we found based on other tuning process.
"""


class TunableParameter:
    @abstractmethod
    def get_value(self, not_previous: bool = False):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)


class RandomTunableParameter(TunableParameter, ABC):
    def __init__(self, seed: int):
        self.random_generator = Random(seed)


class RandomTunableSteppedParameter(RandomTunableParameter):
    def __init__(self, min_value: int | float, max_value: int | float, step: int | float,
                 seed: int, multiply: bool = False):
        """
        Returns a random value in the given range with a step equal to a multiple of the step given.
        @param min_value: The minimum value.
        @param max_value: The maximum value.
        @param step: The smallest step to perform.
        @param seed: Seed to randomly select one element.
        """
        super().__init__(seed)

        # Range of allowed values
        self.min_value = min_value
        self.max_value = max_value

        # Smallest difference between values
        self.step = step

        self.multiply = multiply

        # The highest number of steps we can perform
        self.step_max_gen = int((self.max_value - self.min_value) / self.step)

    def get_value(self, not_previous: bool = False):
        if self.multiply:
            return_value = self.min_value * (max(self.random_generator.randint(0, self.step_max_gen) * self.step, 1))
            return return_value if return_value <= self.max_value else self.max_value

        return self.min_value + self.step * self.random_generator.randint(0, self.step_max_gen)


class RandomTunableDiscreteParameter(RandomTunableParameter):
    def __init__(self, values_list: list, seed: int):
        """
        We get a list of possible values from which to pick an element.
        @param values_list: The possible values.
        @param seed: Seed to randomly select one element.
        """
        super().__init__(seed)
        self.values_list = values_list

    def get_value(self, not_previous: bool = False, seed: int = 12):
        return self.values_list[self.random_generator.randint(0, len(self.values_list) - 1)]


class ABAERandomHyperparametersSelectionWrapper:
    @staticmethod
    def create(seed: int | Random = 12) -> ABAERandomHyperparametersSelectionWrapper:
        max_rand_int = 694
        return (
            ABAERandomHyperparametersSelectionWrapper()
            # We try different values. A good rule of thumb I found is: (2 * aspects + 2).
            .__add_parameter("aspect_size", RandomTunableSteppedParameter(
                min_value=12, max_value=20, step=2, seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            .__add_parameter('embedding_size', RandomTunableDiscreteParameter(
                [100, 150, 200, 300], seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            # Lower learning rates won't be favoured. But this makes sense, I want my training procedure to be fast.
            .__add_parameter("epochs", RandomTunableDiscreteParameter(
                [8, 10, 12, 15], seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            .__add_parameter("batch_size", RandomTunableDiscreteParameter(
                [32, 64, 128], seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            # A good heuristic for tuning learning rates is to use a logarithmic scale rather than a linear step between
            # the minimum and maximum values. Thus, we use a stepped learning rate selector.
            # .__add_parameter("learning_rate", RandomTunableSteppedParameter(
            #     0.001, 0.1, 5, multiply=True, seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            # ))
            .__add_parameter("learning_rate", RandomTunableDiscreteParameter(
                np.logspace(-4, -1, 5).tolist(), seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            # We want to use a learning rate scheduler.
            .__add_parameter("decay_rate", RandomTunableSteppedParameter(
                0.9, 0.99, 0.015, seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            .__add_parameter("momentum", RandomTunableSteppedParameter(
                0.90, 0.99, 0.01, seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
            .__add_parameter("negative_sample_size", RandomTunableDiscreteParameter(
                [10, 15, 20], seed=seed if type(seed) is int else seed.randint(0, max_rand_int)
            ))
        )

    def __init__(self):
        self.parameters = dict()

    def __add_parameter(self, name: str, parameter: TunableParameter) -> ABAERandomHyperparametersSelectionWrapper:
        self.parameters[name] = parameter
        return self

    def __getitem__(self, item: str):
        return self.parameters[item]

    def __next__(self):
        return_dictionary = dict()

        for key in self.parameters:
            return_dictionary[key] = self.parameters[key]()

        return return_dictionary


class HyperparameterTuningManager:
    def __init__(self, hyperparameters: ABAERandomHyperparametersSelectionWrapper, corpus_file: str,
                 # In case someone wants to load an existing set of configurations to avoid
                 seen_configurations_path: str, seen_configurations_filename: str = None):
        self.hyperparameters = hyperparameters

        # Path to the corpus considered for the tuning process
        self.corpus_file = corpus_file
        # Has to be lower than 50. For now, it is fixed
        self.top_n = 10

        self.configurations_path = f"{seen_configurations_path}/seen_configurations.json"
        # Create the folder if it does not exist
        Path(seen_configurations_path).mkdir(parents=True, exist_ok=True)

        # Should I make a class to manage configurations?
        self.seen_configurations = set()
        if seen_configurations_filename is not None:
            self.configurations_path = f"{seen_configurations_path}/{seen_configurations_filename}.json"

        self.__load_previous_configurations()

    def __call__(self, different_configurations: int, repeat: int):
        tuning_process_result_list = []

        for i in range(different_configurations):
            parameters = next(self.hyperparameters)

            while self.seen_configurations.__contains__(frozenset(parameters.items())):
                print(f"We already worked on configuration: {parameters}")
                parameters = next(self.hyperparameters)  # In case we fetch the same config more than once.

            print(f"Working on configuration: {parameters}")
            self.seen_configurations.add(frozenset(parameters.items()))
            tuning_process_result_list.append(self.__run_config(parameters, repeat))

        self.__store_seen_configurations()
        return tuning_process_result_list

    def __load_previous_configurations(self):
        # If previous configs exist we read them.
        if Path(self.configurations_path).is_file():
            print(f"Loading previous configurations from: {self.configurations_path}")
            obj = json.load(open(self.configurations_path))

            for configuration in obj:
                self.seen_configurations.add(frozenset(configuration.items()))

    def __run_config(self, parameters: dict, repeat: int) -> dict:
        # Prepare the ABAE configuration
        config = AbaeModelConfiguration(corpus_file=self.corpus_file, model_name=f"tuning_{uuid4()}", **parameters)

        manager = AbaeModelManager(config)
        vocab = manager.embedding_model.vocabulary()

        ds = PositiveNegativeCommentGeneratorDataset(
            config.corpus_file, vocabulary=vocab, negative_size=config.negative_sample_size
        )

        train, validation = torch.utils.data.random_split(ds, [0.75, 0.25], generator=torch.Generator().manual_seed(42))
        test_dataloader = DataLoader(dataset=validation, batch_size=config.batch_size, shuffle=True)

        model_results = dict(evaluation_loss=[], scores=[], aspect_coherence=[], coherence=[], params=parameters)

        # The embeddings model won't be overridden only our ABAE.
        for run in range(repeat):
            print(f"Training process in progress.. ({run + 1}/{repeat})")
            _, iteration_model = manager.run_train_process(train)

            print("Evaluating the model")
            evaluation_results = iteration_model.evaluate(test_dataloader)

            # Evaluate scores:
            word_emb = normalize(iteration_model.get_layer('word_embedding').weights[0].value.data)
            aspect_embeddings = normalize(iteration_model.get_layer('aspect_embedding').w)
            inv_vocab = manager.embedding_model.model.wv.index_to_key

            aspects_top_k_words = [get_aspect_top_k_words(a, word_emb, inv_vocab, top_k=50) for a in aspect_embeddings]
            aspect_words = [[word[0] for word in aspect] for aspect in aspects_top_k_words]  # Remap

            coherence, coherence_model = coherence_per_aspect(
                aspects=aspect_words, text_dataset=ds.text_ds.loc[validation.indices], topn=self.top_n
            )

            # Data to track for our trial. It will be averaged.
            model_results['evaluation_loss'].append(evaluation_results)
            model_results['scores'].append(evaluation_results)
            model_results['aspect_coherence'].append(coherence)
            model_results['coherence'].append(coherence_model.get_coherence())

        json.dump(model_results, open(f"{manager.output_path}/run_results.json", "w"))
        return dict(coherence=np.mean(model_results['scores']), loss=np.mean(model_results['evaluation_loss']))

    def __store_seen_configurations(self):
        write_object = []
        for config in self.seen_configurations:
            # Frozen sets are not JSON Serializable, dictionaries are
            write_object.append(dict(config))
        # Persist the configurations to the file path.
        json.dump(write_object, open(self.configurations_path, "w"))
