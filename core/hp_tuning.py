from __future__ import annotations
import gc
import os

import uuid
from abc import abstractmethod, ABC
from random import Random

import torch
from torch.utils.data import ConcatDataset, Subset, DataLoader

from core.dataset import PositiveNegativeCommentGeneratorDataset
from core.train import AbaeModelManager, AbaeModelConfiguration

"""
    I feel like this goes out of the scope of the project.
    The approach is what is to be considered and not the results.
    
    While the idea to tune the parameters is a real work case scenario it is very
    time consuming. I will leave this here but probably won't dive too deep into
    parameter tuning and just use some default good enough values based on other studies.
    
    I still have made the experience on  hp tuning at the Statistical Methods for ML course.
    todo: Decidi se procedere comunque.
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
    def __init__(self, min_value: int | float, max_value: int | float, step: int | float, seed: int = 12):
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

        # The highest number of steps we can perform
        self.step_max_gen = int((self.max_value - self.min_value) / self.step)

    def get_value(self, not_previous: bool = False):
        return self.min_value + self.step * self.random_generator.randint(0, self.step_max_gen)


class RandomTunableDiscreteParameter(RandomTunableParameter):
    def __init__(self, values_list: list, seed: int = 12):
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
    def create() -> ABAERandomHyperparametersSelectionWrapper:
        return (
            ABAERandomHyperparametersSelectionWrapper()
            .__add_parameter(RandomTunableSteppedParameter(14, 20, 1), 'aspect_size')
            .__add_parameter(RandomTunableDiscreteParameter([100, 150, 200]), 'embedding_size')
            .__add_parameter(RandomTunableDiscreteParameter([5, 10, 15, 20]), 'epochs')
            .__add_parameter(RandomTunableDiscreteParameter([32, 64, 128]), 'batch_size')
        )

    def __init__(self):
        self.parameters = dict()

    def __add_parameter(self, parameter: TunableParameter, name: str) -> ABAERandomHyperparametersSelectionWrapper:
        self.parameters[name] = parameter
        return self

    def __getitem__(self, item: str):
        return self.parameters[item]

    def __next__(self):
        return_dictionary = dict()

        for key in self.parameters:
            return_dictionary[key] = self.parameters[key]()

        return return_dictionary


class KFoldDatasetWrapper:
    def __init__(self, k: int):
        """
        The class handles K-Fold cross validation as we will use the same configuration for multiple models so
        why not implement a handy structure to make things easier?
        :param k: Number of folds. Cannot be changed but why not? It ain't hard
        """
        self.k: int = k

        self.dataset: [Subset] = []
        self.split_dataset: [Subset] = []

        self.fold_fraction: float = 0

    def load_data(self, dataset):
        self.dataset = dataset

        # This way we can use automatic round-robin procedure on random_split for the splitting if
        # the dataset cannot be entirely divided by the estimated fold size.
        self.fold_fraction = 1 / self.k
        self.split_dataset = torch.utils.data.random_split(dataset, [self.fold_fraction for i in range(self.k)])

    def get_data_for_fold(self, ignored_fold: int) -> tuple[ConcatDataset, Subset]:
        """
        :param ignored_fold:
        :return: train and test datasets with current k fold selected to be the validation one
        """
        return (ConcatDataset([x for i, x in enumerate(self.split_dataset) if i != ignored_fold]),
                self.split_dataset[ignored_fold])

    def run_k_fold_cv(self, model_generator: AbaeModelManager, batch_size: int = 32, epochs=10):
        """
        Runs k-fold CV and returns the tuple of [estimates, test_size].
        :param learning_parameters_template: Learning parameters that compile the model
        :param model_generator: A model generator to create a new instance each k iteration
        :param input_shape: The expected input shape is C x W x H
        :param batch_size: Batch size for training
        :return: the tuple of [estimates, test_size].
        """
        test_performances = []
        test_fold_sizes = []

        for i in range(self.k):
            print(f"Starting procedure for fold {i}")
            # To avoid going OOM
            torch.cuda.empty_cache()
            gc.collect()

            train, test = self.get_data_for_fold(i)
            train, validation = torch.utils.data.random_split(train, [0.875, 0.125])

            train_dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=True)

            iteration_model = model_generator.prepare_training_model()

            iteration_model.fit(train_dataloader, validation_data=validation_dataloader, epochs=epochs, verbose=1)
            test_dataloader = DataLoader(dataset=test, shuffle=True)

            # todo: Valutazioni su modello come coerenza e altro.
            test_performances.append(iteration_model.evaluate(test_dataloader, verbose=1))
            test_fold_sizes.append(len(test))

        return test_performances, test_fold_sizes


# todo move scripts
if __name__ == "__main__":
    # os.environ['KERAS_BACKEND'] = "torch"
    ## Parameters scouting. We scout on our main dataset.
    corpus_file = "../data/processed-dataset/full/64k.preprocessed.csv"

    # Using the whole dataset ensures representativeness and better alignment with real-world performance but can be computationally expensive.
    # A common approach is to start with subsets for initial tuning and refine on the full dataset when computationally feasible.

    hp = ABAERandomHyperparametersSelectionWrapper.create()
    k_fold = KFoldDatasetWrapper(k=5)

    n = 15  # Amount of different test configurations. todo pass args?
    random_process_id = uuid.uuid4()

    for i in range(n):
        parameters = next(hp)
        print(parameters)

        config = AbaeModelConfiguration(
            corpus_file=corpus_file, model_name=f"hp_{i}",
            output_path=f"./output/{random_process_id}",
            embedding_size=parameters["embedding_size"],
            aspect_embedding_size=parameters["embedding_size"],
            aspect_size=parameters["aspect_size"],
            epochs=parameters["epochs"],
            batch_size=parameters["batch_size"]
        )

        manager = AbaeModelManager(config)
        train_dataset = PositiveNegativeCommentGeneratorDataset(
            vocabulary=manager.embedding_model.vocabulary(),
            csv_dataset_path=config.corpus_file, negative_size=15
        )

        k_fold.load_data(train_dataset)
        k_fold.run_k_fold_cv(manager, config.batch_size, config.epochs)