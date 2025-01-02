import gc
import os
import uuid
from abc import abstractmethod
from random import Random

import torch
from torch.utils.data import ConcatDataset, Subset, DataLoader

from core.dataset import PositiveNegativeCommentGeneratorDataset
from core.train import AbaeModelManager, AbaeModelConfiguration


class TunableParameter:
    @abstractmethod
    def get_value(self, not_previous: bool = False):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)


class RandomTunableSteppedParameter(TunableParameter):
    def __init__(self, min_value: int | float, max_value: int | float, step: int | float, seed: int = 12):
        self.min_value = min_value
        self.max_value = max_value

        self.step = step

        self.random_generator = Random(seed)
        self.step_max_gen = int((self.max_value - self.min_value) / self.step)
        self.generated_values = []

    def get_value(self, not_previous: bool = False):
        value = self.min_value + self.step * self.random_generator.randint(0, self.step_max_gen)
        self.generated_values.append(value)
        return value


class RandomTunableDiscreteParameter(TunableParameter):
    def __init__(self, values_list: list, seed: int = 12):
        self.values_list = values_list
        self.returned_value_indexes = []

        self.random_generator = Random(seed)

    def get_value(self, not_previous: bool = False, seed: int = 12):
        index = self.random_generator.randint(0, len(self.values_list) - 1)
        self.returned_value_indexes.append(index)
        return self.values_list[index]


class ABAEHyperparametersWrapper:
    def __init__(self):
        self.parameters = dict(
            aspect_size=RandomTunableSteppedParameter(14, 20, 1),
            embedding_size=RandomTunableDiscreteParameter([64, 128, 192]),
            epochs=RandomTunableDiscreteParameter([5, 7, 10, 14, 20]),
            batch_size=RandomTunableDiscreteParameter([32, 64, 128])
        )

    def __getitem__(self, item: str):
        return self.parameters[item]

    def __next__(self):
        embedding_size = self.parameters["embedding_size"]()
        return dict(
            aspect_size=self.parameters["aspect_size"](),
            embedding_size=embedding_size,
            aspect_embedding_size=embedding_size,
            epochs=self.parameters["epochs"](),
            batch_size=self.parameters["batch_size"](),
        )


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


if __name__ == "__main__":
    # os.environ['KERAS_BACKEND'] = "torch"
    ## Parameters scouting. We scout on our main dataset.
    corpus_file = "../data/processed-dataset/full/64k.preprocessed.csv"

    # Using the whole dataset ensures representativeness and better alignment with real-world performance but can be computationally expensive.
    # A common approach is to start with subsets for initial tuning and refine on the full dataset when computationally feasible.

    hp = ABAEHyperparametersWrapper()
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
        )

        manager = AbaeModelManager(config)
        train_dataset = PositiveNegativeCommentGeneratorDataset(
            vocabulary=manager.embedding_model.vocabulary(),
            csv_dataset_path=config.corpus_file, negative_size=15
        )

        k_fold.load_data(train_dataset)
        k_fold.run_k_fold_cv(manager, parameters["batch_size"], parameters["epochs"])
