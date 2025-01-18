from __future__ import annotations

from abc import abstractmethod, ABC
from random import Random

import numpy as np

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
