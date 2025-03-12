import json
from abc import abstractmethod, ABC
from pathlib import Path
from random import Random

from pandas import DataFrame


class TunableParameter:
    @abstractmethod
    def get_value(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.get_value()


class RandomTunableParameter(TunableParameter, ABC):
    def __init__(self, seed: int):
        self.random_generator = Random(seed)


class RandomTunableOffsetParameter(RandomTunableParameter):
    def __init__(self, value_range: tuple[int | float, int | float], step: int | float, seed: int):
        """
        Returns a random value in the given range with a step equal to a multiple of the step given.
        @param value_range: Range of allowed values. For example: [3,5]
        @param step: The smallest step to perform.
        @param seed: Seed to randomly select one element.
        """
        super().__init__(seed)
        # Range of allowed values
        self.value_range = value_range
        # Smallest difference between values
        self.step = step
        self.generated_values = set()
        # The highest number of steps we can perform
        self.step_max_gen = int((value_range[1] - value_range[0]) / self.step)

    def get_value(self):
        val = self.value_range[0] + self.step * self.random_generator.randint(0, self.step_max_gen)
        self.generated_values.add(val)
        return val


class RandomTunableDiscreteParameter(RandomTunableParameter):
    def __init__(self, values_list: list, seed: int):
        """
        We get a list of possible values from which to pick an element.
        @param values_list: The possible values.
        @param seed: Seed to randomly select one element.
        """
        super().__init__(seed)
        self.values_list = values_list

    def get_value(self):
        return self.values_list[self.random_generator.randint(0, len(self.values_list) - 1)]


class HyperparametersConfigGenerator:
    def __init__(self):
        self.parameters = dict()

    def add_parameter(self, name: str, parameter: TunableParameter):
        self.parameters[name] = parameter

    def __getitem__(self, item: str):
        return self.parameters[item]

    def __next__(self):
        return {key: self.parameters[key]() for key in self.parameters}


class UniqueParametersConfigGenerator(HyperparametersConfigGenerator):
    def __init__(self, patience: int):
        super().__init__()
        # Seen configurations are not returned again
        self.seen_configurations = set()
        self.patience = patience

    def __next__(self):
        config = super().__next__()
        patience_load = self.patience

        while frozenset(config.items()) in self.seen_configurations and patience_load > 0:
            patience_load -= 1  # Decrease the local patience
            print(f"We already generated the configuration: {config}")

            config = super().__next__()

        if patience_load == 0 and frozenset(config.items()) in self.seen_configurations:
            return None  # No configuration is returned as we failed to generate a valid one

        self.seen_configurations.add(frozenset(config.items()))
        return config


class UniqueParametersConfigFsGenerator(UniqueParametersConfigGenerator):
    def __init__(self, patience: int, seen_configurations_path: str, seen_configurations_filename: str = None):
        super().__init__(patience)

        file_name = "seen_configurations.json" if seen_configurations_filename is None else seen_configurations_filename
        self.configurations_path = f"{seen_configurations_path}/{file_name}"
        Path(seen_configurations_path).mkdir(parents=True, exist_ok=True)

        self.__load_previous_configurations()

    def __next__(self):
        configuration = super().__next__()
        self.__store_seen_configurations()
        return configuration

    def __load_previous_configurations(self):
        if Path(self.configurations_path).is_file():
            print(f"Loading previous configurations from: {self.configurations_path}")
            obj = json.load(open(self.configurations_path))
            [self.seen_configurations.add(frozenset(configuration.items())) for configuration in obj]

    def __store_seen_configurations(self):
        # Frozen sets are not JSON Serializable, dictionaries are
        write_object = [dict(config) for config in self.seen_configurations]
        # Persist the configurations to the file path.
        json.dump(write_object, open(self.configurations_path, "w"))


class TuningProcedure(ABC):
    def __init__(self, generator: HyperparametersConfigGenerator):
        self.generator = generator

    @abstractmethod
    def run(self, data: DataFrame, configurations: int) -> list:
        pass
