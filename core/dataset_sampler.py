from abc import abstractmethod
from pathlib import Path

import pandas as pd


class DatasetSampler:
    def __init__(self, target_size: int | float, output_dir: str, random_state: int = 42):
        self.target_size: int | float = target_size
        self.output_dir = output_dir

        self.random_state = random_state

    @abstractmethod
    def apply_sample_rule(self, records: int, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    def make_sample_of_data(self, corpus_file: str, target_file: str):
        full_dataset = pd.read_csv(corpus_file)
        quantity = self.target_size

        if type(self.target_size) is float:
            if self.target_size > 1 or self.target_size < 0:
                raise ValueError("The target size must be a float between 0 and 1.")
            quantity = int(len(full_dataset) * self.target_size)

        new_dataset = self.apply_sample_rule(quantity, full_dataset)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Dataset with a total of {len(new_dataset)} rows has been generated.")
        print(f"Storing the dataset under: {self.output_dir}/{target_file}\n")
        new_dataset.to_csv(f"{self.output_dir}/{target_file}", mode="w", header=True, index=False)


class BggDatasetRandomBalancedSampler(DatasetSampler):
    """
    It's random because we sample randomly.
    It's balanced because we try to keep the same amount of reviews per game.
    """

    def apply_sample_rule(self, records: int, dataset: pd.DataFrame) -> pd.DataFrame:
        grouped_dataset = dataset.groupby(["game_id"], group_keys=False)
        reviews_per_game = int(records / len(grouped_dataset.count())) + 1

        print(f"I have a total of {len(grouped_dataset.count())} games with reviews. "
              f"We want to be ~{self.target_size} reviews so we take {reviews_per_game} reviews per game.")

        return grouped_dataset[dataset.columns].apply(
            lambda x: x.sample(min(len(x), reviews_per_game), random_state=self.random_state)
        )


class BggDatasetLongestSampler(DatasetSampler):
    """
    Compared to the random sampler we cannot assure that we keep track of reviews of any kind of game.
    I expect more complex games to receive longer reviews and therefore my network to better
    learn to recognize that exact topi in a review.
    """

    def apply_sample_rule(self, records: int, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.sort_values(by="comments", ascending=False, key=lambda x: x.str.len())[0: records]
