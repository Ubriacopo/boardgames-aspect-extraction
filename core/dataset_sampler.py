from abc import abstractmethod
from typing import Generator

import pandas as pd
from pandas import DataFrame


class ConsumingDatasetSampler:
    def __init__(self, batch_size: int, corpus_file_path: str, random_state: int = 42):
        self.batch_size: int = batch_size
        self.random_state = random_state

        # Of course the corpus has to exist else this raises an exception
        self.full_dataset = pd.read_csv(corpus_file_path)

    # Once consumed the sampler has to be thrown to the trash
    def generator(self) -> Generator[DataFrame, None, None]:
        # This is a Python generator.
        while len(self.full_dataset) > 0:
            # Return a new batch if you can.
            rows = self.apply_sample_rule(self.full_dataset)
            # I have to remove elements to avoid re-sampling them.
            self.full_dataset.drop(rows.index, inplace=True)
            # Return the sampled rows.
            yield rows

    @abstractmethod
    def apply_sample_rule(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass


class BggDatasetRandomBalancedSampler(ConsumingDatasetSampler):
    """
    It's random because we sample randomly.
    It's balanced because we try to keep the same amount of reviews per game.
    For each game at least one review is pulled, if possible, up to sample_size/len(ds) + 1
    """

    def apply_sample_rule(self, dataset: pd.DataFrame) -> pd.DataFrame:
        grouped_dataset = dataset.groupby(["game_id"], group_keys=False)
        reviews_per_game = int(self.batch_size / len(grouped_dataset.count())) + 1

        print(f"I have a total of {len(grouped_dataset.count())} games with reviews. "
              f"We take {reviews_per_game} reviews per game.")

        return grouped_dataset[dataset.columns].apply(
            lambda x: x.sample(min(len(x), reviews_per_game), random_state=self.random_state)
        )


class BggDatasetLongestSampler(ConsumingDatasetSampler):
    """
    Compared to the random sampler we cannot assure that we keep track of reviews of any kind of game.
    I expect more complex games to receive longer reviews and therefore my network to better
    learn to recognize that exact topi in a review.
    """

    def __init__(self, batch_size: int, corpus_file_path: str, random_state: int = 42):
        super().__init__(batch_size, corpus_file_path, random_state)
        self.full_dataset = self.full_dataset.sort_values(by=["comments"], ascending=False, key=lambda x: x.str.len())

    def apply_sample_rule(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset[0: self.batch_size]