from abc import abstractmethod
from typing import Generator

import pandas as pd
from pandas import DataFrame


class ConsumingDatasetSampler:
    def __init__(self, batch_size: int, corpus: str, rand_state: int = 42):
        self.batch_size: int = batch_size
        self.rand_state: int = rand_state
        # Of course the corpus has to exist else this raises an exception
        self.ds = pd.read_csv(corpus)

    @abstractmethod
    def apply_sample_rule(self, ds: pd.DataFrame):
        pass

    def generator(self) -> Generator[DataFrame, None, None]:
        while len(self.ds) > 0:
            # Return a new batch if you can.
            rows = self.apply_sample_rule(self.ds)
            # I have to remove elements to avoid re-sampling them.
            self.ds.drop(rows.index, inplace=True)
            # Return the sampled rows.
            yield rows


class BggDatasetRandomBalancedSampler(ConsumingDatasetSampler):
    def apply_sample_rule(self, ds: pd.DataFrame):
        grouped_dataset = ds.groupby(["game_id"], group_keys=False)
        reviews_per_game = int(self.batch_size / len(grouped_dataset.count())) + 1

        print(f"I have a total of {len(grouped_dataset.count())} games with reviews. "
              f"We take {reviews_per_game} reviews per game.")

        return (
            grouped_dataset[ds.columns]
            .apply(lambda x: x.sample(min(len(x), reviews_per_game), random_state=self.rand_state))
        )
