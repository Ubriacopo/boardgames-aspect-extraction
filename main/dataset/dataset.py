import numpy as np
import pandas as pd

from pandas import DataFrame
from torch.utils.data import Dataset

import swifter


class BaseBoardgameDataset(Dataset):
    def encode(self, entry: list):
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, dataset: DataFrame | list | str, vocabulary: dict):
        # This dataset is not padded
        super().__init__()
        self.vocabulary = vocabulary
        self.dataset: DataFrame = self.process_dataset(dataset)

    def process_dataset(self, dataset: DataFrame | list | str):
        if type(dataset) == str:
            dataset = pd.read_csv(dataset)
        elif type(dataset) == list:
            dataset = DataFrame({"comments": dataset})
        print("Generating numeric representation for each word of ds.")
        return dataset['comments'].swifter.apply(lambda x: self.encode(x.split(' ')))

    def __getitem__(self, item):
        return np.array(self.dataset[item]), 0  # Return an assigned label. Unsupervised so we have None.

    def __len__(self):
        return len(self.dataset) - 1
