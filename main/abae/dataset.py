import numpy as np
import pandas as pd
from pandas import DataFrame

from keras import preprocessing as pre

from main.dataset.dataset import BaseBoardgameDataset


# I could just name it "PaddedSequencesDataset"
class ABAEDataset(BaseBoardgameDataset):
    def __init__(self, dataset: DataFrame | list | str, vocabulary: dict, max_seq_length: int, use_lowest_pad: bool = False):
        self.max_seq_length = max_seq_length

        # If this flag is true the padding will go to the min between (max_seq_length, max(s)) where
        # max(s) is the longest found string (divided in words) we found in our documents.
        self.use_lowest_pad = use_lowest_pad
        super().__init__(dataset, vocabulary)

    def process_dataset(self, dataset: DataFrame | list):
        temp_ds = super().process_dataset(dataset)
        # ABAE takes padded sequences only.

        print("Max sequence length calculation in progress...")
        padding_size = self.max_seq_length

        max_found_length = temp_ds.map(lambda x: len(x)).max()
        with_lost_information = temp_ds.map(lambda x: len(x) > self.max_seq_length).sum()

        with_lost_information > 0 and print(
            f"We loose information on {with_lost_information}({with_lost_information / len(temp_ds) * 100}% of ds)."
        )

        if self.use_lowest_pad:
            padding_size = min(max_found_length, self.max_seq_length)

        return pd.Series(pre.sequence.pad_sequences(temp_ds, maxlen=padding_size).tolist())


class PositiveNegativeABAEDataset(ABAEDataset):
    def __init__(self, dataset: DataFrame | list | str, vocabulary: dict, max_seq_length: int,
                 negative_size: int, use_lowest_pad: bool = False):
        self.negative_size = negative_size
        super().__init__(dataset, vocabulary, max_seq_length, use_lowest_pad)

    def __getitem__(self, index):
        sample = np.array(self.dataset.at[index])
        negative_samples = self.dataset.sample(n=self.negative_size + 1)

        # We drop the current element index if it was sampled.
        negative_samples = negative_samples.drop(index=index, errors='ignore')
        # Now we might have one negative sample too much, we omit the last one:
        negative_samples = np.stack(negative_samples.head(self.negative_size).to_numpy())
        return [sample, negative_samples], 0
