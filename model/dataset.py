import os
import numpy
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset
from keras import ops as K
from keras import preprocessing as pre
import numpy as np


class LazyCommentDataset(Dataset):
    nlp = spacy.blank("en")

    def __init__(self, csv_file_path: str):
        """
        This one is a lot slower than the CommentDataset, but it also has zero memory footprint.
        If we struggle with RAM we might consider to opt for this or explore better solutions.

        @param csv_file_path: The path where the file can be found
        """
        self.csv_file_path = csv_file_path
        self.len = len(pd.read_csv(self.csv_file_path, names=["comments"]).dropna())

    def __getitem__(self, index):
        # We skip the header from comments so index + 1
        read_line = (pd.read_csv(self.csv_file_path, skiprows=index + 1, nrows=1, names=["comments"]).dropna())
        return [token.text for token in self.nlp(read_line.at[0, "comments"])]

    def __len__(self):
        return self.len


class CommentDataset(Dataset):
    # To avoid overheads
    nlp = spacy.blank("en")
    """
    https://discuss.pytorch.org/t/tensordataset-with-lazy-loading/204191
    
    TensorDataset with lazy loading?
    Yes, but you can construct this huge file or split it in several big files for convenience. Alternatively, you also have hfd5 format that allows lazy loading and carrying metadata.

    Anyway, it seem your use case would be better solved with a custom dataset that loads each file on-the-fly.Have a look at:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html 24
    
    So the short answer is:
    If huge dataset / array, use hdf5 or memory map.
    If hundreds of small files, use a custom dataset.
    
    https://stackoverflow.com/questions/27717776/lazy-loading-csv-with-pandas
    """

    def __init__(self, vocabulary: dict, csv_dataset_path: str = ""):
        super(CommentDataset).__init__()
        self.vocabulary: dict = vocabulary
        # Is way faster than having to reload it at each iteration.
        self.dataset = pd.read_csv(csv_dataset_path, names=["comments"]).dropna()
        self.dataset = self.dataset["comments"].swifter.apply(lambda x: [token.text for token in self.nlp(x)])
        print(self.dataset.memory_usage(deep=True))
        print(self.dataset.info(memory_usage='deep'))

    def __getitem__(self, index):
        indexes = []
        # Map each word to the correct representation. If it does not exist 0 is returned as it is the <unk> key.
        zero = torch.tensor(0, dtype=torch.int16)
        for token in self.dataset.at[index + 1]:
            indexes.append(
                torch.tensor(self.vocabulary[token], dtype=torch.int16) if token in self.vocabulary else zero)
        return indexes

    def __len__(self):
        return len(self.dataset)

    def get_raw_sentence(self, index):
        return self.dataset.at[index + 1]


class PositiveNegativeCommentGeneratorDataset(Dataset):

    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist 0 is returned as it is the <unk> key.
        return np.array([self.vocabulary[token] if token in self.vocabulary else 0 for token in entry])

    def __init__(self, csv_dataset_path: str, vocabulary: dict, negative_size: int):
        super(PositiveNegativeCommentGeneratorDataset).__init__()
        self.nlp = spacy.blank("en")
        self.vocabulary: dict = vocabulary
        # Is way faster than having to reload it at each iteration.
        self.dataset = pd.read_csv(csv_dataset_path, names=["comments"]).dropna()
        self.dataset = self.dataset["comments"].head(1000).swifter.apply(
            lambda x: self.generate_numeric_representation([token.text for token in self.nlp(x)])
        )

        self.max_seq_length = self.dataset.map(lambda x: len(x)).max()
        self.dataset = pd.Series(pre.sequence.pad_sequences(self.dataset, maxlen=self.max_seq_length).tolist())
        self.negative_size = negative_size

    def __getitem__(self, index):
        # For each input sentence, we randomly sample m sentences from our training data as negative samples
        # Stack to get rid of lists and create nice numpy arrays to be elaborated
        return [np.array(self.dataset.at[index + 1]), np.stack(self.dataset.sample(n=self.negative_size).to_numpy())], 0

    def __len__(self):
        return len(self.dataset)
