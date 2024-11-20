import numpy as np
import pandas as pd
import spacy
from keras import preprocessing as pre
from torch.utils.data import Dataset
import swifter


class PositiveNegativeCommentGeneratorDataset(Dataset):
    # todo give maxsequence len
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

    def get_text_item(self, index: int):
        pass

    def __getitem__(self, index: int):
        # For each input sentence, we randomly sample m sentences from our training data as negative samples
        # Stack to get rid of lists and create nice numpy arrays to be elaborated/\
        """
        TODO: Found error, embedding are empty for some rows
        """
        sample = np.array(self.dataset.at[index + 1])
        negative_samples = np.stack(self.dataset.sample(n=self.negative_size).to_numpy())
        return [sample, negative_samples], [0, 0]

    def __len__(self):
        # return len(self.dataset)
        return 999  # Testing the full cycle
