import gensim
import numpy as np
import pandas as pd
import spacy
from keras import preprocessing as pre
from torch.utils.data import Dataset
import swifter


class PositiveNegativeCommentGeneratorDataset(Dataset):
    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, csv_dataset_path: str, vocabulary: dict, negative_size: int, max_seq_length: int = 256):
        """

        @param csv_dataset_path:
        @param vocabulary:
        @param negative_size:
        @param max_seq_length: We set as a maximum 512 tokens for each sequence as it is a good standard.
        I would love to do
        """
        super(PositiveNegativeCommentGeneratorDataset).__init__()

        print("Loading spacy model.")
        self.nlp = spacy.blank("en")
        self.vocabulary: dict = vocabulary
        self.negative_size: int = negative_size

        self.max_seq_length: int = max_seq_length

        print(f"Loading dataset from file: {csv_dataset_path}")
        self.dataset = pd.read_csv(csv_dataset_path).dropna()

        print("Generating numeric representation for each word of ds.")

        self.text_ds = self.dataset["original_text"]
        self.id_ds = self.dataset["game_id"]

        self.dataset = self.dataset["comments"].swifter.apply(
            lambda x: self.generate_numeric_representation([token.text for token in self.nlp(x)])
        )
        print("Max sequence length calculation in progress...")
        # optimal_length = int(np.percentile(lengths, 95))  # 95th percentile. 512 is above that. (525/169973) ~ 0.3 %
        max_found_length = self.dataset.map(lambda x: len(x)).max()
        with_lost_information = self.dataset.map(lambda x: len(x) > max_seq_length).sum()
        print(f"We loose information on {with_lost_information} points."
              f"This is {with_lost_information / len(self.dataset) * 100}% of the dataset.")

        print(f"Padding sequences to max length ({max_seq_length}).")
        self.dataset = pd.Series(pre.sequence.pad_sequences(self.dataset, maxlen=max_seq_length).tolist())

        print("Max sequence length is: ", max_found_length, f" but we will limit sequences to {max_seq_length} tokens.")

    def get_text_item(self, index: int):
        return self.text_ds.at[index]

    def get_associated_game_id(self, index: int):
        return self.id_ds.at[index]

    def __getitem__(self, index: int):
        # For each input sentence, we randomly sample m sentences from our training data as negative samples
        # Stack to get rid of lists and create nice numpy arrays to be elaborated/\
        sample = np.array(self.dataset.at[index])
        negative_samples = np.stack(self.dataset.sample(n=self.negative_size).to_numpy())
        return [sample, negative_samples], 0

    def __len__(self):
        return len(self.dataset) - 1


class EmbeddingsDataset(Dataset):

    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, csv_dataset_path: str, embeddings_model: gensim.models.Word2Vec, max_seq_length: int = 256):
        super(EmbeddingsDataset).__init__()
        self.embeddings_model = embeddings_model
        self.nlp = spacy.blank("en")

        self.vocabulary = self.embeddings_model.wv.key_to_index

        print(f"Loading dataset from file: {csv_dataset_path}")
        self.dataset = pd.read_csv(csv_dataset_path).dropna()

        print("Generating numeric representation for each word of ds.")

        self.text_ds = self.dataset["original_text"]
        self.id_ds = self.dataset["game_id"]

        self.dataset = self.dataset["comments"].swifter.apply(
            lambda x: self.generate_numeric_representation([token.text for token in self.nlp(x)])
        )
        print("Max sequence length calculation in progress...")
        # optimal_length = int(np.percentile(lengths, 95))  # 95th percentile. 512 is above that. (525/169973) ~ 0.3 %
        max_found_length = self.dataset.map(lambda x: len(x)).max()
        with_lost_information = self.dataset.map(lambda x: len(x) > max_seq_length).sum()
        print(f"We loose information on {with_lost_information} points."
              f"This is {with_lost_information / len(self.dataset) * 100}% of the dataset.")

        print(f"Padding sequences to max length ({max_seq_length}).")
        self.dataset = pd.Series(pre.sequence.pad_sequences(self.dataset, maxlen=max_seq_length).tolist())

    def __getitem__(self, index):
        return np.array(self.dataset.at[index])

    def __len__(self):
        return len(self.dataset) - 1

    def get_full_dataset(self):
        return np.array([np.array(e) for e in self.dataset])
