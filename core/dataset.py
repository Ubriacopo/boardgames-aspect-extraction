from abc import abstractmethod

import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import preprocessing as pre
from pandas import DataFrame
from torch.utils.data import Dataset
import swifter


class BaseBoardgameDataset(Dataset):
    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, dataset: DataFrame, vocabulary: dict, max_seq_length: int = 256):
        """

        @param csv_dataset_path:
        @param vocabulary:
        @param negative_size:
        @param max_seq_length: We set as a maximum 512 tokens for each sequence as it is a good standard.
        I would love to do
        """
        super().__init__()
        self.vocabulary: dict = vocabulary
        self.dataset = dataset

        print("Generating numeric representation for each word of ds.")
        self.original_review_ds = self.dataset["original_text"]
        self.text_ds = self.dataset["comments"]

        self.id_ds = self.dataset["game_id"]

        self.dataset = self.dataset["comments"].swifter.apply(
            lambda x: self.generate_numeric_representation(x.split(' '))
        )
        print("Max sequence length calculation in progress...")
        max_found_length = self.dataset.map(lambda x: len(x)).max()
        print("Max sequence length is: ", max_found_length, f". The limit is set to {max_seq_length} tokens.")
        # optimal_length = int(np.percentile(lengths, 95))  # 95th percentile. 512 is above that. (525/169973) ~ 0.3 %

        with_lost_information = self.dataset.map(lambda x: len(x) > max_seq_length).sum()

        # To what size we have to pad our sequences:
        padding_size = np.min([max_found_length, max_seq_length])

        if with_lost_information > 0:
            print(f"We loose information on {with_lost_information} points."
                  f"This is {with_lost_information / len(self.dataset) * 100}% of the dataset.")

        print(f"Padding sequences to length ({padding_size}).")
        self.dataset = pd.Series(pre.sequence.pad_sequences(self.dataset, maxlen=padding_size).tolist())

    def get_text_sentence(self, index: int):
        return self.text_ds.at[index]  # Returns the review this sentence belongs to.

    def get_review_by_index(self, index: int):
        return self.original_review_ds.at[index]

    def get_associated_game_id(self, index: int):
        return self.id_ds.at[index]

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.dataset) - 1


class PositiveNegativeCommentGeneratorDataset(BaseBoardgameDataset):
    def __init__(self, csv_dataset_path: str, vocabulary: dict, negative_size: int, max_seq_length: int = 80):
        print(f"Loading dataset from file: {csv_dataset_path}")
        dataset = pd.read_csv(csv_dataset_path)
        self.negative_size = negative_size
        super().__init__(dataset, vocabulary, max_seq_length)

    def __getitem__(self, index: int):
        # For each input sentence, we randomly sample m sentences from our training data as negative samples
        # Stack to get rid of lists and create nice numpy arrays to be elaborated/\
        sample = np.array(self.dataset.at[index])
        negative_samples = np.stack(self.dataset.sample(n=self.negative_size).to_numpy())
        return [sample, negative_samples], 0


class EmbeddingsDataset(BaseBoardgameDataset):
    def __init__(self, csv_dataset_path: str, embeddings_model: gensim.models.Word2Vec, max_seq_length: int = 80):
        dataset = pd.read_csv(csv_dataset_path)
        self.embeddings_model = embeddings_model
        super().__init__(dataset, embeddings_model.wv.key_to_index, max_seq_length)

    def __getitem__(self, index: int):
        return np.array(self.dataset.at[index])


class TokenizedDataset(BaseBoardgameDataset):
    def __init__(self, csv_dataset_path: str, vocabulary: dict, max_seq_length: int = 80):
        dataset = pd.read_csv(csv_dataset_path)
        super().__init__(dataset, vocabulary, max_seq_length)

    def __getitem__(self, index: int):
        return np.array(self.dataset.at[index])


# Label-less ds
class SimpleWord2VecEmbeddingsDataset(Dataset):
    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, corpus: list, embeddings_model: Word2Vec, max_seq_length: int = 80):
        self.vocabulary = embeddings_model.wv.key_to_index

        ds = map(lambda x: self.generate_numeric_representation(x.split(' ')), corpus)
        max_found_length = len(max(ds, key=len))

        self.dataset = pre.sequence.pad_sequences(ds, maxlen=min(max_found_length, max_seq_length)).tolist()
        self.dataset = list(ds)

    def __getitem__(self, index: int):
        # Output: Point + Label
        return np.array(self.dataset[index]), 0

    def __len__(self):
        return len(self.dataset) - 1


class PositiveNegativeWord2VecEmbeddingsDataset(SimpleWord2VecEmbeddingsDataset):
    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, corpus: list, embeddings_model: Word2Vec, negative_size: int, max_seq_length: int = 80):
        self.negative_size = negative_size
        super().__init__(corpus, embeddings_model, max_seq_length)

    def __getitem__(self, item):
        # todo: Vedi se è lento.
        positive_sample = self.dataset[item]
        indexes = filter(lambda x: x != item, np.random.choice(self.__len__(), self.negative_size + 1, replace=False))

        negative_samples = [self.dataset[index] for index in list(indexes)[0:self.negative_size]]
        return [np.array(positive_sample), np.array(negative_samples)], 0


class PandasNumericTextDataset(Dataset):
    def generate_numeric_representation(self, entry):
        # Map each word to the correct representation. If it does not exist <UNK> value is returned.
        return np.array([self.vocabulary[t] if t in self.vocabulary else self.vocabulary['<UNK>'] for t in entry])

    def __init__(self, dataframe: DataFrame, vocabulary: dict, max_seq_length: int = 80):
        self.vocabulary = vocabulary
        self.ds = dataframe["comments"].switfter.appl(lambda x: self.generate_numeric_representation(x.split(' ')))

        print("Max sequence length calculation in progress...")
        max_found_length = self.ds.map(lambda x: len(x)).max()
        print("Max sequence length is: ", max_found_length, f". The limit is set to {max_seq_length} tokens.")
        with_lost_information = self.ds.map(lambda x: len(x) > max_seq_length).sum()

        padding_size = min(max_found_length, max_seq_length)

        if with_lost_information > 0:
            print(f"We loose information on {with_lost_information} points."
                  f"This is {with_lost_information / len(self.ds) * 100}% of the dataset.")
        print(f"Padding sequences to length ({padding_size}).")
        self.ds = pd.Series(pre.sequence.pad_sequences(self.ds, maxlen=padding_size).tolist())

    def __getitem__(self, index: int):
        return np.array(self.ds.at[index]), 0

    def __len__(self):
        return len(self.ds) - 1


class PandasPositiveNegativeNumericTextDataset(PandasNumericTextDataset):
    def __init__(self, dataframe: DataFrame, vocabulary: dict, negative_size: int, max_seq_length: int = 80):
        self.negative_size = negative_size
        super().__init__(dataframe, vocabulary, max_seq_length)

    def __getitem__(self, index: int):
        positive_sample = np.array(self.ds.at[index])
        # No duplicates assure us that with +1 we get the correct amount
        negative_samples = self.ds.sample(n=self.negative_size + 1)
        negative_samples = negative_samples.drop(index=index)  # We drop the current element index

        # Todo: Trasferisci comportamento su altri classi
        if len(negative_samples) > self.negative_size:
            # Drop a random element
            index = negative_samples.index
            negative_samples.drop(np.random.choice(index, 1))

        negative_samples = np.stack(negative_samples.to_numpy())
        return [positive_sample, negative_samples], 0
