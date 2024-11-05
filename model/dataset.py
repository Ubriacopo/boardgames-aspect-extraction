import os

import pandas as pd
import spacy
from torch.utils.data import Dataset


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

    def __init__(self, csv_dataset_path: str = ""):
        # Is way faster than having to reload it at each iteration.
        self.dataset = pd.read_csv(csv_dataset_path, names=["comments"]).dropna()
        self.dataset = self.dataset["comments"].swifter.apply(lambda x: [token.text for token in self.nlp(x)])

        # Generate the vocabulary or load it if it exists
        if os.path.exists(f"{csv_dataset_path}.vocab"):
            self.vocab = pd.read_csv(f"{csv_dataset_path}.vocab")
        else:
            self.vocab = pd.DataFrame(list(self.__generate_vocabulary().items()), columns=["token", "times"])
            self.vocab.sort_values(by="times", inplace=True, ascending=False)

            # Index is the important element. Times is just an enrichment but could be dropped as
            # the info it carries is not really required from now on.
            self.vocab.to_csv(f"{csv_dataset_path}.vocab", index=True)

            # Should I save already processed dataset?

        print(self.dataset.memory_usage(deep=True))
        print(self.dataset.info(memory_usage='deep'))

    def __generate_vocabulary(self) -> dict:
        # Default two terms. (We do zero masking?)
        vocabulary = {"<unk>": 0, "<pad>": 1}
        # todo empty spaces ci sono ancora
        for entry in self.dataset:
            for e in entry:  # We factorized the text in order to have arrays of words
                vocabulary[e] = 1 if e not in vocabulary else vocabulary.get(e) + 1
        return vocabulary

    def __getitem__(self, index):
        # todo vedi
        return [self.vocab[token]["index"] for token in self.dataset.at[index + 1]]

    def __len__(self):
        return len(self.dataset)


class VocabularyDataset(Dataset):
    nlp = spacy.blank("en")

    # todo end
    def __init__(self, vocabulary, max_sequence_length: int = 256, batch_size: int = 30, csv_dataset_path: str = ""):
        """

        @param word2vec_model: gensim.Word2Vec model
        @param max_sequence_length:
        @param batch_size:
        @param csv_dataset_path:
        """
        super(VocabularyDataset).__init__()

        if not os.path.exists(csv_dataset_path):
            raise FileNotFoundError("To process a corpus the file has to exist. Please pre-process the corpus "
                                    "before starting. As the procedure might take some time, it won't be launched now.")

        self.max_sequence_length = max_sequence_length
        self.csv_dataset_path = csv_dataset_path
        self.chunk_size = batch_size
        self.vocabulary = vocabulary

    def __getitem__(self, index):
        # Read the batch of rows of our dataset
        x = next(pd.read_csv(self.csv_dataset_path, skiprows=index * self.chunk_size + 1, chunksize=self.chunk_size))
        # Tokenize them.
        x = [self.nlp(e) for e in x]

        # Filter only relevant entries (this is to mitigate the problem we had in preprocessing and will be removed)
        # todo Rimuovi questa istruzione quando rifai il pre-processed dataset.
        x = [[i.text for i in row if not i.is_punct and not i.is_currency] for row in x]

        # Load previous model to get the index representation of the word to return to the model

        # todo handle unknown e pad
        # wv.vocab[i].index todo vedi il tipo di vocabulary
        return [self.vocabulary.index for i in x]
