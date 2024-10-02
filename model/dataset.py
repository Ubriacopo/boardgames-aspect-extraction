import spacy
import pandas as pd


class BoardGameCorpusDataset:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")

    def load_corpus(self, file_name: str):
        # todo Run preprocessing if file does not exist.
        dataframe = pd.read_csv(file_name)

        # todo vedi se questa call giusta
        #series_of_data = dataframe["comments"].apply(lambda x: self.nlp.tokenizer(x))

        # Dovrebbe essere giusto cosi. Swifter makes faster
        return dataframe["comments"].swifter.apply(lambda x: [i.text for i in self.nlp.tokenizer(x)])
