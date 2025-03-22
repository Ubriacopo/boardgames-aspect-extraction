import keras
import numpy as np
import pandas as pd
import torch
import swifter
from gensim import corpora
from gensim.models import CoherenceModel
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from main.abae.dataset import ABAEDataset


class ABAEEvaluationProcessor:
    # Classe utility per fare misurazioni metriche. This class is initialization only.
    def __init__(self, test_ds: str | DataFrame, model: keras.Model, inverse_vocab, vocab, max_sequence_length: int):
        # To allow passing the file reference
        if type(test_ds) is str:
            test_ds = pd.read_csv(test_ds)

        self.df: DataFrame = test_ds  # df as for dataframe as the set is a dataframe or a path to a dataframe

        # Word Vector (Like Gensim names)
        self.wv = normalize(model.get_layer(index=1).weights[0].value.data, dim=-1)
        # Aspect Vector (Like Gensim names)
        self.av = normalize(model.get_layer(index=7).w, dim=-1)
        self.calculated_aspects: list = []

        self.vocabulary = vocab
        self.inverse_vocabulary = inverse_vocab
        self.max_sequence_length = max_sequence_length

    def word_topic_relation(self, word: str):
        index = self.vocabulary[word]
        similarity = self.wv[index] @ self.av.T
        return similarity

    def extract_top_k_words(self, aspect_index: int, top_k: int, verbose=False) -> list:
        """

        @param aspect_index: The index of the aspect we want to extract top k words from
        @param top_k:
        @param verbose: If we want to print out the top k words for the current aspect.
        @return:
        """
        if aspect_index >= len(self.av):
            raise IndexError("Aspect index out of range.")

        similarity = self.wv @ self.av[aspect_index]
        sorted_words = torch.argsort(similarity, descending=True)
        for w in sorted_words[:top_k]:
            verbose and print("Word: ", self.inverse_vocabulary[w], f"({similarity[w]})")
            yield self.inverse_vocabulary[w], similarity[w]

    def __prepare_aspects(self, top_n: int):
        if len(self.calculated_aspects) >= top_n:
            return self.calculated_aspects
        n = len(self.av)  # Number of aspect vectors (av is named like wv - aspect_vector)
        # Extract top k words and map to only the word actual value and to list as the methods gives a generator.
        self.calculated_aspects = [list(map(lambda x: x[0], self.extract_top_k_words(i, top_n))) for i in range(n)]
        return self.calculated_aspects

    def u_mass_coherence_model(self, top_n: int, aspects: list[list] = None) -> CoherenceModel:
        if aspects is None or len(aspects) == 0 or len(aspects[0]) < top_n:
            aspects = self.__prepare_aspects(top_n)

        dictionary = corpora.Dictionary()
        # BOW creation
        df = self.df['comments'].swifter.apply(lambda x: x.split(' '))
        corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in df]

        return CoherenceModel(topics=aspects, corpus=corpus, dictionary=dictionary, coherence='u_mass', topn=top_n)

    def c_npmi_coherence_model(self, top_n: int, aspects: list[list] = None) -> CoherenceModel:
        if aspects is None or len(aspects) == 0 or len(aspects[0]) < top_n:
            aspects = self.__prepare_aspects(top_n)
        df = self.df['comments'].swifter.apply(lambda x: x.split(' '))
        dictionary = corpora.Dictionary(df.to_list())
        return CoherenceModel(topics=aspects, texts=df, dictionary=dictionary, coherence='c_npmi', topn=top_n)

    def c_v_coherence_model(self, top_n: int, aspects: list[list] = None) -> CoherenceModel:
        if aspects is None or len(aspects) == 0 or len(aspects[0]) < top_n:
            aspects = self.__prepare_aspects(top_n)
        df = self.df['comments'].swifter.apply(lambda x: x.split(' '))
        dictionary = corpora.Dictionary(df.to_list())
        return CoherenceModel(topics=aspects, texts=df, dictionary=dictionary, coherence='c_v', topn=top_n)

    def silhouette_score(self, model: keras.Model, inference_model: keras.Model):
        if self.df is None:
            raise AttributeError("test_ds is not set but is required to evaluate the silhouette score")

        ds = ABAEDataset(self.df, self.vocabulary, self.max_sequence_length)
        att, labels = inference_model.predict(DataLoader(ds, batch_size=512))

        embeddings = model.get_layer(index=1)(np.stack(ds.dataset.map(lambda x: np.array(x))))
        w_embs = [(att[..., np.newaxis] * emb.cpu().numpy()).sum(0) for emb, att in zip(embeddings, att)]
        return float(silhouette_score(w_embs, np.argmax(labels, axis=1), metric='cosine'))

    def get_aspects(self, top_n: int):
        n = len(self.av)  # Number of aspect vectors (av is named like wv - aspect_vector)
        return [list(self.extract_top_k_words(i, top_n)) for i in range(n)]
