import keras
import numpy as np
import torch
from gensim import corpora
from gensim.models import CoherenceModel
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from main.abae.dataset import ABAEDataset
from main.abae.model_manager import ABAEManager


class ABAEEvaluationProcessor:
    # Classe utility per fare misurazioni metriche. This class is initialization only.
    def __init__(self, manager: ABAEManager, test_ds: str | DataFrame):
        self.manager = manager
        self.df = test_ds  # df as for dataframe as the set is a dataframe or a path to a dataframe

        model = manager.get_compiled_model()
        # Word Vector (Like Gensim names)
        self.__wv = normalize(model.get_layer(index=1).weights[0].value.data, dim=-1)
        # Aspect Vector (Like Gensim names)
        self.__av = normalize(model.get_layer(index=7).w, dim=-1)

        self.__inverse_vocabulary = manager.generator.emb_model.model.wv.index_to_key

    def extract_top_k_words(self, aspect_index: int, top_k: int, verbose=False) -> list:
        """

        @param aspect_index: The index of the aspect we want to extract top k words from
        @param top_k:
        @param verbose: If we want to print out the top k words for the current aspect.
        @return:
        """
        if aspect_index >= len(self.__av):
            raise IndexError("Aspect index out of range.")

        similarity = self.__wv @ self.__av[aspect_index]
        sorted_words = torch.argsort(similarity, descending=True)
        for w in sorted_words[:top_k]:
            verbose and print("Word: ", self.__inverse_vocabulary[w], f"({similarity[w]})")
            yield self.__inverse_vocabulary[w], similarity[w]

    def __prepare_aspects(self, top_n: int, aspects: list[list]):
        if aspects is None or len(aspects) == 0 or len(aspects[0]) < top_n:
            n = len(self.__av)
            # Extract top k words and map to only the word actual value and to list as the methods gives a generator.
            return [list(map(lambda x: x[0], self.extract_top_k_words(i, top_n))) for i in range(n)]

        return aspects

    def u_mass_coherence_model(self, top_n: int, ds, aspects: list[list] = None) -> CoherenceModel:
        aspects = self.__prepare_aspects(top_n, aspects)

        dictionary = corpora.Dictionary()
        # BOW creation
        corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in ds]

        return CoherenceModel(topics=aspects, corpus=corpus, dictionary=dictionary, coherence='u_mass', topn=top_n)

    def c_npmi_coherence_model(self, top_n: int, ds, aspects: list[list] = None) -> CoherenceModel:
        aspects = self.__prepare_aspects(top_n, aspects)

        dictionary = corpora.Dictionary(ds.to_list())
        return CoherenceModel(topics=aspects, texts=ds, dictionary=dictionary, coherence='c_npmi')

    def c_v_coherence_model(self, top_n: int, ds, aspects: list[list] = None) -> CoherenceModel:
        aspects = self.__prepare_aspects(top_n, aspects)

        dictionary = corpora.Dictionary(ds.to_list())
        return CoherenceModel(topics=aspects, texts=ds, dictionary=dictionary, coherence='c_v')

    def silhouette_score(self):
        if self.df is None:
            raise AttributeError("test_ds is not set but is required to evaluate the silhouette score")

        model = self.manager.get_compiled_model()
        inference_model = self.manager.get_inference_model()

        vocabulary = self.manager.generator.emb_model.vocabulary()
        ds = ABAEDataset(self.df, vocabulary, self.manager.c.max_seq_len)

        att, labels = inference_model.predict(DataLoader(ds, batch_size=self.manager.c.batch_size))

        embeddings = model.get_layer(index=1)(np.stack(ds.dataset.map(lambda x: np.array(x))))
        w_embs = [(att[..., np.newaxis] * emb.numpy()).sum(0) for emb, att in zip(embeddings, att)]
        return silhouette_score(w_embs, np.argmax(labels, axis=1), metric='cosine')