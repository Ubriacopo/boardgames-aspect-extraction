import keras
import torch
from gensim import corpora
from gensim.models import CoherenceModel
from torch.nn.functional import normalize


class ABAEEvaluationProcessor:
    # Classe utility per fare misurazioni metriche. This class is initialization only.
    def __init__(self, word_embeddings, aspect_embeddings, inverse_vocabulary: dict):
        self.__word_embeddings = word_embeddings
        self.__aspect_embeddings = aspect_embeddings
        self.__inverse_vocabulary = inverse_vocabulary

    def extract_top_k_words(self, aspect_index: int, top_k: int, verbose=False) -> list:
        """

        @param aspect_index: The index of the aspect we want to extract top k words from
        @param top_k:
        @param verbose: If we want to print out the top k words for the current aspect.
        @return:
        """
        if aspect_index >= len(self.__aspect_embeddings):
            raise IndexError("Aspect index out of range.")

        similarity = self.__word_embeddings @ self.__aspect_embeddings[aspect_index]
        sorted_words = torch.argsort(similarity, descending=True)
        for w in sorted_words[:top_k]:
            verbose and print("Word: ", self.__inverse_vocabulary[w], f"({similarity[w]})")
            yield self.__inverse_vocabulary[w], similarity[w]

    def __prepare_aspects(self, top_n: int, aspects: list[list]):
        if aspects is None or len(aspects) == 0 or len(aspects[0]) < top_n:
            n = len(self.__aspect_embeddings)
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

    @staticmethod
    def generate_for_model(model: keras.Model, inverse_vocabulary: dict):
        word_embeddings = model.get_layer('word_embedding').weights[0].value.data
        word_embeddings = normalize(word_embeddings, dim=-1)
        aspect_embeddings = model.get_layer('weighted_aspect_embedding').w
        aspect_embeddings = normalize(aspect_embeddings, dim=-1)

        return ABAEEvaluationProcessor(word_embeddings, aspect_embeddings, inverse_vocabulary)
