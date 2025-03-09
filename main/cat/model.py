class CAt:
    def __init__(self, attention):
        self.attention = attention

    def get_aspects(self, fragments, embeddings, n_adj_seed, n_nouns, min_count):
        pass

    def get_scores(self, instances, aspects, r, labels, remove_oov=False, **kwargs):
        a = self.attention # todo use
        pass
