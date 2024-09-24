from bgg_service.bgg_corpus_service import BggCorpusService


class ModelGenerator:
    corpus_service: BggCorpusService

    def __init__(self, corpus_service: BggCorpusService):
        self.corpus_service = corpus_service
