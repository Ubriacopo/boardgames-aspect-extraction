from bgg_service.bgg_corpus_service import BggCorpusService
from bgg_service.bgg_retriever_service import BggRetrieverService

corpus_service = BggCorpusService(BggRetrieverService(BggCorpusService.BGG_URL), "./../../resources/2024-08-18.csv")
corpus_service.download_corpus(20)
