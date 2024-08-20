import bgg_service
import corpus_service
service = bgg_service.BggRetrieverService()
information = service.load_next()
print(information)
information = service.load_more()

corpus_service = corpus_service.BggCorpusService(service)
corpus = corpus_service.generate_corpus()
print(corpus)