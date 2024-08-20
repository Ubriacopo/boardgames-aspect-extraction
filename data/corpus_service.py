# Estrae il testo importante dai oggetti json complessi e genera il corpus.
from data.bgg_service import BggRetrieverService


# todo: Rename to BggCorpusService
class BggCorpusService:
    def __init__(self, bgg_retriever_service: BggRetrieverService):
        self.bgg_retriever_service: BggRetrieverService = bgg_retriever_service

    @staticmethod
    def get_comments(information: dict) -> list[str]:
        """
        Extracts a list of all comments from the BGG API results. Only "relevant" comments are considered.
        (Avoid Kickstarter, Bought, etc. as comments)
        """
        try:
            comments_array = [comment['value'] for comment in information['items']['item']['comments']['comment']]
            return list(filter(lambda comment: len(comment) > 15, comments_array))
        except Exception as e:
            print(e)
            return []

    # todo: Cambio strategico sarebbe scarcicare in batch piu giochi come suggerito:
    # https://stackoverflow.com/questions/63124365/throttling-an-api-that-429s-after-too-many-requests
    # e fare solo poche pagine ciascuno. Poi va salvato perchè grosso

    # Might online learning be the best idea and j
    ## todo su service handle error object: OrderedDict({'error': OrderedDict({'message': 'Rate limit exceeded.'})})
    def generate_corpus(self) -> list[str]:

        corpus: list[str] = []

        while self.bgg_retriever_service.has_next():
            current_game = self.bgg_retriever_service.load_next()
            corpus.extend(BggCorpusService.get_comments(current_game))

            # todo vedi cosa mi da. meglio fare con pages? introduci anche delay per evitare
            while current_game := self.bgg_retriever_service.load_more():
                comments = BggCorpusService.get_comments(current_game)
                if len(comments) == 0:
                    break
                corpus.extend(BggCorpusService.get_comments(current_game))

        return corpus
