"""
    Service that loads the reviews from BGG for a given game. (We use BGG-APIv2)

    Postman methods reference:
    https://www.postman.com/1toddlewis/workspace/boardgamegeek/request/1583548-73846a92-fb66-4b71-9f07-c7b9c124ac89

    Game list of at least 100 scores: https://github.com/beefsack/bgg-ranking-historicals
    These will be further reduced in the learning phase to have at least 1k reviews (?) -> todo: Decide a minimum.

    Example url: https://boardgamegeek.com/xmlapi2/thing?id={csv-id}&stats=1&comments=1&page=3

    todo: Should I limit the number of comments per game? Is it possible that mostly strategy games get comments
        and review on BGG and so I'd be creating a low bias towards those games?
"""


class BggService:
    def __init__(self, endpoint: str, game_list_csv_path: str = "./resources/2024-08-18.csv"):
        # https://boardgamegeek.com/xmlapi2/thing?id={csv-id}&stats=1&comments=1&page=3
        self.endpoint: str = endpoint

        # Load csv data with Pandas.

    def load_next(self):
        """
        Returns comments of the next boardgame in list from the loaded csv. (State has to be persisted)
        """
        pass
