from typing import Final

import requests

import pandas as pd
from pandas import DataFrame, Series

from xmljson import yahoo
from xml.etree.ElementTree import fromstring

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

# To fill Url (Cannot be overwritten)
BGG_URL: Final[str] = "https://boardgamegeek.com/xmlapi2/thing?id={id}&stats=1&comments=1&page={page}"


class BggService:
    games_dataframe: DataFrame = None
    latest_game: Series = None

    def __init__(self, endpoint: str, game_list_csv_path: str = "./resources/2024-08-18.csv"):
        self.endpoint: str = endpoint

        # Todo: Filter out the games with less than 150 reviews?
        self.games_dataframe = pd.read_csv(game_list_csv_path)

        self.current_game_index = 0
        self.current_game_meta_page = 1

    def load_next(self):
        """
        Returns comments of the next boardgame in list from the loaded csv. (State has to be persisted)
        """
        self.latest_game = self.games_dataframe.loc[self.current_game_index]
        self.current_game_meta_page = 1

        response = requests.get(BGG_URL.format(id=self.latest_game["ID"], page=self.current_game_meta_page))
        return yahoo.data(fromstring(response.content))

    def load_more(self):
        """
        Goes to next page of selected game metadata (comments, etc.)
        """
        if self.latest_game is None:
            raise Exception("No more information for the current game is available")

        self.current_game_meta_page += 1

        response = requests.get(BGG_URL.format(id=self.latest_game["ID"], page=self.current_game_meta_page))
        return yahoo.data(fromstring(response.content))
