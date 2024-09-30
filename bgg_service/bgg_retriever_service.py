# XML Structure of what we desire
import logging
import sys
from dataclasses import dataclass

import pandas
import pandas as pd
from pandas import DataFrame

import requests

from xmljson import yahoo
from xml.etree.ElementTree import fromstring

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


@dataclass
class GameCommentsRecord:
    game_id: int  # Reference to the game this comments belongs to
    comments: list[str]  # The extracted comments of the game


class BggRetrieverService:
    def __init__(self, base_url: str):
        self.base_url = base_url

    @staticmethod
    def __extract_game_record(game: dict) -> GameCommentsRecord:
        comments = [comment['value'] for comment in game['comments']['comment']]
        # Remove the very short ones (Kickstarter, wished, etc.)
        comments = list(filter(lambda comment: len(comment) > 20, comments))
        return GameCommentsRecord(game_id=game['id'], comments=comments)

    @staticmethod
    def _make_dataframe(game_information: dict) -> DataFrame:
        if 'comment' not in game_information['comments']:
            return DataFrame()  # Nothing to populate.

        comments = [comment['value'] for comment in game_information['comments']['comment']]
        comments = list(filter(lambda comment: len(comment) > 15, comments))
        return DataFrame({"comments": comments, "game_id": [game_information['id'] for i in range(len(comments))]})

    def load(self, game_ids: list[int], page: int = 0) -> DataFrame:

        try:
            game_ids_string = "".join([f"{game_id}," for game_id in game_ids])[:-1]
            logging.debug(f"{self.__class__.__name__}: Start looking for game ids: {game_ids_string}")

            response = requests.get(self.base_url.format(id=game_ids_string, page=page))
            data = yahoo.data(fromstring(response.content))

            logging.debug(f"{self.__class__.__name__}: retrieved data from server extracting now the game infos")
            return pd.concat(
                [self._make_dataframe(game) for game in data['items']['item']], ignore_index=True, sort=False
            )

        except Exception as exception:
            logging.error(exception)
            logging.debug(f"{self.__class__.__name__}: thrown exception, returning empty dataframe")
            return pd.DataFrame()  # Return empty array as we could not load for those games
