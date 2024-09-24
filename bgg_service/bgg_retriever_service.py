# XML Structure of what we desire
from dataclasses import dataclass

import requests

from xmljson import yahoo
from xml.etree.ElementTree import fromstring


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
        comments = list(filter(lambda comment: len(comment) > 15, comments))
        return GameCommentsRecord(game_id=game['id'], comments=comments)

    def load(self, game_ids: list[int], page: int = 0) -> list[GameCommentsRecord]:

        try:
            game_ids_string = "".join([f"{game_id}," for game_id in game_ids])[:-1]
            response = requests.get(self.base_url.format(id=game_ids_string, page=page))
            data = yahoo.data(fromstring(response.content))
            return [BggRetrieverService.__extract_game_record(game) for game in data['items']['item']]

        except Exception as exception:
            print(exception)
            return []  # Return empty array as we could not load for those games
