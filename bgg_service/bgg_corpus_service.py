import json
import os
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from bgg_service.bgg_retriever_service import BggRetrieverService


class BggCorpusService:
    BGG_URL: Final[str] = "https://boardgamegeek.com/xmlapi2/thing?id={id}&stats=1&comments=1&page={page}"

    def __init__(self, retriever_service: BggRetrieverService, game_list_csv_path: str = "../../resources/2024-08-18.csv"):
        self.retriever_service = retriever_service
        self.games_dataframe: DataFrame = pd.read_csv(game_list_csv_path)

    def download_corpus(self, download_batch_size: int = 19):
        i = 0  # 'i' is the currently elaborated batch
        size = self.games_dataframe.shape[0]
        while i * download_batch_size < size:
            # Work on batches.
            print(f"Working on batch {i}/{size / download_batch_size}")
            next_batch_end = (i + 1) * download_batch_size
            up_to = size if size < next_batch_end else next_batch_end
            ids = self.games_dataframe[i * download_batch_size:up_to]["ID"]

            # Download multiple pages
            page = 0
            while docs := self.retriever_service.load(game_ids=ids, page=page):
                print(f"Working page {page} of batch {i}/{size / download_batch_size}")
                page += 1
                for doc in docs:
                    if not doc.comments:
                        # No more comments for game, so we make it more lightweight for responses
                        ids.remove(doc.game_id)

                    # Save the comments somewhere (Single file?)
                    # Poi fai Train/Test/Validation splits ? We are not in supervised paradigm >:(
                    mode = 'a' if os.path.exists("./../data/corpus.txt") else 'w+'
                    with open("./../../data/corpus.txt", mode) as corpus:
                        json.dump([dict(comment=comment, game_id=doc.game_id) for comment in doc.comments], corpus)
