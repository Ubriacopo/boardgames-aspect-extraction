import logging
import os
import sys
from typing import Final

import pandas as pd
from pandas import DataFrame, Series

from bgg_service.bgg_retriever_service import BggRetrieverService

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

# todo change name? to download service o altro
class BggCorpusService:
    BGG_URL: Final[str] = "https://boardgamegeek.com/xmlapi2/thing?id={id}&stats=1&comments=1&page={page}"

    def __init__(self, retriever_service: BggRetrieverService,
                 game_list_csv_path: str = "../../resources/2024-08-18.csv",
                 download_file_path: str = "./../../data/corpus.csv"):
        """

        @param retriever_service:
        @param game_list_csv_path:
        @param download_file_path:
        """
        self.retriever_service = retriever_service
        self.game_list_csv_path = game_list_csv_path
        self.games_dataframe: DataFrame = pd.read_csv(game_list_csv_path)
        self.download_file_path: str = download_file_path

    def _download_page(self, page: int, ids: Series | list) -> int:
        docs = self.retriever_service.load(game_ids=ids, page=page)
        logging.info(f"{self.__class__.__name__}:download_corpus: "
                     f"Working page {page} of the batch")

        has_header = not os.path.exists(self.download_file_path)
        logging.info(f"{self.__class__.__name__}:download_corpus: "
                     f"Storing to csv file the downloaded data")
        docs.to_csv(self.download_file_path, mode="a", header=has_header, index=False)
        logging.info(f"{self.__class__.__name__}:download_corpus: "
                     f"Stored a total of {len(docs)} records in .csv")

        # The first time we run for the current game we record it as downloaded.
        # This allows us to stop without repetitions if we want to split the process with a loss
        # of information relative to the missing pages (up to 24).
        if page == 0:
            for game_id in ids:
                # Downloaded games are registered as downloaded.
                # I can this way split the download process in multiple instances
                self.games_dataframe.loc[self.games_dataframe.ID == game_id, 'downloaded'] = True

        return page + 1

    def download_corpus(self, download_batch_size: int = 19):
        i = 0  # 'i' is the currently elaborated batch
        non_downloaded_dataframe = self.games_dataframe.query('downloaded.isnull()')
        size = self.games_dataframe.shape[0]
        while i * download_batch_size < size:

            logging.info(f"{self.__class__.__name__}:download_corpus: "
                         f"Working on batch {i}/{size / download_batch_size}")
            next_batch_end = (i + 1) * download_batch_size
            up_to = size if size < next_batch_end else next_batch_end

            ids = non_downloaded_dataframe[i * download_batch_size:up_to]["ID"]
            # Download multiple pages (We set custom max of 20 -> 2.5k comments per game max)
            # I might have to stop early as no more page are available? todo
            for page in range(25):
                self._download_page(page, ids)

            logging.info(f"{self.__class__.__name__}:download_corpus: "
                         f"Current set of ids is being set as done: {ids}")

            self.games_dataframe.to_csv(self.game_list_csv_path, mode="w", header=True, index=False)
            logging.info(f"{self.__class__.__name__}:download_corpus: "
                         f"Done storing updated game list csv")
            i += 1
