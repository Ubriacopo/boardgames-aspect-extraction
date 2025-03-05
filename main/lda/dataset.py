from gensim import corpora
from pandas import DataFrame

from main.dataset.dataset import BaseBoardgameDataset


class LdaDataset(BaseBoardgameDataset):
    def encode(self, entry: list):
        return entry

    def __init__(self, dataset: DataFrame | list):
        self.dict: corpora.Dictionary | None = None
        super().__init__(dataset, dict())


    def process_dataset(self, dataset: DataFrame | list):
        temp_ds = super().process_dataset(dataset)
        self.dict = corpora.Dictionary(temp_ds)
        return [self.dict.doc2bow(doc, allow_update=True) for doc in temp_ds]
