from gensim import corpora
from pandas import DataFrame

from main.dataset.dataset import BaseBoardgameDataset


class LdaDataset(BaseBoardgameDataset):
    def encode(self, entry: list):
        return entry

    def __init__(self, dataset: DataFrame | list | str, stop_words: list[str] = None):
        self.dict: corpora.Dictionary | None = None
        self.stop_words: list[str] = stop_words if stop_words else []
        super().__init__(dataset, dict())

    def process_dataset(self, dataset: DataFrame | list | str):
        temp_ds = super().process_dataset(dataset)
        temp_ds = temp_ds.map(lambda x: [w for w in x if w not in self.stop_words])
        temp_ds = temp_ds[temp_ds.apply(len) > 0]

        self.dict = corpora.Dictionary(temp_ds)
        return [self.dict.doc2bow(doc, allow_update=True) for doc in temp_ds]
