import json
from abc import abstractmethod
from keras import ops as K
import pandas as pd
import swifter
from pandas import DataFrame


class DataLoaderUtility:
    @abstractmethod
    def load(self, file_path: str) -> list:
        pass


class CorpusLoaderUtility(DataLoaderUtility):
    def __init__(self, column_name: str | None = "comments"):
        # The referenced object we seek should have this col name
        self.column_name = column_name

    def load(self, corpus: str | DataFrame) -> list:
        # We suppose that this utility is called on pre-processed data that has
        # just to be split on spaces.
        if type(corpus) is str:
            corpus = pd.read_csv(corpus)
        return corpus[self.column_name].swifter.apply(lambda x: x.split()).tolist()


def max_margin_loss(y_true, y_pred):
    """
    The max margin loss function is used to train the model.
    It is a hinge loss function that is used to train the model to maximize the
    margin between the correct class and the other classes.

    @param y_true: The true labels.
    @param y_pred: The predicted labels.
    @return: The loss value.
    """
    return K.mean(y_pred, axis=-1)



class ModelAspectMapper:
    LUCK: str = "luck"
    BOOKKEEPING: str = "bookkeeping"
    DOWNTIME: str = "downtime"
    INTERACTION: str = "interaction"
    BASH: str = "bash"
    COMPLEX_COMPLICATED: str = "complex/complicated"
    MISC: str = "misc"

    def __init__(self, aspects: int):
        """
        Class that maps the output of a model to the gold standard classes based on a mapping
        @param aspects: Number of aspects identified by the model
        """
        self.aspect_size = aspects
        self.mappings = {i: '' for i in range(aspects)}
        self.gold_aspects = ["luck", "bookkeeping", "downtime", "interaction", "bash", "complex/complicated", "misc"]

    def assign(self, aspect_index: int, class_name: str):
        if class_name not in self.gold_aspects:
            raise "Given class is invalid"
        self.mappings[aspect_index] = class_name

    def store(self, target_folder):
        with open(f"{target_folder}/mappings.json", 'w') as f:
            json.dump(self.mappings, f)

    def map_to_gold(self, scores: list[float]) -> pd.DataFrame:
        if len(scores) != self.aspect_size:
            raise Exception("Scores did not match aspect size")
        # Scores
        return_object = [{"score": 0, "label": gold, "sources": []} for gold in self.gold_aspects]
        for i in range(len(scores)):
            aspect = self.mappings[i]
            index = self.gold_aspects.index(aspect)
            return_object[index]['score'] += scores[i]
            return_object[index]['sources'].append(i)

        return pd.DataFrame(return_object)

    @classmethod
    def load_from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            objects = json.load(f)

        instance = cls(len(objects))
        instance.mappings = objects
        return instance
