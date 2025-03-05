from dataclasses import dataclass

import pandas as pd
from gensim.models import LdaModel

from main.lda.dataset import LdaDataset
from main.model import ModelGenerator


@dataclass
class LdaGeneratorConfig:
    topics: int = 14
    corpus_file_path: str = ""


class LdaModelGenerator(ModelGenerator):
    def generate_inference_model(self, custom_objects: dict, existing_model_path: str = None):
        return self.generate_training_model(custom_objects, existing_model_path)

    def __init__(self, config: LdaGeneratorConfig):
        self.c: LdaGeneratorConfig = config

    def generate_training_model(self, custom_objects: dict = None, existing_model_path: str = None):
        if existing_model_path is not None:
            return LdaModel.load(existing_model_path)

        random_state = custom_objects['random_state'] if hasattr(custom_objects, 'random_state') else 42
        chunk_size = custom_objects['chunk_size'] if hasattr(custom_objects, 'chunk_size') else 1000

        lda_dataset = LdaDataset(pd.read_csv(self.c.corpus_file_path))

        return LdaModel(corpus=lda_dataset.dataset, id2word=lda_dataset.dict, num_topics=self.c.topics,
                        random_state=random_state, chunksize=chunk_size, passes=10)
