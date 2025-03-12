from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

import keras
import pandas as pd
from keras import Optimizer
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from pandas import DataFrame
from torch.utils.data import DataLoader

from core.utils import max_margin_loss
from main.abae.dataset import PositiveNegativeABAEDataset
from main.embedding import Word2VecWrapper
from main.abae.embedding import AspectEmbedding
from main.abae.model import BaseABAE, ABAE, ABAEGeneratorConfig
from main.utils import CorpusLoaderUtility


@dataclass
class ABAEManagerConfig(ABAEGeneratorConfig):
    min_word_count: int = 5
    max_vocab_size: int | None = None
    # todo add optimizer
    batch_size: int = 128
    epochs: int = 15
    model_name: str = "ABAE"  # Cannot be required for dataclass hierarchy
    output_folder: str = "./output"

    @classmethod
    def from_configuration(cls, model_name: str, object: dict):
        instance = cls()
        [instance.__setattr__(f.name, object[f.name]) for f in fields(instance) if f.name in object]
        instance.model_name = model_name  # Required name
        return instance

    def output_path(self):
        path = f"{self.output_folder}/{self.model_name}"
        # If the folder does not exist we take care of it immediately.
        Path(path).mkdir(parents=True, exist_ok=True)
        return path


class ABAEManager:
    custom_objects = {'max_margin_loss': max_margin_loss}

    def __init__(self, config: ABAEManagerConfig, model_generator: BaseABAE):
        self.c = config  # Store configuration
        self.generator = model_generator

        # Where everything will be eventually stored.
        self.considered_path = f"{self.c.output_path()}/{self.c.model_name}.keras"
        self.__train_model: keras.Model | None = None
        self.__inference_model: keras.Model | None = None

    @classmethod
    def from_scratch(cls, config: ABAEManagerConfig, corpus_path: str, override: bool = False, model_class=ABAE):
        # Make embeddings
        corpus = CorpusLoaderUtility(column_name="comments").load(corpus_path)
        embeddings_file = f"{config.output_path()}/{config.model_name}.embeddings.model"

        emb_model: Word2VecWrapper
        if not override and Path(embeddings_file).exists():
            emb_model = Word2VecWrapper.from_existing(embeddings_file)
        else:
            emb_model = Word2VecWrapper(config.embedding_size, config.min_word_count, config.max_vocab_size)
            emb_model.generate(corpus)
            emb_model.persist(embeddings_file)

        aspect_embeddings_file = f"{config.output_path()}/{config.model_name}.aspect_embeddings.model"
        aspect_model: AspectEmbedding = AspectEmbedding(config.aspect_size, config.embedding_size)

        if not override and Path(aspect_embeddings_file).exists():
            aspect_model.load_existing(aspect_embeddings_file)
        else:
            aspect_model.generate(emb_model.weights())
            aspect_model.persist(aspect_embeddings_file)

        return cls(config, model_class(config, emb_model, aspect_model))

    def get_inference_model(self, refresh: bool = False) -> keras.Model:
        if refresh or self.__inference_model is None:
            self.__inference_model = self.generator.generate_inference_model(self.custom_objects, self.considered_path)

        return self.__inference_model

    def get_compiled_model(self, optimizer: str | Optimizer = 'adam', load_existing: bool = False,
                           refresh: bool = True):
        if not refresh and self.__train_model is not None:
            return self.__train_model

        path = self.considered_path if load_existing else None
        self.__train_model = self.generator.generate_training_model(self.custom_objects, path)
        self.__train_model.compile(optimizer=optimizer, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        return self.__train_model

    def train(self, df: str | DataFrame, verbose: int = 1):
        if type(df) is str:
            df = pd.read_csv(df)

        vocabulary = self.generator.emb_model.vocabulary()
        ds = PositiveNegativeABAEDataset(df, vocabulary, self.c.max_seq_len, self.c.negative_sample_size)

        # Just a utility function, one can directly work on the model.
        self.get_compiled_model(refresh=False)
        train_dataloader = DataLoader(dataset=ds, batch_size=self.c.batch_size, shuffle=True)

        history = self.__train_model.fit(train_dataloader, epochs=self.c.epochs, verbose=verbose, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.c.model_name}.keras", monitor='max_margin'),
            EarlyStopping(monitor='loss', baseline=8, start_from_epoch=2),  # It for sure is bad
            EarlyStopping(monitor='loss', baseline=5, start_from_epoch=3),
            EarlyStopping(monitor='loss', start_from_epoch=4, patience=3)
        ])

        self.__train_model.save(self.considered_path)
        return history, self.get_inference_model(refresh=True)
