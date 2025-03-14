import warnings
from pathlib import Path

import keras
import pandas as pd
from keras import Optimizer
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from pandas import DataFrame
from torch.utils.data import DataLoader

from core.utils import max_margin_loss
from main.abae.config import ABAEManagerConfig
from main.abae.dataset import PositiveNegativeABAEDataset
from main.embedding import Word2VecWrapper
from main.abae.embedding import AspectEmbedding
from main.abae.model import BaseABAE, ABAE
from main.utils import CorpusLoaderUtility


class MetricAboveThresholdStopping(EarlyStopping):
    def __init__(self, threshold, **kwargs):
        super(MetricAboveThresholdStopping, self).__init__(**kwargs)
        self.threshold = threshold  # threshold for validation loss

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        # implement your own logic here
        if (epoch >= self.start_from_epoch) & (current >= self.threshold):
            print(f"\nEarly stopping triggered by metric {self.monitor} as value {current} > {self.threshold}")
            self.stopped_epoch = epoch
            self.model.stop_training = True


class ABAEManager:
    custom_objects = {'max_margin_loss': max_margin_loss}

    def __init__(self, config: ABAEManagerConfig, model_generator: BaseABAE):
        self.c = config  # Store configuration
        self.generator = model_generator

        # Where everything will be eventually stored.
        self.considered_path = f"{self.c.output_path()}/{self.c.name}.keras"
        self.__train_model: keras.Model | None = None
        self.__inference_model: keras.Model | None = None

    @classmethod
    def from_scratch(cls, config: ABAEManagerConfig, corpus_path: str, override: bool = False, model_class=ABAE):
        # Make embeddings
        corpus = CorpusLoaderUtility(column_name="comments").load(corpus_path)
        embeddings_file = f"{config.output_path()}/{config.name}.embeddings.model"

        emb_model: Word2VecWrapper
        if not override and Path(embeddings_file).exists():
            emb_model = Word2VecWrapper.from_existing(embeddings_file)
        else:
            emb_model = Word2VecWrapper(config.embedding_size, config.min_word_count, config.max_vocab_size)
            emb_model.generate(corpus)
            emb_model.persist(embeddings_file)

        aspect_embeddings_file = f"{config.output_path()}/{config.name}.aspect_embeddings.model"
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
        self.get_compiled_model(refresh=False, optimizer=keras.optimizers.Adam(learning_rate=self.c.learning_rate))
        train_dataloader = DataLoader(dataset=ds, batch_size=self.c.batch_size, shuffle=True)

        history = self.__train_model.fit(train_dataloader, epochs=self.c.epochs, verbose=verbose, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.c.name}.keras", monitor='max_margin_loss'),
            MetricAboveThresholdStopping(monitor='max_margin_loss', threshold=10., start_from_epoch=1),
            # It for sure is bad
            MetricAboveThresholdStopping(monitor='max_margin_loss', threshold=6.5, start_from_epoch=6),
            EarlyStopping(monitor='max_margin_loss', start_from_epoch=4, patience=3, mode='min')
        ])

        self.__train_model.save(self.considered_path)
        return history, self.get_inference_model(refresh=True)
