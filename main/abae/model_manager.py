import os
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
from main.abae.evaluation import ABAEEvaluationProcessor
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

    def get_compiled_model(self, opt: str | Optimizer = 'adam', load_existing: bool = False, refresh: bool = True):
        if not refresh and self.__train_model is not None:
            # We want to get the current model directly not a new instance
            return self.__train_model

        path = self.considered_path if load_existing else None
        print(f"Generating a new compiled model from {'scratch' if path is None else 'fs'}")
        self.__train_model = self.generator.generate_training_model(self.custom_objects, path)
        self.__train_model.compile(optimizer=opt, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        return self.__train_model

    def train(self, df: str | DataFrame, verbose: int = 1):
        if type(df) is str:
            df = pd.read_csv(df)

        vocabulary = self.generator.emb_model.vocabulary()
        ds = PositiveNegativeABAEDataset(df, vocabulary, self.c.max_seq_len, self.c.negative_sample_size)

        # Just a utility function, one can directly work on the model.
        self.get_compiled_model(refresh=False, opt=keras.optimizers.Adam(learning_rate=self.c.learning_rate))
        num_cores = os.cpu_count()
        train_dataloader = DataLoader(
            dataset=ds, batch_size=self.c.batch_size, shuffle=True, num_workers=int(num_cores / 2), pin_memory=True
        )
        print("Training is starting:")
        history = self.__train_model.fit(train_dataloader, epochs=self.c.epochs, verbose=verbose, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            # ModelCheckpoint(filepath=f"./tmp/ckpt/{self.c.name}.keras", monitor='max_margin_loss'),
            MetricAboveThresholdStopping(monitor='max_margin_loss', threshold=10., start_from_epoch=1),
            # It for sure is bad
            MetricAboveThresholdStopping(monitor='max_margin_loss', threshold=8, start_from_epoch=5),
            EarlyStopping(monitor='max_margin_loss', start_from_epoch=4, patience=3, mode='min')
        ])

        self.__train_model.save(self.considered_path)
        return history, self.get_inference_model(refresh=True)

    def evaluate(self, tops: list[int], test_corpus: str | pd.DataFrame):
        # Where run results are stored.
        results = dict(coherence=[], top=tops)

        # Test set max_margin evaluation
        vocabulary = self.generator.emb_model.vocabulary()
        max_seq_length = self.c.max_seq_len
        negative_size = self.c.negative_sample_size
        test_ds = PositiveNegativeABAEDataset(test_corpus, vocabulary, max_seq_length, negative_size)
        results['loss'] = self.__train_model.evaluate(DataLoader(test_ds, batch_size=self.c.batch_size))

        # Other metrics
        inverse_vocab = self.generator.emb_model.model.wv.index_to_key
        vocab = self.generator.emb_model.vocabulary()
        ev_processor = ABAEEvaluationProcessor(
            test_corpus, self.__train_model, inverse_vocab, vocab, max_sequence_length=self.c.max_seq_len
        )

        results['silhouette_score'] = ev_processor.silhouette_score(self.__train_model, self.__inference_model)

        for top in results['top']:
            # Measured coherence is u_mass as stated is most reliable in topic association
            results['coherence'].append(ev_processor.u_mass_coherence_model(top_n=top).get_coherence())

        return results
