from dataclasses import dataclass
from pathlib import Path

import keras
from keras import Optimizer
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers.schedules import ExponentialDecay
from torch.utils.data import DataLoader

from core.dataset import BaseBoardgameDataset
from core.embeddings import WordEmbedding, AspectEmbedding
from core.model import ABAEGenerator, ModelGenerator, ABAE
from core.utils import LoadCorpusUtility, zero_loss
from core.utils import max_margin_loss


@dataclass
class AbaeModelConfiguration:
    """
      Configuration for the ABAE model.

      Attributes:
          corpus_file: Path to the corpus file.
          model_name: str Name of model
          max_vocab_size: Maximum size of the vocabulary.
          embedding_size: Size of the word embeddings.
          aspect_size: Number of aspects.
          max_sequence_length: Maximum length of the input sequences.
          negative_sample_size: Number of negative samples.
          output_path: Where files are stored
    """
    corpus_file: str
    model_name: str

    aspect_size: int = 16
    max_vocab_size: int = None
    embedding_size: int = 200

    learning_rate: float = 0.01
    decay_rate: float = 0.95
    momentum: float = 0.9

    max_sequence_length: int = 80
    negative_sample_size: int = 15

    batch_size: int = 64
    epochs: int = 10

    output_path: str = "./output"


class AbaeModelManager:
    def __init__(self, config: AbaeModelConfiguration, override_existing: bool = False):
        super(AbaeModelManager, self).__init__()
        self.config = config

        self.output_path = f"{self.config.output_path}/{self.config.model_name}"
        # We organize the model in a folder to have the files in the right spot.
        Path(f"{self.config.output_path}/{self.config.model_name}").mkdir(parents=True, exist_ok=True)

        # Load the Embeddings
        self.embedding_model: WordEmbedding | None = None
        self.aspect_model: AspectEmbedding | None = None
        self.model_generator: ModelGenerator | None = None
        # The training model instance.
        self._t_model: keras.Model | None = None
        self._ev_model: keras.Model | None = None

        self.load_embeddings(corpus_file=config.corpus_file, override_existing=override_existing)

    def load_embeddings(self, corpus_file: str = None, persist: bool = True, override_existing: bool = False):
        """

        @param corpus_file:
        @param persist:
        @param override_existing:
        @return:
        """
        keep = not override_existing

        load_utility = LoadCorpusUtility(column_name="comments")
        corpus = load_utility.load_data(corpus_file if corpus_file is not None else self.config.corpus_file)

        self.embedding_model = WordEmbedding(
            self.config.embedding_size, self.output_path, max_vocab_size=self.config.max_vocab_size
        )
        self.embedding_model.generate(corpus=corpus, sg=True, persist=persist, load_existing=keep)

        self.aspect_model = AspectEmbedding(self.config.aspect_size, self.config.embedding_size, self.output_path)

        emb_weights = self.embedding_model.weights()
        self.aspect_model.generate(embedding_weights=emb_weights, persist=persist, load_existing=keep)

        # Now we can initialize the model generator.
        self.model_generator: ModelGenerator = ABAEGenerator(
            self.config.max_sequence_length, self.config.negative_sample_size, self.embedding_model, self.aspect_model
        )

        # Reset the models. We had an override of the embeddings!
        self._t_model = None

    def run_train_process(self, dataset: BaseBoardgameDataset,
                          consider_stored: bool = False, optimizer: str | Optimizer = None):

        considered_path = f"{self.output_path}/{self.config.model_name}.keras"

        # Get a trainable model structure.
        self._t_model = self.model_generator.generate_model(considered_path if consider_stored else None)

        # We leave the choice of the optimizer open in case, but we will probably use this.
        if optimizer is None:
            steps_per_epoch = len(dataset) / self.config.batch_size
            total_steps = self.config.epochs * steps_per_epoch

            # Why SGD? Well there are reasons! Check the paper I put in the notes to see.
            # SGD has fewer Hyperparameters and also better generalizes in most cases.
            optimizer = keras.optimizers.SGD(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=self.config.learning_rate,
                    # I guess that by bounding this we learn the best configuration with this assumption.
                    decay_steps=.1 * total_steps,  # Every 10% of training process we reduce the learning rate,
                    decay_rate=self.config.decay_rate
                ),
                momentum=self.config.momentum,
            )

        self._t_model.compile(optimizer=optimizer, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        train_dataloader = DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True)

        # Now run the training process and return the process history.
        history = self._t_model.fit(train_dataloader, epochs=self.config.epochs, verbose=2, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.config.model_name}.keras", monitor='max_margin')
        ])

        # Persist the model to the work directory
        self._t_model.save(considered_path)
        # Return history and re-initialize the ev_model
        return history, self.get_model(force_refresh=True)

    def get_model(self, force_refresh: bool = False):
        """
        You can only get a model if there has been a train procedure on it before!
        @param force_refresh: If we want to force a refresh of the ev model
        @return: A model instance if it exists on fs else it raises an exception.
        """
        if force_refresh is True or self._ev_model is None:
            model_file_path = f"{self.output_path}/{self.config.model_name}.keras"
            self._ev_model = self.model_generator.generate_model(model_file_path, False)

        return self._ev_model


class ABAEManager:
    def __init__(self, config: AbaeModelConfiguration, override_existing: bool = False):
        self.config = config

        self.output_path = f"{self.config.output_path}/{self.config.model_name}"
        # We organize the model in a folder to have the files in the right spot.
        Path(f"{self.config.output_path}/{self.config.model_name}").mkdir(parents=True, exist_ok=True)

        # Load the Embeddings
        self.embedding_model: WordEmbedding | None = None
        self.aspect_model: AspectEmbedding | None = None
        self.model_generator: ABAE | None = None
        # The training model instance.
        self._t_model: keras.Model | None = None
        self._ev_model: keras.Model | None = None

        self.load_embeddings(corpus_file=config.corpus_file, override_existing=override_existing)

    def load_embeddings(self, corpus_file: str = None, persist: bool = True, override_existing: bool = False):
        """

        @param corpus_file:
        @param persist:
        @param override_existing:
        @return:
        """
        keep = not override_existing

        load_utility = LoadCorpusUtility(column_name="comments")
        corpus = load_utility.load_data(corpus_file if corpus_file is not None else self.config.corpus_file)

        self.embedding_model = WordEmbedding(
            self.config.embedding_size, self.output_path, max_vocab_size=self.config.max_vocab_size
        )
        self.embedding_model.generate(corpus=corpus, sg=True, persist=persist, load_existing=keep)

        self.aspect_model = AspectEmbedding(self.config.aspect_size, self.config.embedding_size, self.output_path)

        emb_weights = self.embedding_model.weights()
        self.aspect_model.generate(embedding_weights=emb_weights, persist=persist, load_existing=keep)

        # Now we can initialize the model generator.
        self.model_generator: ABAE = ABAE(
            self.config.max_sequence_length, self.config.negative_sample_size, self.aspect_model, self.embedding_model,
            self.aspect_model.aspect_size
        )

        # Reset the models. We had an override of the embeddings!
        self._t_model = None

    def run_train_process(self, dataset: BaseBoardgameDataset,
                          consider_stored: bool = False, optimizer: str | Optimizer = None):

        considered_path = f"{self.output_path}/{self.config.model_name}.keras"

        # Get a trainable model structure.
        self._t_model = self.model_generator.make_trainable_model(considered_path if consider_stored else None)

        # We leave the choice of the optimizer open in case, but we will probably use this.
        if optimizer is None:
            steps_per_epoch = len(dataset) / self.config.batch_size
            total_steps = self.config.epochs * steps_per_epoch

            # Why SGD? Well there are reasons! Check the paper I put in the notes to see.
            # SGD has fewer Hyperparameters and also better generalizes in most cases.
            optimizer = keras.optimizers.SGD(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=self.config.learning_rate,
                    # I guess that by bounding this we learn the best configuration with this assumption.
                    decay_steps=.1 * total_steps,  # Every 10% of training process we reduce the learning rate,
                    decay_rate=self.config.decay_rate
                ),
                momentum=self.config.momentum,
            )

        self._t_model.compile(optimizer=optimizer, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        train_dataloader = DataLoader(dataset=dataset, batch_size=self.config.batch_size, shuffle=True)

        # Now run the training process and return the process history.
        history = self._t_model.fit(train_dataloader, epochs=self.config.epochs, verbose=2, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.config.model_name}.keras", monitor='max_margin')
        ])

        # Persist the model to the work directory
        self._t_model.save(considered_path)
        # Return history and re-initialize the ev_model
        return history, self.get_model(force_refresh=True)

    def get_model(self, force_refresh: bool = False):
        """
        You can only get a model if there has been a train procedure on it before!
        @param force_refresh: If we want to force a refresh of the ev model
        @return: A model instance if it exists on fs else it raises an exception.
        """
        if force_refresh is True or self._ev_model is None:
            model_file_path = f"{self.output_path}/{self.config.model_name}.keras"
            self._ev_model = self.model_generator.generate_model(model_file_path, False)

        return self._ev_model
