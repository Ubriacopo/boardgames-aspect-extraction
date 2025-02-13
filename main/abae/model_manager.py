from dataclasses import dataclass
from pathlib import Path

import keras
from keras import Optimizer
from keras.src.callbacks import ModelCheckpoint
from keras.src.optimizers.schedules import ExponentialDecay
from torch.utils.data import DataLoader

from core.utils import max_margin_loss
from main.abae.dataset import PositiveNegativeABAEDataset
from main.abae.embedding import WordEmbedding, AspectEmbedding
from main.abae.model import ABAEConfig, ABAE
from main.utils import CorpusLoaderUtility


@dataclass
class ABAEGeneratorConfig:
    max_seq_len: int = 80
    embedding_size: int = 100
    aspect_size: int = 14
    negative_sample_size: int = 20

    train_ds_expected_size: int | None = None

    min_word_count: int = 5
    max_vocab_size: int | None = None

    learning_rate: float = 1e-2
    decay_rate: float = 0.95
    momentum: float = 0.9

    batch_size: int = 128
    epochs: int = 15

    output_folder: str = "./output"
    model_name: str = "ABAE"

    def output_path(self):
        path = f"{self.output_folder}/{self.model_name}"
        # If the folder does not exist we take care of it immediately.
        Path(path).mkdir(parents=True, exist_ok=True)
        return path


class ABAEModelManager:
    def __init__(self, config: ABAEGeneratorConfig):
        self.c = config  # Generator config. (self)
        self.mc = ABAEConfig()  # Model config (ABAEModel)

        # The generator is initialized but not all the objets. Any call on it before generating the embedding
        # will of course fail. The generate_embeddings_models fn has to be called first.
        self.model_generator: ABAE | None = ABAE(self.mc)

        self.__train_model: keras.Model | None = None
        self.__inference_model: keras.Model | None = None

    def considered_path(self):
        return f"{self.c.output_path()}/{self.c.model_name}.keras"

    def generate_embeddings_models(self, corpus_file: str, persist: bool = True, override_existing: bool = False):
        keep = not override_existing
        corpus = CorpusLoaderUtility(column_name="comments").load(corpus_file)
        # Create word embeddings model (word2vec in our implementation currently)
        self.mc.embeddings_model = WordEmbedding(
            self.c.embedding_size, self.c.output_path(), "word_embedding", self.c.min_word_count, self.c.max_vocab_size
        )
        self.mc.embeddings_model.generate(corpus, persist, False, keep)

        # Initialize the aspect embedding matrix based on the previous word embeddings.
        self.mc.aspect_embeddings_model = AspectEmbedding(
            self.c.aspect_size, self.c.embedding_size, self.c.output_path(), "aspect_embedding"
        )
        self.mc.aspect_embeddings_model.generate(self.mc.embeddings_model.weights(), persist, keep)

    def get_default_optimizer(self, ds_length: int) -> Optimizer:
        return keras.optimizers.SGD(
            learning_rate=ExponentialDecay(
                initial_learning_rate=self.c.learning_rate,
                # Every 10% of training process we reduce the learning rate,
                decay_steps=.1 * self.c.epochs * (ds_length / self.c.batch_size),
                decay_rate=self.c.decay_rate
            ),
            momentum=self.c.momentum,
        ) if ds_length is not None else 'adam'  # If it was not passed we haven't tuned yet to we lean to the adam optimizer

    def get_compiled_model(self, optimizer: str | Optimizer = None, load_existing: bool = False, refresh: bool = True):
        # Returns the latest init of the model as we do not want to refresh
        if not refresh and self.__train_model is not None:
            return self.__train_model

        # Create the model (if needed load from fs as asked by load_existing.
        self.__train_model = self.model_generator.generate_training_model(
            custom_objects={'max_margin_loss': max_margin_loss},
            existing_model_path=self.considered_path() if load_existing else None
        )

        opt = self.get_default_optimizer(self.c.train_ds_expected_size) if optimizer is None else optimizer
        self.__train_model.compile(optimizer=opt, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        return self.__train_model

    def get_inference_model(self, refresh: bool = False):
        if refresh or self.__inference_model is None:
            self.__inference_model = self.model_generator.generate_inference_model(
                custom_objects={'max_margin_loss': max_margin_loss},
                existing_model_path=f"{self.c.output_path()}/{self.c.model_name}.keras"
            )

        return self.__inference_model

    def train(self, ds: PositiveNegativeABAEDataset):
        self.get_compiled_model(refresh=False)  # In case this called wasn't done before.
        train_dataloader = DataLoader(dataset=ds, batch_size=self.c.batch_size, shuffle=True)

        history = self.__train_model.fit(train_dataloader, epochs=self.c.epochs, verbose=1, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.c.model_name}.keras", monitor='max_margin')
        ])

        self.__train_model.save(self.considered_path())
        return history, self.get_inference_model(refresh=True)
