from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import keras
from keras import Optimizer
from keras.src.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from core.utils import max_margin_loss
from main.abae.dataset import PositiveNegativeABAEDataset
from main.embedding import Word2VecWrapper
from main.abae.embedding import AspectEmbedding
from main.abae.model import BaseABAE, ABAE, ABAEGeneratorConfig, SelfAttentionABAE
from main.utils import CorpusLoaderUtility


@dataclass
class ABAEManagerConfig(ABAEGeneratorConfig):
    min_word_count: int = 5
    max_vocab_size: int | None = None
    # todo add optimizer
    batch_size: int = 128
    epochs: int = 15

    corpus_file_path: str = ""
    model_name: str = "ABAE"
    output_folder: str = "./output"

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

    def train(self, ds: PositiveNegativeABAEDataset):
        # Just a utility function, one can directly work on the model.
        self.get_compiled_model(refresh=False)
        train_dataloader = DataLoader(dataset=ds, batch_size=self.c.batch_size, shuffle=True)

        history = self.__train_model.fit(train_dataloader, epochs=self.c.epochs, verbose=1, callbacks=[
            # Every epoch the model is persisted on the FS. (tmp)
            ModelCheckpoint(filepath=f"./tmp/ckpt/{self.c.model_name}.keras", monitor='max_margin')
        ])

        self.__train_model.save(self.considered_path)
        return history, self.get_inference_model(refresh=True)


class ABAEManagerFactory:
    @abstractmethod
    def factory_method(self, config: ABAEManagerConfig, override_existing: bool = False) -> ABAEManager:
        pass

    @staticmethod
    def make_emb_wrapper(corpus: list, config: ABAEManagerConfig, override: bool) -> Word2VecWrapper:

        embeddings_file = f"{config.output_path()}/{config.model_name}.embeddings.model"

        if not override and Path(embeddings_file).exists():
            return Word2VecWrapper.from_existing(embeddings_file)

        emb_model = Word2VecWrapper(config.embedding_size, config.min_word_count, config.max_vocab_size)
        emb_model.generate(corpus)
        emb_model.persist(embeddings_file)
        return emb_model

    @staticmethod
    def make_aspect_emb_wrapper(weights, config: ABAEManagerConfig, override: bool) -> AspectEmbedding:

        aspect_embeddings_file = f"{config.output_path()}/{config.model_name}.aspect_embeddings.model"
        aspect_model: AspectEmbedding = AspectEmbedding(config.aspect_size, config.embedding_size)

        if not override and Path(aspect_embeddings_file).exists():
            aspect_model.load_existing(aspect_embeddings_file)

        aspect_model.generate(weights)
        aspect_model.persist(aspect_embeddings_file)
        return aspect_model

    def make_embeddings(self, config: ABAEManagerConfig, override: bool = False) -> tuple[
        Word2VecWrapper, AspectEmbedding]:

        corpus = CorpusLoaderUtility(column_name="comments").load(config.corpus_file_path)
        emb_model: Word2VecWrapper = self.make_emb_wrapper(corpus, config, override)
        aspect_model: AspectEmbedding = self.make_aspect_emb_wrapper(emb_model.weights(), config, override)
        return emb_model, aspect_model


class ABAEDefaultManagerFactory(ABAEManagerFactory):
    def factory_method(self, config: ABAEManagerConfig, override_existing: bool = False) -> ABAEManager:
        e, a = self.make_embeddings(config, override_existing)
        return ABAEManager(config, ABAE(config, e, a))


class ABAESelfAttentionManagerFactory(ABAEManagerFactory):
    def factory_method(self, config: ABAEManagerConfig, override_existing: bool = False) -> ABAEManager:
        e, a = self.make_embeddings(config, override_existing)
        return ABAEManager(config, SelfAttentionABAE(config, e, a))
