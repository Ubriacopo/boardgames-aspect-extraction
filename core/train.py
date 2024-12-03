from torch.utils.data import DataLoader

from core.utils import max_margin_loss
from core.embeddings import WordEmbedding, AspectEmbedding
from core.model import ABAEGenerator
from core.utils import LoadCorpusUtility


class CoreModelManager:
    # Could be very well a simple datacalss to pass todo
    def __init__(self, corpus_file: str, embeddings_file: str, aspects_file: str, model_file: str,
                 mav_vocab_size: int = 10000, word_embedding_size: int = 128, aspect_embedding_size: int = 128):
        self.embedding_model: WordEmbedding | None = None
        self.aspect_model: AspectEmbedding | None = None

        self.corpus_file: str = corpus_file
        self.embeddings_file: str = embeddings_file
        self.aspects_file: str = aspects_file
        self.model_file: str = model_file
        self.max_vocab_size: int = mav_vocab_size
        self.word_embedding_size: int = word_embedding_size
        self.aspect_embedding_size: int = aspect_embedding_size
        self.aspect_size: int | None = None

        self._train_model = None
        self.trained = False
        self.generator: ABAEGenerator | None = None

    def prepare_embeddings_step(self, override_existing: bool = False, aspect_size: int = 10):
        self.aspect_size = aspect_size

        self.embedding_model = WordEmbedding(
            LoadCorpusUtility(column_name="comments"),
            max_vocab_size=self.max_vocab_size,
            embedding_size=self.word_embedding_size,
            target_model_file=self.embeddings_file,
            corpus_file=self.corpus_file
        )

        self.aspect_model = AspectEmbedding(
            aspect_size=self.aspect_size,
            embedding_size=self.aspect_embedding_size,
            target_model_file=self.aspects_file,
            base_embeddings=self.embedding_model
        )

        self.embedding_model.load_model(override=override_existing)
        self.aspect_model.load_model(override=override_existing)

    def train_model(self, dataloader: DataLoader, max_sequence_length: int = 256, negative_sample_size: int = 15,
                    epochs: int = 10, optimizer: str = 'SGD', persist_model: bool = True):
        if self.aspect_model is None or self.embedding_model is None:
            raise ValueError("You need to prepare the embeddings before training the model.")

        self.generator = ABAEGenerator(max_sequence_length, negative_sample_size, self.embedding_model, self.aspect_model)
        self._train_model = self.generator.make_training_model(existing_model_path=self.model_file)

        # Compute the model with the max margin loss function and custom choice Optimizer
        self._train_model.compile(optimizer=optimizer, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        history = self._train_model.fit(x=dataloader, batch_size=64, epochs=epochs)
        self.trained = True

        if persist_model:
            self._train_model.save(self.model_file)

        return history

    def get_trained_model(self):
        return self.generator.make_model(self.model_file)

