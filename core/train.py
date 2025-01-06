import argparse

from dataclasses import dataclass
from pathlib import Path

from core.utils import max_margin_loss
from core.embeddings import WordEmbedding, AspectEmbedding
from core.model import ABAEGenerator
from core.utils import LoadCorpusUtility


# todo fix more params
@dataclass
class AbaeModelConfiguration:
    """
      Configuration for the ABAE model.

      Attributes:
          corpus_file: Path to the corpus file.
          model_name: str Name of model
          max_vocab_size: Maximum size of the vocabulary.
          embedding_size: Size of the word embeddings.
          aspect_embedding_size: Size of the aspect embeddings.
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
    aspect_embedding_size: int = 200

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

        self.__load_embeddings(override_existing)
        self.model_generator: ABAEGenerator = ABAEGenerator(
            self.config.max_sequence_length, self.config.negative_sample_size, self.embedding_model, self.aspect_model
        )

        # The training model instance.
        self._t_model = None
        self._ev_model = None

    def __load_embeddings(self, override_existing: bool):
        c = self.config  # To make it more readable.
        keep = not override_existing

        load_utility = LoadCorpusUtility(column_name="comments")

        self.embedding_model = WordEmbedding(c.embedding_size, self.output_path, max_vocab_size=c.max_vocab_size)
        self.embedding_model.generate(corpus=load_utility.load_data(c.corpus_file), sg=True, load_existing=keep)

        self.aspect_model = AspectEmbedding(c.aspect_size, c.aspect_embedding_size, self.output_path)
        self.aspect_model.generate(embedding_weights=self.embedding_model.weights(), load_existing=keep)

    def prepare_evaluation_model(self):
        model_file_path = f"{self.output_path}/{self.config.model_name}.keras"
        self._ev_model = self.model_generator.make_model(model_file_path)
        return self._ev_model

    def prepare_training_model(self, consider_stored: bool = False, optimizer: str = 'SGD'):
        considered_path = f"{self.output_path}/{self.config.model_name}.keras" if consider_stored else None
        self._t_model = self.model_generator.make_training_model(considered_path)
        self._t_model.compile(optimizer=optimizer, loss=[max_margin_loss], metrics={'max_margin': max_margin_loss})
        return self._t_model

    def persist_model(self):
        if self._t_model is not None:
            self._t_model.save(f"{self.output_path}/{self.config.model_name}.keras")

    def prepare_model(self, train: bool = False):
        return self.prepare_training_model() if train else self.prepare_evaluation_model()


# Main run script. TODO move to scripts folder
if __name__ == "__main__":
    # os.environ['KERAS_BACKEND'] = "torch"
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=64)
    p.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=10)
    p.add_argument("-c", "--corpus", dest="corpus_file", type=str, metavar='<str>', required=True)
    p.add_argument("-m", "--model-name", dest="model_name", type=str, metavar='<str>', required=True)
    p.add_argument("--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=16)
    p.add_argument("--max-vocab-size", dest="max_vocab_size", type=int, metavar='<int>', default=None)
    p.add_argument("--embedding-size", dest="embedding_size", type=int, metavar='<int>', default=128)
    p.add_argument("--aspect-embedding-size", dest="aspect_embedding_size", type=int, metavar='<int>', default=128)
    p.add_argument("--max-sequence-length", dest="max_sequence_length", type=int, metavar='<int>', default=80)
    p.add_argument("--negative-sample-size", dest="negative_sample_size", type=int, metavar='<int>', default=15)
    p.add_argument("--output-path", dest="output_path", type=str, metavar='<str>', default="./output")
    a = p.parse_args()

    config = AbaeModelConfiguration(
        corpus_file=a.corpus_file,
        model_name=a.model_name,
        aspect_size=a.aspect_size,
        max_vocab_size=a.max_vocab_size,
        embedding_size=a.embedding_size,
        aspect_embedding_size=a.aspect_embedding_size,
        max_sequence_length=a.max_sequence_length,
        negative_sample_size=a.negative_sample_size,
        output_path=a.output_path,
        batch_size=a.batch_size,
        epochs=a.epochs,
    )
    print(config)

    manager = AbaeModelManager(config)
    train_model = manager.prepare_training_model('adam')

    from core.dataset import PositiveNegativeCommentGeneratorDataset
    from torch.utils.data import DataLoader

    train_dataset = PositiveNegativeCommentGeneratorDataset(
        vocabulary=manager.embedding_model.vocabulary(),
        csv_dataset_path=config.corpus_file, negative_size=15
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_model.fit(train_dataloader, epochs=config.epochs)

    manager.persist_model()
