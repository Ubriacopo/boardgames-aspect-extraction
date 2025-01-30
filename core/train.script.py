import argparse

from core.train import ABAEModelConfiguration, ABAEModelManager
from core.dataset import PositiveNegativeCommentGeneratorDataset

from torch.utils.data import DataLoader

# todo fai function e butta gli script. Sono scomodi e non mi piacciono.
if __name__ == "__main__":
    # os.environ['KERAS_BACKEND'] = "torch"
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=64)
    p.add_argument("-e", "--epochs", dest="epochs", type=int, metavar='<int>', default=10)
    p.add_argument("-c", "--corpus", dest="corpus_file", type=str, metavar='<str>', required=True)
    p.add_argument("-m", "--model-name", dest="model_name", type=str, metavar='<str>', required=True)
    p.add_argument("--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=16)
    p.add_argument("--max-vocab-size", dest="max_vocab_size", type=int, metavar='<int>', default=None)
    p.add_argument("--embedding-size", dest="embedding_size", type=int, metavar='<int>', default=200)
    p.add_argument("--max-sequence-length", dest="max_sequence_length", type=int, metavar='<int>', default=80)
    p.add_argument("--negative-sample-size", dest="negative_sample_size", type=int, metavar='<int>', default=15)
    p.add_argument("--output-path", dest="output_path", type=str, metavar='<str>', default="../output")
    a = p.parse_args()

    config = ABAEModelConfiguration(
        corpus_file=a.corpus_file,
        model_name=a.model_name,
        aspect_size=a.aspect_size,
        max_vocab_size=a.max_vocab_size,
        embedding_size=a.embedding_size,
        max_sequence_length=a.max_sequence_length,
        negative_sample_size=a.negative_sample_size,
        output_path=a.output_path,
        batch_size=a.batch_size,
        epochs=a.epochs,
    )
    print(config)

    manager = ABAEModelManager(config)
    train_model = manager.__prepare_training_model('adam')
    train_dataset = PositiveNegativeCommentGeneratorDataset(
        vocabulary=manager.embedding_model.vocabulary(),
        csv_dataset_path=config.corpus_file, negative_size=15
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_model.fit(train_dataloader, epochs=config.epochs)
