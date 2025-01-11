import argparse
from core.train import AbaeModelManager, AbaeModelConfiguration
from core.evaluation import normalize, get_aspect_top_k_words, coherence_per_aspect
import core.dataset as dataset

if __name__ == "__main__":
    # todo add missing args
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model-name", dest="model_name", type=str, metavar='<str>', required=True,
                        help="The name of the model")
    parser.add_argument("-a", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', required=True,
                        help="The number of aspects to use")
    parser.add_argument("-c", "--corpus-file", dest="corpus_file", type=str, metavar='<str>',
                        default="../data/processed-dataset/full/256k.preprocessed.csv",
                        help="The path to the corpus file (default: '../data/processed-dataset/full/256k.preprocessed.csv')")

    a = parser.parse_args()

    config = AbaeModelConfiguration(
        corpus_file=a.corpus_file, model_name=a.model_name, aspect_size=a.aspect_size, output_path="../output"
    )

    manager = AbaeModelManager(config)
    inference_model = manager.__prepare_evaluation_model()

    word_emb = normalize(inference_model.get_layer('word_embedding').weights[0].value.data)
    aspect_embeddings = normalize(inference_model.get_layer('weighted_aspect_emb').W)

    print(f"Word embeddings shape: {word_emb.shape}")
    inv_vocab = manager.embedding_model.model.wv.index_to_key

    aspects_top_k_words = [get_aspect_top_k_words(a, word_emb, inv_vocab, top_k=50) for a in aspect_embeddings]

    vocab = manager.embedding_model.model.wv.key_to_index
    ds = dataset.PositiveNegativeCommentGeneratorDataset(config.corpus_file, vocab, 15)

    aspect_words = [[word[0] for word in aspect] for aspect in aspects_top_k_words]
    coherence, coherence_model = coherence_per_aspect(aspect_words, ds.text_ds, 10)
    print(coherence)
