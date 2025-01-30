import os

os.environ['KERAS_BACKEND'] = "torch"

from uuid import uuid4

from core.evaluation import normalize, get_aspect_top_k_words, coherence_per_aspect
from core.hp_tuning import ABAERandomHyperparametersSelectionWrapper
from core.train import ABAEModelManager, ABAEModelConfiguration
from core.dataset import PositiveNegativeCommentGeneratorDataset

from torch.utils.data import DataLoader

hp_wrapper = ABAERandomHyperparametersSelectionWrapper.create()
configurations = 15  # We try 15 different configurations

seen_configurations = set()
scores = list()

corpus_file = "../output/dataset/pre-processed/tuning.preprocessed.csv"

for i in range(configurations):
    uuid = uuid4()
    parameters = next(hp_wrapper)
    while seen_configurations.__contains__(frozenset(parameters.items())):
        print(f"We already worked on configuration: {parameters}")
        parameters = next(hp_wrapper)  # In case we fetch the same config more than once.
    print(f"Working on configuration: {parameters}")
    seen_configurations.add(frozenset(parameters.items()))

    # Train process
    config = ABAEModelConfiguration(corpus_file=corpus_file, model_name=f"tuning_{uuid}", **parameters)
    manager = ABAEModelManager(config)  # todo pass "persist". We dont want to persist these

    # The dataset generation depends on the embedding model
    ds = PositiveNegativeCommentGeneratorDataset(
        vocabulary=manager.embedding_model.vocabulary(),
        csv_dataset_path=config.corpus_file, negative_size=15
    )

    train_dataloader = DataLoader(dataset=ds, batch_size=config.batch_size, shuffle=True)
    iteration_model = manager.__prepare_training_model()
    iteration_model.fit(train_dataloader, epochs=config.epochs)

    # Evaluate the model
    # We evaluate on the relative coherence between topics.
    print("Evaluating model")
    word_emb = normalize(iteration_model.get_layer('word_embedding').weights[0].value.data)

    aspect_embeddings = normalize(iteration_model.get_layer('weighted_aspect_emb').W)
    print(f"Word embeddings shape: {word_emb.shape}")
    inv_vocab = manager.embedding_model.model.wv.index_to_key

    aspects_top_k_words = [get_aspect_top_k_words(a, word_emb, inv_vocab, top_k=50) for a in aspect_embeddings]

    aspect_words = [[word[0] for word in aspect] for aspect in aspects_top_k_words]
    coherence, coherence_model = coherence_per_aspect(aspect_words, ds.text_ds, 10)
    scores.append(dict(coherence=coherence_model.get_coherence(), parameters=parameters))

# End done.
print(scores)
