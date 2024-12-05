import itertools
import math

import numpy as np
import torch

from core.dataset import PositiveNegativeCommentGeneratorDataset


def normalize_embedding_matrix(embedding_matrix: torch.tensor) -> torch.tensor:
    return embedding_matrix / torch.linalg.norm(embedding_matrix, dim=-1, keepdim=True)


def extract_top_k_words_of_aspect(aspect: torch.tensor, word_embeddings,
                                  inverse_vocabulary: dict, top_k: int = 25, verbose: bool = True) -> list:
    """
    This function will extract the top k words of each aspect.
    @return: A list of aspects with their top k words. [str, float, int]
    """
    aspect = aspect.cpu()
    similarity = word_embeddings.matmul(aspect).detach().numpy()

    ordered_words = np.argsort(similarity)[::-1]
    top_k_words = [(inverse_vocabulary[w], similarity[w], w) for w in ordered_words[:top_k]]
    # Normalize the data for comparison

    if verbose:
        print("\nGiven aspect most representative words are:")
        for i in top_k_words:
            # hr][/i is not a valid word. meh. todo: Process better the words.
            print("Word: ", i[0], f"({i[1]})")

    return top_k_words


def coherence(top_n_words: list, corpus: PositiveNegativeCommentGeneratorDataset):
    coherence_score = 0

    document_frequency_values = dict()
    for w in top_n_words:
        document_frequency_values[w] = document_frequency(w, corpus)

    # Exponential complexity. We can optimize this.
    document_co_frequency_values = dict()
    for w_i, w_j in itertools.combinations(top_n_words, 2):
        co_occurrence = document_co_occurrence(w_i, w_j, corpus)
        document_co_frequency_values[(w_i, w_j)] = co_occurrence
        document_co_frequency_values[(w_j, w_i)] = co_occurrence

    for i in range(2, len(top_n_words)):
        w_i = top_n_words[i]

        for j in range(i - 1):
            w_j = top_n_words[j]

            coherence_score += np.log((document_co_frequency_values[(w_i, w_j)] + 1) / document_frequency_values[w_j])

    return coherence_score / len(top_n_words)


def document_frequency(a: str | int, corpus: PositiveNegativeCommentGeneratorDataset) -> float:
    return corpus.dataset.apply(lambda x: a in x).value_counts().get(True, 0)


def document_co_occurrence(a: str | int, b: str | int, corpus: PositiveNegativeCommentGeneratorDataset) -> float:
    return corpus.dataset.apply(lambda x: a in x and b in x).value_counts().get(True, 0)
