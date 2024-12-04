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
    @return: A list of aspects with their top k words.
    """
    # We will extract the top k words of each aspect
    top_k_words = []

    aspect = aspect.cpu()
    similarity = word_embeddings.matmul(aspect).detach().numpy()

    ordered_words = np.argsort(similarity)[::-1]
    desc_list = [(inverse_vocabulary[w], similarity[w]) for w in ordered_words[:top_k]]
    # Normalize the data for comparison

    if verbose:
        print("\nGiven aspect most representative words are:")
        for i in desc_list:
            # hr][/i is not a valid word. meh. todo: Process better the words.
            print("Word: ", i[0], f"({i[1]})")

    return top_k_words


def coherence(top_n_words: list, corpus: PositiveNegativeCommentGeneratorDataset):
    coherence_score = 0
    itertools.combinations(top_n_words, 2)
    for w_i, w_j in itertools.combinations(top_n_words, 2):
        coherence_score += np.log((document_co_occurrence(w_i, w_j, corpus) + 1) / document_frequency(w_j, corpus))
    return coherence_score


def document_frequency(a: str | int, corpus: PositiveNegativeCommentGeneratorDataset) -> float:
    return corpus.dataset.apply(lambda x: a in x).value_counts()[1]


def document_co_occurrence(a: str | int, b: str | int, corpus: PositiveNegativeCommentGeneratorDataset) -> float:
    return corpus.dataset.apply(lambda x: a in x and b in x).value_counts()[1]
