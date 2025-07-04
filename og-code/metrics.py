from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor


@dataclass
class WordMetrics:
    """A dataclass to store the metrics of an experiment."""

    word: str
    average_pairwise_cosine_similarity: float
    mev: float
    similarities: np.ndarray
    intergroup_similarity: float
    intragroup_similarity: float


@dataclass
class WordResults:
    """A dataclass to store the results of an experiment."""

    word: str
    embeddings: Tensor
    contexts: list
    metrics: WordMetrics


@dataclass
class ExperimentResults:
    """A dataclass to store the results of an experiment."""

    params: dict
    results_by_word: dict  # Key: target word, Value: WordResults


def get_metrics(embeddings: Tensor, word: str) -> WordMetrics:
    """Gets the desired metrics from the embeddings.

    Args:
        embeddings (torch.Tensor): Tensor of embeddings.
        word (str): The word to get the metrics for.

    Returns:
        Metrics: A dataclass containing the metrics.
    """
    similarities, average_similarity, std_similarity = average_pairwise_cosine_similarity(embeddings)
    mev_score = mev(embeddings)

    metrics = WordMetrics(
        word=word,
        average_pairwise_cosine_similarity=average_similarity,
        mev=mev_score,
        similarities=similarities,
        intergroup_similarity=None,
        intragroup_similarity=None,
    )

    return metrics


def average_pairwise_cosine_similarity(embeddings: Tensor) -> float:
    """Computed using the average pairwise cosine similarities of the representations of
    its instances in that layer. layer does not contextualize the representations at
    all, then the self similarity is 1 (i.e., the representations are identical across
    all contexts)

    Args:
        embeddings (Tensor): The embeddings of a word across all contexts

    Returns:
        float: The average pairwise cosine similarity of the representations of its
            instances in that layer
    """
    similarities = cosine_similarity(embeddings)
    upper_triangle = np.triu(similarities)
    indices = np.triu_indices(len(similarities), k=1)
    upper_vector = upper_triangle[indices]
    average_similarity = np.nanmean(upper_vector)
    std_similarity = np.nanstd(upper_vector)

    return similarities, average_similarity, std_similarity


def mev(word_embeddings: np.ndarray) -> float:
    """The Maximum Explainable Variance (MEV) metric aims to provide an upper bound on
    how well a static embedding could replace a word's contextualized representations in
    a given layer of a model. We quantify the proportion of variance in the word's
    contextualized representations that can be explained by their first principal
    component replacement for the contextualized representations of the word in that
    layer. If mev is 1, it indicates that a static embedding would be a perfect
    replacement for the contextualized representations

    Args:
        word_embeddings (np.ndarray): The embeddings of a word across all contexts

    Returns:
        float: The Maximum Explainable Variance (MEV) metric
    """
    pca = PCA(n_components=1)
    pca.fit(word_embeddings)
    proportion_variance_explained = pca.explained_variance_ratio_[0]

    return proportion_variance_explained


def intra_inter_similarity_og(embeddings: np.ndarray, labels: list) -> tuple:
    n = embeddings.shape[0]
    intra_similarity_sum = 0
    inter_similarity_sum = 0
    intra_count = 0
    inter_count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]

                # Ensure similarity is between 0 and 1
                # similarity = (similarity + 1) / 2
                similarity = (similarity - (-1)) / (1 - (-1))

                if labels[i] == labels[j]:
                    intra_similarity_sum += similarity
                    intra_count += 1
                else:
                    inter_similarity_sum += similarity
                    inter_count += 1

    intra_similarity = intra_similarity_sum / intra_count
    inter_similarity = inter_similarity_sum / inter_count

    return intra_similarity, inter_similarity

def intra_inter_similarity(embeddings, labels):
    # sample 100 embeddings randomly
    n = embeddings.shape[0]
    indices = np.random.choice(n, 100, replace=False)
    embeddings = embeddings[indices]
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix = (similarity_matrix + 1) / 2

    labels = np.array(labels)
    intra_mask = (labels[:, None] == labels)
    inter_mask = ~intra_mask

    np.fill_diagonal(intra_mask, False)
    np.fill_diagonal(inter_mask, False)

    intra_similarity_sum = np.sum(similarity_matrix[intra_mask])
    inter_similarity_sum = np.sum(similarity_matrix[inter_mask])

    intra_count = np.sum(intra_mask)
    inter_count = np.sum(inter_mask)

    intra_similarity = intra_similarity_sum / intra_count
    inter_similarity = inter_similarity_sum / inter_count

    return intra_similarity, inter_similarity


def anisotropic_sim_mev(experiment_data_all, word_list, samples=100):
    sample_embeddings = []
    for word in word_list:
        embeddings = experiment_data_all.results_by_word[word].embeddings
        sample_embeddings.append(embeddings[np.random.choice(embeddings.shape[0], samples, replace=False)])

    sample_embeddings = np.concatenate(sample_embeddings)

    _, anisotropic_sim, _ = average_pairwise_cosine_similarity(sample_embeddings)
    anisotropic_mev = mev(sample_embeddings)

    for word in word_list:
        experiment_data_all.results_by_word[word].metrics.anisotropic_sim_adjusted = experiment_data_all.results_by_word[word].metrics.average_pairwise_cosine_similarity - anisotropic_sim
        experiment_data_all.results_by_word[word].metrics.anisotropic_mev_adjusted = experiment_data_all.results_by_word[word].metrics.mev - anisotropic_mev

    return anisotropic_sim, anisotropic_mev, experiment_data_all