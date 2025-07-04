import concurrent.futures
import glob
import multiprocessing
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from .embedding import BNCContextEmbedder
from .metrics import (
    ExperimentResults,
    WordMetrics,
    WordResults,
    get_metrics,
    intra_inter_similarity,
)


def get_embeddings_and_metrics(
    params: dict, dump: bool = True, verbose: bool = True
) -> ExperimentResults:
    """Run an experiment using the parameters provided.

    Args:
        params (dict): A dictionary of parameters for the experiment.
        dump (bool, optional): Whether to dump the data or return it. Defaults to True.

    Returns:
        Results: A dataclass containing the results of the experiment.
    """
    embedder = BNCContextEmbedder(model_name=params["model_name"])

    files = sorted(glob.glob(f"{params['text_folder']}*.txt"))

    results_by_word = {}

    target_words = params["target_word"]
    if not isinstance(target_words, list):
        target_words = [target_words]

    for n, target_word in enumerate(target_words):
        if verbose:
            print(
                f"Processing embeddings for target word ({n+1} of {len(target_words)}) : '{target_word}'..."
            )
            disable_tqdm = False
        else:
            disable_tqdm = True

        context_data = {}

        for i, file in enumerate(tqdm(files, disable=disable_tqdm)):
            embeddings, contexts, indices = embedder(file, target_word, params["context_window"])

            context_data[i] = {"indices": indices, "contexts": contexts, "embeddings": embeddings}

        non_empty_embeddings, non_empty_contexts = [], []
        for file in context_data.values():
            for i, context in enumerate(file["contexts"]):
                if len(context) > 0:
                    non_empty_embeddings.append(file["embeddings"][i])
                    non_empty_contexts.append(context)

        non_empty_embeddings = torch.stack(non_empty_embeddings)

        # decode the contexts
        contexts = [embedder.tokenizer.decode(context) for context in non_empty_contexts]

        metrics = get_metrics(non_empty_embeddings, target_word)

        word_results = WordResults(
            word=target_word,
            embeddings=non_empty_embeddings,
            contexts=contexts,
            metrics=metrics,
        )

        results_by_word[target_word] = word_results

    payload = ExperimentResults(
        params=params,
        results_by_word=results_by_word,
    )

    if dump:
        print("Dumping data...")
        # check to see if experiments folder exists
        if not os.path.exists(params["output_folder"]):
            os.mkdir(params["output_folder"])
        torch.save(
            payload,
            f"{params['output_folder']}{params['model_name']}_{params['context_window']}_{params['text_folder'].split('/')[-2]}.pt",
        )
        print("Data dumped!")

    if verbose:
        print("Experiment complete!")

    return payload


def load_results():
    if os.path.exists('results.pkl'):
        with open('results.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return {}
    
def update_and_save_results(results):
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

def job(context_window, grid, KEYWORD):

    exp_params = {
        'model_name': 'bert-large-uncased',
        'target_word': [KEYWORD],
        'context_window': context_window,
        'text_folder': '../data/bnc_dataset/', # remove first dot depending on where you are running this notebook/storing the data...
        'output_folder': '../experiments/',
    }

    experiment_data = get_embeddings_and_metrics(exp_params, dump=False, verbose=False)

    best_score = -1
    best_params = None

    for params in grid:
        pca_embeddings = PCA(params['n_components']).fit_transform(experiment_data.results_by_word[KEYWORD].embeddings.detach().numpy())
        gmm = GaussianMixture(n_components=params['n_clusters'], random_state=0, covariance_type='full').fit(pca_embeddings)

        score = silhouette_score(pca_embeddings, gmm.predict(pca_embeddings), metric='cosine')

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score, context_window


def multi_context(params):
    word_list = params['word_list']
    context_windows = params['context_windows']
    param_grid = params['param_grid']

    num_cores = multiprocessing.cpu_count()

    grid = ParameterGrid(param_grid)
    results = load_results()

    for KEYWORD in tqdm(word_list, total=len(word_list), ascii="░▒█", colour='GREEN', desc="Running..."):

        keyword_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(job, context_window, grid, KEYWORD) for context_window in context_windows]
        
            for future in concurrent.futures.as_completed(futures):
                best_params, best_score, context_window = future.result()
                keyword_results[context_window] = {'best_params': best_params, 'best_score': best_score}

        results[KEYWORD] = keyword_results

        update_and_save_results(results)

    return results


def get_consensus_labels_and_ARI(keyword,
                                 experiment_data_all,
                                 optimal_params,
                                 forced_components=False,
                                 labels=True):
    
    if forced_components:
        n_components = forced_components
    else:
        n_components = optimal_params['n_components']
    

    n_clusters = optimal_params['n_clusters']

    X = experiment_data_all.results_by_word[keyword].embeddings.detach().numpy()

    X_pca = PCA(n_components=n_components).fit_transform(X)

    # Parameters
    n_init = 1000 

    # Store cluster assignments for each run
    cluster_assignments = []
    Zs = []

    for _ in range(n_init):
        gmm = GaussianMixture(n_components=n_clusters, n_init=1, random_state=np.random.randint(100000))
        gmm.fit(X_pca)
        labels = gmm.predict(X=X_pca)
        cluster_assignments.append(labels)

        if forced_components:
            x_min = X_pca[:, 0].min() - 1
            x_max = X_pca[:, 0].max() + 1
            y_min = X_pca[:, 1].min() - 1
            y_max = X_pca[:, 1].max() + 1

            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            XX, Y = np.meshgrid(x, y)
            XXX = np.array([XX.ravel(), Y.ravel()]).T
            Z = -gmm.score_samples(XXX)
            Z = Z.reshape(XX.shape)
            Zs.append(Z)

    # Assessing stability using Adjusted Rand Index (ARI)
    ari_scores = []

    for i in range(n_init):
        for j in range(i + 1, n_init):
            ari = adjusted_rand_score(cluster_assignments[i], cluster_assignments[j])
            ari_scores.append(ari)

    # Consensus Clustering
    consensus_matrix = np.zeros((X_pca.shape[0], X_pca.shape[0]))

    for labels in cluster_assignments:
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
                    if i != j:
                        consensus_matrix[j, i] += 1

    consensus_matrix /= n_init

    # Perform hierarchical clustering on the consensus matrix
    link = linkage(consensus_matrix, method='average')
    consensus_labels = fcluster(link, t=n_clusters, criterion='maxclust')

    intra_similarity, inter_similarity = intra_inter_similarity(X_pca, consensus_labels)

    return X_pca, ari_scores, consensus_labels, Zs, intra_similarity, inter_similarity
        