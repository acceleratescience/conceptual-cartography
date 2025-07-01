import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, Any



@dataclass
class Landscape:
    """Results from landscape analysis of embeddings.
    
    Represents a 2D conceptual landscape generated from high-dimensional embeddings
    through PCA reduction and Gaussian mixture modeling.
    
    Attributes:
        grid_x: X coordinates of the landscape density grid
        grid_y: Y coordinates of the landscape density grid  
        density_surface: Z values representing concept density across the landscape
        pca_embeddings: 2D PCA-reduced embeddings of the original data points
        cluster_labels: Consensus cluster assignments for each embedding
        ari_scores: Adjusted Rand Index scores measuring clustering consistency
        optimization_results: Dictionary containing optimization parameters and scores
    """
    grid_x: np.ndarray
    grid_y: np.ndarray
    density_surface: np.ndarray
    pca_embeddings: np.ndarray
    cluster_labels: np.ndarray
    ari_scores: list
    optimization_results: Dict[str, Any]


class LandscapeComputer:
    """Compute conceptual landscapes from embeddings.
    
    This class generates 2D conceptual landscapes by:
    1. Optimizing PCA dimensions and clustering parameters
    2. Running multiple clustering iterations for consensus
    3. Computing density surfaces and cluster assignments
    
    Usage:
        computer = LandscapeComputer(embeddings)
        landscape = computer(
            pca_components_range=range(2, 6),
            n_clusters_range=range(2, 6)
        )
    """
    
    def __init__(self, embeddings: np.ndarray):
        """Initialize the landscape computer.
        
        Args:
            embeddings: Array of embeddings to analyze
        """
        self.embeddings = embeddings
        
    def __call__(self, 
                 pca_components_range: range,
                 n_clusters_range: range,
                 covariance_types: list = None,
                 n_init: int = 100) -> Landscape:
        """Compute landscape and return structured results.
        
        Args:
            pca_components_range: Range of PCA components to test
            n_clusters_range: Range of cluster numbers to test
            covariance_types: List of GMM covariance types to test
            n_init: Number of initialization runs for consensus clustering
            
        Returns:
            Landscape containing all computed landscape data
        """
        if covariance_types is None:
            covariance_types = ['full', 'tied', 'diag', 'spherical']
            
        best_params, optimization_df = self._optimize_clustering(
            pca_components_range, n_clusters_range, covariance_types
        )
        
        landscape = self._generate_landscape(
            best_params['best_params'], n_init
        )
        
        landscape.optimization_results = {
            'best_params': best_params['best_params'],
            'best_score': best_params['best_score'],
            'all_results': optimization_df.to_dict('records')
        }
        
        return landscape
    
    def _optimize_clustering(self, 
                           pca_components_range: range,
                           n_clusters_range: range, 
                           covariance_types: list) -> tuple:
        """Optimize clustering parameters using silhouette score."""
        results = []
        best_score = -1
        best_params = None
        best_pca = None
        best_gmm = None

        param_grid = {
            'n_components': pca_components_range,
            'n_clusters': n_clusters_range,
            'covariance_type': covariance_types
        }
        
        for params in ParameterGrid(param_grid):
            pca = PCA(n_components=int(params['n_components']))
            reduced_data = pca.fit_transform(self.embeddings)
            
            gmm = GaussianMixture(
                n_components=int(params['n_clusters']),
                covariance_type=params['covariance_type']
            )
            
            labels = gmm.fit_predict(reduced_data)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(reduced_data, labels)
                
                results.append({
                    'n_pca_components': int(params['n_components']),
                    'n_clusters': int(params['n_clusters']),
                    'covariance_type': params['covariance_type'],
                    'silhouette_score': float(score),
                    'explained_variance_ratio': np.sum(pca.explained_variance_ratio_)
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_pca = pca
                    best_gmm = gmm
        
        # Format best paramsS
        # create dataclass?
        
        best_score = float(best_score)
        best_params['n_components'] = int(best_params['n_components'])
        best_params['n_clusters'] = int(best_params['n_clusters'])

        results_df = pd.DataFrame(results)

        best = {
            'best_params': best_params,
            'best_score': best_score,
            'best_pca': best_pca,
            'best_gmm': best_gmm
        }
        
        return best, results_df
    
    def _generate_landscape(self, optimal_params: dict, n_init: int = 100) -> Landscape:
        """Generate the actual landscape using optimal parameters."""
        n_components = optimal_params['n_components']
        n_clusters = optimal_params['n_clusters']
        
        X_pca = PCA(n_components=2).fit_transform(self.embeddings)
        
        cluster_assignments = []
        density_surfaces = []
        
        for _ in range(n_init):
            gmm = GaussianMixture(
                n_components=n_clusters, 
                n_init=1, 
                random_state=np.random.randint(100000)
            )
            gmm.fit(X_pca)
            labels = gmm.predict(X_pca)
            cluster_assignments.append(labels)
            
            # Landscape generation (Z)
            x_min = X_pca[:, 0].min() - 1
            x_max = X_pca[:, 0].max() + 1
            y_min = X_pca[:, 1].min() - 1
            y_max = X_pca[:, 1].max() + 1
            
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            XX, YY = np.meshgrid(x, y)
            grid_points = np.array([XX.ravel(), YY.ravel()]).T
            Z = -gmm.score_samples(grid_points)
            Z = Z.reshape(XX.shape)
            density_surfaces.append(Z)
        
        # ARI for robustness
        ari_scores = []
        for i in range(n_init):
            for j in range(i + 1, n_init):
                ari = adjusted_rand_score(cluster_assignments[i], cluster_assignments[j])
                ari_scores.append(ari)
        
        # Build consensus matrix
        consensus_matrix = np.zeros((X_pca.shape[0], X_pca.shape[0]))
        
        for labels in cluster_assignments:
            for i in range(len(labels)):
                for j in range(i, len(labels)):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        if i != j:
                            consensus_matrix[j, i] += 1
        
        consensus_matrix /= n_init
        
        # Generate consensus clustering
        link = linkage(consensus_matrix, method='average')
        consensus_labels = fcluster(link, t=n_clusters, criterion='maxclust')
        
        # Compute final grid for visualization
        x_min = X_pca[:, 0].min() - 1
        x_max = X_pca[:, 0].max() + 1
        y_min = X_pca[:, 1].min() - 1
        y_max = X_pca[:, 1].max() + 1
        
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        grid_x, grid_y = np.meshgrid(x, y)
        
        # Average density surface across all runs
        mean_density = np.mean(density_surfaces, axis=0)
        
        return Landscape(
            grid_x=grid_x,
            grid_y=grid_y,
            density_surface=mean_density,
            pca_embeddings=X_pca,
            cluster_labels=consensus_labels,
            ari_scores=ari_scores,
            optimization_results={}  # Will be filled by caller
        )


# Backwards compatibility aliases (can be removed once all code is updated)
optimize_clustering = None  # Mark as deprecated
get_landscape = None  # Mark as deprecated