from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MetricsResult:
    """Results from embedding metrics computation."""
    mev: float
    average_similarity: float
    similarity_std: float
    similarity_matrix: np.ndarray
    intra_similarity: Optional[float] = None
    inter_similarity: Optional[float] = None


class MetricsComputer:
    """Compute metrics from embeddings"""
    

@dataclass
class EmbeddingMetrics:
    """
    Compute and store embedding metrics with optional anisotropic correction.
    
    Metrics are computed lazily - only when accessed. Anisotropic baselines are
    computed once and reused across all metrics that need them.
    """
    embeddings: np.ndarray
    labels: Optional[np.ndarray] = None
    anisotropic_embeddings: Optional[np.ndarray] = None
    sample_size: Optional[int] = None
    
    # Cached results
    _similarity_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _anisotropic_baselines: Optional[Tuple[float, float]] = field(default=None, init=False, repr=False)
    _mev_raw: Optional[float] = field(default=None, init=False, repr=False)
    _avg_similarity_raw: Optional[float] = field(default=None, init=False, repr=False)
    _intra_inter_raw: Optional[Tuple[float, float]] = field(default=None, init=False, repr=False)
    
    @property
    def anisotropic_baselines(self) -> Tuple[float, float]:
        """Compute anisotropic baselines once and cache."""
        if self._anisotropic_baselines is None:
            if self.anisotropic_embeddings is None:
                raise ValueError("anisotropic_embeddings must be provided for anisotropic correction")
            
            # Use the baseline calculation functions
            _, aniso_sim, _ = average_pairwise_cosine_similarity(self.anisotropic_embeddings)
            aniso_mev = mev(self.anisotropic_embeddings)
            self._anisotropic_baselines = (aniso_sim, aniso_mev)
        
        return self._anisotropic_baselines
    
    @property
    def similarity_matrix(self) -> np.ndarray:
        """Compute similarity matrix once and cache."""
        if self._similarity_matrix is None:
            self._similarity_matrix = cosine_similarity(self.embeddings)
        return self._similarity_matrix
    
    # Raw metrics (no anisotropic correction)
    @property
    def mev_raw(self) -> float:
        """Maximum Explained Variance (raw)."""
        if self._mev_raw is None:
            self._mev_raw = mev(self.embeddings)
        return self._mev_raw
    
    @property
    def average_similarity_raw(self) -> Tuple[float, float]:
        """Average pairwise cosine similarity (raw)."""
        if self._avg_similarity_raw is None:
            similarity_matrix, avg_sim, std_sim = average_pairwise_cosine_similarity(self.embeddings)
            self._avg_similarity_raw = (avg_sim, std_sim)
        return self._avg_similarity_raw
    
    @property
    def intra_inter_similarity_raw(self) -> Tuple[float, float]:
        """Intra and inter cluster similarity (raw)."""
        if self.labels is None:
            raise ValueError("Labels required for intra/inter similarity")
        
        if self._intra_inter_raw is None:
            intra, inter = intra_inter_similarity(self.embeddings, self.labels)
            self._intra_inter_raw = (intra, inter)
        return self._intra_inter_raw
    
    # Anisotropic-corrected metrics
    @property
    def mev_corrected(self) -> float:
        """MEV with anisotropic correction."""
        _, aniso_mev = self.anisotropic_baselines
        return self.mev_raw - aniso_mev
    
    @property
    def average_similarity_corrected(self) -> float:
        """Average similarity with anisotropic correction."""
        aniso_sim, _ = self.anisotropic_baselines
        avg_sim, std_sim = self.average_similarity_raw
        return avg_sim - aniso_sim
    
    @property
    def intra_inter_similarity_corrected(self) -> Tuple[float, float]:
        """Intra/inter similarity with anisotropic correction."""
        aniso_sim, _ = self.anisotropic_baselines
        intra_raw, inter_raw = self.intra_inter_similarity_raw
        return (intra_raw - aniso_sim, inter_raw - aniso_sim)
    
    def get_metrics(self, corrected: bool = True, include: Optional[list] = None) -> dict:
        suffix = "_corrected" if corrected else "_raw"
        
        all_metrics = {
            "mev": getattr(self, f"mev{suffix}"),
        }

        all_metrics['similarity_matrix'] = self.similarity_matrix
        
        # Handle average_similarity differently for raw vs corrected
        if corrected:
            all_metrics["average_similarity"] = self.average_similarity_corrected
            all_metrics["similarity_std"] = self.average_similarity_raw[1]  # std is always raw
        else:
            avg_sim, std_sim = self.average_similarity_raw
            all_metrics["average_similarity"] = avg_sim
            all_metrics["similarity_std"] = std_sim
        
        # Add intra/inter metrics if labels available
        if self.labels is not None:
            intra, inter = getattr(self, f"intra_inter_similarity{suffix}")
            all_metrics.update({
                "intra_similarity": intra,
                "inter_similarity": inter,
            })
        
        # Filter if specific metrics requested
        if include:
            all_metrics = {k: v for k, v in all_metrics.items() if k in include}
        
        return all_metrics


def average_pairwise_cosine_similarity(embeddings):
    """Original function - kept for compatibility."""
    similarities = cosine_similarity(embeddings)
    upper_triangle = np.triu(similarities)
    indices = np.triu_indices(len(similarities), k=1)
    upper_vector = upper_triangle[indices]
    average_similarity = np.nanmean(upper_vector)
    std_similarity = np.nanstd(upper_vector)
    return similarities, average_similarity, std_similarity

def mev(word_embeddings: np.ndarray) -> float:
    """Original function - kept for compatibility."""
    pca = PCA(n_components=1)
    pca.fit(word_embeddings)
    return pca.explained_variance_ratio_[0]

def intra_inter_similarity(embeddings, labels, sample=None):
    """Original function - kept for compatibility."""
    if sample:
        n = embeddings.shape[0]
        indices = np.random.choice(n, sample, replace=False)
        embeddings = embeddings[indices]
        labels = np.array(labels)[indices]
    
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix = (similarity_matrix + 1) / 2
    
    labels = np.array(labels)
    intra_mask = (labels[:, None] == labels)
    inter_mask = ~intra_mask
    
    np.fill_diagonal(intra_mask, False)
    np.fill_diagonal(inter_mask, False)
    
    intra_similarity = np.sum(similarity_matrix[intra_mask]) / np.sum(intra_mask)
    inter_similarity = np.sum(similarity_matrix[inter_mask]) / np.sum(inter_mask)
    
    return intra_similarity, inter_similarity