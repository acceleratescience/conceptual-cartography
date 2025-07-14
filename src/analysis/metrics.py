from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from src import ContextEmbedder

@dataclass
class MetricsResult:
    """Results from embedding metrics computation."""
    mev: float
    average_similarity: float
    similarity_std: float
    similarity_matrix: np.ndarray
    intra_similarity: Optional[float] = None
    inter_similarity: Optional[float] = None
    mev_correction: Optional[float] = None
    sim_correction: Optional[float] = None


class MetricsComputer:
    """Compute metrics from embeddings with optional anisotropic correction.
    
    This class computes various metrics from embeddings including MEV (Maximum Explained Variance),
    average pairwise cosine similarity, and intra/inter cluster similarities.
    
    Usage:
        computer = MetricsComputer(embeddings, labels=cluster_labels)
        result = computer(corrected=True, anisotropic_embeddings=baseline_embeddings)
    """
    
    def __init__(self,
                 embeddings: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 embedder: Optional[ContextEmbedder] = None,
                 sentences: Optional[list[str]] = None,
                 context_window: Optional[int] = None):
        """Initialize the metrics computer.
        
        Args:
            embeddings: Array of embeddings to analyze
            labels: Optional cluster labels for intra/inter similarity computation
        """
        self.embeddings = embeddings
        self.labels = labels
        self.embedder = embedder
        self.sentences = sentences
        self.context_window = context_window
        
        # Cached computations
        self._anisotropic_baselines: Optional[Tuple[float, float]] = None

    def _generate_anisotropic_embeddings(self, sample_size: Optional[int] = 1000) -> np.ndarray:
        pass

    def __call__(self, 
                 corrected: bool = True,
                 sample_size: Optional[np.ndarray] = 1000,
                 include: Optional[list] = None) -> MetricsResult:
        """Compute metrics and return structured results.
        
        Args:
            corrected: Whether to apply anisotropic correction
            anisotropic_embeddings: Baseline embeddings for anisotropic correction
            include: Optional list of specific metrics to include
            
        Returns:
            MetricsResult containing all computed metrics
        """
        # Compute similarity matrix (cached)
        similarity_matrix = self._get_similarity_matrix()
        
        # Compute raw metrics
        mev_raw = self._compute_mev()
        avg_sim_raw, std_sim = self._compute_average_similarity()

        mev_correction = None
        sim_correction = None

        if corrected:
            if self.embedder is None or self.sentences is None:
                raise ValueError("embedder and sentences required for anisotropy correction")
            
            aniso_embeddings = self._generate_anisotropic_embeddings(sample_size)
            aniso_sim, aniso_mev = self._get_anisotropic_baselines(aniso_embeddings)
            
            # Apply corrections
            mev_final = mev_raw - aniso_mev
            avg_sim_final = avg_sim_raw - aniso_sim
            
            # Store correction values
            mev_correction = aniso_mev
            sim_correction = aniso_sim
        else:
            mev_final = mev_raw
            avg_sim_final = avg_sim_raw

        # Compute intra/inter similarities if labels provided
        intra_sim = None
        inter_sim = None
        if self.labels is not None:
            intra_raw, inter_raw = self._compute_intra_inter_similarity(sample=100)
            if corrected:
                intra_sim = intra_raw - aniso_sim
                inter_sim = inter_raw - aniso_sim
            else:
                intra_sim = intra_raw
                inter_sim = inter_raw
        
        # Create result
        result = MetricsResult(
            mev=mev_final,
            average_similarity=avg_sim_final,
            similarity_std=std_sim,
            similarity_matrix=similarity_matrix,
            intra_similarity=intra_sim,
            inter_similarity=inter_sim,
            mev_correction=mev_correction,
            sim_correction=sim_correction
        )
        
        # Filter results if specific metrics requested
        if include:
            # Create a new result with only requested metrics
            filtered_data = {}
            for field_name in include:
                if hasattr(result, field_name):
                    filtered_data[field_name] = getattr(result, field_name)
            # Return a new result with only the requested fields
            # This is sort of a hack, we should really be only calculating the
            # requested metrics, not gathering them AFTER calculating all of them.
        
        return result

    def _get_similarity_matrix(self) -> np.ndarray:
        """Get cosine similarity matrix (cached)."""
        similarity_matrix = cosine_similarity(self.embeddings)
        return similarity_matrix

    def _compute_mev(self) -> float:
        """Compute Maximum Explained Variance."""
        pca = PCA(n_components=1)
        pca.fit(self.embeddings)
        return float(pca.explained_variance_ratio_[0])
    
    def _compute_average_similarity(self) -> Tuple[float, float]:
        """Compute average pairwise cosine similarity and standard deviation."""
        similarities = self._get_similarity_matrix()
        # Normalize to [0, 1] range for consistency with intra/inter similarity
        similarities = (similarities + 1) / 2
        upper_triangle = np.triu(similarities)
        indices = np.triu_indices(len(similarities), k=1)
        upper_vector = upper_triangle[indices]
        average_similarity = np.nanmean(upper_vector)
        std_similarity = np.nanstd(upper_vector)
        return float(average_similarity), float(std_similarity)
    
    def _compute_intra_inter_similarity(self, sample: Optional[int] = None) -> Tuple[float, float]:
        """Compute intra and inter cluster similarity."""
        if self.labels is None:
            raise ValueError("Labels required for intra/inter similarity computation")
        
        embeddings = self.embeddings
        labels = self.labels
        
        # Optional sampling to reduce computation time
        if sample:
            n = embeddings.shape[0]
            indices = np.random.choice(n, sample, replace=False)
            embeddings = embeddings[indices]
            labels = np.array(labels)[indices]
        
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = (similarity_matrix + 1) / 2  # Normalize to [0, 1]
        
        labels = np.array(labels)
        intra_mask = (labels[:, None] == labels)
        inter_mask = ~intra_mask
        
        # Remove diagonal (self-similarity)
        np.fill_diagonal(intra_mask, False)
        np.fill_diagonal(inter_mask, False)
        
        intra_similarity = np.sum(similarity_matrix[intra_mask]) / np.sum(intra_mask)
        inter_similarity = np.sum(similarity_matrix[inter_mask]) / np.sum(inter_mask)
        
        return float(intra_similarity), float(inter_similarity)
    
    def _get_anisotropic_baselines(self, anisotropic_embeddings: np.ndarray) -> Tuple[float, float]:
        """Compute anisotropic baselines (cached)."""
        if self._anisotropic_baselines is None:
            # Compute similarity baseline
            similarities = cosine_similarity(anisotropic_embeddings)
            # Normalize to [0, 1] range for consistency with other similarity metrics
            similarities = (similarities + 1) / 2
            upper_triangle = np.triu(similarities)
            indices = np.triu_indices(len(similarities), k=1)
            upper_vector = upper_triangle[indices]
            aniso_sim = np.nanmean(upper_vector)
            
            # Compute MEV baseline
            pca = PCA(n_components=1)
            pca.fit(anisotropic_embeddings)
            aniso_mev = pca.explained_variance_ratio_[0]
            
            self._anisotropic_baselines = (float(aniso_sim), float(aniso_mev))
        
        return self._anisotropic_baselines
