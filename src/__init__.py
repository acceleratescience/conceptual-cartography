"""Conceptual Cartography - Embedding landscape analysis toolkit.

A comprehensive toolkit for analyzing the conceptual landscape of language models
through embedding analysis, metrics computation, and interactive visualization.

Example Usage:
    >>> from src import AppConfig, ContextEmbedder, EmbeddingMetrics
    >>> from src.core import load_config_from_yaml
    >>> 
    >>> # Load configuration
    >>> config = load_config_from_yaml("config.yaml")
    >>> 
    >>> # Generate embeddings
    >>> embedder = ContextEmbedder(config.model.model_name)
    >>> embeddings = embedder(sentences, target_word="bank")
    >>> 
    >>> # Compute metrics
    >>> metrics = EmbeddingMetrics(embeddings['final_embeddings'])
    >>> results = metrics.get_metrics()
"""

# Main configuration and utilities
from .core import (
    AppConfig,
    ModelConfigs,
    DataConfigs, 
    ExperimentConfigs,
    MetricConfigs,
    LandscapeConfigs,
    ContextEmbedder,
    load_config_from_yaml,
    load_sentences,
    save_output,
    save_metrics,
    save_landscape,
)

# Analysis functionality
from .analysis import (
    EmbeddingMetrics,
    Landscape,
    get_landscape,
    optimize_clustering,
    average_pairwise_cosine_similarity,
    mev,
    intra_inter_similarity,
)

__version__ = "0.1.0"

# Define what gets imported with "from src import *"
__all__ = [
    # Core configuration
    "AppConfig",
    "ModelConfigs",
    "DataConfigs", 
    "ExperimentConfigs",
    "MetricConfigs",
    "LandscapeConfigs",
    
    # Core functionality
    "ContextEmbedder",
    "load_config_from_yaml",
    "load_sentences",
    "save_output", 
    "save_metrics",
    "save_landscape",
    
    # Analysis
    "EmbeddingMetrics",
    "Landscape",
    "get_landscape",
    "optimize_clustering",
    "average_pairwise_cosine_similarity",
    "mev",
    "intra_inter_similarity",
    
    # Metadata
    "__version__",
]
