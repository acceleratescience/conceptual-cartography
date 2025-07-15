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
    EmbeddingResult,
    load_config_from_yaml,
    load_sentences,
    save_output,
    save_metrics,
    save_landscape,
)

# Analysis functionality
from .analysis import (
    MetricsComputer,
    MetricsResult,
    Landscape,
    LandscapeComputer,
)

__version__ = "0.1.0"


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
    "EmbeddingResult",
    "load_config_from_yaml",
    "load_sentences",
    "save_output", 
    "save_metrics",
    "save_landscape",
    
    # Analysis
    "MetricsComputer",
    "MetricsResult",
    "Landscape",
    "LandscapeComputer",
    
    # Metadata
    "__version__",
]
