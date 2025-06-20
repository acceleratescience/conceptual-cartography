"""Analysis functionality for conceptual cartography.

This module contains components for computing embedding metrics
and generating conceptual landscapes.
"""

from .metrics import (
    EmbeddingMetrics,
    average_pairwise_cosine_similarity,
    mev,
    intra_inter_similarity,
)
from .landscapes import (
    Landscape,
    get_landscape,
    optimize_clustering,
    wrap_text_with_highlight,
)

__all__ = [
    # Metrics functionality
    "EmbeddingMetrics",
    "average_pairwise_cosine_similarity",
    "mev",
    "intra_inter_similarity",
    # Landscape functionality
    "Landscape",
    "get_landscape",
    "optimize_clustering", 
    "wrap_text_with_highlight",
]
