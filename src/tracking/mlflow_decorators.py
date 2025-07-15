# mlflow_decorators.py
import functools
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import mlflow
import mlflow.pytorch
import torch
from ..core.config import AppConfig
from ..analysis.metrics import MetricsResult
from ..analysis.landscapes import Landscape


def track_experiment(experiment_name: Optional[str] = None):
    """Decorator to track an entire experiment run with MLflow."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config if available
            config = None
            for arg in args:
                if isinstance(arg, AppConfig):
                    config = arg
                    break
            
            # Start MLflow run
            with mlflow.start_run(run_name=experiment_name):
                # Log configuration parameters
                if config:
                    mlflow.log_params({
                        "model_name": config.model.model_name,
                        "target_word": config.experiment.target_word,
                        "context_window": config.experiment.context_window,
                        "batch_size": config.experiment.model_batch_size,
                        "anisotropy_correction": config.metrics.anisotropy_correction,
                        "layers": str(config.metrics.layers),
                        "metrics": config.metrics.metrics
                    })
                
                # Execute original function
                result = func(*args, **kwargs)
                
                return result
        return wrapper
    return decorator


def track_embeddings(log_artifacts: bool = True):
    """Decorator to track embedding generation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if mlflow.active_run():
                # Log embedding metrics
                if hasattr(result, 'final_embeddings'):
                    mlflow.log_metrics({
                        "num_embeddings": result.final_embeddings.shape[0],
                        "embedding_dim": result.final_embeddings.shape[1],
                        "num_layers": result.hidden_embeddings.shape[1] if hasattr(result, 'hidden_embeddings') else 1
                    })
                
                # Log artifacts if requested and output_path is available
                if log_artifacts and len(args) > 0:
                    output_path = args[0]  # Assuming first arg is output_path
                    if isinstance(output_path, (str, Path)) and Path(output_path).exists():
                        mlflow.log_artifacts(str(output_path))
            
            return result
        return wrapper
    return decorator


def track_metrics(prefix: str = ""):
    """Decorator to track metrics computation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if mlflow.active_run() and isinstance(result, (MetricsResult, dict)):
                metrics_to_log = {}
                
                if isinstance(result, MetricsResult):
                    metrics_to_log = {
                        f"{prefix}mev": result.mev,
                        f"{prefix}average_similarity": result.average_similarity,
                        f"{prefix}similarity_std": result.similarity_std,
                    }
                    if result.intra_similarity is not None:
                        metrics_to_log[f"{prefix}intra_similarity"] = result.intra_similarity
                    if result.inter_similarity is not None:
                        metrics_to_log[f"{prefix}inter_similarity"] = result.inter_similarity
                
                elif isinstance(result, dict):
                    metrics_to_log = {f"{prefix}{k}": v for k, v in result.items() 
                                    if isinstance(v, (int, float))}
                
                mlflow.log_metrics(metrics_to_log)
            
            return result
        return wrapper
    return decorator


def track_landscapes(log_artifacts: bool = True):
    """Decorator to track landscape generation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if mlflow.active_run() and isinstance(result, Landscape):
                # Log landscape metrics
                mlflow.log_metrics({
                    "pca_components": result.pca_components or 0,
                    "cluster_count": result.cluster_count or 0,
                    "mean_ari_score": sum(result.ari_scores) / len(result.ari_scores) if result.ari_scores else 0
                })
                
                # Log artifacts if requested
                if log_artifacts and len(args) > 0:
                    output_path = args[0]
                    if isinstance(output_path, (str, Path)) and Path(output_path).exists():
                        mlflow.log_artifacts(str(output_path))
            
            return result
        return wrapper
    return decorator


def track_model(model_name: str = "model"):
    """Decorator to track model artifacts."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if mlflow.active_run():
                # Look for model in result or args
                model = None
                if hasattr(result, 'model'):
                    model = result.model
                elif hasattr(result, 'embedder') and hasattr(result.embedder, 'model'):
                    model = result.embedder.model
                
                if model is not None:
                    mlflow.pytorch.log_model(model, model_name)
            
            return result
        return wrapper
    return decorator
