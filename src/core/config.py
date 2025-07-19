# config.py
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from transformers import AutoConfig
from requests.exceptions import HTTPError


class ModelConfigs(BaseModel):
    model_name: str = 'distilbert-base-uncased'


class DataConfigs(BaseModel):
    sentences_path: str = 'data/testing_data/test.txt'
    output_path: str = 'output'


class MetricConfigs(BaseModel):
    metrics_provided: bool = False
    anisotropy_correction: bool = False
    layers: list | str = 'final'
    metrics: list = ['similarity_matrix', 'mev', 'inter_similarity', 'intra_similarity', 'average_similarity', 'similarity_std']

    @field_validator('metrics', mode='before')
    @classmethod
    def check_metrics(cls, v: Any) -> Any:
        valid_metrics = ['similarity_matrix', 'mev', 'inter_similarity', 'intra_similarity', 'average_similarity', 'similarity_std']
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError(f"Metrics must be a list or a string, got {type(v)}")
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")
        return v

    @field_validator('layers', mode='before')
    @classmethod
    def check_layers_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.lower() == 'final':
                return v.lower()
            elif v.lower() == 'all':
                return v.lower()
            else:
                # Allows for comma-separated string like "0, 1, 11"
                try:
                    return [int(x.strip()) for x in v.split(',')]
                except ValueError:
                    raise ValueError("Layer string must be 'final' or a comma-separated list of integers.")
        if isinstance(v, list):
            if not all(isinstance(i, int) for i in v):
                raise ValueError("Layer list must contain only integers.")
            return v
        if isinstance(v, int): # Allow single integer
            return [v]
        raise ValueError("Layers must be 'final', an integer, or a list of integers.")

class LandscapeConfigs(BaseModel):
    landscapes_provided: bool = False
    pca_min: int = Field(default=2, ge=1)
    pca_max: int = Field(default=5, ge=1)
    pca_step: int = Field(default=1, gt=0)
    cluster_min: int = Field(default=2, ge=1)
    cluster_max: int = Field(default=5, ge=1)
    cluster_step: int = Field(default=1, gt=0)
    generate_all: bool = True
    save_optimization: bool = True

class ExperimentConfigs(BaseModel):
    model_batch_size: int = Field(default=32, gt=0)
    context_window: Optional[int] = Field(default=10, ge=0)
    target_word: str = 'bank'

    @field_validator('context_window', mode='before')
    @classmethod
    def preprocess_context_window(cls, v: Any) -> Optional[Any]:
        if isinstance(v, str):
            if v.lower() == 'none':
                return None 
            return v
        return v


class AppConfig(BaseModel):
    model: ModelConfigs = Field(default_factory=ModelConfigs)
    data: DataConfigs = Field(default_factory=DataConfigs)
    experiment: ExperimentConfigs = Field(default_factory=ExperimentConfigs)
    metrics: MetricConfigs = Field(default_factory=MetricConfigs)
    landscapes: LandscapeConfigs = Field(default_factory=LandscapeConfigs)

    @model_validator(mode='after')
    def validate_layers_against_model(self) -> 'AppConfig':
        # Only validate layers if metrics are actually provided
        if not self.metrics.metrics_provided:
            return self
            
        model_name = self.model.model_name
        layers_to_check = self.metrics.layers

        # Skip validation for string values like 'final'
        if isinstance(layers_to_check, str):
            if layers_to_check not in ['final', 'all']:
                raise ValueError("layers must be 'final', 'all', or a list of integers")
            return self

        try:
            config = AutoConfig.from_pretrained(model_name)

            num_layers = getattr(config, 'num_hidden_layers', None)
            if num_layers is None:
                # Fallback for encoder-decoder models?
                num_layers = getattr(config, 'num_encoder_layers', 0)

            if num_layers == 0:
                 # Just in case the model has no layers for some reason...
                 print(f"Warning: Could not determine number of layers for {model_name}. Skipping layer validation.")
                 return self

            if layers_to_check == 'all':
                layers_to_check = range(num_layers)
                self.metrics.layers = layers_to_check

            # Only validate if we have a list/range of integers
            if hasattr(layers_to_check, '__iter__') and not isinstance(layers_to_check, str):
                for layer_index in layers_to_check:
                    if not (0 <= layer_index < num_layers):
                        raise ValueError(
                            f"\nInvalid layer index: {layer_index}.\n"
                            f"Model '{model_name}' has {num_layers} layers (indexed 0 to {num_layers - 1})."
                        )

        except (OSError, HTTPError):
            raise ValueError(f"Could not fetch configuration for model: '{model_name}'. Please ensure the name is correct and you are online.")

        return self
