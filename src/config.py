# config.py
from dataclasses import dataclass, field
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


class ModelConfigs(BaseModel):
    model_name: str = 'distilbert-base-uncased'


class DataConfigs(BaseModel):
    sentences_path: str = 'data/testing_data/test.txt'
    output_path: str = 'output'


class MetricConfigs(BaseModel):
    metrics_provided: bool = False
    output_path: str = 'output'
    anisotropy_correction: bool = False
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
    metric: MetricConfigs = Field(default_factory=MetricConfigs)
