# config.py
from dataclasses import dataclass, field
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


class ModelConfigs(BaseModel):
    model_name: str = 'distilbert-base-uncased'


class DataConfigs(BaseModel):
    sentences_path: str = 'data/testing_data/test.txt'
    output_path: str = 'output/test_embeddings'


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
    model_configs: ModelConfigs = Field(default_factory=ModelConfigs)
    data_configs: DataConfigs = Field(default_factory=DataConfigs)
    experiment_configs: ExperimentConfigs = Field(default_factory=ExperimentConfigs)
