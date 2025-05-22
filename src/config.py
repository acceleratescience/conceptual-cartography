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
    context_window: int | None = Field(default=10, ge=0)
    target_word: str = 'bank'

    @field_validator('context_window', mode='before')
    @classmethod
    def none_string_to_none(cls, v: Any):
        if isinstance(v, str) and v.lower == 'none':
            return None
        if not isinstance(v, int) and v is not None:
            raise ValueError("context_window argument must be either a positive integer or 'None'")
        # Actually not sure if this part is needed?
        if v is not None and v < 0:
            raise ValueError("context_window must be ≥ 0")
        return v


class AppConfig(BaseModel):
    model_configs: ModelConfigs = Field(default_factory=ModelConfigs)
    data_configs: DataConfigs = Field(default_factory=DataConfigs)
    experiment_configs: ExperimentConfigs = Field(default_factory=ExperimentConfigs)

# @dataclass
# class ModelConfig:
#     model_name: str = 'bert-base-uncased'

# @dataclass
# class DataConfig:
#     sentences_path: str = 'data/bnc_spoken.txt'
#     output_path: str = 'output/bert_embeddings'

# @dataclass
# class ExperimentConfig:
#     model_batch_size: int = 32
#     context_window: Optional[int] = 10
#     target_word: str = 'bank'

# @dataclass
# class AppConfig:
#     model_config: ModelConfig = field(default_factory=ModelConfig)
#     data_config: DataConfig = field(default_factory=DataConfig)
#     experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)