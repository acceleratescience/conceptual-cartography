# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = 'bert-base-uncased'

@dataclass
class DataConfig:
    sentences_path: str = 'data/bnc_spoken.txt'
    output_path: str = 'output/bert_embeddings'

@dataclass
class ExperimentConfig:
    model_batch_size: int = 32
    context_window: Optional[int] = 10
    target_word: str = 'bank'

@dataclass
class AppConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)