# utils.py
import yaml
from pathlib import Path
from typing import Any
from src.config.config import AppConfig, ModelConfig, DataConfig, ExperimentConfig
import torch

def load_config_from_yaml(config_path: Path) -> AppConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    model_cfg_data = config_dict.get('model_config', {})
    data_cfg_data = config_dict.get('data_config', {})
    exp_cfg_data = config_dict.get('experiment_config', {})
    
    if 'context_window' in exp_cfg_data:
        cw = exp_cfg_data['context_window']
        if isinstance(cw, str) and cw.lower() == 'none':
            exp_cfg_data['context_window'] = None
            
    model_cfg = ModelConfig(**model_cfg_data)
    data_cfg = DataConfig(**data_cfg_data)
    exp_cfg = ExperimentConfig(**exp_cfg_data)
    
    return AppConfig(
        model_config=model_cfg,
        data_config=data_cfg,
        experiment_config=exp_cfg
    )


def load_sentences(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def save_embeddings(output_path: str, embeddings: torch.Tensor, contexts: list, indices: list):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, Path(output_path) / "embeddings.pt")
    
    with open(Path(output_path) / "contexts.txt", "w", encoding="utf-8") as f:
        for context in contexts:
            f.write(f"{context}\n")
            
    with open(Path(output_path) / "indices.txt", "w", encoding="utf-8") as f:
        for index in indices:
            f.write(f"{index}\n")
    print(f"Embeddings and associated data saved to {output_path}")