# utils.py
import yaml
from pathlib import Path
from src.config import AppConfig
from pydantic import ValidationError

import torch

def load_config_from_yaml(config_path: Path) -> AppConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    try:
        app_config = AppConfig(**config_dict)
        return app_config
    except ValidationError as e:
        print(f"❌ Configuration Error in '{config_path}':")
        print(e)
        raise


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
    