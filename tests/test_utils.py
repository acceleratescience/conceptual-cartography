import pytest
import torch
from pathlib import Path
from pydantic import ValidationError
from src.core.utils import load_config_from_yaml, load_sentences, save_output
from src.config import AppConfig, ModelConfigs, DataConfigs, ExperimentConfigs

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_config_from_yaml_valid():
    """Test loading a valid YAML configuration file."""
    config_path = FIXTURES_DIR / "bert-test.yaml"
    app_config = load_config_from_yaml(config_path)

    assert isinstance(app_config, AppConfig)
    assert isinstance(app_config.model_configs, ModelConfigs)
    assert isinstance(app_config.data_configs, DataConfigs)
    assert isinstance(app_config.experiment_configs, ExperimentConfigs)
    
    assert app_config.model_configs.model_name == 'bert-base-uncased'
    assert app_config.data_configs.sentences_path == 'test-data/test.txt'
    assert app_config.experiment_configs.target_word == 'spring'
    assert app_config.experiment_configs.context_window is None
    assert app_config.experiment_configs.model_batch_size == 32


def test_load_config_from_yaml_file_not_found():
    """Test loading a non-existent YAML file."""
    config_path = Path("non_existent_config.yaml")
    with pytest.raises(FileNotFoundError):
        load_config_from_yaml(config_path)


def test_load_sentences_valid_file():
    """Test loading sentences from a file with content."""
    file_path =  file_path = str(FIXTURES_DIR / "test.txt")
    sentences = load_sentences(file_path)
    expected_sentences = [
        "Bank.",
        "The Spring River banking was covered in wildflowers during the spring season.",
        "I need to go to the bank to deposit my paycheck before it closes; I hope to spring for a new bike soon.",
        "The pilot had to bank the aircraft sharply to avoid the storm clouds. The old mattress had a broken spring.",
        "Many economists predict the central bank will raise interest rates next month, which might cause the market to spring back."
    ]
    assert sentences[:5] == expected_sentences
    assert len(sentences) == 26


def test_load_sentences_empty_file():
    """Test loading sentences from an empty file."""
    file_path = str(FIXTURES_DIR / "empty.txt")
    sentences = load_sentences(file_path)
    assert sentences == []
    assert len(sentences) == 0


def test_load_sentences_file_not_found():
    """Test loading sentences from a non-existent file."""
    file_path = "non_existent_sentences.txt"
    with pytest.raises(FileNotFoundError):
        load_sentences(file_path)


# TODO: Need to test all broken yaml fields
def test_load_config_from_yaml_invalid():
    """Test loading an invalid YAML that should raise ValidationError."""
    config_path = FIXTURES_DIR / "broken-config.yaml"
    with pytest.raises(ValidationError) as excinfo:
        load_config_from_yaml(config_path)
    assert "context_window" in str(excinfo.value).lower()
    assert "model_batch_size" in str(excinfo.value).lower()


# Test saving
# TODO: this needs updating
def test_save_output_creates_files_and_verifies_content(tmp_path):
    """
    Test that save_output creates the expected directory and files
    with correct content in a temporary location.
    tmp_path is a pytest fixture providing a temporary directory path object.
    """
    output_dir_in_tmp = tmp_path / "experiment_results" / "run1"
    
    # Create dummy output dictionary matching the new format
    dummy_output = {
        'final_embeddings': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        'hidden_embeddings': torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
        'valid_contexts': [[101, 102, 103], [201, 202]],
        'valid_indices': [0, 1]
    }

    save_output(str(output_dir_in_tmp), dummy_output)

    assert output_dir_in_tmp.exists(), "Output directory should be created"
    assert output_dir_in_tmp.is_dir(), "Output path should be a directory"
    
    final_embeddings_file = output_dir_in_tmp / "final_embeddings.pt"
    hidden_embeddings_file = output_dir_in_tmp / "hidden_embeddings.pt"
    contexts_file = output_dir_in_tmp / "contexts.txt"
    indices_file = output_dir_in_tmp / "indices.txt"

    assert final_embeddings_file.exists(), "final_embeddings.pt should be created"
    assert hidden_embeddings_file.exists(), "hidden_embeddings.pt should be created"
    assert contexts_file.exists(), "contexts.txt should be created"
    assert indices_file.exists(), "indices.txt should be created"

    # Verify final_embeddings.pt
    loaded_final_embeddings = torch.load(final_embeddings_file)
    assert torch.equal(loaded_final_embeddings, dummy_output['final_embeddings']), "Loaded final embeddings should match saved ones"

    # Verify hidden_embeddings.pt
    loaded_hidden_embeddings = torch.load(hidden_embeddings_file)
    assert torch.equal(loaded_hidden_embeddings, dummy_output['hidden_embeddings']), "Loaded hidden embeddings should match saved ones"

    # Verify contexts.txt
    with open(contexts_file, "r", encoding="utf-8") as f:
        saved_contexts_lines = f.read().splitlines() 
    
    expected_contexts_lines = [str(context) for context in dummy_output['valid_contexts']]
    assert saved_contexts_lines == expected_contexts_lines, "Content of contexts.txt is incorrect"

    # Verify indices.txt
    with open(indices_file, "r", encoding="utf-8") as f:
        saved_indices_lines = f.read().splitlines()
        
    expected_indices_lines = [str(index) for index in dummy_output['valid_indices']]
    assert saved_indices_lines == expected_indices_lines, "Content of indices.txt is incorrect"