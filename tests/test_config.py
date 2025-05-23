import pytest
from pydantic import ValidationError

from src.config import ModelConfigs, DataConfigs, ExperimentConfigs, AppConfig


"""                                                                  
The testing file `test.txt` contains 26 sentences. The word `bank` occurs in isolation exactly 22
times. `bank` occurs twice in 1 sentence, and the remaining occurrences are `banking`, `banks`, 
and `Embankment`. These uses should not occur in the embeddings.   
The word `spring` occurs in isolation 13 times. It occurs twice in 1 sentence, and a remaining
occurence is `springing`.

The default config word is `bank`, with `spring` being an override test word
"""

# Test Defaults

def test_model_config_defaults():
    """Test that ModelConfig instantiates with default values."""
    config = ModelConfigs()
    assert config.model_name == 'distilbert-base-uncased'

def test_data_config_defaults():
    """Test DataConfig with default values."""
    config = DataConfigs()
    assert config.sentences_path == 'data/testing_data/test.txt'
    assert config.output_path == 'output/test_embeddings'

def test_experiment_config_defaults():
    """Test ExperimentConfig with default values."""
    config = ExperimentConfigs()
    assert config.model_batch_size == 32
    assert config.context_window == 10 # need to also test None
    assert config.target_word == 'bank'

def test_app_config_defaults():
    """Test AppConfig instantiates with default sub-configs."""
    config = AppConfig()
    assert isinstance(config.model_configs, ModelConfigs)
    assert isinstance(config.data_configs, DataConfigs)
    assert isinstance(config.experiment_configs, ExperimentConfigs)
    assert config.model_configs.model_name == 'distilbert-base-uncased' 

# Test Manual Override

def test_model_config_custom_name():
    """Test ModelConfig with a custom model_name."""
    config = ModelConfigs(model_name="terrible-model")
    assert config.model_name == "terrible-model"

def test_data_config_custom_paths():
    """Test DataConfig with custom paths."""
    config = DataConfigs(
        sentences_path="custom/path/file.txt",
        output_path="custom/output_path"
    )
    assert config.sentences_path == "custom/path/file.txt"
    assert config.output_path == "custom/output_path"

def test_experiment_config_custom_values():
    """Test ExperimentConfig with valid custom values."""
    config = ExperimentConfigs(
        model_batch_size=32,
        context_window=20,
        target_word="spring"
    )
    assert config.model_batch_size == 32
    assert config.context_window == 20
    assert config.target_word == "spring"

def test_app_config_with_custom_dicts():
    """Test AppConfig instantiation with provided dictionaries."""
    raw_config = {
        "model_configs": {"model_name": "custom-model"},
        "data_configs": {"sentences_path": "custom/data.txt"},
        "experiment_configs": {"target_word": "custom_target"}
    }
    config = AppConfig(**raw_config)
    assert config.model_configs.model_name == "custom-model"
    assert config.data_configs.sentences_path == "custom/data.txt"
    assert config.experiment_configs.target_word == "custom_target"
    assert config.experiment_configs.model_batch_size == 32

# Experiment config with None, 'None', batch_size = 0, neg context_window, invalid context_window
def test_experiment_config_context_window_string_none():
    """Test ExperimentConfig context_window validator with 'None' string."""
    config = ExperimentConfigs(context_window='None')
    assert config.context_window is None

    config_lower = ExperimentConfigs(context_window="none")
    assert config_lower.context_window is None

def test_experiment_config_context_window_actual_none():
    """Test ExperimentConfig context_window with actual None."""
    config = ExperimentConfigs(context_window=None)
    assert config.context_window is None

def test_experiment_config_invalid_batch_size():
    """Test that a batch_size <= 0 raises a ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfigs(model_batch_size=0)

    with pytest.raises(ValidationError):
        ExperimentConfigs(model_batch_size=-1)

def test_experiment_config_invalid_context_window_negative():
    """Test that a negative context_window raises a ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfigs(context_window=-5)

def test_experiment_config_invalid_context_window_string():
    """Test that an invalid string for context_window raises ValidationError."""
    with pytest.raises(ValidationError):
        ExperimentConfigs(context_window="not_an_integer")