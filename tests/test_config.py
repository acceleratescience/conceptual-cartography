import pytest
from pydantic import ValidationError

from src.config import ModelConfigs, DataConfigs, ExperimentConfigs, AppConfig


"""Test defaults                                                                          
The testing file `test.txt` contains 26 sentences. The word `bank` occurs in isolation exactly 22
times. `bank` occurs twice in 1 sentence, and the remaining occurrences are `banking`, `banks`, 
and `Embankment`. These uses should not occur in the embeddings.   
The word `spring` occurs in isolation 13 times. It occurs twice in 1 sentence, and a remaining
occurence is `springing`.

The default config word is `bank`, with `spring` being an override test word
"""

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