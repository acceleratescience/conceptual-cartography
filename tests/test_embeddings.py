import pytest
import torch
from typing import Dict, List, Tuple
from unittest.mock import MagicMock
import re

from pathlib import Path
from src.utils import load_sentences
from src.embeddings import ContextEmbedder


# Easiest thing to test is the find_occurences?
@pytest.mark.parametrize("args, expected_output", [
    # Test case 1: Empty sentence
    ({"sentence_text": "", "pattern_str": r"\btest\b"}, []),
    # Test case 2: Sentence with no occurrences
    ({"sentence_text": "This is a sample sentence.", "pattern_str": r"\btest\b"}, []),
    # Test case 3: Sentence with one occurrence
    ({"sentence_text": "This is a test sentence.", "pattern_str": r"\btest\b"}, [("This is a test sentence.", 10)]),
    # Test case 4: Sentence with multiple occurrences
    ({"sentence_text": "test one, test two.", "pattern_str": r"\btest\b"}, [("test one, test two.", 0), ("test one, test two.", 10)]),
    # Test case 5: Case insensitivity (pattern is already lowercased in __call__)
    ({"sentence_text": "This is a Test sentence.", "pattern_str": r"\btest\b"}, [("This is a Test sentence.", 10)]),
    # Test case 6: Whole word matching (pattern includes \b)
    ({"sentence_text": "testing is not test.", "pattern_str": r"\btest\b"}, [("testing is not test.", 15)]),
    # Test case 7: Only spaces in sentence
    ({"sentence_text": " ", "pattern_str": r"\btest\b"}, []),
])
def test_find_occurrences_in_sentence_static(args: Dict, expected_output: List[Tuple[str, int]]):
    """Test the _find_occurrences_in_sentence_static method."""
    results = ContextEmbedder._find_occurrences_in_sentence_static(args)
    assert results == expected_output

# TODO: find a way to test the embedding...