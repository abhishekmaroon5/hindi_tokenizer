"""
Tests for the Hindi tokenizer core functionality.
"""

import os
import tempfile
import pytest
from hindi_tokenizer.core import HindiTokenizer


def test_tokenizer_initialization():
    """Test that the tokenizer can be initialized."""
    tokenizer = HindiTokenizer()
    assert tokenizer is not None
    assert tokenizer.model is None


def test_encode_decode_with_model(tmp_path):
    """Test encoding and decoding with a mock model."""
    # This test would normally require a trained model
    # For unit testing purposes, we'll just check that the methods exist
    tokenizer = HindiTokenizer()
    
    # Check that methods exist
    assert hasattr(tokenizer, 'encode')
    assert hasattr(tokenizer, 'encode_as_pieces')
    assert hasattr(tokenizer, 'decode')
    
    # Check that methods raise appropriate errors when model is not loaded
    with pytest.raises(ValueError):
        tokenizer.encode("यह एक हिंदी वाक्य है।")
    
    with pytest.raises(ValueError):
        tokenizer.encode_as_pieces("यह एक हिंदी वाक्य है।")
    
    with pytest.raises(ValueError):
        tokenizer.decode([1, 2, 3, 4])
