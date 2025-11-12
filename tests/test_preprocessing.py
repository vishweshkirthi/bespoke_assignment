#!/usr/bin/env python3
"""
Tests for text preprocessing functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import preprocess_text, validate_text_requirements


def test_preprocess_text():
    """Test text preprocessing function"""
    
    # Test basic functionality
    text = "This is a SAMPLE text with  extra   spaces."
    result = preprocess_text(text)
    assert result == "this is a sample text with extra spaces."
    
    # Test Unicode normalization
    text = "Smart "quotes" and 'apostrophes' — with dashes…"
    result = preprocess_text(text)
    assert '"' in result and "'" in result and '-' in result and '...' in result
    
    # Test empty input
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
    
    print("✓ Text preprocessing tests passed")


def test_validate_text_requirements():
    """Test text validation function"""
    
    # Valid text
    valid_text = "This is a sufficiently long text with more than ten words to meet requirements."
    is_valid, error = validate_text_requirements(valid_text)
    assert is_valid == True
    assert error == ""
    
    # Too short
    is_valid, error = validate_text_requirements("Short")
    assert is_valid == False
    assert "too short" in error.lower()
    
    # Empty text
    is_valid, error = validate_text_requirements("")
    assert is_valid == False
    assert "empty" in error.lower()
    
    # Too few words
    few_words = "A" * 60  # Long but few words
    is_valid, error = validate_text_requirements(few_words)
    assert is_valid == False
    assert "words" in error.lower()
    
    print("✓ Text validation tests passed")


if __name__ == "__main__":
    test_preprocess_text()
    test_validate_text_requirements()
    print("✓ All tests passed!")