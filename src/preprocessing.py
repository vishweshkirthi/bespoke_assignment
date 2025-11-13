"""
Text preprocessing utilities for FastText document quality classification
"""

import re
import unicodedata


def preprocess_text(text: str) -> str:
    """Standardize and preprocess text for FastText"""
    if not text:
        return ""
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace and normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep basic punctuation
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # Normalize common punctuation using string replacement
    text = text.replace('"', '"').replace('"', '"')  # Smart quotes to regular quotes
    text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to regular apostrophes  
    text = text.replace('–', '-').replace('—', '-')  # Em/en dashes to hyphens
    text = text.replace('…', '...')  # Ellipsis to three dots
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{4,}', '...', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Separate punctuation from words by adding spaces
    # This helps FastText treat punctuation as separate tokens
    text = re.sub(r'([a-zA-Z0-9])([.!?:;,()"\[\]{}])', r'\1 \2', text)  # After alphanumeric
    text = re.sub(r'([.!?:;,()"\[\]{}])([a-zA-Z0-9])', r'\1 \2', text)  # Before alphanumeric
    
    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def validate_text_requirements(text: str) -> tuple[bool, str]:
    """
    Validate text meets minimum requirements
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) < 50:
        return False, "Text too short (minimum 50 characters required)"
    
    
    word_count = len(text.split())
    if word_count < 10:
        return False, "Text must contain at least 10 words"
    
    return True, ""