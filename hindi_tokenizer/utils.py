"""
Utility functions for Hindi tokenizer.
"""

import re
import unicodedata


def normalize_hindi_text(text):
    """
    Normalize Hindi text for consistent tokenization.
    
    Args:
        text (str): Input Hindi text.
        
    Returns:
        str: Normalized Hindi text.
    """
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def is_hindi_char(char):
    """
    Check if a character is a Hindi character (Devanagari script).
    
    Args:
        char (str): A single character.
        
    Returns:
        bool: True if the character is in Devanagari script.
    """
    # Devanagari Unicode range: U+0900 to U+097F
    return '\u0900' <= char <= '\u097F'


def contains_hindi(text):
    """
    Check if text contains Hindi characters.
    
    Args:
        text (str): Input text.
        
    Returns:
        bool: True if the text contains Hindi characters.
    """
    return any(is_hindi_char(char) for char in text)


def count_tokens(text, tokenizer):
    """
    Count the number of tokens in a text.
    
    Args:
        text (str): Input text.
        tokenizer (HindiTokenizer): Initialized tokenizer.
        
    Returns:
        int: Number of tokens.
    """
    return len(tokenizer.encode(text))
