"""
Utility functions for Khmer OCR
CTC decoding, text-to-ID conversion, and other helpers
"""

import torch


def text_to_ids(text, char_to_id):
    """
    Convert text to a sequence of character IDs
    
    Args:
        text: Input text string
        char_to_id: Dictionary mapping characters to IDs
    
    Returns:
        List of character IDs
    """
    return [char_to_id.get(char, 0) for char in text]


def ids_to_text(ids, id_to_char):
    """
    Convert sequence of IDs back to text
    
    Args:
        ids: List of character IDs
        id_to_char: Dictionary mapping IDs to characters
    
    Returns:
        Text string
    """
    return ''.join([id_to_char.get(id, '') for id in ids])


def ctc_decode(predictions, blank_id=0):
    """
    Decode CTC predictions
    
    Args:
        predictions: Model output predictions
        blank_id: ID representing the blank label
    
    Returns:
        Decoded sequence
    """
    # TODO: Implement CTC decoding logic
    # Remove duplicate consecutive characters
    # Remove blank labels
    pass


def create_char_mappings(texts):
    """
    Create character-to-ID and ID-to-character mappings
    
    Args:
        texts: List of text samples
    
    Returns:
        Tuple of (char_to_id, id_to_char) dictionaries
    """
    unique_chars = sorted(set(''.join(texts)))
    char_to_id = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    char_to_id['<blank>'] = 0
    id_to_char = {idx: char for char, idx in char_to_id.items()}
    return char_to_id, id_to_char
