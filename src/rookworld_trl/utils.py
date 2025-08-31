"""
Utility functions for RookWorld TRL training
"""

import re
from typing import List


def normalize_spacing(text: str) -> str:
    """
    Normalize all sequences of 2+ spaces to single spaces.
    
    This is critical for GRPO training to prevent artificial KL divergence
    inflation due to spacing differences between model and reference outputs.
    
    Args:
        text: Input text with potentially excessive spacing
        
    Returns:
        Text with normalized spacing (max 1 space between elements)
        
    Examples:
        >>> normalize_spacing("M: e2e4      d2d4        E: 0.5")
        "M: e2e4 d2d4 E: 0.5"
        
        >>> normalize_spacing("                    M: g1f3")  
        "M: g1f3"
    """
    if not text:
        return text
        
    # Replace sequences of 2+ spaces (including tabs, newlines) with single space
    normalized = re.sub(r'\s{2,}', ' ', text)
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def normalize_completion_batch(completions: List[str]) -> List[str]:
    """
    Normalize spacing in a batch of completions.
    
    Args:
        completions: List of completion strings with potential spacing issues
        
    Returns:
        List of completions with normalized spacing
    """
    return [normalize_spacing(completion) for completion in completions]


def normalize_prompt_completion_pair(prompt: str, completion: str) -> tuple[str, str]:
    """
    Normalize spacing in both prompt and completion for consistent processing.
    
    Args:
        prompt: Input prompt
        completion: Generated completion
        
    Returns:
        Tuple of (normalized_prompt, normalized_completion)
    """
    return normalize_spacing(prompt), normalize_spacing(completion)