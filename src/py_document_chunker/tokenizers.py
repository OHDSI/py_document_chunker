"""
Provides utility functions for creating token-based length functions from
popular tokenizer libraries.

This module is designed to be optional and only imports dependencies when the
utility functions are actually called, keeping the core package's dependency
footprint minimal.
"""

from typing import Callable

# A cache to store initialized tokenizer objects to avoid redundant setup.
_tokenizer_cache = {}


def from_tiktoken(encoding_name: str) -> Callable[[str], int]:
    """
    Creates a length function from a `tiktoken` encoding.

    This function allows any `TextSplitter` to measure chunk size in terms of
    tokens, which is crucial for aligning text segments with the constraints of
    Large Language Models.

    This utility requires the `tiktoken` library to be installed.
    e.g., `pip install tiktoken` or `pip install advanced-text-segmentation[tokenizers]`

    Args:
        encoding_name: The name of the `tiktoken` encoding to use,
            (e.g., "cl100k_base" for GPT-4, "p50k_base" for GPT-3.5).

    Returns:
        A callable function that takes a string and returns its length in tokens.

    Raises:
        ImportError: If the `tiktoken` library is not installed.
        Exception: If the specified encoding name is not found.
    """
    global _tokenizer_cache

    if encoding_name in _tokenizer_cache:
        return _tokenizer_cache[encoding_name]

    try:
        import tiktoken
    except ImportError:
        raise ImportError(
            "The 'tiktoken' library is required to use this tokenizer. "
            "Please install it by running 'pip install tiktoken' or "
            "'pip install advanced-text-segmentation[tokenizers]'."
        )

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        raise Exception(f"Failed to load tiktoken encoding '{encoding_name}': {e}")

    def length_function(text: str) -> int:
        return len(encoding.encode(text))

    # Cache the function for future calls
    _tokenizer_cache[encoding_name] = length_function
    return length_function
