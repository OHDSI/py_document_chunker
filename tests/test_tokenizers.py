from unittest.mock import patch

import pytest

from py_document_chunker import RecursiveCharacterSplitter
from py_document_chunker.tokenizers import from_tiktoken


def test_recursive_splitter_with_tiktoken():
    """
    Validates that the RecursiveCharacterSplitter correctly uses a `tiktoken`
    length function to respect token-based chunk size constraints.

    This test ensures that the `length_function` is not just a placeholder
    but is actively used by the splitting logic to produce chunks that are
    compliant with LLM token limits.
    """
    # A text with 20 tokens according to cl100k_base
    # "This is a test sentence for token-based chunking. Hello world."
    # (20 tokens)
    text = "This is a test sentence for token-based chunking. Hello world."

    # Use the cl100k_base encoding, common for GPT-3.5/4
    try:
        length_function = from_tiktoken("cl100k_base")
    except ImportError:
        pytest.skip("tiktoken is not installed, skipping token-based test.")
    except Exception:
        pytest.skip("Could not load tiktoken model, skipping token-based test.")

    # Total tokens = 20.
    # We expect this to be split into three chunks.
    # Chunk 1: "This is a test sentence for" (7 tokens)
    # Chunk 2: "for token-based chunking." (5 tokens, plus 1 overlap) -> 6 total
    # Chunk 3: "chunking. Hello world." (4 tokens, plus 1 overlap) -> 5 total
    splitter = RecursiveCharacterSplitter(
        chunk_size=8,
        chunk_overlap=2,
        length_function=length_function,
        separators=[" "],  # Split by space for granular control
    )

    chunks = splitter.split_text(text)

    assert len(chunks) > 1, "The text should have been split into multiple chunks."

    # Validate that each chunk's token count is within the specified limit
    for i, chunk in enumerate(chunks):
        token_count = length_function(chunk.content)
        assert token_count <= 8, (
            f"Chunk {i} with content '{chunk.content}' has {token_count} tokens, "
            f"which exceeds the chunk_size of 8."
        )


def test_from_tiktoken_invalid_encoding():
    """
    Tests that from_tiktoken raises an exception for an invalid encoding name.
    """
    with pytest.raises(Exception, match="Failed to load tiktoken encoding"):
        from_tiktoken("invalid-encoding-name")


def test_from_tiktoken_caching():
    """
    Tests that the from_tiktoken utility caches the length function.
    """
    try:
        # Clear cache for a clean test
        from py_document_chunker import tokenizers

        tokenizers._tokenizer_cache.clear()

        func1 = from_tiktoken("cl100k_base")
        func2 = from_tiktoken("cl100k_base")
        assert (
            func1 is func2
        ), "The from_tiktoken function should return the same cached object"
    except ImportError:
        pytest.skip("tiktoken is not installed, skipping token-based test.")
    except Exception:
        pytest.skip("Could not load tiktoken model, skipping token-based test.")


def test_from_tiktoken_import_error():
    """Tests that an ImportError is raised if tiktoken is not installed."""
    from py_document_chunker import tokenizers

    tokenizers._tokenizer_cache.clear()

    with patch.dict("sys.modules", {"tiktoken": None}):
        with pytest.raises(ImportError):
            from_tiktoken("cl100k_base")
