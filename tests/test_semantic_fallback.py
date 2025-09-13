from typing import List

import pytest

# Attempt to import numpy to determine if tests should be skipped.
try:
    import numpy as np

    from py_document_chunker.strategies.semantic import SemanticSplitter

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# A mock embedding function for testing.
def mock_embedding_function(texts: List[str]) -> "np.ndarray":
    embeddings = []
    for text in texts:
        if "Topic A" in text:
            embeddings.append([1.0, 0.0, 0.0])
        elif "Topic B" in text:
            embeddings.append([0.0, 1.0, 0.0])
        else:  # transition or noise
            embeddings.append([0.0, 0.0, 1.0])
    return np.array(embeddings)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_fallback_splits_long_sentence_and_preserves_overlap():
    """
    Tests that the fallback mechanism correctly splits a long sentence that exceeds
    the chunk size and that the overlap between the generated chunks is correct.
    """
    text = (
        "This is a very long sentence about Topic A that is designed to be split "
        "by the fallback mechanism into multiple chunks. "
        "This is a second sentence about Topic B that should be in its own chunk."
    )
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        chunk_size=80,
        chunk_overlap=20,
    )
    chunks = splitter.split_text(text)

    assert len(chunks) == 3
    # The first chunk should be the beginning of the first sentence.
    assert chunks[0].content == (
        "This is a very long sentence about Topic A that is designed to be split by the "
    )
    # The second chunk should be the end of the first sentence, with overlap.
    assert chunks[1].content == (
        "to be split by the fallback mechanism into multiple chunks."
    )
    # The third chunk should be the second sentence.
    assert chunks[2].content == (
        "This is a second sentence about Topic B that should be in its own chunk."
    )

    # Check the overlap between the first two chunks.
    assert chunks[0].overlap_content_next == "to be split by the "
    assert chunks[1].overlap_content_previous == "to be split by the "

    # The second chunk should not have overlap with the third, because they are
    # from different semantic groups.
    assert chunks[1].overlap_content_next is None
    assert chunks[2].overlap_content_previous is None
