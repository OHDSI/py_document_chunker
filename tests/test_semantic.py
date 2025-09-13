from typing import List
from unittest.mock import patch

import pytest

# Attempt to import numpy to determine if tests should be skipped.
try:
    import numpy as np

    from py_document_chunker.strategies.semantic import SemanticSplitter

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# A mock embedding function for testing.
# Sentences about topic A get similar vectors.
# Sentences about topic B get similar vectors, but different from A.
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


TEXT_FOR_SEMANTIC_SPLIT = (
    "This is a sentence about Topic A. Here is another sentence on Topic A. "
    "This sentence is a transition. "
    "Now we talk about Topic B. And here is more on Topic B."
)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_identifies_breakpoints():
    """
    Tests that the SemanticSplitter correctly identifies multiple breakpoints
    based on embedding similarity drop-offs.
    """
    # Similarities: [1.0, 0.0, 0.0, 1.0]. 50th percentile is 0.5.
    # Breakpoints should be triggered for the two 0.0 similarity scores.
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="percentile",
        breakpoint_threshold=50,
    )
    chunks = splitter.split_text(TEXT_FOR_SEMANTIC_SPLIT)

    # Expecting 3 chunks based on the two breakpoints found.
    # Chunk 1: Sentences about Topic A
    # Chunk 2: The transition sentence
    # Chunk 3: Sentences about Topic B
    assert len(chunks) == 3
    assert (
        chunks[0].content
        == "This is a sentence about Topic A. Here is another sentence on Topic A."
    )
    assert chunks[1].content == "This sentence is a transition."
    assert (
        chunks[2].content == "Now we talk about Topic B. And here is more on Topic B."
    )
    # No overlap should be generated when not using a fallback splitter with overlap
    assert chunks[0].overlap_content_next is None
    assert chunks[1].overlap_content_previous is None


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_std_dev():
    """
    Tests that the SemanticSplitter correctly identifies breakpoints using the
    standard deviation method.
    """
    # Similarities: [1.0, 0.0, 0.0, 1.0]. Mean=0.5, Std=0.5.
    # A threshold of 1.0 std dev below the mean is 0.0.
    # Breakpoints should be triggered for similarities < 0.0.
    # With a threshold of 0.9, the 0.0 similarities will be < (0.5 - 0.9*0.5) = 0.05
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="std_dev",
        breakpoint_threshold=0.9,
    )
    chunks = splitter.split_text(TEXT_FOR_SEMANTIC_SPLIT)
    assert len(chunks) == 3


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_single_sentence_uses_fallback():
    """
    Tests that if only one sentence is present, the fallback splitter is used.
    """
    text = "This is a single sentence, but it is very long, so it should be split by the fallback mechanism."
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function, chunk_size=50, chunk_overlap=10
    )
    chunks = splitter.split_text(text)
    # The fallback splitter should have been invoked.
    assert len(chunks) > 1
    assert len(chunks[0].content) <= 50


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_invalid_percentile_raises_error():
    """
    Tests that an invalid percentile value raises a ValueError.
    """
    with pytest.raises(
        ValueError, match="Percentile threshold must be between 0 and 100"
    ):
        SemanticSplitter(
            embedding_function=mock_embedding_function,
            breakpoint_method="percentile",
            breakpoint_threshold=101,
        )


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_absolute():
    """
    Tests that the SemanticSplitter correctly identifies breakpoints using the
    absolute value method.
    """
    # Similarities: [1.0, 0.0, 0.0, 1.0].
    # Breakpoints should be triggered for similarities < 0.5.
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
    )
    chunks = splitter.split_text(TEXT_FOR_SEMANTIC_SPLIT)
    assert len(chunks) == 3


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_fallback_populates_overlap():
    """
    Tests that when a semantic chunk is too large and a fallback splitter with
    overlap is used, the final chunk list has overlap metadata correctly populated.
    This is the main test for the fix to the metadata population.
    """
    # This whole text is one semantic topic, so it won't be split by similarity.
    # However, it's larger than chunk_size, so it will be passed to the fallback.
    text = "This is a very long sentence about Topic A part one. This is a very long sentence about Topic A part two. This is a very long sentence about Topic A part three."

    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="percentile",
        breakpoint_threshold=95,  # High threshold, should not split
        chunk_size=100,
        chunk_overlap=20,  # This overlap is for the fallback splitter
    )

    chunks = splitter.split_text(text)

    # The fallback RecursiveCharacterSplitter should have been used.
    assert len(chunks) > 1
    # This is the key assertion: the overlap metadata should now be populated.
    assert chunks[0].overlap_content_next is not None
    assert chunks[1].overlap_content_previous is not None
    assert chunks[0].overlap_content_next == chunks[1].overlap_content_previous
    # Make the assertion less brittle, just check for a known part of the overlapping sentence.
    assert "Topic A part two" in chunks[1].content
    assert "about Topic A" in chunks[0].overlap_content_next


def test_dependency_import_error():
    """
    Tests that an ImportError is raised if numpy is not installed.
    """
    import importlib
    import sys

    try:
        with patch.dict(sys.modules, {"numpy": None}):
            import py_document_chunker.strategies.semantic

            importlib.reload(py_document_chunker.strategies.semantic)

            from py_document_chunker.strategies.semantic import SemanticSplitter

            with pytest.raises(ImportError, match="numpy is not installed"):
                SemanticSplitter(embedding_function=lambda x: [])
    finally:
        # Restore the module to its original state
        import py_document_chunker.strategies.semantic

        importlib.reload(py_document_chunker.strategies.semantic)


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_fallback_on_last_group():
    """
    Tests that the fallback splitter is correctly applied to the last group of
    sentences if it is oversized.
    """
    # The last sentence is very long and will form its own group, which is oversized.
    text = (
        "This is a sentence about Topic A. "
        "This is a very long sentence about Topic B that will exceed the chunk size."
    )
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_text(text)
    # The first sentence is one chunk.
    # The second sentence is split into multiple chunks by the fallback.
    assert len(chunks) > 2
    assert chunks[0].content == "This is a sentence about Topic A."
    assert "Topic B" in chunks[1].content


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_with_runt_handling():
    """
    Tests that small chunks created by semantic splitting are correctly handled
    by merging them with the previous chunk.
    """
    # Topic A is long, the transition is very short (a runt), and Topic B is long.
    text = (
        "This is a long sentence about Topic A. It has plenty of content. "
        "Go. "
        "This is a long sentence about Topic B. It also has plenty of content."
    )

    # The mock embeddings will create a clear semantic break around "Go."
    def mock_embedding_function_runt(texts: List[str]) -> "np.ndarray":
        # This mock function is intentionally simple and order-dependent for this test.
        # It simulates a clear semantic break.
        embeddings = []
        for i, text in enumerate(texts):
            if i < 2:  # First two sentences are Topic A
                embeddings.append([1.0, 0.0, 0.0])
            elif i == 2:  # The "Go." sentence is its own topic
                embeddings.append([0.0, 1.0, 0.0])
            else:  # The remaining sentences are Topic B
                embeddings.append([0.0, 0.0, 1.0])
        return np.array(embeddings)

    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function_runt,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        minimum_chunk_size=10,  # "Go." has length 3.
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks = splitter.split_text(text)

    # Without runt handling, we would expect 3 chunks.
    # With merging, the "Go." chunk should be merged with the "Topic A" chunk.
    # The "Topic B" chunk should remain separate.
    # So we expect 2 chunks.
    assert len(chunks) == 2
    assert "Go." in chunks[0].content
    assert "Topic A" in chunks[0].content
    assert "Topic B" in chunks[1].content


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_fallback_on_intermediate_group():
    """
    Tests that the fallback splitter is correctly applied to an intermediate group.
    """
    text = (
        "This is a very long sentence about Topic A that will exceed the chunk size. "
        "This is a sentence about Topic B."
    )
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_text(text)
    assert len(chunks) > 2
    assert "Topic A" in chunks[0].content
    assert "Topic B" in chunks[-1].content
