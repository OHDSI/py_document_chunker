import pytest
from typing import List

# Attempt to import numpy to determine if tests should be skipped.
try:
    import numpy as np
    from py_document_chunker.strategies.semantic import SemanticSplitter
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="Numpy is not available")
def test_semantic_splitting_with_multiple_topics_and_fallback():
    """
    Tests the SemanticSplitter in a complex scenario involving:
    1.  Multiple distinct semantic topics (A, B, C).
    2.  A mock embedding function that creates clear breakpoints.
    3.  An oversized semantic chunk (Topic B) that forces a fallback to the
        RecursiveCharacterSplitter.
    4.  Verification of chunk content, count, and overlap metadata.
    """
    # 1. Define a document with three distinct topics. Topic B is long.
    text = (
        "Topic A: sentence one. Topic A: sentence two. "
        "Topic B: this is a very long sentence, part one. "
        "Topic B: this is a very long sentence, part two. "
        "Topic B: this is a very long sentence, part three. "
        "Topic C: sentence one. Topic C: sentence two."
    )

    # 2. Create a mock embedding function for the topics.
    def mock_embedding_function(texts: List[str]) -> "np.ndarray":
        embeddings = []
        for text in texts:
            if "Topic A" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "Topic B" in text:
                embeddings.append([0.0, 1.0, 0.0])
            elif "Topic C" in text:
                embeddings.append([0.0, 0.0, 1.0])
            else:
                embeddings.append([0.5, 0.5, 0.5])  # Should not happen in this test
        return np.array(embeddings)

    # 3. Instantiate the splitter.
    # The chunk_size is set to force the "Topic B" group to be split by the fallback.
    # The breakpoint_threshold is set to reliably split between topics.
    splitter = SemanticSplitter(
        embedding_function=mock_embedding_function,
        breakpoint_method="absolute",
        breakpoint_threshold=0.5,
        chunk_size=100,
        chunk_overlap=20,
    )

    # 4. Run the splitter.
    chunks = splitter.split_text(text)

    # 5. Add assertions.
    # We expect the following chunks:
    # - 1 chunk for Topic A.
    # - Multiple chunks for Topic B (fallback).
    # - 1 chunk for Topic C.
    # Based on the text and chunk size, Topic B should be split into 2 chunks.
    # So, total chunks = 1 (A) + 2 (B) + 1 (C) = 4.
    assert len(chunks) == 4, f"Expected 4 chunks, but got {len(chunks)}"

    # Check Topic A chunk
    assert "Topic A" in chunks[0].content
    assert "Topic B" not in chunks[0].content
    assert "Topic C" not in chunks[0].content
    assert chunks[0].chunking_strategy_used == "semantic"

    # Check Topic B chunks (created by fallback)
    assert "Topic B" in chunks[1].content
    assert "Topic A" not in chunks[1].content
    assert "Topic C" not in chunks[1].content
    assert "part one" in chunks[1].content
    assert "part three" not in chunks[1].content
    # The fallback splitter should set this field.
    assert chunks[1].chunking_strategy_used == "recursive_character"

    assert "Topic B" in chunks[2].content
    assert "part three" in chunks[2].content
    assert chunks[2].chunking_strategy_used == "recursive_character"

    # Check Topic C chunk
    assert "Topic C" in chunks[3].content
    assert "Topic A" not in chunks[3].content
    assert "Topic B" not in chunks[3].content
    assert chunks[3].chunking_strategy_used == "semantic"

    # Verify overlap metadata for the fallback-split chunks
    assert chunks[1].overlap_content_next is not None
    assert chunks[2].overlap_content_previous is not None
    assert chunks[1].overlap_content_next == chunks[2].overlap_content_previous
