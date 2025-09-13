import pytest

from py_document_chunker import Chunk


def test_chunk_post_init_validations():
    """
    Tests that the Chunk __post_init__ method raises ValueErrors for invalid inputs.
    """
    with pytest.raises(ValueError, match="start_index must be a non-negative integer."):
        Chunk(content="test", start_index=-1, end_index=10, sequence_number=0)

    with pytest.raises(
        ValueError, match="end_index must not be less than start_index."
    ):
        Chunk(content="test", start_index=10, end_index=5, sequence_number=0)

    with pytest.raises(
        ValueError, match="sequence_number must be a non-negative integer."
    ):
        Chunk(content="test", start_index=0, end_index=10, sequence_number=-1)


def test_chunk_to_dict():
    """
    Tests that the to_dict method correctly converts a Chunk object to a dictionary.
    """
    chunk = Chunk(
        content="test content",
        start_index=10,
        end_index=22,
        sequence_number=1,
        source_document_id="doc1",
        hierarchical_context={"H1": "Header 1"},
        overlap_content_previous="prev",
        overlap_content_next="next",
        chunking_strategy_used="test_strategy",
        metadata={"key": "value"},
    )
    chunk_dict = chunk.to_dict()

    assert chunk_dict["content"] == "test content"
    assert chunk_dict["start_index"] == 10
    assert chunk_dict["end_index"] == 22
    assert chunk_dict["sequence_number"] == 1
    assert chunk_dict["source_document_id"] == "doc1"
    assert chunk_dict["hierarchical_context"] == {"H1": "Header 1"}
    assert chunk_dict["overlap_content_previous"] == "prev"
    assert chunk_dict["overlap_content_next"] == "next"
    assert chunk_dict["chunking_strategy_used"] == "test_strategy"
    assert chunk_dict["metadata"] == {"key": "value"}
    assert "chunk_id" in chunk_dict
