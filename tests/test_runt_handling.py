import copy
from typing import List, Optional

import pytest

from py_document_chunker.base import TextSplitter
from py_document_chunker.core import Chunk

# ==========================================
# Test Setup
# ==========================================


class DummySplitter(TextSplitter):
    """A concrete implementation of TextSplitter for testing the base class logic."""

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """This method is not used in these tests but is required by the ABC."""
        return [
            Chunk(content=text, start_index=0, end_index=len(text), sequence_number=0)
        ]


def create_chunks_from_content(contents: list[str]) -> tuple[list[Chunk], str]:
    """Helper to create a list of Chunk objects from a list of strings."""
    chunks = []
    start_index = 0
    full_text_parts = []
    for i, content in enumerate(contents):
        end_index = start_index + len(content)
        chunks.append(
            Chunk(
                content=content,
                start_index=start_index,
                end_index=end_index,
                sequence_number=i,
            )
        )
        full_text_parts.append(content)
        start_index = end_index + 1  # Assume space separator

    original_text = " ".join(full_text_parts)
    return chunks, original_text


@pytest.fixture
def sample_chunks_and_text():
    """Provides a standard set of chunks for testing."""
    return create_chunks_from_content(
        [
            "This is the first major chunk.",  # 30 chars
            "runt",  # 4 chars
            "This is the second major chunk.",  # 31 chars
            "runt2",  # 5 chars
            "runt3",  # 5 chars
            "This is the final major chunk.",  # 30 chars
        ]
    )


# ==========================================
# Tests for 'merge_with_previous' Strategy
# ==========================================


def test_merge_previous_single_runt_in_middle():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "runt", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "A long piece of text. runt"
    assert result[0].end_index == chunks[1].end_index
    assert result[1].content == "Another long piece."
    assert result[0].sequence_number == 0 and result[1].sequence_number == 1


def test_merge_previous_runt_at_end():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "Another long piece.", "runt"]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[1].content == "Another long piece. runt"
    assert result[1].end_index == chunks[2].end_index
    assert result[0].content == "A long piece of text."


def test_merge_previous_runt_at_start():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks, text = create_chunks_from_content(
        ["runt", "A long piece of text.", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "runt A long piece of text."
    assert result[0].start_index == chunks[0].start_index
    assert result[0].end_index == chunks[1].end_index


def test_merge_previous_multiple_consecutive_runts():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "runt1", "runt2", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "A long piece of text. runt1 runt2"
    assert result[0].end_index == chunks[2].end_index


def test_merge_previous_does_not_merge_if_exceeds_chunk_size():
    splitter = DummySplitter(
        chunk_size=30,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_previous",
    )
    chunks, text = create_chunks_from_content(["This chunk is 27 characters.", "runt"])

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "This chunk is 27 characters."
    assert result[1].content == "runt"


# ==========================================
# Tests for 'merge_with_next' Strategy
# ==========================================


def test_merge_next_single_runt_in_middle():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_next",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "runt", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "A long piece of text."
    assert result[1].content == "runt Another long piece."
    assert result[1].start_index == chunks[1].start_index


def test_merge_next_runt_at_start():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_next",
    )
    chunks, text = create_chunks_from_content(
        ["runt", "A long piece of text.", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "runt A long piece of text."
    assert result[0].start_index == chunks[0].start_index


def test_merge_next_runt_at_end():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_next",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "Another long piece.", "runt"]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "A long piece of text."
    assert result[1].content == "Another long piece. runt"
    assert result[1].end_index == chunks[2].end_index


def test_merge_next_multiple_consecutive_runts():
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_next",
    )
    chunks, text = create_chunks_from_content(
        ["A long piece of text.", "runt1", "runt2", "Another long piece."]
    )

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "A long piece of text."
    assert result[1].content == "runt1 runt2 Another long piece."
    assert result[1].start_index == chunks[1].start_index


def test_merge_next_does_not_merge_if_exceeds_chunk_size():
    splitter = DummySplitter(
        chunk_size=30,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="merge_with_next",
    )
    chunks, text = create_chunks_from_content(["runt", "This chunk is 27 characters."])

    result = splitter._enforce_minimum_chunk_size(chunks, text)

    assert len(result) == 2
    assert result[0].content == "runt"
    assert result[1].content == "This chunk is 27 characters."


# ==========================================
# Tests for 'discard' Strategy
# ==========================================


def test_discard_strategy(sample_chunks_and_text):
    splitter = DummySplitter(
        chunk_size=100,
        chunk_overlap=0,
        minimum_chunk_size=10,
        min_chunk_merge_strategy="discard",
    )
    chunks, text = sample_chunks_and_text

    result = splitter._enforce_minimum_chunk_size(copy.deepcopy(chunks), text)

    assert len(result) == 3
    assert "runt" not in result[0].content
    assert "runt2" not in result[1].content
    assert "runt3" not in result[2].content

    assert result[0].sequence_number == 0
    assert result[1].sequence_number == 1
    assert result[2].sequence_number == 2
