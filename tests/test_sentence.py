from unittest.mock import patch

import pytest

# Attempt to import NLTK to determine if tests should be skipped.
try:
    import nltk

    from py_document_chunker import SentenceSplitter

    # We also need to check for the 'punkt' model.
    try:
        nltk.data.find("tokenizers/punkt")
        NLTK_AVAILABLE = True
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

TEXT = "This is the first sentence. This is the second sentence. A third one follows. And a fourth. Finally, the fifth."


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_sentence_basic_splitting():
    """Tests basic sentence splitting and aggregation into chunks."""
    splitter = SentenceSplitter(chunk_size=80, overlap_sentences=0)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert (
        chunks[0].content
        == "This is the first sentence. This is the second sentence. A third one follows."
    )
    assert chunks[0].start_index == 0
    assert chunks[1].content == "And a fourth. Finally, the fifth."
    assert chunks[1].start_index == 78


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_sentence_overlap():
    """Tests the sentence-based overlap functionality."""
    splitter = SentenceSplitter(chunk_size=80, overlap_sentences=1)
    chunks = splitter.split_text(TEXT)

    assert len(chunks) == 2
    assert (
        chunks[0].content
        == "This is the first sentence. This is the second sentence. A third one follows."
    )
    assert chunks[1].content == "A third one follows. And a fourth. Finally, the fifth."
    assert chunks[0].overlap_content_next is not None


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_oversized_sentence_fallback():
    """Tests that a single sentence larger than chunk_size is split by the fallback mechanism."""
    long_sentence = "This is a single very long sentence that is designed to be much larger than the tiny chunk size we are going to set for this specific test case."
    splitter = SentenceSplitter(chunk_size=40, overlap_sentences=0)
    chunks = splitter.split_text(long_sentence)

    assert len(chunks) > 1
    assert "".join(c.content for c in chunks) == long_sentence
    assert chunks[0].content.startswith("This is a single very long sentence")
    assert chunks[1].start_index > 0


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_no_sentences_found():
    """Tests that an empty list is returned if no sentences are found."""
    splitter = SentenceSplitter()
    assert splitter.split_text("         ") == []


def test_punkt_not_downloaded():
    """
    Tests that a RuntimeError is raised if the 'punkt' model is not downloaded.
    """
    with patch("nltk.data.find", side_effect=LookupError()):
        with pytest.raises(RuntimeError):
            SentenceSplitter()


def test_dependency_import_error():
    """
    Tests that an ImportError is raised if NLTK is not installed.
    We use unittest.mock to simulate its absence.
    """
    import importlib
    import sys

    # Simulate that the `nltk` package is not available
    with patch.dict(sys.modules, {"nltk": None}):
        # The module that imports `nltk` must be reloaded for the patch to take effect
        import py_document_chunker.strategies.sentence

        importlib.reload(py_document_chunker.strategies.sentence)

        from py_document_chunker import SentenceSplitter

        with pytest.raises(ImportError, match="NLTK is not installed"):
            SentenceSplitter()

    # The with block automatically restores sys.modules, so the module is
    # usable by other tests. Another reload is not necessary.


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_invalid_nlp_backend_raises_error():
    with pytest.raises(
        ValueError, match="Currently, only the 'nltk' backend is supported."
    ):
        SentenceSplitter(nlp_backend="unsupported")


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_negative_overlap_sentences_raises_error():
    import importlib

    import py_document_chunker.strategies.sentence

    importlib.reload(py_document_chunker.strategies.sentence)
    from py_document_chunker import SentenceSplitter

    with pytest.raises(
        ValueError, match="overlap_sentences must be a non-negative integer."
    ):
        SentenceSplitter(overlap_sentences=-1)


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_empty_text_returns_empty_list():
    splitter = SentenceSplitter()
    assert splitter.split_text("") == []


@pytest.mark.skipif(
    not NLTK_AVAILABLE, reason="NLTK or its 'punkt' model is not available"
)
def test_whitespace_preservation():
    """Tests that whitespace between sentences is preserved in the chunk content."""
    text_with_newlines = "This is the first sentence.\n\nThis is the second.\n\n  And a third."
    splitter = SentenceSplitter(chunk_size=100, overlap_sentences=0)
    chunks = splitter.split_text(text_with_newlines)

    assert len(chunks) == 1
    chunk = chunks[0]

    # The chunk content should be an exact slice from the original text.
    expected_content = text_with_newlines[chunk.start_index:chunk.end_index]
    assert chunk.content == expected_content
