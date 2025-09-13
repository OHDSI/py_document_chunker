from py_document_chunker import RecursiveCharacterSplitter

TEXT = """
Title: The Great Adventure

Chapter 1: The Beginning

It was a dark and stormy night. A lone traveler approached the village.
He was looking for a place to stay.

The innkeeper told him the inn was full.
"""


def test_recursive_basic_splitting():
    """Tests basic recursive splitting with default separators."""
    splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(TEXT)

    # Check that the text is split into multiple chunks and the content is preserved.
    assert len(chunks) > 1
    assert "".join(c.content for c in chunks) == TEXT


def test_recursive_custom_separators():
    """Tests splitting with a custom regex separator."""
    splitter = RecursiveCharacterSplitter(
        separators=[r"Chapter \d+:"],
        keep_separator=False,
        chunk_size=150,
        chunk_overlap=0,
    )
    chunks = splitter.split_text(TEXT)

    # Check that the split happened and the separator is removed.
    assert len(chunks) > 1
    assert "Chapter 1:" not in "".join(c.content for c in chunks)


def test_recursive_fallback_splitting():
    """Tests the fallback mechanism for text that can't be split by separators."""
    text = "a" * 200
    splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=5)
    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert chunks[0].content == "a" * 50


def test_recursive_length_function():
    """Tests that the splitter respects the length function."""
    # Each word is 5 tokens
    text = "word1 word2 word3 word4 word5 word6"

    # chunk_size is 12 tokens, so it should fit 2 words (10 tokens)
    splitter = RecursiveCharacterSplitter(
        chunk_size=12, chunk_overlap=0, length_function=lambda x: len(x.split()) * 5
    )
    chunks = splitter.split_text(text)

    assert len(chunks) == 3
    assert chunks[0].content.strip() == "word1 word2"
    assert chunks[1].content.strip() == "word3 word4"
    assert chunks[2].content.strip() == "word5 word6"
