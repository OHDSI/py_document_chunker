import pytest

try:
    from py_document_chunker.strategies.structure.html import HTMLSplitter

    BS4_LXML_AVAILABLE = True
except ImportError:
    BS4_LXML_AVAILABLE = False

HTML_TEXT = """
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>This is the introduction.</p>
    <div class="section">
        <h2>Section 1</h2>
        <p>Here is the content for the first section.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
    </div>
    <div class="section">
        <h2>Section 2</h2>
        <p>This is the second section.</p>
    </div>
</body>
</html>
"""


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_basic_splitting():
    """Tests that HTML is split by block tags."""
    splitter = HTMLSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_text(HTML_TEXT)

    assert len(chunks) > 1
    # Check that chunks are based on block elements
    assert "Main Title" in chunks[0].content
    assert "This is the introduction" in chunks[0].content
    assert "Section 1" in chunks[0].content


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_hierarchical_context():
    """Tests that the hierarchical_context is correctly populated from parent headers."""
    splitter = HTMLSplitter(chunk_size=500, chunk_overlap=50)  # Use a large chunk size
    chunks = splitter.split_text(HTML_TEXT)

    assert len(chunks) > 0

    # Find the chunk containing "List item 1"
    list_chunk = next(c for c in chunks if "List item 1" in c.content)

    # The direct parent header is H2, but it should also find the H1
    assert list_chunk.hierarchical_context.get("H1") == "Main Title"
    assert list_chunk.hierarchical_context.get("H2") == "Section 1"


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_remove_tags():
    """Tests that the `remove_tags` parameter correctly strips specified tags."""
    text_with_script = '<body><p>Some content.</p><script>alert("you are hacked");</script><footer>Footer</footer></body>'
    # Test with default removal list, which includes 'footer'
    splitter_default = HTMLSplitter()
    chunks_default = splitter_default.split_text(text_with_script)
    assert "alert" not in chunks_default[0].content
    assert "Footer" not in chunks_default[0].content

    # Test with custom removal list. For 'Footer' to be found, 'footer' must also
    # be considered a block tag. We set a chunk_size that is smaller than the
    # combined blocks but larger than the first block to force a split.
    splitter_custom = HTMLSplitter(
        chunk_size=15,
        chunk_overlap=0,
        remove_tags=["script"],
        block_tags=["p", "footer"],
    )
    chunks_custom = splitter_custom.split_text(text_with_script)
    # The content should now be two separate chunks
    assert len(chunks_custom) == 2
    assert chunks_custom[0].content == "Some content."
    assert chunks_custom[1].content == "Footer"


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_strip_all_tags_mode():
    """Tests the `strip_all_tags=True` mode for plain text splitting."""
    splitter = HTMLSplitter(
        chunk_size=30,
        chunk_overlap=0,  # Set to 0 to allow for clean join assertion
        strip_all_tags=True,
    )
    chunks = splitter.split_text(HTML_TEXT)
    # The <head> tag, which contains "Test Page", is stripped by default.
    full_text = "Main Title This is the introduction. Section 1 Here is the content for the first section. List item 1 List item 2 Section 2 This is the second section."

    assert len(chunks) > 1
    # This assertion is robust: it checks that no HTML is left and that the
    # combined content is correct, without being brittle about exact split points.
    for chunk in chunks:
        assert "<" not in chunk.content and ">" not in chunk.content
    # We join and replace spaces to make the comparison robust to minor whitespace diffs
    assert "".join(c.content for c in chunks).replace(" ", "") == full_text.replace(
        " ", ""
    )


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_start_index_correctness():
    """Tests that the start_index metadata is accurate."""
    splitter = HTMLSplitter(chunk_size=50, chunk_overlap=5)
    chunks = splitter.split_text(HTML_TEXT)

    # This is a bit harder to test precisely due to whitespace handling in HTML parsing.
    # A good enough check is that the text content of the chunk can be found at or near
    # the recorded start_index.
    for chunk in chunks:
        # We can't do a direct string comparison because of whitespace differences.
        # Instead, we check that the first few words of the chunk content appear in
        # a window of the original text around the start_index. This is a good
        # heuristic for correctness.
        words = chunk.content.strip().split()
        if not words:
            continue

        # Check for the first word in a window around the start index
        window_start = max(0, chunk.start_index - 50)
        window_end = chunk.start_index + 50
        window = HTML_TEXT[window_start:window_end]

        assert (
            words[0] in window
        ), f"First word '{words[0]}' not found near start_index {chunk.start_index}"


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_html_parser_fallback(monkeypatch):
    """
    Tests that the splitter falls back to lxml if html5lib fails.
    """
    from bs4 import BeautifulSoup as OriginalBeautifulSoup

    def mock_beautiful_soup(*args, **kwargs):
        if "html5lib" in args or kwargs.get("features") == "html5lib":
            raise Exception("html5lib failed")
        return OriginalBeautifulSoup(*args, **kwargs)

    monkeypatch.setattr(
        "py_document_chunker.strategies.structure.html.BeautifulSoup",
        mock_beautiful_soup,
    )

    splitter = HTMLSplitter()
    # This should not raise an exception because it falls back to lxml
    chunks = splitter.split_text(HTML_TEXT)
    assert len(chunks) > 0


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_empty_and_whitespace_html():
    """Tests that the splitter handles empty or whitespace-only HTML gracefully."""
    splitter = HTMLSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n \t ") == []
    assert splitter.split_text("<html><body></body></html>") == []


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_empty_block_tags_are_ignored():
    """Tests that empty block tags do not create empty chunks."""
    text = "<h1>Title</h1><p></p><p>  </p><h2>Subtitle</h2>"
    splitter = HTMLSplitter()
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert "Title" in chunks[0].content
    assert "Subtitle" in chunks[0].content


@pytest.mark.skipif(
    not BS4_LXML_AVAILABLE, reason="BeautifulSoup4 or lxml not installed"
)
def test_oversized_block_uses_fallback():
    """Tests that a block larger than chunk_size is split by the fallback."""
    text = f"<p>{'a' * 50}</p><p>{'b' * 50}</p>"
    splitter = HTMLSplitter(chunk_size=40, chunk_overlap=10)
    chunks = splitter.split_text(text)
    # Each <p> tag is oversized and should be split by the fallback.
    assert len(chunks) == 4  # 2 for each <p> tag
    assert chunks[0].content == "a" * 40
    assert chunks[1].content.startswith("a" * 10)
    assert chunks[2].content == "b" * 40
    assert chunks[3].content.startswith("b" * 10)
