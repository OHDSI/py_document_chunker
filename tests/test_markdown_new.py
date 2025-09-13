import pytest

from py_document_chunker import MarkdownSplitter

# FRD Requirement Being Tested:
# R-3.4.2 (extended): The strategy MUST recognize structural hierarchies, including
# different block-level elements like paragraphs, lists, and blockquotes, and
# should not merge them into a single chunk if it compromises semantic boundaries.

MARKDOWN_WITH_DIFFERENT_BLOCKS = """# Section with Mixed Content

This is the first paragraph. It is a distinct semantic unit.

- This is the first list item.
- This is the second list item.

This is the second paragraph, appearing after the list."""


def test_splits_by_different_block_types():
    """
    Tests that the splitter correctly creates new chunks for different
    block-level elements (e.g., a paragraph followed by a list) even when
    they would otherwise fit in a single chunk.
    """
    splitter = MarkdownSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = splitter.split_text(MARKDOWN_WITH_DIFFERENT_BLOCKS)

    # Expect 3 chunks:
    # 1. Heading + first paragraph
    # 2. The list
    # 3. The second paragraph
    assert (
        len(chunks) == 3
    ), "Expected to split content into 3 chunks based on block type"

    # Check chunk 1 (Heading + Paragraph)
    assert chunks[0].content.strip().startswith("# Section with Mixed Content")
    assert "first paragraph" in chunks[0].content
    assert "list item" not in chunks[0].content

    # Check chunk 2 (List)
    assert chunks[1].content.strip().startswith("- This is the first list item.")
    assert "first paragraph" not in chunks[1].content
    assert "second paragraph" not in chunks[1].content

    # Check chunk 3 (Paragraph)
    assert chunks[2].content.strip().startswith("This is the second paragraph")


COMPLEX_MARKDOWN = """# Advanced Document

This is a paragraph introducing a table.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

> This is a blockquote.
> It should be a single chunk.

And here is a final paragraph.
"""


def test_handles_complex_block_types():
    """
    Tests that the splitter correctly handles more complex block types
    like tables and blockquotes as distinct chunks.
    """
    splitter = MarkdownSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = splitter.split_text(COMPLEX_MARKDOWN)

    assert len(chunks) == 4, "Expected to split content into 4 chunks based on block type"

    # Chunk 1: Heading and intro paragraph
    assert chunks[0].content.strip().startswith("# Advanced Document")
    assert "introducing a table" in chunks[0].content
    assert "| Header 1 |" not in chunks[0].content
    assert "> This is a blockquote" not in chunks[0].content
    assert chunks[0].hierarchical_context == {"H1": "Advanced Document"}

    # Chunk 2: Table
    assert chunks[1].content.strip().startswith("| Header 1 |")
    assert "Cell 4" in chunks[1].content
    assert "> This is a blockquote" not in chunks[1].content
    assert chunks[1].hierarchical_context == {"H1": "Advanced Document"}

    # Chunk 3: Blockquote
    assert chunks[2].content.strip().startswith("> This is a blockquote.")
    assert "final paragraph" not in chunks[2].content
    assert chunks[2].hierarchical_context == {"H1": "Advanced Document"}

    # Chunk 4: Final paragraph
    assert chunks[3].content.strip().startswith("And here is a final paragraph.")
    assert chunks[3].hierarchical_context == {"H1": "Advanced Document"}
