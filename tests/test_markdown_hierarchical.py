from unittest.mock import patch

import pytest

from py_document_chunker import MarkdownSplitter

# FRD Requirement Being Tested:
# R-3.4.3: The strategy MUST prioritize splitting at higher-level structural boundaries.
# R-3.4.2: The strategy MUST recognize different block types (e.g. paragraphs, lists).

HIERARCHICAL_MD = """# Section 1 Title

This is the first paragraph of the first section.

## Section 1.1 Title

This is a subsection.

# Section 2 Title

This is a paragraph directly under the H1.

## Section 2.1 Title

This is the first subsection of Section 2.

## Section 2.2 Title

This is the second subsection of Section 2. It contains a list.

- List item 1
- List item 2

This is a paragraph after the list.

# Section 3 Title

This is a very long section with no subheadings. It is designed to be much larger than the chunk size to test the fallback mechanism. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. More and more and more text to ensure it is very long indeed.
"""


def test_hierarchical_and_block_type_splitting():
    """
    Tests that the splitter correctly splits based on header hierarchy AND block type.
    This test is updated to reflect the new, more granular splitting logic.
    """
    splitter = MarkdownSplitter(chunk_size=400, chunk_overlap=0)
    chunks = splitter.split_text(HIERARCHICAL_MD)

    # With the new logic, we expect 9 chunks.
    # 7 chunks from the structured part of the document, and 2 from the
    # fallback splitting of the very long Section 3.
    assert len(chunks) == 9

    # Chunk 1: Section 1
    assert chunks[0].content.strip().startswith("# Section 1 Title")
    assert "first paragraph" in chunks[0].content
    assert "Section 1.1" not in chunks[0].content
    assert chunks[0].hierarchical_context == {"H1": "Section 1 Title"}

    # Chunk 2: Section 1.1
    assert chunks[1].content.strip().startswith("## Section 1.1 Title")
    assert chunks[1].hierarchical_context == {
        "H1": "Section 1 Title",
        "H2": "Section 1.1 Title",
    }

    # Chunk 3: Section 2 Title + first paragraph
    assert chunks[2].content.strip().startswith("# Section 2 Title")
    assert "paragraph directly under the H1" in chunks[2].content
    assert "Section 2.1" not in chunks[2].content
    assert chunks[2].hierarchical_context == {"H1": "Section 2 Title"}

    # Chunk 4: Section 2.1
    assert chunks[3].content.strip().startswith("## Section 2.1 Title")
    assert chunks[3].hierarchical_context == {
        "H1": "Section 2 Title",
        "H2": "Section 2.1 Title",
    }

    # Chunk 5: Section 2.2 Title + paragraph before list
    assert chunks[4].content.strip().startswith("## Section 2.2 Title")
    assert "contains a list" in chunks[4].content
    assert "List item 1" not in chunks[4].content
    assert chunks[4].hierarchical_context == {
        "H1": "Section 2 Title",
        "H2": "Section 2.2 Title",
    }

    # Chunk 6: The list itself (this is the new behavior)
    assert chunks[5].content.strip().startswith("- List item 1")
    assert "paragraph after the list" not in chunks[5].content
    assert chunks[5].hierarchical_context == {
        "H1": "Section 2 Title",
        "H2": "Section 2.2 Title",
    }

    # Chunk 7: The paragraph after the list
    assert chunks[6].content.strip().startswith("This is a paragraph after the list.")
    assert chunks[6].hierarchical_context == {
        "H1": "Section 2 Title",
        "H2": "Section 2.2 Title",
    }

    # Chunks 8 & 9: Fallback split of Section 3
    assert chunks[7].content.strip().startswith("# Section 3 Title")
    assert "Lorem ipsum" in chunks[7].content
    assert "More and more" not in chunks[7].content  # Check that it was split
    assert chunks[7].hierarchical_context == {"H1": "Section 3 Title"}
    assert chunks[7].chunking_strategy_used == "markdown-fallback"

    assert "More and more" in chunks[8].content
    assert chunks[8].hierarchical_context == {"H1": "Section 3 Title"}
    assert chunks[8].chunking_strategy_used == "markdown-fallback"


def test_import_error_if_markdown_it_not_installed():
    """
    Tests that an ImportError is raised if markdown-it-py is not installed.
    """
    with patch("py_document_chunker.strategies.structure.markdown.MarkdownIt", None):
        with pytest.raises(ImportError):
            MarkdownSplitter()


def test_empty_and_whitespace_text():
    """Tests that the splitter handles empty or whitespace-only text gracefully."""
    splitter = MarkdownSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n \t ") == []


def test_private_get_node_text_with_nested_children():
    """
    Tests the private method _get_node_text with a more complex, nested node
    to ensure the recursive search for the end line works correctly.
    """
    from unittest.mock import Mock
    splitter = MarkdownSplitter()

    COMPLEX_MD = """# L1
## L2
- Item
"""
    line_indices = splitter._get_line_start_indices(COMPLEX_MD)

    # Mock SyntaxTreeNode structure to mimic markdown-it's output for COMPLEX_MD
    grandchild_node = Mock()
    grandchild_node.map = (2, 3)
    grandchild_node.children = []

    child_node_h2 = Mock()
    child_node_h2.map = (1, 3)
    child_node_h2.children = [grandchild_node]

    top_level_node = Mock()
    top_level_node.map = (0, 3)
    top_level_node.children = [child_node_h2]

    # The while loop in _get_node_text should be tested by this.
    text, start, end = splitter._get_node_text(top_level_node, COMPLEX_MD, line_indices)

    assert text.strip() == "## L2\n- Item"
    assert start == 5  # Start of "## L2"
    assert end == 18 # End of "- Item\n"

def test_markdown_splitting_without_trailing_newline():
    """
    Tests that the splitter correctly handles markdown text that does not
    end with a trailing newline. This covers the `else len(text)` branch.
    """
    splitter = MarkdownSplitter(chunk_size=1000)
    text = "# Section 1\n\nSome text" # No trailing newline

    chunks = splitter.split_text(text)

    assert len(chunks) == 1
    assert chunks[0].content.strip().startswith("# Section 1")
    assert chunks[0].content.strip().endswith("Some text")
    assert chunks[0].end_index == len(text)

def test_type_boundary_after_heading():
    """
    Tests the edge case where a heading is followed by a different block type,
    ensuring they are not split if they fit within the chunk size.
    """
    splitter = MarkdownSplitter(chunk_size=1024)
    text = "# Heading\n- List Item"

    chunks = splitter.split_text(text)

    # Expectation: The heading and the list are kept together in one chunk
    # because the logic explicitly prevents a type-based split after a heading.
    assert len(chunks) == 1
    assert chunks[0].content == "# Heading\n- List Item"
