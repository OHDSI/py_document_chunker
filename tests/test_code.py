import pytest

from py_document_chunker import CodeSplitter

# Sample code snippets for testing
PYTHON_CODE = '''
def my_function(a, b):
    """This is a sample function."""
    return a + b

class MyClass:
    """A sample class."""
    def __init__(self, x):
        self.x = x

    def get_x(self):
        return self.x
'''


def test_split_python_by_toplevel_constructs():
    """
    Tests that the splitter correctly identifies top-level constructs
    (function and class) in Python code and splits them into separate chunks.
    """
    splitter = CodeSplitter(language="python", chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_text(PYTHON_CODE)

    assert len(chunks) == 2, "Should split into two chunks (function and class)"

    # Check content of the first chunk (the function)
    assert "def my_function(a, b):" in chunks[0].content
    assert "return a + b" in chunks[0].content
    assert "class MyClass:" not in chunks[0].content

    # Check content of the second chunk (the class)
    assert "class MyClass:" in chunks[1].content
    assert "def __init__(self, x):" in chunks[1].content
    assert "def my_function(a, b):" not in chunks[1].content

    # Verify start and end indices
    assert chunks[0].start_index == 1  # Ignoring the initial newline
    assert chunks[0].end_index > 1
    assert chunks[1].start_index > chunks[0].end_index


# A single large function that is larger than the chunk_size
LARGE_PYTHON_FUNCTION = """
def very_large_function():
    # This function is intentionally long to test the fallback mechanism.
    # It should be split into multiple smaller chunks by the recursive splitter.
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    # More lines to make it exceed the chunk size...
    print(a, b, c, d, e, f, g, h, i, j)
    # And even more lines...
    return a + b + c + d + e + f + g + h + i + j
"""


def test_fallback_splitter_for_large_node():
    """
    Tests that if a single code construct (like a function) is larger than
    the chunk_size, it gets passed to the fallback splitter.
    """
    splitter = CodeSplitter(language="python", chunk_size=150, chunk_overlap=0)
    # The LARGE_PYTHON_FUNCTION is > 150 chars
    chunks = splitter.split_text(LARGE_PYTHON_FUNCTION)

    assert len(chunks) > 1, "The large function should be split into multiple chunks"

    # Verify that no chunk exceeds the chunk_size
    for chunk in chunks:
        assert len(chunk.content) <= 150


JS_CODE = """
function greet(name) {
  return `Hello, ${name}!`;
}

const add = (a, b) => {
  // A simple arrow function
  return a + b;
};
"""


def test_split_javascript_code():
    """
    Tests that the splitter works correctly for a different language (JS).
    """
    splitter = CodeSplitter(language="javascript", chunk_size=100, chunk_overlap=0)
    chunks = splitter.split_text(JS_CODE)

    assert len(chunks) == 2
    assert "function greet(name)" in chunks[0].content
    assert "const add = (a, b)" in chunks[1].content


def test_unsupported_language_raises_error():
    """
    Tests that initializing the splitter with an unsupported language
    raises a ValueError.
    """
    with pytest.raises(ValueError, match="is not supported or could not be loaded"):
        CodeSplitter(language="not_a_real_language")


def test_import_error_if_tree_sitter_not_installed(monkeypatch):
    """
    Tests that an ImportError is raised if tree-sitter is not installed.
    """
    monkeypatch.setattr("py_document_chunker.strategies.code.Parser", None)
    with pytest.raises(ImportError, match="tree-sitter is not installed"):
        CodeSplitter(language="python")


def test_empty_and_whitespace_code():
    """Tests that empty or whitespace-only code returns an empty list."""
    splitter = CodeSplitter(language="python")
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n \t ") == []


def test_no_chunkable_nodes_uses_fallback():
    """
    Tests that if no high-level chunkable nodes are found, the fallback
    splitter is used on the entire file.
    """
    # This Python code only contains simple statements, no functions or classes.
    code = "a = 1\nb = 2\nprint(a + b)"
    splitter = CodeSplitter(language="python", chunk_size=10, chunk_overlap=5)
    chunks = splitter.split_text(code)
    # The fallback should have been used.
    assert len(chunks) > 1
    assert chunks[0].content.strip() == "a = 1"
