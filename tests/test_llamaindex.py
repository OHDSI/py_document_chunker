from unittest.mock import MagicMock, patch

import pytest

from py_document_chunker.base import TextSplitter
from py_document_chunker.core import Chunk
from py_document_chunker.integrations.llamaindex import LlamaIndexWrapper

# Mock LlamaIndex classes if not installed
try:
    from llama_index.core.callbacks.base import CallbackManager
    from llama_index.core.schema import BaseNode, Document, TextNode
except ImportError:
    # Create dummy classes for testing if llama_index is not available
    class BaseNode:
        def __init__(self, text="", metadata=None, node_id="1"):
            self.text = text
            self.metadata = metadata or {}
            self.node_id = node_id

        def get_content(self):
            return self.text

        def as_related_node_info(self):
            return {"node_id": self.node_id}

    class TextNode(BaseNode):
        def __init__(self, text="", metadata=None, relationships=None, **kwargs):
            super().__init__(text=text, metadata=metadata)
            self.relationships = relationships or {}

    class Document(BaseNode):
        pass

    class CallbackManager:
        pass


# Mock TextSplitter for testing
class MockTextSplitter(TextSplitter):
    def __init__(self, chunk_size=100, chunk_overlap=50, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text, source_document_id=None):
        words = text.split()
        chunks = []
        for i in range(0, len(words), 2):
            content = " ".join(words[i : i + 2])
            chunks.append(
                Chunk(
                    content=content,
                    start_index=0,
                    end_index=len(content),
                    sequence_number=i // 2,
                )
            )
        return chunks


@pytest.fixture
def mock_splitter():
    """Fixture for a mock text splitter."""
    return MockTextSplitter()


def test_llamaindex_wrapper_init(mock_splitter):
    """Tests the initialization of the LlamaIndexWrapper."""
    wrapper = LlamaIndexWrapper(mock_splitter)
    assert wrapper.splitter is mock_splitter


def test_from_defaults(mock_splitter):
    """Tests the factory method."""
    wrapper = LlamaIndexWrapper.from_defaults(mock_splitter)
    assert isinstance(wrapper, LlamaIndexWrapper)
    assert wrapper.splitter is mock_splitter


@patch("py_document_chunker.integrations.llamaindex.TextNode")
def test_chunk_to_node(mock_text_node, mock_splitter):
    """Tests conversion of a Chunk to a TextNode."""
    chunk = Chunk(content="Hello world", start_index=0, end_index=11, sequence_number=0)
    source_node = MagicMock(spec=BaseNode)
    source_node.metadata = {"source": "doc1"}
    source_node.as_related_node_info.return_value = {"node_id": "1"}

    wrapper = LlamaIndexWrapper(mock_splitter)
    wrapper._ensure_llamaindex_is_installed = MagicMock()

    wrapper._chunk_to_node(chunk, source_node)

    mock_text_node.assert_called_once()
    assert mock_text_node.call_args[1]["text"] == "Hello world"


@patch("py_document_chunker.integrations.llamaindex.TextNode")
def test_parse_nodes(mock_text_node, mock_splitter):
    """Tests parsing a sequence of nodes."""
    nodes = [MagicMock(spec=BaseNode)]
    nodes[0].get_content.return_value = "this is a test"
    nodes[0].node_id = "1"
    nodes[0].metadata = {}
    nodes[0].as_related_node_info.return_value = {"node_id": "1"}

    wrapper = LlamaIndexWrapper(mock_splitter)
    wrapper._ensure_llamaindex_is_installed = MagicMock()

    wrapper._parse_nodes(nodes)

    assert mock_text_node.call_count == 2


def test_class_name():
    """Tests the class_name method."""
    assert LlamaIndexWrapper.class_name() == "LlamaIndexWrapper"


def test_import_error(mock_splitter):
    """
    Tests that an ImportError is raised if llama-index-core is not installed.
    """
    with patch("py_document_chunker.integrations.llamaindex.NodeParser", object):
        wrapper = LlamaIndexWrapper(mock_splitter)
        wrapper.callback_manager = MagicMock()
        with pytest.raises(ImportError):
            wrapper._parse_nodes([])
