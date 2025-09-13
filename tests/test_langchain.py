from unittest.mock import MagicMock, patch

import pytest

from py_document_chunker.base import TextSplitter
from py_document_chunker.core import Chunk
from py_document_chunker.integrations.langchain import LangChainWrapper


# Mock TextSplitter for testing
class MockTextSplitter(TextSplitter):
    def __init__(self, chunk_size=100, chunk_overlap=50, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text, source_document_id=None):
        # Simple split by space for predictable testing
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


def test_langchain_wrapper_init(mock_splitter):
    """Tests the initialization of the LangChainWrapper."""
    wrapper = LangChainWrapper(mock_splitter)
    assert wrapper.splitter is mock_splitter


@patch("py_document_chunker.integrations.langchain.Document")
def test_chunk_to_document(mock_document, mock_splitter):
    """Tests the conversion of a Chunk to a LangChain Document."""
    chunk = Chunk(
        content="Hello world",
        start_index=0,
        end_index=11,
        sequence_number=0,
        chunk_id="1",
        metadata={"key": "value"},
    )
    wrapper = LangChainWrapper(mock_splitter)

    # Don't raise an error if langchain is not installed
    wrapper._ensure_langchain_is_installed = MagicMock()

    wrapper._chunk_to_document(chunk)

    mock_document.assert_called_once()
    assert mock_document.call_args[1]["page_content"] == "Hello world"


def test_split_text(mock_splitter):
    """Tests the basic split_text method."""
    wrapper = LangChainWrapper(mock_splitter)
    text = "this is a test"
    result = wrapper.split_text(text)
    assert result == ["this is", "a test"]


@patch("py_document_chunker.integrations.langchain.Document")
def test_create_documents(mock_document, mock_splitter):
    """Tests the creation of documents from texts."""
    wrapper = LangChainWrapper(mock_splitter)
    wrapper._ensure_langchain_is_installed = MagicMock()
    texts = ["this is a test", "another one"]
    wrapper.create_documents(texts)

    assert mock_document.call_count == 3


@patch("py_document_chunker.integrations.langchain.Document")
def test_create_documents_with_metadata(mock_document, mock_splitter):
    """Tests creating documents with metadata."""
    wrapper = LangChainWrapper(mock_splitter)
    wrapper._ensure_langchain_is_installed = MagicMock()
    texts = ["this is a test"]
    metadatas = [{"source": "doc1"}]
    wrapper.create_documents(texts, metadatas=metadatas)

    assert mock_document.call_count == 2
    assert mock_document.call_args_list[0][1]["metadata"]["source"] == "doc1"


@patch("py_document_chunker.integrations.langchain.Document")
def test_split_documents(mock_document, mock_splitter):
    """Tests splitting a list of LangChain Documents."""
    wrapper = LangChainWrapper(mock_splitter)
    wrapper._ensure_langchain_is_installed = MagicMock()

    doc_class = type(
        "Document",
        (),
        {"page_content": "this is a test", "metadata": {"source": "doc1"}},
    )

    docs = [doc_class(), doc_class()]
    wrapper.split_documents(docs)

    assert mock_document.call_count == 4


def test_import_error(mock_splitter):
    """
    Tests that an ImportError is raised if langchain-core is not installed.
    """
    with patch(
        "py_document_chunker.integrations.langchain.LangChainTextSplitter", object
    ):
        wrapper = LangChainWrapper(mock_splitter)
        with pytest.raises(ImportError):
            wrapper.split_documents([])
