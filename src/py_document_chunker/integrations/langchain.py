from typing import Any, List, Optional

from ..base import TextSplitter as AdvancedTextSplitter
from ..core import Chunk

try:
    from langchain_core.documents import Document
    from langchain_core.text_splitter import TextSplitter as LangChainTextSplitter
except ImportError:
    # Allow the package to be imported even if langchain is not installed.
    # A runtime error will be raised only when the class is used.
    LangChainTextSplitter = object
    Document = object


class LangChainWrapper(LangChainTextSplitter):
    """
    A wrapper to use any text splitter from the `advanced-text-segmentation`
    package seamlessly within the LangChain ecosystem.

    This class conforms to the `langchain_core.text_splitter.TextSplitter`
    interface, allowing it to be used in any LangChain pipeline that expects
    a text splitter.
    """

    def __init__(self, splitter: AdvancedTextSplitter, **kwargs: Any):
        """
        Initializes the LangChain wrapper.

        Args:
            splitter: An instance of a text splitter from this package (e.g.,
                `SentenceSplitter`, `MarkdownSplitter`).
            **kwargs: Additional arguments to be passed to the LangChain
                base TextSplitter, although they are not used by this wrapper's
                core logic.
        """
        # We don't call super().__init__ with splitter args, as the wrapped
        # splitter already has them. We pass LangChain's expected args.
        super().__init__(**kwargs)
        self.splitter = splitter

    def _ensure_langchain_is_installed(self):
        """Checks if langchain-core is installed and raises an error if not."""
        if LangChainTextSplitter is object:
            raise ImportError(
                "langchain-core is not installed. Please install it via `pip install "
                '"py_document_chunker[langchain]"` or `pip install langchain-core`.'
            )

    def _chunk_to_document(self, chunk: Chunk) -> Document:
        """Converts a native Chunk object to a LangChain Document."""
        self._ensure_langchain_is_installed()
        # The main content goes into page_content.
        # All other rich metadata from the chunk goes into the metadata dict.
        metadata = chunk.to_dict()
        page_content = metadata.pop("content")

        # LangChain's Document expects a flat metadata dictionary.
        # Our hierarchical_context is already a dict, which is fine.
        return Document(page_content=page_content, metadata=metadata)

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into a list of strings.

        Note: This method loses the rich metadata from the splitter.
        For most use cases, `create_documents` is preferred.
        """
        chunks = self.splitter.split_text(text)
        return [chunk.content for chunk in chunks]

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Create LangChain Document objects from a list of texts.
        """
        self._ensure_langchain_is_installed()
        documents = []
        # If no metadatas are provided, create a list of empty dicts
        metadatas = metadatas or ([{}] * len(texts))

        for i, text in enumerate(texts):
            source_metadata = metadatas[i]
            # The source_document_id can be passed from the metadata.
            # We look for a common key like 'source' or 'document_id'.
            source_doc_id = source_metadata.get("source") or source_metadata.get(
                "document_id"
            )

            chunks = self.splitter.split_text(
                text, source_document_id=str(source_doc_id) if source_doc_id else None
            )
            for chunk in chunks:
                # Combine original metadata with the chunk's rich metadata
                # The chunk's metadata takes precedence.
                new_metadata = source_metadata.copy()
                new_metadata.update(chunk.to_dict())
                page_content = new_metadata.pop("content")

                documents.append(
                    Document(page_content=page_content, metadata=new_metadata)
                )

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a list of LangChain Documents, preserving metadata."""
        self._ensure_langchain_is_installed()
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas=metadatas)
