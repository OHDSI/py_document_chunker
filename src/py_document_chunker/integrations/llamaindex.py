from typing import Any, List, Optional, Sequence

from ..base import TextSplitter as AdvancedTextSplitter
from ..core import Chunk

try:
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.callbacks.base import CallbackManager
    from llama_index.core.node_parser.interface import NodeParser
    from llama_index.core.schema import BaseNode, TextNode
except ImportError:
    # Allow import even if llama_index is not installed.
    def NodeParser(**kwargs):
        return None

    BaseNode = object

    def Field(**kwargs):
        return None

    CallbackManager = Any
    TextNode = Any


class LlamaIndexWrapper(NodeParser):
    """
    A wrapper to use any text splitter from the `advanced-text-segmentation`
    package seamlessly within the LlamaIndex ecosystem.

    This class conforms to the `llama_index.core.node_parser.NodeParser`
    interface, allowing it to be used in any LlamaIndex pipeline that expects
    a node parser (which is LlamaIndex's term for a text splitter).
    """

    splitter: AdvancedTextSplitter = Field(
        description="The advanced-text-segmentation splitter to use."
    )

    def __init__(
        self,
        splitter: AdvancedTextSplitter,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        """
        Initializes the LlamaIndex wrapper.

        Args:
            splitter: An instance of a text splitter from this package.
            callback_manager: LlamaIndex callback manager.
        """
        super().__init__(splitter=splitter, callback_manager=callback_manager, **kwargs)

    def _ensure_llamaindex_is_installed(self):
        """Checks if llama-index-core is installed and raises an error if not."""
        if NodeParser is object:
            raise ImportError(
                "llama-index-core is not installed. Please install it via `pip install "
                '"py_document_chunker[llamaindex]"` or `pip install llama-index-core`.'
            )

    @classmethod
    def from_defaults(
        cls, splitter: AdvancedTextSplitter, **kwargs: Any
    ) -> "LlamaIndexWrapper":
        """A factory method to create the wrapper from a splitter instance."""
        return cls(splitter=splitter, **kwargs)

    def _chunk_to_node(self, chunk: Chunk, source_node: BaseNode) -> TextNode:
        """Converts a native Chunk object to a LlamaIndex TextNode."""
        self._ensure_llamaindex_is_installed()
        metadata = chunk.to_dict()
        text = metadata.pop("content")

        # Combine source document metadata with the chunk's metadata
        final_metadata = source_node.metadata.copy()
        final_metadata.update(metadata)

        return TextNode(
            text=text,
            metadata=final_metadata,
            relationships={
                "source": source_node.as_related_node_info(),
            },
        )

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Parses a sequence of LlamaIndex nodes into more granular text nodes.
        """
        self._ensure_llamaindex_is_installed()
        all_new_nodes: List[BaseNode] = []
        for source_node in nodes:
            text = source_node.get_content()

            # Use the wrapped splitter to get chunks
            chunks = self.splitter.split_text(
                text, source_document_id=source_node.node_id
            )

            for chunk in chunks:
                new_node = self._chunk_to_node(chunk, source_node)
                all_new_nodes.append(new_node)

        return all_new_nodes

    # LlamaIndex's from_defaults expects a class method
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LlamaIndexWrapper"
