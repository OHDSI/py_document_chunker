from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..base import TextSplitter
from ..core import Chunk
from .recursive import RecursiveCharacterSplitter

try:
    from tree_sitter import Language, Node, Parser
    from tree_sitter_language_pack import get_language, get_parser
except ImportError:
    Parser = None  # type: ignore

# Language-specific queries to find top-level chunkable nodes.
# This can be expanded for more languages and more granular control.
LANGUAGE_QUERIES: Dict[str, Set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",  # const/let variables
        "export_statement",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "enum_item",
        "impl_item",
        "trait_item",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
    },
    # Add other languages here
}


class CodeSplitter(TextSplitter):
    """
    Splits source code into chunks based on its syntactic structure.

    This strategy uses the `tree-sitter` library to parse source code into a
    concrete syntax tree. It then traverses the tree to split the code along
    syntactic boundaries, such as functions, classes, or methods. This ensures
    that chunks are syntactically complete and logically coherent.
    """

    def __init__(self, language: str, *args: Any, **kwargs: Any):
        """
        Initializes the CodeSplitter.

        Args:
            language: The programming language of the code (e.g., 'python', 'javascript').
            *args, **kwargs: Additional arguments for the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)
        if Parser is None:
            raise ImportError(
                "tree-sitter is not installed. Please install it via `pip install "
                '"pyDocumentChunker[code]"` or `pip install tree-sitter tree-sitter-languages`.'
            )
        try:
            self.language: Language = get_language(language)
        except Exception:
            raise ValueError(
                f"Language '{language}' is not supported or could not be loaded. "
                "Please ensure it is a valid language supported by tree-sitter-languages."
            )

        self.parser: Parser = get_parser(language)
        self._chunkable_nodes: Set[str] = LANGUAGE_QUERIES.get(language, set())
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def _find_chunkable_nodes(self, node: Node) -> List[Node]:
        """
        Performs a pre-order traversal to find the highest-level chunkable nodes.

        If a node is of a chunkable type, it is added to the list, and its
        children are not traversed further. This prioritizes high-level constructs
        (like whole classes) over their smaller constituents (like methods).
        """
        if node.type in self._chunkable_nodes:
            return [node]

        chunkable_children = []
        for child in node.children:
            chunkable_children.extend(self._find_chunkable_nodes(child))
        return chunkable_children

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the source code using its syntax tree."""
        text = self._preprocess(text)
        if not text.strip():
            return []

        text_bytes = text.encode("utf-8")
        tree = self.parser.parse(text_bytes)
        root_node = tree.root_node

        if not root_node.children:
            # If the document is empty or just whitespace
            return self._fallback_splitter.split_text(text, source_document_id)

        # 1. Find all the highest-level syntactic nodes that are designated as chunkable.
        chunkable_nodes = self._find_chunkable_nodes(root_node)

        # If no chunkable nodes are found, fallback to splitting the whole document.
        if not chunkable_nodes:
            return self._fallback_splitter.split_text(text, source_document_id)

        chunks: List[Chunk] = []
        for node in chunkable_nodes:
            node_text = text_bytes[node.start_byte : node.end_byte].decode("utf-8")

            # 2. If a high-level node is still larger than chunk_size, use the
            #    fallback splitter on just the text of that node.
            if self.length_function(node_text) > self.chunk_size:
                fallback_chunks = self._fallback_splitter.split_text(
                    node_text, source_document_id
                )
                # Adjust indices of fallback chunks to be relative to the whole document
                for chunk in fallback_chunks:
                    chunk.start_index += node.start_byte
                    chunk.end_index += node.start_byte
                    chunk.chunking_strategy_used = "code-fallback"
                chunks.extend(fallback_chunks)
            else:
                # 3. Otherwise, create a single chunk from the node.
                chunk = Chunk(
                    content=node_text,
                    start_index=node.start_byte,
                    end_index=node.end_byte,
                    sequence_number=0,  # Placeholder, will be re-sequenced
                    source_document_id=source_document_id,
                    chunking_strategy_used="code",
                )
                chunks.append(chunk)

        # Re-assign sequence numbers for the final list of chunks
        for i, chunk in enumerate(chunks):
            chunk.sequence_number = i

        # Populate overlap metadata and handle runts as post-processing steps
        from ..utils import _populate_overlap_metadata

        _populate_overlap_metadata(chunks, text)
        return self._enforce_minimum_chunk_size(chunks, text)
