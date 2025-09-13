from typing import Any, Dict, List, Optional, Tuple

from ...base import TextSplitter
from ...core import Chunk
from ..recursive import RecursiveCharacterSplitter

try:
    from markdown_it import MarkdownIt
    from markdown_it.tree import SyntaxTreeNode
except ImportError:
    MarkdownIt = None  # type: ignore
    SyntaxTreeNode = Any  # type: ignore


class MarkdownSplitter(TextSplitter):
    """
    Splits a Markdown document based on its structural elements in a hierarchical manner.

    This strategy uses `markdown-it-py` to parse the Markdown into a syntax tree.
    It then recursively traverses the tree to split the document, prioritizing
    higher-level structural boundaries (like H1, H2) over lower-level ones, in
    accordance with FRD R-3.4.3.

    If a semantic section is larger than the chunk size, it is recursively
    split by its sub-headers. If a section has no sub-headers and is still too
    large, a fallback `RecursiveCharacterSplitter` is used.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the MarkdownSplitter."""
        super().__init__(*args, **kwargs)
        if MarkdownIt is None:
            raise ImportError(
                "markdown-it-py is not installed. Please install it via `pip install "
                '"pyDocumentChunker[markdown]"` or `pip install markdown-it-py`.'
            )
        self.md_parser = MarkdownIt("commonmark", {"sourcepos": True}).enable("table")
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=False,
            unicode_normalize=False,
        )

    def _get_line_start_indices(self, text: str) -> List[int]:
        """Calculates the character start index of each line in the text."""
        indices = [0]
        for line in text.splitlines(keepends=True):
            indices.append(indices[-1] + len(line))
        return indices

    def _get_node_text(
        self, node: SyntaxTreeNode, text: str, line_indices: List[int]
    ) -> Tuple[str, int, int]:
        """Extracts the raw text content of a node, including all its children."""
        if not node.map or not node.children:
            start_line, end_line = node.map or (0, 0)
            start_char = line_indices[start_line]
            end_char = (
                line_indices[end_line] if end_line < len(line_indices) else len(text)
            )
            return text[start_char:end_char], start_char, end_char

        start_line, _ = node.children[0].map or (0, 0)
        end_line_node = node.children[-1]

        # Find the true end line by looking at the last descendant
        while end_line_node.children:
            end_line_node = end_line_node.children[-1]
        _, end_line = end_line_node.map or (0, 0)

        start_char = line_indices[start_line]
        end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
        return text[start_char:end_char], start_char, end_char

    def _get_nodes_text(
        self, nodes: List[SyntaxTreeNode], text: str, line_indices: List[int]
    ) -> Tuple[str, int, int]:
        """Extracts the raw text content of a list of nodes."""
        if not nodes:
            return "", 0, 0
        start_line, _ = nodes[0].map
        _, end_line = nodes[-1].map
        start_char = line_indices[start_line]
        end_char = line_indices[end_line] if end_line < len(line_indices) else len(text)
        return text[start_char:end_char], start_char, end_char

    def _extract_blocks(
        self, text: str
    ) -> List[Tuple[str, Dict[str, Any], int, int, str]]:
        """
        Pass 1: Parse the document and create a flat list of semantic blocks
        with their content, context, character indices, and block type.
        """
        tokens = self.md_parser.parse(text)
        root_node = SyntaxTreeNode(tokens)
        line_indices = self._get_line_start_indices(text)

        blocks = []
        header_context: Dict[str, Any] = {}

        for node in root_node.children:
            if not node.map:
                continue

            start_line, end_line = node.map
            start_char = line_indices[start_line]
            end_char = (
                line_indices[end_line] if end_line < len(line_indices) else len(text)
            )
            content = text[start_char:end_char]
            block_type = node.type

            if node.type == "heading":
                level = int(node.tag[1:])
                # Clear deeper or same-level headers from context
                keys_to_del = [k for k in header_context if int(k[1:]) >= level]
                for k in keys_to_del:
                    del header_context[k]
                header_content = (
                    node.children[0].content.strip() if node.children else ""
                )
                header_context[f"H{level}"] = header_content

            if content.strip():
                blocks.append(
                    (content, header_context.copy(), start_char, end_char, block_type)
                )

        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the Markdown text using a two-pass, non-recursive method."""
        processed_text = self._preprocess(text)
        if not processed_text.strip():
            return []

        # Pass 1: Extract all semantic blocks with their context and type.
        blocks = self._extract_blocks(processed_text)
        if not blocks:
            return []

        # Pass 2: Group blocks into chunks.
        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, Dict[str, Any], int, int, str]] = []

        for block in blocks:
            # If the current chunk is empty, start it with the current block.
            if not current_chunk_blocks:
                current_chunk_blocks.append(block)
                continue

            # Get data from the running chunk and the new block
            potential_text = "".join(b[0] for b in current_chunk_blocks) + block[0]
            current_context = current_chunk_blocks[-1][1]
            current_block_type = current_chunk_blocks[-1][4]
            new_context = block[1]
            new_block_type = block[4]

            # --- Boundary Condition Checks ---
            # Condition 1: The combined text exceeds the chunk size.
            size_boundary = self.length_function(potential_text) > self.chunk_size

            # Condition 2: The new block starts a new major section (e.g., H2 after H2).
            header_boundary = False
            if new_context != current_context:
                new_header_level = max(
                    (int(k[1:]) for k in new_context if k.startswith("H")), default=99
                )
                current_header_level = max(
                    (int(k[1:]) for k in current_context if k.startswith("H")),
                    default=99,
                )
                if new_header_level <= current_header_level:
                    header_boundary = True

            # Condition 3: The block type changes (e.g., paragraph -> bullet_list).
            # We don't want to split on every paragraph, so we allow paragraph->paragraph.
            type_boundary = current_block_type != new_block_type and not (
                current_block_type == "paragraph" and new_block_type == "paragraph"
            )
            # Edge case: A heading should always be grouped with the content that immediately
            # follows it, so we don't create a type boundary after a heading.
            if current_block_type == "heading":
                type_boundary = False

            # --- Boundary Condition Checks ---
            # A heading should always be merged with the content that follows it.
            is_header_only_chunk = (
                len(current_chunk_blocks) == 1
                and current_chunk_blocks[0][4] == "heading"
            )

            # If any boundary condition is met, finalize the current chunk.
            # EXCEPTION: Never split a header from its content due to size.
            # The oversized chunk will be handled by the fallback mechanism.
            if (
                (size_boundary and not is_header_only_chunk)
                or header_boundary
                or type_boundary
            ):
                chunk_content = "".join(b[0] for b in current_chunk_blocks)
                start_index = current_chunk_blocks[0][2]
                end_index = current_chunk_blocks[-1][3]

                chunks.append(
                    Chunk(
                        content=chunk_content,
                        start_index=start_index,
                        end_index=end_index,
                        sequence_number=len(chunks),
                        source_document_id=source_document_id,
                        hierarchical_context=current_chunk_blocks[0][1],
                        chunking_strategy_used="markdown",
                    )
                )
                # Start a new chunk with the current block
                current_chunk_blocks = [block]
            else:
                # Merge the block into the current chunk
                current_chunk_blocks.append(block)

        # Add the last remaining chunk
        if current_chunk_blocks:
            chunk_content = "".join(b[0] for b in current_chunk_blocks)
            start_index = current_chunk_blocks[0][2]
            end_index = current_chunk_blocks[-1][3]
            chunks.append(
                Chunk(
                    content=chunk_content,
                    start_index=start_index,
                    end_index=end_index,
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    hierarchical_context=current_chunk_blocks[0][1],
                    chunking_strategy_used="markdown",
                )
            )

        # Pass 3: Use fallback for any chunks that are still too large.
        final_chunks: List[Chunk] = []
        for chunk in chunks:
            if self.length_function(chunk.content) > self.chunk_size:
                # This chunk, likely a single large block, is too big.
                # We need to split it using the fallback mechanism.
                # The fallback splitter will return chunks with indices relative to `chunk.content`.
                # We must correct them to be relative to the original `processed_text`.
                fallback_chunks = self._fallback_splitter.split_text(chunk.content)
                for fb_chunk in fallback_chunks:
                    # Correct the indices of the fallback chunk
                    fb_start_index = chunk.start_index + fb_chunk.start_index
                    fb_end_index = chunk.start_index + fb_chunk.end_index

                    final_chunks.append(
                        Chunk(
                            content=fb_chunk.content,
                            start_index=fb_start_index,
                            end_index=fb_end_index,
                            sequence_number=len(final_chunks),
                            source_document_id=source_document_id,
                            hierarchical_context=chunk.hierarchical_context,
                            chunking_strategy_used="markdown-fallback",
                        )
                    )
            else:
                # This chunk is already valid.
                chunk.sequence_number = len(final_chunks)
                final_chunks.append(chunk)

        from ...utils import _populate_overlap_metadata

        _populate_overlap_metadata(final_chunks, processed_text)

        return self._enforce_minimum_chunk_size(final_chunks, processed_text)
