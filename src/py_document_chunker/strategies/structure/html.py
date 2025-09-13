import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from ...base import TextSplitter
from ...core import Chunk
from ..recursive import RecursiveCharacterSplitter

# List of tags that are typically block-level and contain content.
DEFAULT_BLOCK_TAGS = [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "th",
    "td",
    "caption",
]
# Tags to be removed completely from the document before processing.
DEFAULT_STRIP_TAGS = ["script", "style", "head", "nav", "footer", "aside"]


class HTMLSplitter(TextSplitter):
    """
    Splits an HTML document based on its structural tags.

    This strategy offers two main modes as per FR-2.2.3:
    1.  **Structure-Aware (default):** Parses HTML to identify structural elements
        (like paragraphs, headers, list items) and uses them as the primary basis
        for splitting. This is effective for preserving logical sections.
    2.  **Text-Only:** If `strip_all_tags` is True, it removes all HTML tags and
        splits the resulting plain text using a recursive character splitter.

    The splitter requires `BeautifulSoup4` and `lxml` to be installed.
    """

    def __init__(
        self,
        block_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        strip_all_tags: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the HTMLSplitter.

        Args:
            block_tags: A list of block-level HTML tags to serve as split points
                in structure-aware mode (FR-3.4.2). Defaults to a standard list.
            remove_tags: A list of tags (e.g., 'script', 'style') to completely
                remove before any processing.
            strip_all_tags: If True, strips all HTML tags to process raw text
                instead of using structure-aware splitting (FR-2.2.3).
            *args: Positional arguments for the base TextSplitter.
            **kwargs: Keyword arguments for the base TextSplitter.
        """
        super().__init__(*args, **kwargs)

        self.block_tags = block_tags or DEFAULT_BLOCK_TAGS
        self.remove_tags = remove_tags or DEFAULT_STRIP_TAGS
        self.strip_all_tags = strip_all_tags
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            # Whitespace is normalized by BeautifulSoup in get_text, so we avoid
            # double-processing here unless we are in text-only mode.
            normalize_whitespace=self.normalize_whitespace and self.strip_all_tags,
            unicode_normalize=self.unicode_normalize,
        )

    def _extract_blocks(
        self, soup: BeautifulSoup, text: str, line_start_indices: List[int]
    ) -> List[Tuple[str, int, int, Dict[str, Any]]]:
        """Extracts text blocks from the parsed soup, with metadata."""
        blocks = []
        for tag in soup.find_all(self.block_tags):
            # Get the raw text to calculate original end index
            raw_text = tag.get_text(separator=" ", strip=False)
            if not raw_text.strip():
                continue

            # Preprocess the text for the chunk content
            processed_content = self._preprocess(
                tag.get_text(separator=" ", strip=True)
            )
            if not processed_content:
                continue

            start_index = text.find(str(tag))
            if start_index == -1:
                start_index = 0

            end_index = start_index + len(raw_text)

            header_context: Dict[str, Any] = {}
            current = tag
            while current:
                # Find preceding headers for context
                for sibling in current.find_previous_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        level = sibling.name.upper()
                        if level not in header_context:
                            header_context[level] = self._preprocess(
                                sibling.get_text(separator=" ", strip=True)
                            )
                current = current.parent

            sorted_context = dict(
                sorted(header_context.items(), key=lambda item: item[0])
            )
            blocks.append((processed_content, start_index, end_index, sorted_context))

        return blocks

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the HTML text into structure-aware or plain-text chunks."""
        if not text.strip():
            return []

        # --- Parsing Setup ---
        try:
            soup = BeautifulSoup(text, "html5lib")
        except Exception as e:
            warnings.warn(
                f"Failed to parse HTML with html5lib, falling back to lxml. Error: {e}"
            )
            soup = BeautifulSoup(text, "lxml")

        # --- Tag Removal ---
        # Remove specified non-content tags (e.g., <script>, <style>)
        for tag_name in self.remove_tags:
            for tag_obj in soup.select(tag_name):
                tag_obj.decompose()

        # --- Branching Logic: Strip All Tags vs. Structure-Aware ---
        if self.strip_all_tags:
            # FR-2.2.3: Option to strip all tags and process raw text
            plain_text = soup.get_text(separator=" ", strip=True)
            # Preprocess the entire plain text at once
            processed_text = self._preprocess(plain_text)
            return self._fallback_splitter.split_text(
                processed_text, source_document_id=source_document_id
            )

        # --- Structure-Aware Splitting (Default) ---
        # Pre-calculate start index of each line for approximate position mapping
        line_start_indices = [0] + [m.end() for m in re.finditer("\n", text)]
        blocks = self._extract_blocks(soup, text, line_start_indices)

        chunks: List[Chunk] = []
        current_chunk_blocks: List[Tuple[str, int, int, Dict[str, Any]]] = []
        sequence_number = 0

        def flush_current_chunk():
            nonlocal sequence_number, current_chunk_blocks
            if not current_chunk_blocks:
                return

            content = " ".join(b[0] for b in current_chunk_blocks)
            start_idx = current_chunk_blocks[0][1]
            end_idx = current_chunk_blocks[-1][2]
            merged_context = {}
            for b in reversed(current_chunk_blocks):
                merged_context.update(b[3])

            chunks.append(
                Chunk(
                    content=content,
                    start_index=start_idx,
                    end_index=end_idx,
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    hierarchical_context=merged_context,
                    chunking_strategy_used="html",
                )
            )
            sequence_number += 1
            current_chunk_blocks = []

        for block_text, block_start, block_end, block_context in blocks:
            if self.length_function(block_text) > self.chunk_size:
                flush_current_chunk()
                # Split the oversized block using the fallback
                fallback_chunks = self._fallback_splitter.split_text(block_text)
                for fb_chunk in fallback_chunks:
                    chunks.append(
                        Chunk(
                            content=fb_chunk.content,
                            start_index=block_start,
                            end_index=block_end,
                            sequence_number=sequence_number,
                            source_document_id=source_document_id,
                            hierarchical_context=block_context,
                            chunking_strategy_used="html-fallback",
                        )
                    )
                    sequence_number += 1
                continue

            # Check if adding the next block would exceed the chunk size.
            potential_content = " ".join(
                b[0] for b in current_chunk_blocks + [(block_text, 0, 0, {})]
            )
            if (
                self.length_function(potential_content) > self.chunk_size
                and current_chunk_blocks
            ):
                flush_current_chunk()

            current_chunk_blocks.append(
                (block_text, block_start, block_end, block_context)
            )

        flush_current_chunk()

        # Post-process to add overlap metadata, fulfilling FRD requirements R-5.2.7
        # and R-5.2.8. Even though this strategy doesn't create overlapping chunks
        # by design, the fallback splitter does, and runt merging can also
        # introduce overlaps. Therefore, this step is essential for consistency.
        from ...utils import _populate_overlap_metadata

        _populate_overlap_metadata(chunks, text)

        # Enforce the minimum chunk size. This call now includes the original text
        # so that it can correctly recalculate overlap metadata after merging runts.
        final_chunks = self._enforce_minimum_chunk_size(chunks, text)

        # The sequence numbers are re-assigned within _enforce_minimum_chunk_size,
        # so we can just return the result.
        return final_chunks
