import copy
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .core import Chunk


class TextSplitter(ABC):
    """
    Abstract Base Class for all text splitters in the package.

    This class establishes a common interface for all chunking strategies, ensuring
    modularity and extensibility. It also handles core configuration parameters
    common across most strategies.

    Attributes:
        chunk_size (int): The target maximum size of each chunk.
        chunk_overlap (int): The amount of overlap between consecutive chunks.
        length_function (Callable[[str], int]): Function to measure text length.
        normalize_whitespace (bool): If True, normalizes all whitespace.
        unicode_normalize (Optional[str]): The form for Unicode normalization
            (e.g., 'NFC', 'NFKC').
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        normalize_whitespace: bool = False,
        unicode_normalize: Optional[str] = None,
        minimum_chunk_size: Optional[int] = None,
        min_chunk_merge_strategy: str = "merge_with_previous",
        strip_control_chars: bool = False,
    ):
        """
        Initializes the TextSplitter.

        Args:
            chunk_size: The maximum size of a chunk.
            chunk_overlap: The overlap between consecutive chunks.
            length_function: The function to measure text length. Defaults to `len`.
            normalize_whitespace: If True, collapses consecutive whitespace, newlines,
                and tabs into a single space.
            unicode_normalize: The Unicode normalization form to apply. Can be one
                of 'NFC', 'NFKC', 'NFD', 'NFKD'. Defaults to None.
            minimum_chunk_size: Optional integer. If a generated chunk's size is
                below this, it will be handled by the specified strategy.
            min_chunk_merge_strategy: How to handle chunks smaller than
                `minimum_chunk_size`. Can be 'discard', 'merge_with_previous',
                or 'merge_with_next'.
            strip_control_chars: If True, removes all Unicode control characters
                (categories starting with 'C') from the text before processing.

        Raises:
            ValueError: If `chunk_overlap` is not smaller than `chunk_size`.
            ValueError: If `unicode_normalize` is not a valid normalization form.
            ValueError: If `min_chunk_merge_strategy` is not a valid strategy.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) must be smaller than "
                f"chunk size ({chunk_size})."
            )
        if unicode_normalize and unicode_normalize not in [
            "NFC",
            "NFKC",
            "NFD",
            "NFKD",
        ]:
            raise ValueError(
                f"Invalid unicode_normalize form: {unicode_normalize}. "
                "Must be one of 'NFC', 'NFKC', 'NFD', 'NFKD'."
            )
        if minimum_chunk_size and minimum_chunk_size >= chunk_size:
            raise ValueError(
                f"minimum_chunk_size ({minimum_chunk_size}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )
        valid_merge_strategies = ["discard", "merge_with_previous", "merge_with_next"]
        if min_chunk_merge_strategy not in valid_merge_strategies:
            raise ValueError(
                f"Invalid min_chunk_merge_strategy: {min_chunk_merge_strategy}. "
                f"Must be one of {valid_merge_strategies}."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.normalize_whitespace = normalize_whitespace
        self.unicode_normalize = unicode_normalize
        self.strip_control_chars = strip_control_chars
        self.minimum_chunk_size = minimum_chunk_size or 0
        self.min_chunk_merge_strategy = min_chunk_merge_strategy

    def _enforce_minimum_chunk_size(
        self, chunks: List[Chunk], original_text: str
    ) -> List[Chunk]:
        """
        Enforces the minimum chunk size by merging or discarding small chunks.

        This method is called as a post-processing step by concrete splitters.
        """
        if self.minimum_chunk_size <= 0:
            return chunks

        if self.min_chunk_merge_strategy == "discard":
            merged_chunks = [
                c
                for c in chunks
                if self.length_function(c.content) >= self.minimum_chunk_size
            ]
        else:
            # Create a new list of copies to avoid modifying original chunk objects
            merged_chunks = [copy.copy(c) for c in chunks]

            if self.min_chunk_merge_strategy == "merge_with_previous":
                # Iterate forward and merge runts with the previous chunk.
                i = 1
                while i < len(merged_chunks):
                    if (
                        self.length_function(merged_chunks[i].content)
                        < self.minimum_chunk_size
                    ):
                        prev_chunk = merged_chunks[i - 1]
                        current_chunk = merged_chunks[i]
                        if (
                            self.length_function(prev_chunk.content)
                            + self.length_function(current_chunk.content)
                        ) <= self.chunk_size:
                            prev_chunk.content = original_text[
                                prev_chunk.start_index : current_chunk.end_index
                            ]
                            prev_chunk.end_index = current_chunk.end_index
                            merged_chunks.pop(i)
                            # After merging, stay at the same index `i` in case the
                            # next chunk is also a runt that needs to be merged into
                            # the same previous chunk.
                            continue
                    i += 1
                # Fallback for a runt at the beginning of the list.
                if (
                    len(merged_chunks) > 1
                    and self.length_function(merged_chunks[0].content)
                    < self.minimum_chunk_size
                ):
                    if (
                        self.length_function(merged_chunks[0].content)
                        + self.length_function(merged_chunks[1].content)
                    ) <= self.chunk_size:
                        merged_chunks[1].content = original_text[
                            merged_chunks[0].start_index : merged_chunks[1].end_index
                        ]
                        merged_chunks[1].start_index = merged_chunks[0].start_index
                        merged_chunks.pop(0)

            elif self.min_chunk_merge_strategy == "merge_with_next":
                # Iterate backward and merge runts with the next chunk.
                i = len(merged_chunks) - 2
                while i >= 0:
                    if (
                        self.length_function(merged_chunks[i].content)
                        < self.minimum_chunk_size
                    ):
                        current_chunk = merged_chunks[i]
                        next_chunk = merged_chunks[i + 1]
                        if (
                            self.length_function(current_chunk.content)
                            + self.length_function(next_chunk.content)
                        ) <= self.chunk_size:
                            next_chunk.content = original_text[
                                current_chunk.start_index : next_chunk.end_index
                            ]
                            next_chunk.start_index = current_chunk.start_index
                            merged_chunks.pop(i)
                    i -= 1
                # Fallback for a runt at the end of the list.
                if (
                    len(merged_chunks) > 1
                    and self.length_function(merged_chunks[-1].content)
                    < self.minimum_chunk_size
                ):
                    if (
                        self.length_function(merged_chunks[-1].content)
                        + self.length_function(merged_chunks[-2].content)
                    ) <= self.chunk_size:
                        merged_chunks[-2].content = original_text[
                            merged_chunks[-2].start_index : merged_chunks[-1].end_index
                        ]
                        merged_chunks[-2].end_index = merged_chunks[-1].end_index
                        merged_chunks.pop(-1)

        # Re-assign sequence numbers
        for i, chunk in enumerate(merged_chunks):
            chunk.sequence_number = i

        # After merging or discarding, the overlap metadata of the affected chunks
        # may be stale. We must recalculate it to ensure consistency and correctness.
        # This addresses a bug where merging runts would not update the overlap
        # content of neighboring chunks.
        from .utils import _populate_overlap_metadata

        _populate_overlap_metadata(merged_chunks, original_text)

        return merged_chunks

    def _preprocess(self, text: str) -> str:
        """
        Applies configured preprocessing steps to the input text.

        This method is called by concrete splitter implementations before chunking.
        """
        if self.strip_control_chars:
            # Remove all characters in Unicode categories starting with 'C' (Other).
            # This includes Cc (Control), Cf (Format), Cs (Surrogate),
            # Co (Private Use), and Cn (Unassigned).
            # This is a robust way to remove null bytes, BOM, and other non-printing
            # characters that can interfere with processing.
            text = "".join(
                char for char in text if unicodedata.category(char)[0] != "C"
            )

        if self.unicode_normalize:
            text = unicodedata.normalize(self.unicode_normalize, text)
        if self.normalize_whitespace:
            # Collapse consecutive whitespace (spaces, tabs, newlines) into a single space
            text = re.sub(r"\s+", " ", text).strip()
        return text

    @abstractmethod
    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Abstract method to split a document text into a list of `Chunk` objects.

        Every concrete implementation of a chunking strategy must implement this
        method.

        Args:
            text: The input text of the document to be split.
            source_document_id: An optional identifier for the source document,
                which will be attached to each resulting chunk.

        Returns:
            A list of `Chunk` objects, each representing a segment of the input text.
        """
        pass  # pragma: no cover

    def chunk(self, text: str, source_document_id: Optional[str] = None) -> List[Chunk]:
        """
        A convenience method that provides a more intuitive alias for `split_text`.

        Args:
            text: The input text to be chunked.
            source_document_id: An optional identifier for the source document.

        Returns:
            A list of `Chunk` objects.
        """
        return self.split_text(text, source_document_id=source_document_id)
