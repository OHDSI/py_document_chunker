import re
from typing import Any, List, Optional, Tuple

from ..base import TextSplitter
from ..core import Chunk
from ..utils import _populate_overlap_metadata


class RecursiveCharacterSplitter(TextSplitter):
    """
    Splits text recursively based on a prioritized list of character separators.

    This strategy is one of the most common and versatile for text chunking. It
    works by attempting to split the text hierarchically using a list of
    separators. If a split results in a segment that is still too large, the
    strategy recursively attempts to split that segment using the next separator
    in the list.

    This implementation follows a two-stage process to ensure that the `start_index`
    metadata is accurately preserved:
    1.  **Splitting:** The text is recursively broken down into small pieces using the
        provided separators. The output of this stage is a list of text fragments,
        each with its start index relative to the original document.
    2.  **Merging:** The small, indexed fragments are then merged back together into
        chunks that respect the `chunk_size` and `chunk_overlap` parameters.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the RecursiveCharacterSplitter.

        Args:
            separators: A prioritized list of strings or regex patterns to split on.
                Defaults to `["\\n\\n", "\\n", ". ", " ", ""]`.
            keep_separator: If True, the separator is kept as part of the preceding
                chunk. This is generally recommended to preserve context.
            *args, **kwargs: Additional arguments passed to the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._keep_separator = keep_separator

    def _recursive_split(
        self, text: str, separators: List[str], start_index: int
    ) -> List[Tuple[str, int]]:
        """Recursively splits text and returns fragments with their start indices."""
        final_fragments = []
        if not text:
            return []

        # If the text is small enough, we're done with this path.
        if self.length_function(text) <= self.chunk_size:
            return [(text, start_index)]

        # If we've run out of separators, do a hard split.
        if not separators:
            step = self.chunk_size - self.chunk_overlap
            for i in range(0, len(text), step):
                chunk_text = text[i : i + self.chunk_size]
                final_fragments.append((chunk_text, start_index + i))
            return final_fragments

        current_separator = separators[0]
        remaining_separators = separators[1:]

        # Perform the split using the current separator.
        try:
            sub_splits = re.split(f"({current_separator})", text)
        except re.error:
            # If the regex is invalid, try the next separator in the list.
            return self._recursive_split(text, remaining_separators, start_index)

        # If the separator did not split the text, try the next one.
        if len(sub_splits) <= 1:
            return self._recursive_split(text, remaining_separators, start_index)

        # The core logic: iterate through the splits (part1, sep1, part2, sep2, ...)
        # and recurse on the parts, while correctly tracking the character offset.
        current_offset = 0
        for i in range(0, len(sub_splits), 2):
            part = sub_splits[i]
            separator = sub_splits[i + 1] if i + 1 < len(sub_splits) else ""

            # Determine the text to recurse upon based on `keep_separator`.
            # The start index for the recursion is always the start of the `part`.
            if self._keep_separator:
                text_to_recurse_on = part + separator
            else:
                text_to_recurse_on = part

            if text_to_recurse_on:
                part_start_index = start_index + current_offset
                final_fragments.extend(
                    self._recursive_split(
                        text_to_recurse_on, remaining_separators, part_start_index
                    )
                )

            # Advance the offset by the length of the original part and separator
            # to correctly calculate the start index for the next iteration.
            current_offset += len(part) + len(separator)

        return final_fragments

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits the input text using the recursive and merging strategy.

        Args:
            text: The text to be split.
            source_document_id: Optional identifier for the source document.

        Returns:
            A list of `Chunk` objects.
        """
        text = self._preprocess(text)
        if not text:
            return []

        # Stage 1: Recursively split text into indexed fragments
        fragments = self._recursive_split(text, self._separators, 0)

        # Stage 2: Merge fragments into chunks, with correct length calculation.
        chunks: List[Chunk] = []
        current_chunk_fragments: List[Tuple[str, int]] = []
        sequence_number = 0

        for fragment_text, fragment_start_index in fragments:
            # Check if adding the next fragment would exceed the chunk size.
            # This is done by joining the content and measuring, which is correct
            # To check if the next fragment fits, we must calculate the potential
            # chunk content by slicing the original text, just as we do when finalizing it.
            if not current_chunk_fragments:
                potential_content = fragment_text
            else:
                start_idx = current_chunk_fragments[0][1]
                end_idx = fragment_start_index + len(fragment_text)
                potential_content = text[start_idx:end_idx]

            if (
                self.length_function(potential_content) > self.chunk_size
                and current_chunk_fragments
            ):
                # Finalize the current chunk.
                # The content is sliced from the original text to preserve separators.
                start_idx = current_chunk_fragments[0][1]
                last_fragment_text, last_fragment_start_index = current_chunk_fragments[
                    -1
                ]
                end_idx = last_fragment_start_index + len(last_fragment_text)
                content = text[start_idx:end_idx]

                chunk = Chunk(
                    content=content,
                    start_index=start_idx,
                    end_index=end_idx,
                    sequence_number=sequence_number,
                    source_document_id=source_document_id,
                    chunking_strategy_used="recursive_character",
                )
                chunks.append(chunk)
                sequence_number += 1

                # Start a new chunk, handling overlap.
                # We slide a window backwards from the end of the last chunk's fragments.
                overlap_fragments_start_idx = len(current_chunk_fragments) - 1
                overlap_fragments: List[Tuple[str, int]] = []

                while overlap_fragments_start_idx >= 0:
                    current_fragment = current_chunk_fragments[
                        overlap_fragments_start_idx
                    ]
                    overlap_fragments.insert(0, current_fragment)

                    # To measure overlap, we must use the original text, not the joined fragments.
                    overlap_start_idx = overlap_fragments[0][1]
                    last_ov_frag_text, last_ov_frag_start = overlap_fragments[-1]
                    overlap_end_idx = last_ov_frag_start + len(last_ov_frag_text)
                    overlap_content = text[overlap_start_idx:overlap_end_idx]

                    if self.length_function(overlap_content) > self.chunk_overlap:
                        overlap_fragments.pop(0)
                        break
                    overlap_fragments_start_idx -= 1

                current_chunk_fragments = overlap_fragments

            # Add the new fragment to the current chunk
            current_chunk_fragments.append((fragment_text, fragment_start_index))

        # Add the last remaining chunk
        if current_chunk_fragments:
            start_idx = current_chunk_fragments[0][1]
            last_fragment_text, last_fragment_start_index = current_chunk_fragments[-1]
            end_idx = last_fragment_start_index + len(last_fragment_text)
            content = text[start_idx:end_idx]

            chunk = Chunk(
                content=content,
                start_index=start_idx,
                end_index=end_idx,
                sequence_number=sequence_number,
                source_document_id=source_document_id,
                chunking_strategy_used="recursive_character",
            )
            chunks.append(chunk)

        # Post-process to add overlap metadata using the shared utility.
        _populate_overlap_metadata(chunks, text)

        return self._enforce_minimum_chunk_size(chunks, text)
