from typing import List

from .core import Chunk


def _populate_overlap_metadata(chunks: List[Chunk], original_text: str) -> None:
    """
    Populates the overlap metadata fields for a list of chunks in place.

    This function calculates the exact text segment shared between consecutive chunks
    and populates the `overlap_content_next` and `overlap_content_previous`
    attributes of the `Chunk` objects.

    This utility is essential for meeting FRD requirements R-5.2.7 and R-5.2.8.

    Args:
        chunks: A list of `Chunk` objects, assumed to be in sequential order.
        original_text: The original source text from which the chunks were created.
    """
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # Check if there is an overlap based on character indices
        if next_chunk.start_index < current_chunk.end_index:
            # The overlapping text is the slice from the start of the next chunk
            # to the end of the current chunk.
            overlap_content = original_text[
                next_chunk.start_index : current_chunk.end_index
            ]

            if overlap_content:
                current_chunk.overlap_content_next = overlap_content
                next_chunk.overlap_content_previous = overlap_content
