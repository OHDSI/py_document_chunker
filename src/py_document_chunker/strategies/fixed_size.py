from typing import Any, List, Optional

from ..core import Chunk
from .recursive import RecursiveCharacterSplitter


class FixedSizeSplitter(RecursiveCharacterSplitter):
    """
    Splits text into chunks of a fixed size, respecting the length_function.

    This splitter is a specialized version of the RecursiveCharacterSplitter.
    It functions by splitting the text into individual characters ("") and then
    merging them back together into chunks that adhere to the specified
    `chunk_size`, as measured by the `length_function`. This approach ensures
    that even when using token-based length functions, the resulting chunks
    do not exceed the size limit.

    This strategy guarantees adherence to R-3.1.1 and R-3.1.2 from the FRD.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initializes the FixedSizeSplitter.

        This constructor overrides the `separators` argument and forces it to `[""]`
        to ensure splitting happens at the character level, then passes all other
        arguments to the parent `RecursiveCharacterSplitter`.

        Args:
            *args: Positional arguments passed to the parent splitter.
            **kwargs: Keyword arguments passed to the parent splitter. `separators`
                      will be ignored if provided.
        """
        # FRD-Gap-Analysis: The original implementation ignored the length_function.
        # By inheriting from RecursiveCharacterSplitter and forcing the separator
        # to be the empty string, we get a robust implementation that correctly
        # uses the length_function and populates all metadata fields (like overlap)
        # for free, thus fixing R-3.1.1 and R-5.2.7/R-5.2.8.

        # Force the separator to be by character, which is the essence of a "fixed size" split
        # that doesn't respect semantic boundaries.
        if "separators" in kwargs:
            del kwargs["separators"]
        if "keep_separator" in kwargs:
            del kwargs["keep_separator"]

        super().__init__(separators=[""], keep_separator=True, *args, **kwargs)

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Splits the text and adjusts the chunk metadata to reflect the strategy.
        """
        # Use the parent's robust splitting logic
        chunks = super().split_text(text, source_document_id=source_document_id)

        # Set the correct strategy name for metadata purposes
        for chunk in chunks:
            chunk.chunking_strategy_used = "fixed_size"

        return chunks
