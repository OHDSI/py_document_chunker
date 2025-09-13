import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """
    A data class representing a chunk of text with enriched metadata, as specified
    in the Functional Requirements Document (FRD) section 5.2.

    Attributes:
        content (str): The text content of the chunk.
        start_index (int): The character start position (0-indexed) of the chunk
            relative to the source document (R-5.2.3).
        end_index (int): The character end position of the chunk relative to the
            source document (R-5.2.4).
        sequence_number (int): The ordinal position (0-indexed) of the chunk
            within the document sequence (R-5.2.5).

        chunk_id (str): A unique identifier (UUID v4) for the chunk (R-5.2.1).
        source_document_id (Optional[str]): An identifier for the original document
            from which the chunk was derived (R-5.2.2).
        hierarchical_context (Dict[str, Any]): A structured representation of the
            logical hierarchy (e.g., headers) the chunk belongs to. Populated by
            Structure-Aware strategies (R-5.2.6).
        overlap_content_previous (Optional[str]): The exact text segment shared
            between this chunk and the immediately preceding one (R-5.2.7).
        overlap_content_next (Optional[str]): The exact text segment shared
            between this chunk and the immediately subsequent one (R-5.2.8).
        chunking_strategy_used (Optional[str]): The name of the strategy
            configuration that generated this chunk (R-5.2.9).
        metadata (Dict[str, Any]): A flexible dictionary for any additional,
            unstructured metadata.
    """

    content: str
    start_index: int
    end_index: int
    sequence_number: int

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_document_id: Optional[str] = None
    hierarchical_context: Dict[str, Any] = field(default_factory=dict)
    overlap_content_previous: Optional[str] = None
    overlap_content_next: Optional[str] = None
    chunking_strategy_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Performs basic validation after the object is initialized."""
        if self.start_index < 0:
            raise ValueError("start_index must be a non-negative integer.")
        if self.end_index < self.start_index:
            raise ValueError("end_index must not be less than start_index.")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be a non-negative integer.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Chunk object to a dictionary."""
        return {
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "sequence_number": self.sequence_number,
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "hierarchical_context": self.hierarchical_context,
            "overlap_content_previous": self.overlap_content_previous,
            "overlap_content_next": self.overlap_content_next,
            "chunking_strategy_used": self.chunking_strategy_used,
            "metadata": self.metadata,
        }
