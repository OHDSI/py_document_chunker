from typing import Any, List, Optional, Tuple

from ..base import TextSplitter
from ..core import Chunk
from ..utils import _populate_overlap_metadata
from .recursive import RecursiveCharacterSplitter

try:
    import spacy

    # Load the English model once
    NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    spacy = None
    NLP = None


class SpacySentenceSplitter(TextSplitter):
    def __init__(
        self,
        overlap_sentences: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        if "chunk_overlap" not in kwargs:
            chunk_size = kwargs.get("chunk_size", 1024)
            kwargs["chunk_overlap"] = int(chunk_size * 0.1)

        super().__init__(*args, **kwargs)

        if spacy is None or NLP is None:
            raise ImportError(
                "Spacy is not installed or the model 'en_core_web_sm' could not be loaded. "
                'Please install it via `pip install "pyDocumentChunker[spacy]"` and download the model '
                "via `python -m spacy download en_core_web_sm`."
            )

        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")
        self.overlap_sentences = overlap_sentences

        # The fallback splitter should have a small overlap to ensure it can
        # handle sentences that are just over the chunk size.
        fallback_overlap = kwargs.get("chunk_overlap", 0)
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=fallback_overlap,
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        text = self._preprocess(text)
        if not text.strip():
            return []

        doc = NLP(text)
        all_sentences = [
            (sent.text, sent.start_char, sent.end_char) for sent in doc.sents
        ]

        if not all_sentences:
            return []

        chunks: List[Chunk] = []
        current_chunk_sents: List[Tuple[str, int, int]] = []

        for sentence_text, start, end in all_sentences:
            # If the sentence itself is larger than the chunk size, we have to handle it separately.
            if self.length_function(sentence_text) > self.chunk_size:
                # First, finalize any existing sentences in the current chunk.
                if current_chunk_sents:
                    final_content = " ".join(s[0] for s in current_chunk_sents)
                    chunks.append(
                        Chunk(
                            content=final_content,
                            start_index=current_chunk_sents[0][1],
                            end_index=current_chunk_sents[-1][2],
                            sequence_number=len(chunks),
                            source_document_id=source_document_id,
                            chunking_strategy_used="spacy_sentence",
                        )
                    )
                    # Reset for the next iteration.
                    current_chunk_sents = []

                # Now, split the oversized sentence using the fallback splitter.
                fallback_chunks = self._fallback_splitter.split_text(sentence_text)
                for fb_chunk in fallback_chunks:
                    # Adjust indices to be relative to the original document
                    fb_chunk.start_index += start
                    fb_chunk.end_index += start
                    fb_chunk.sequence_number = len(chunks)
                    fb_chunk.chunking_strategy_used = "spacy_sentence_fallback"
                    chunks.append(fb_chunk)
                continue

            # Check if adding the new sentence would exceed the chunk size.
            potential_content = " ".join(
                s[0] for s in current_chunk_sents + [(sentence_text, start, end)]
            )
            if (
                self.length_function(potential_content) > self.chunk_size
                and current_chunk_sents
            ):
                # Finalize the current chunk.
                final_content = " ".join(s[0] for s in current_chunk_sents)
                chunks.append(
                    Chunk(
                        content=final_content,
                        start_index=current_chunk_sents[0][1],
                        end_index=current_chunk_sents[-1][2],
                        sequence_number=len(chunks),
                        source_document_id=source_document_id,
                        chunking_strategy_used="spacy_sentence",
                    )
                )

                # Handle overlap by keeping some of the previous sentences.
                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:]

            # Add the new sentence to the current chunk.
            current_chunk_sents.append((sentence_text, start, end))

        if current_chunk_sents:
            final_content = " ".join(s[0] for s in current_chunk_sents)
            chunks.append(
                Chunk(
                    content=final_content,
                    start_index=current_chunk_sents[0][1],
                    end_index=current_chunk_sents[-1][2],
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    chunking_strategy_used="spacy_sentence",
                )
            )

        _populate_overlap_metadata(chunks, text)

        return self._enforce_minimum_chunk_size(chunks, text)
