from typing import Any, List, Optional, Tuple

from ..base import TextSplitter
from ..core import Chunk
from ..utils import _populate_overlap_metadata
from .recursive import RecursiveCharacterSplitter

try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
except ImportError:
    nltk = None  # type: ignore


class SentenceSplitter(TextSplitter):
    def __init__(
        self,
        overlap_sentences: int = 1,
        nlp_backend: str = "nltk",
        *args: Any,
        **kwargs: Any,
    ):
        if "chunk_overlap" not in kwargs:
            chunk_size = kwargs.get("chunk_size", 1024)
            kwargs["chunk_overlap"] = int(chunk_size * 0.1)

        super().__init__(*args, **kwargs)

        if nlp_backend != "nltk":
            raise ValueError("Currently, only the 'nltk' backend is supported.")
        if nltk is None:
            raise ImportError(
                'NLTK is not installed. Please install it via `pip install "pyDocumentChunker[nlp]"` '
                "or `pip install nltk` to use the SentenceSplitter."
            )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            raise RuntimeError(
                "NLTK's 'punkt' tokenizer model is not downloaded. Please run "
                "`python -c \"import nltk; nltk.download('punkt')\"` to download it."
            )

        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")
        self.overlap_sentences = overlap_sentences

        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,  # No overlap in fallback since sentences are merged later
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        text = self._preprocess(text)
        if not text:
            return []

        import string

        tokenizer: PunktSentenceTokenizer = nltk.data.load(
            "tokenizers/punkt/english.pickle"
        )
        sentence_spans = list(tokenizer.span_tokenize(text))

        all_sentences = [
            (text[start:end], start, end)
            for start, end in sentence_spans
            if text[start:end].strip(string.punctuation + string.whitespace)
        ]

        if not all_sentences:
            return []

        processed_sentences = []
        for sentence_text, start, end in all_sentences:
            if self.length_function(sentence_text) > self.chunk_size:
                # This text is already preprocessed, so we don't want the fallback
                # to do it again. We can create a temporary splitter or just
                # call the internal split method.
                # The fallback splitter was initialized with the same preprocessing
                # settings, which is not ideal as it will double-process.
                # A cleaner way is to ensure the fallback doesn't re-process.
                # For now, let's assume the fallback's preprocessing is idempotent.
                fallback_chunks = self._fallback_splitter.split_text(sentence_text)
                for fb_chunk in fallback_chunks:
                    processed_sentences.append(
                        (
                            fb_chunk.content,
                            start + fb_chunk.start_index,
                            start + fb_chunk.end_index,
                        )
                    )
            else:
                processed_sentences.append((sentence_text, start, end))

        chunks: List[Chunk] = []
        current_chunk_sents: List[Tuple[str, int, int]] = []

        for sent, start, end in processed_sentences:
            # To check the length, we should measure the length of the text slice,
            # not a string joined with spaces, to be accurate.
            potential_start_index = current_chunk_sents[0][1] if current_chunk_sents else start
            potential_end_index = end
            potential_content_slice = text[potential_start_index:potential_end_index]

            if (
                self.length_function(potential_content_slice) > self.chunk_size
                and current_chunk_sents
            ):
                # Finalize the current chunk
                start_index = current_chunk_sents[0][1]
                end_index = current_chunk_sents[-1][2]
                final_content = text[start_index:end_index]
                chunks.append(
                    Chunk(
                        content=final_content,
                        start_index=start_index,
                        end_index=end_index,
                        sequence_number=len(chunks),
                        source_document_id=source_document_id,
                        chunking_strategy_used="sentence",
                    )
                )

                # Start a new chunk with overlap
                overlap_idx = max(0, len(current_chunk_sents) - self.overlap_sentences)
                current_chunk_sents = current_chunk_sents[overlap_idx:]

            current_chunk_sents.append((sent, start, end))

        # Add the last remaining chunk
        if current_chunk_sents:
            start_index = current_chunk_sents[0][1]
            end_index = current_chunk_sents[-1][2]
            final_content = text[start_index:end_index]
            chunks.append(
                Chunk(
                    content=final_content,
                    start_index=start_index,
                    end_index=end_index,
                    sequence_number=len(chunks),
                    source_document_id=source_document_id,
                    chunking_strategy_used="sentence",
                )
            )

        # Post-process to add overlap metadata using the shared utility.
        _populate_overlap_metadata(chunks, text)

        return self._enforce_minimum_chunk_size(chunks, text)
