from typing import Any, Callable, List, Literal, Optional

from ..base import TextSplitter
from ..core import Chunk
from ..utils import _populate_overlap_metadata
from .recursive import RecursiveCharacterSplitter

try:
    import string

    import nltk
    import numpy as np
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    from numpy.linalg import norm
except ImportError:
    np = None  # type: ignore
    nltk = None  # type: ignore

BreakpointMethod = Literal["percentile", "std_dev", "absolute"]


class SemanticSplitter(TextSplitter):
    """
    Splits text into semantically coherent chunks based on embedding similarity.

    This advanced strategy works by splitting the text into sentences, embedding each
    sentence, and then identifying breakpoints where the semantic similarity between
    adjacent sentences drops significantly. This results in chunks that are topically
    related.

    The user must provide a function that can convert a list of text strings into
    embedding vectors.
    """

    def __init__(
        self,
        embedding_function: Callable[[List[str]], Any],
        breakpoint_method: BreakpointMethod = "percentile",
        breakpoint_threshold: float = 95.0,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initializes the SemanticSplitter.

        Args:
            embedding_function: A callable that takes a list of strings and returns
                a list or numpy array of their embeddings.
            breakpoint_method: The method for determining breakpoints ('percentile',
                'std_dev', 'absolute').
            breakpoint_threshold: The sensitivity for the breakpoint method.
                - For 'percentile', this is the percentile (0-100) of similarity scores.
                - For 'std_dev', it's the number of standard deviations from the mean.
                - For 'absolute', it's the fixed similarity value.
            *args, **kwargs: Additional arguments for the base `TextSplitter`.
        """
        super().__init__(*args, **kwargs)

        if np is None:
            raise ImportError(
                "numpy is not installed. Please install it via `pip install "
                '"pyDocumentChunker[semantic]"` or `pip install numpy`.'
            )
        self.embedding_function = embedding_function
        self.breakpoint_method = breakpoint_method
        self.breakpoint_threshold = breakpoint_threshold
        if self.breakpoint_method == "percentile":
            if not (0 <= self.breakpoint_threshold <= 100):
                raise ValueError("Percentile threshold must be between 0 and 100.")

        # This splitter manages its own chunking logic, so we pass a default size
        # to the parent, but it won't be directly used for splitting.
        # This splitter manages its own sentence tokenization.
        # We only need the fallback splitter for oversized chunks.
        self._fallback_splitter = RecursiveCharacterSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            normalize_whitespace=self.normalize_whitespace,
            unicode_normalize=self.unicode_normalize,
        )

    def _calculate_similarities(self, embeddings: "np.ndarray") -> "np.ndarray":
        """Calculates cosine similarity between adjacent embedding vectors."""
        normalized_embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)
        return np.sum(normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1)

    def split_text(
        self, text: str, source_document_id: Optional[str] = None
    ) -> List[Chunk]:
        """Splits the text based on semantic breakpoints."""
        text = self._preprocess(text)
        if not text:
            return []

        # 1. Split the text into individual sentences with their start/end indices.
        # This logic is intentionally copied from SentenceSplitter to ensure this
        # strategy gets a true list of sentences, not pre-chunked groups.
        try:
            tokenizer: PunktSentenceTokenizer = nltk.data.load(
                "tokenizers/punkt/english.pickle"
            )
        except LookupError:
            raise RuntimeError(
                "NLTK's 'punkt' tokenizer model is not downloaded. Please run "
                "`python -c \"import nltk; nltk.download('punkt')\"` to download it."
            )
        sentence_spans = list(tokenizer.span_tokenize(text))

        raw_sentences = [
            (text[start:end], start, end)
            for start, end in sentence_spans
            if text[start:end].strip(string.punctuation + string.whitespace)
        ]

        if not raw_sentences:
            return []

        # Create Chunk objects for each sentence for easier processing
        sentences = [
            Chunk(
                content=sent_text, start_index=start, end_index=end, sequence_number=i
            )
            for i, (sent_text, start, end) in enumerate(raw_sentences)
        ]

        if len(sentences) < 2:
            return self._fallback_splitter.split_text(text, source_document_id)

        sentence_texts = [s.content for s in sentences]
        embeddings = np.array(self.embedding_function(sentence_texts))
        similarities = self._calculate_similarities(embeddings)

        # 2. Determine the similarity threshold for a breakpoint.
        if self.breakpoint_method == "percentile":
            threshold = np.percentile(similarities, self.breakpoint_threshold)
        elif self.breakpoint_method == "std_dev":
            threshold = np.mean(similarities) - self.breakpoint_threshold * np.std(
                similarities
            )
        else:  # 'absolute'
            threshold = self.breakpoint_threshold

        # 3. Identify sentence indices that are potential breakpoints.
        breakpoint_indices = [i for i, s in enumerate(similarities) if s < threshold]

        # 4. Group sentences into chunks based on breakpoints.
        final_chunks = []
        start_sent_idx = 0
        sequence_number = 0
        for bp_idx in breakpoint_indices:
            end_sent_idx = bp_idx + 1
            group = sentences[start_sent_idx:end_sent_idx]
            if not group:
                continue

            content = " ".join(s.content for s in group)
            group_start_char_idx = group[0].start_index

            if self.length_function(content) > self.chunk_size:
                sub_chunks = self._fallback_splitter.split_text(
                    content, source_document_id
                )
                for sub_chunk in sub_chunks:
                    # Adjust indices to be relative to the original document
                    sub_chunk.start_index += group_start_char_idx
                    sub_chunk.end_index += group_start_char_idx
                    sub_chunk.sequence_number = sequence_number
                    final_chunks.append(sub_chunk)
                    sequence_number += 1
            else:
                final_chunks.append(
                    Chunk(
                        content=content,
                        start_index=group_start_char_idx,
                        end_index=group[-1].end_index,
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        chunking_strategy_used="semantic",
                    )
                )
                sequence_number += 1
            start_sent_idx = end_sent_idx

        # Handle the last group of sentences.
        last_group = sentences[start_sent_idx:]
        if last_group:
            content = " ".join(s.content for s in last_group)
            group_start_char_idx = last_group[0].start_index
            if self.length_function(content) > self.chunk_size:
                sub_chunks = self._fallback_splitter.split_text(
                    content, source_document_id
                )
                for sub_chunk in sub_chunks:
                    # Adjust indices to be relative to the original document
                    sub_chunk.start_index += group_start_char_idx
                    sub_chunk.end_index += group_start_char_idx
                    sub_chunk.sequence_number = sequence_number
                    final_chunks.append(sub_chunk)
                    sequence_number += 1
            else:
                final_chunks.append(
                    Chunk(
                        content=content,
                        start_index=group_start_char_idx,
                        end_index=last_group[-1].end_index,
                        sequence_number=sequence_number,
                        source_document_id=source_document_id,
                        chunking_strategy_used="semantic",
                    )
                )

        # Post-process to add overlap metadata using the shared utility.
        _populate_overlap_metadata(final_chunks, text)

        return self._enforce_minimum_chunk_size(final_chunks, text)
