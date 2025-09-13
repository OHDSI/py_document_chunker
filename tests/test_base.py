import pytest

from py_document_chunker import Chunk, FixedSizeSplitter


def create_chunks_from_content(contents: list[str]) -> tuple[list, str]:
    """Helper to create a list of Chunk objects from a list of strings."""
    chunks = []
    start_index = 0
    full_text_parts = []
    for i, content in enumerate(contents):
        end_index = start_index + len(content)
        chunks.append(
            Chunk(
                content=content,
                start_index=start_index,
                end_index=end_index,
                sequence_number=i,
            )
        )
        full_text_parts.append(content)
        # Simulate a space separator for reconstructing the original text
        start_index = end_index + 1

    original_text = " ".join(full_text_parts)
    return chunks, original_text


class TestBaseFunctionality:
    def setup_method(self):
        self.length_function = len

    def test_whitespace_normalization(self):
        text = "This   is a    test.\n\nIt has   extra whitespace."
        splitter = FixedSizeSplitter(
            chunk_size=20, chunk_overlap=5, normalize_whitespace=True
        )
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "  " not in chunk.content
            assert "\n" not in chunk.content

        expected_text = "This is a test. It has extra whitespace."
        assert chunks[0].content in expected_text

    def test_unicode_normalization_nfkc(self):
        text = "ﬁne dining"
        splitter = FixedSizeSplitter(
            chunk_size=20, chunk_overlap=0, unicode_normalize="NFKC"
        )
        chunks = splitter.split_text(text)
        assert chunks[0].content == "fine dining"

    def test_unicode_normalization_nfc(self):
        text = "cafe\u0301"  # café
        splitter = FixedSizeSplitter(
            chunk_size=10, chunk_overlap=0, unicode_normalize="NFC"
        )
        chunks = splitter.split_text(text)
        assert chunks[0].content == "café"

    def test_minimum_chunk_size_discard(self):
        text = "This is a sentence. Runt. This is another."
        splitter = FixedSizeSplitter(
            chunk_size=20,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="discard",
        )
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0].content == "This is a sentence. "
        assert chunks[1].content == "Runt. This is anothe"

    def test_runt_handling_and_metadata_update(self):
        original_text = "aaaaabbbbbccccc"
        c1 = Chunk(content="aaaaa", start_index=0, end_index=5, sequence_number=0)
        c_runt = Chunk(content="bb", start_index=5, end_index=7, sequence_number=1)
        c2 = Chunk(content="bccccc", start_index=6, end_index=12, sequence_number=2)
        c1.overlap_content_next = None
        c_runt.overlap_content_next = "b"
        c2.overlap_content_previous = "b"

        splitter = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=10,
            minimum_chunk_size=3,
            min_chunk_merge_strategy="merge_with_previous",
        )
        initial_chunks = [c1, c_runt, c2]
        merged_chunks = splitter._enforce_minimum_chunk_size(
            initial_chunks, original_text
        )
        assert len(merged_chunks) == 2
        merged_c1 = merged_chunks[0]
        next_c = merged_chunks[1]
        assert merged_c1.content == "aaaaabb"
        assert merged_c1.start_index == 0
        assert merged_c1.end_index == 7
        assert merged_c1.overlap_content_next == "b"
        assert next_c.overlap_content_previous == "b"

    def test_invalid_unicode_form_raises_error(self):
        with pytest.raises(ValueError, match="Invalid unicode_normalize form"):
            FixedSizeSplitter(unicode_normalize="INVALID", chunk_overlap=0)

    def test_invalid_merge_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Invalid min_chunk_merge_strategy"):
            FixedSizeSplitter(min_chunk_merge_strategy="INVALID", chunk_overlap=0)

    def test_min_chunk_size_too_large_raises_error(self):
        with pytest.raises(
            ValueError, match="minimum_chunk_size .* must be smaller than"
        ):
            FixedSizeSplitter(chunk_size=100, minimum_chunk_size=100, chunk_overlap=0)

    def test_strip_control_chars_option(self):
        text_with_control_chars = "\ufeffHello\x00 World\x08!\x1f"
        expected_clean_text = "Hello World!"
        splitter_true = FixedSizeSplitter(
            chunk_size=100, chunk_overlap=0, strip_control_chars=True
        )
        chunks_true = splitter_true.split_text(text_with_control_chars)
        assert chunks_true[0].content == expected_clean_text
        splitter_false = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            strip_control_chars=False,
        )
        chunks_false = splitter_false.split_text(text_with_control_chars)
        assert chunks_false[0].content == text_with_control_chars

    def test_runt_handling_at_boundaries(self):
        c_normal_1 = Chunk(
            content="This is a normal chunk.",
            start_index=0,
            end_index=23,
            sequence_number=0,
        )
        c_runt_1 = Chunk(
            content=" Runt.", start_index=23, end_index=29, sequence_number=1
        )
        text_1 = c_normal_1.content + c_runt_1.content
        splitter_next = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="merge_with_next",
        )
        chunks_next = splitter_next._enforce_minimum_chunk_size(
            [c_normal_1, c_runt_1], text_1
        )
        assert len(chunks_next) == 1
        assert chunks_next[0].content == "This is a normal chunk. Runt."
        assert chunks_next[0].start_index == 0
        assert chunks_next[0].end_index == 29
        c_runt_2 = Chunk(
            content="Runt. ", start_index=0, end_index=6, sequence_number=0
        )
        c_normal_2 = Chunk(
            content="This is a normal chunk.",
            start_index=6,
            end_index=29,
            sequence_number=1,
        )
        text_2 = c_runt_2.content + c_normal_2.content
        splitter_prev = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="merge_with_previous",
        )
        chunks_prev = splitter_prev._enforce_minimum_chunk_size(
            [c_runt_2, c_normal_2], text_2
        )
        assert len(chunks_prev) == 1
        assert chunks_prev[0].content == "Runt. This is a normal chunk."
        assert chunks_prev[0].start_index == 0
        assert chunks_prev[0].end_index == 29

    def test_overlap_metadata_is_populated(self):
        text = "This is a sentence. This is another sentence. And a third."
        splitter = FixedSizeSplitter(
            chunk_size=20,
            chunk_overlap=8,
        )
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert chunks[0].overlap_content_previous is None
        assert chunks[-1].overlap_content_next is None
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            assert current_chunk.overlap_content_next is not None
            assert next_chunk.overlap_content_previous is not None
            assert (
                current_chunk.overlap_content_next
                == next_chunk.overlap_content_previous
            )
            overlap_len = len(current_chunk.overlap_content_next)
            assert overlap_len > 0
            assert overlap_len <= splitter.chunk_overlap

    def test_invalid_chunk_overlap_raises_error(self):
        with pytest.raises(ValueError, match="Chunk overlap .* must be smaller than"):
            FixedSizeSplitter(chunk_size=10, chunk_overlap=10)

    def test_preprocess_without_strip_control_chars(self):
        text = "Hello\x00 World"
        splitter = FixedSizeSplitter(strip_control_chars=False)
        processed_text = splitter._preprocess(text)
        assert processed_text == text

    def test_chunk_method(self):
        text = "This is a test."
        splitter = FixedSizeSplitter(chunk_size=10, chunk_overlap=5)
        chunks = splitter.chunk(text)
        assert len(chunks) == 2
        assert chunks[0].content == "This is a "
        assert chunks[1].content == "is a test."

    def test_preprocess_strips_control_chars(self):
        text = "Hello\x00 World"
        splitter = FixedSizeSplitter(strip_control_chars=True)
        processed_text = splitter._preprocess(text)
        assert processed_text == "Hello World"

    def test_preprocess_preserves_control_chars(self):
        text = "Hello\x00 World"
        splitter = FixedSizeSplitter(strip_control_chars=False)
        processed_text = splitter._preprocess(text)
        assert processed_text == text

    def test_runt_handling_with_token_length_function(self):
        def token_length(text: str) -> int:
            return len(text.split())

        splitter = FixedSizeSplitter(
            chunk_size=10,
            chunk_overlap=0,
            minimum_chunk_size=3,
            min_chunk_merge_strategy="merge_with_previous",
            length_function=token_length,
        )
        chunks, original_text = create_chunks_from_content(
            [
                "This is a long piece of text.",
                "runt",
                "another runt",
                "This is the final major chunk.",
            ]
        )
        result = splitter._enforce_minimum_chunk_size(chunks, original_text)
        assert len(result) == 2
        assert result[0].content == "This is a long piece of text. runt another runt"
        assert token_length(result[0].content) == 10
        assert result[1].content == "This is the final major chunk."

    def test_all_chunks_are_runts(self):
        splitter = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="merge_with_previous",
        )
        chunks, text = create_chunks_from_content(["runt1", "runt2", "runt3", "runt4"])
        result = splitter._enforce_minimum_chunk_size(chunks, text)
        assert len(result) == 1
        assert result[0].content == "runt1 runt2 runt3 runt4"

    def test_no_runts_in_chunks(self):
        splitter = FixedSizeSplitter(
            chunk_size=100,
            chunk_overlap=0,
            minimum_chunk_size=10,
            min_chunk_merge_strategy="merge_with_previous",
        )
        chunks, text = create_chunks_from_content(
            ["This is a perfectly sized chunk.", "This one is also fine."]
        )
        original_chunks = chunks[:]
        result = splitter._enforce_minimum_chunk_size(chunks, text)
        assert len(result) == len(original_chunks)
        for i, chunk in enumerate(result):
            assert chunk.content == original_chunks[i].content
            assert chunk.start_index == original_chunks[i].start_index
            assert chunk.end_index == original_chunks[i].end_index
