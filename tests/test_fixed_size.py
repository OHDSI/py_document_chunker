from py_document_chunker import FixedSizeSplitter


class TestFixedSizeSplitter:
    def test_fixed_size_populates_overlap_metadata(self):
        """
        Tests that the new splitter correctly populates overlap metadata,
        thanks to its parent class.
        """
        splitter = FixedSizeSplitter(chunk_size=15, chunk_overlap=5)
        text = "This is a test of the overlap metadata population."
        chunks = splitter.split_text(text)

        assert len(chunks) > 1

        # Check overlap between chunk 0 and 1
        assert chunks[0].overlap_content_next is not None
        assert chunks[1].overlap_content_previous is not None
        assert chunks[0].overlap_content_next == chunks[1].overlap_content_previous

        # Check that the overlap content makes sense
        assert chunks[0].content.endswith(chunks[0].overlap_content_next)
        assert chunks[1].content.startswith(chunks[1].overlap_content_previous)

    def test_fixed_size_splitter_with_tokenizer(self):
        """
        This is the key test. It validates that the overhauled FixedSizeSplitter
        correctly uses a token-based length function.
        """

        # A mock tokenizer where each word is a token of length 1.
        def mock_tokenizer(x: str) -> int:
            return len(x.split())

        splitter = FixedSizeSplitter(
            chunk_size=3,
            chunk_overlap=1,  # 1 word overlap
            length_function=mock_tokenizer,
        )
        text = "one two three four five six seven"  # 7 words
        chunks = splitter.split_text(text)

        # Expected chunks (size=3 words, overlap=1 word):
        # With the corrected logic, the splitter is greedy and overlap includes spaces.
        # 1. "one two three " (length of "one two three" is 3)
        # 2. " three four five " (overlap is " three ", next is "four five ")
        # 3. " five six seven" (overlap is " five ", next is "six seven")
        assert len(chunks) == 3
        assert chunks[0].content == "one two three "
        assert chunks[1].content == " three four five "
        assert chunks[2].content == " five six seven"

        # Verify metadata
        assert chunks[0].overlap_content_next == " three "
        assert chunks[1].overlap_content_previous == " three "
        assert chunks[1].overlap_content_next == " five "
        assert chunks[2].overlap_content_previous == " five "

    def test_fixed_size_splitter_ignores_separators(self):
        """
        Tests that the FixedSizeSplitter ignores the `separators` and `keep_separator` arguments.
        """
        # This should not raise an error and the splitter should still work as expected.
        splitter = FixedSizeSplitter(
            chunk_size=10, chunk_overlap=5, separators=["\n\n"], keep_separator=False
        )
        text = "This is a test."
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0].content == "This is a "
        assert chunks[1].content == "is a test."
