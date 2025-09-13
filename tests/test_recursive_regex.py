from py_document_chunker import RecursiveCharacterSplitter


def test_recursive_splitter_with_variable_width_regex_separator():
    """
    Tests that the RecursiveCharacterSplitter can correctly use a variable-width
    regex pattern as a separator and that the merging logic is correct.
    """
    text = "This is a test document.\n\nIt has multiple sections...    and variable whitespace."
    separators = [r"\s+"]  # Regex for one or more whitespace chars

    # Use a chunk_size that forces merging of the initial words.
    splitter = RecursiveCharacterSplitter(
        chunk_size=15,
        chunk_overlap=2,
        separators=separators,
        keep_separator=False,  # This makes the fragments just the words.
    )
    chunks = splitter.split_text(text)

    chunk_contents = [c.content for c in chunks]

    # EXPECTED BEHAVIOR:
    # Fragments will be: ['This', 'is', 'a', 'test', 'document.', ...]
    # Merging with chunk_size=15:
    # 1. 'This' (len 4) -> ok
    # 2. 'This is' (len 7) -> ok
    # 3. 'This is a' (len 9) -> ok
    # 4. 'This is a test' (len 14) -> ok
    # 5. 'This is a test document.' (len 24) -> too long. Finalize chunk 1.
    # Chunk 1: "This is a test" (from original text slice)
    # Overlap starts. New chunk starts with "test"
    # 6. 'test document.' (len 14) -> ok
    # 7. 'test document. It' (len 17) -> too long. Finalize chunk 2.
    # Chunk 2: "test document."
    # ... and so on.

    expected_first_chunk = "This is a test"
    assert (
        len(chunks) > 1
    ), "The splitter should have split the text into multiple chunks."
    assert (
        chunk_contents[0] == expected_first_chunk
    ), f"The first chunk should be '{expected_first_chunk}', but was '{chunk_contents[0]}'"

    # Verify that no chunk exceeds the chunk_size. This is a critical check.
    for chunk in chunks:
        assert (
            splitter.length_function(chunk.content) <= splitter.chunk_size
        ), f"A chunk exceeded the max size of {splitter.chunk_size}"

    # Verify that the chunk indices are correct
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == 14  # "This is a test"


def test_recursive_splitter_with_keep_separator_true():
    """
    Tests that the splitter correctly keeps the separators when merging,
    which is important for preserving original spacing.
    """
    text = "word1  word2\n\nword3"
    separators = [r"\s+"]

    # keep_separator=True is the default, but we're explicit.
    splitter = RecursiveCharacterSplitter(
        chunk_size=15, chunk_overlap=0, separators=separators, keep_separator=True
    )
    chunks = splitter.split_text(text)

    # EXPECTED BEHAVIOR:
    # The recursive split will produce fragments like ['word1  ', 'word2\n\n', 'word3']
    # Merging with chunk_size=15:
    # 1. 'word1  ' (len 7) -> ok
    # 2. 'word1  word2\n\n' (len 15) -> ok
    # 3. 'word1  word2\n\nword3' (len 20) -> too long. Finalize chunk 1.
    # Chunk 1: "word1  word2\n\n"

    assert chunks[0].content == "word1  word2\n\n"
    assert chunks[1].content == "word3"
