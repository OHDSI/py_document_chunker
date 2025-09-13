import argparse
import glob
import json
import os
import re
import uuid
from typing import Dict, Iterator, List

# For robust sentence tokenization
import nltk

# --- NLTK Setup ---
# Download the 'punkt' tokenizer data if not already present.
# This is a one-time setup.
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    print("Downloading nltk 'punkt' model...")
    nltk.download("punkt")
    print("Download complete.")


def chunk_text_by_sentences(
    text: str, max_chunk_words: int, overlap_sentences: int
) -> List[str]:
    """
    Splits a long text into semantically meaningful chunks based on sentences.

    Args:
        text: The text to be chunked.
        max_chunk_words: The approximate maximum number of words for each chunk.
        overlap_sentences: The number of sentences to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    # 1. Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # 2. Group sentences into chunks
    chunks = []
    current_chunk_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If adding the next sentence exceeds the max word count, finalize the current chunk
        if (
            current_word_count + sentence_word_count > max_chunk_words
            and current_chunk_sentences
        ):
            chunks.append(" ".join(current_chunk_sentences))

            # Start the next chunk with an overlap
            num_overlap = min(len(current_chunk_sentences), overlap_sentences)
            current_chunk_sentences = current_chunk_sentences[-num_overlap:]
            current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
        
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Add the last remaining chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_by_section(
    content: str, max_chunk_words: int, file_path: str, overlap_sentences: int
) -> Iterator[Dict]:
    """
    Splits a Markdown document by sections, then further chunks long sections/paragraphs
    by sentences while maintaining sentence integrity.

    Args:
        content: The full string content of the Markdown file.
        max_chunk_words: The maximum number of words for a chunk.
        file_path: The original file path, used for metadata.
        overlap_sentences: The number of sentences for overlap in long text blocks.

    Yields:
        A dictionary for each chunk with its text and metadata.
    """
    # Regex to split by Markdown headers (## or #).
    sections = re.split(r"(?=^##? )", content, flags=re.MULTILINE)

    def create_chunk_object(text: str, section_header: str) -> Dict:
        """Helper to create the final chunk dictionary with a UUID."""
        return {
            "text": text,
            "metadata": {
                "chunk_id": str(uuid.uuid4()),
                "original_filename": os.path.basename(file_path),
                "section_header": section_header,
            },
        }

    # Handle preamble (text before the first header)
    if sections and not sections[0].strip().startswith("#"):
        preamble = sections.pop(0).strip()
        if preamble:
            if len(preamble.split()) > max_chunk_words:
                for i, sentence_chunk in enumerate(
                    chunk_text_by_sentences(
                        preamble, max_chunk_words, overlap_sentences
                    )
                ):
                    yield create_chunk_object(
                        sentence_chunk, f"Preamble (Part {i+1})"
                    )
            else:
                yield create_chunk_object(preamble, "Preamble")

    # Process the remaining sections
    for section in sections:
        section = section.strip()
        if not section:
            continue

        try:
            header_line, body = section.split("\n", 1)
            header = header_line.strip()
            body = body.strip()
        except ValueError:
            header = section.strip()
            body = ""

        if not body:
            continue

        if len(body.split()) <= max_chunk_words:
            yield create_chunk_object(body, header)
        else:
            # If the section is long, split into paragraphs first
            paragraphs = body.split("\n\n")
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue

                if len(para.split()) > max_chunk_words:
                    # If a paragraph is still too long, chunk it by sentences
                    for j, sentence_chunk in enumerate(
                        chunk_text_by_sentences(
                            para, max_chunk_words, overlap_sentences
                        )
                    ):
                        yield create_chunk_object(
                            sentence_chunk,
                            f"{header} (Paragraph {i+1}, Part {j+1})",
                        )
                else:
                    yield create_chunk_object(
                        para, f"{header} (Paragraph {i+1})"
                    )


def process_documents(
    input_dir: str, output_file: str, max_chunk_words: int, overlap_sentences: int
) -> None:
    """
    Processes all Markdown files in a directory, chunks them, and saves to a JSONL file.
    Adds per-document chunk indexing to the metadata.
    """
    md_files = glob.glob(os.path.join(input_dir, "*.md"))
    print(f"Found {len(md_files)} Markdown files to process.")

    comment_pattern = re.compile(r"^# Parsed Document:.*\n?", flags=re.MULTILINE)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_path in md_files:
            print(f"Processing {file_path}...")
            try:
                # Generate a single, unique ID for the entire document
                document_uuid = str(uuid.uuid4())
                
                with open(file_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()

                content_cleaned = comment_pattern.sub("", content)

                # --- MODIFICATION START ---
                # Step 1: Generate all chunks for the current file and store them in a list.
                chunks_for_file = list(
                    chunk_by_section(
                        content_cleaned, max_chunk_words, file_path, overlap_sentences
                    )
                )
                
                # Step 2: Get the total number of chunks for this document.
                total_chunks = len(chunks_for_file)

                # Step 3: Iterate through the collected chunks to add new metadata and write to file.
                for i, chunk in enumerate(chunks_for_file):

                    chunk["metadata"]["document_id"] = document_uuid
                    

                    # Add the new metadata fields
                    chunk["metadata"]["chunk_index"] = i
                    chunk["metadata"]["total_chunks_in_doc"] = total_chunks
                    
                    f_out.write(json.dumps(chunk) + "\n")
                # --- MODIFICATION END ---
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Chunking complete. Output saved to {output_file}")


def main():
    """Main function to parse command-line arguments and start the chunking process."""
    parser = argparse.ArgumentParser(
        description="Chunk scientific articles in Markdown format using a sentence-aware strategy."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The directory containing the Markdown files (.md) to be chunked.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="chunks_final.jsonl",
        help="The path to the output JSONL file. Defaults to 'chunks_final.jsonl'.",
    )
    parser.add_argument(
        "--max_chunk_words",
        type=int,
        default=384,  # A common token size for embedding models is 512, this is a safe word count
        help="The target maximum number of words for a chunk. Defaults to 384.",
    )
    parser.add_argument(
        "--overlap_sentences",
        type=int,
        default=2,
        help="The number of sentences to overlap between chunks. Defaults to 2.",
    )

    args = parser.parse_args()

    process_documents(
        args.input_dir, args.output_file, args.max_chunk_words, args.overlap_sentences
    )


if __name__ == "__main__":
    main()
