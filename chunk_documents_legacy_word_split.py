import argparse
import glob
import json
import os
import re
from typing import Dict, Iterator


def chunk_by_section(
    content: str, max_chunk_words: int, file_path: str
) -> Iterator[Dict]:
    """
    Splits the document by Markdown headers and then by paragraphs if a section is too long.

    This function implements a hierarchical chunking strategy.
    1. It first splits the document by Markdown headers (# or ##).
    2. For each section, it checks if the word count exceeds max_chunk_words.
    3. If the section is within the limit, it's yielded as a single chunk.
    4. If the section is too long, it's split into paragraphs, and each paragraph
       is yielded as a separate chunk.

    Args:
        content: The full string content of the Markdown file.
        max_chunk_words: The maximum number of words a chunk can have before being split by paragraph.
        file_path: The original file path, used for metadata.

    Yields:
        A dictionary representing a single chunk, with its text and metadata.
    """
    # Regex to split by Markdown headers (## or #).
    # The `(?=^##? )` is a positive lookahead that finds the location of headers
    # without consuming them in the split.
    sections = re.split(r"(?=^##? )", content, flags=re.MULTILINE)

    # The first element might be some text before the first header (e.g., the main title).
    # We'll treat it as a section with a "Preamble" header.
    if sections and not sections[0].strip().startswith("#"):
        preamble = sections.pop(0).strip()
        if preamble:
            # Handle the case where the preamble itself is too long
            if len(preamble.split()) > max_chunk_words:
                paragraphs = preamble.split("\n\n")
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if para:
                        yield {
                            "text": para,
                            "metadata": {
                                "original_filename": os.path.basename(file_path),
                                "section_header": f"Preamble (Paragraph {i+1})",
                            },
                        }
            else:
                yield {
                    "text": preamble,
                    "metadata": {
                        "original_filename": os.path.basename(file_path),
                        "section_header": "Preamble",
                    },
                }

    # Process the remaining sections, which each start with a header.
    for section in sections:
        section = section.strip()
        if not section:
            continue

        try:
            header_line, body = section.split("\n", 1)
            header = header_line.strip()
            body = body.strip()
        except ValueError:
            # This can happen if a section has a header but no body.
            header = section.strip()
            body = ""

        if not body:
            continue

        word_count = len(body.split())

        if word_count <= max_chunk_words:
            # If the section is small enough, yield it as a single chunk.
            yield {
                "text": body,
                "metadata": {
                    "original_filename": os.path.basename(file_path),
                    "section_header": header,
                },
            }
        else:
            # If the section is too long, split it into paragraphs.
            paragraphs = body.split("\n\n")
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para:  # Ensure we don't yield empty paragraphs
                    yield {
                        "text": para,
                        "metadata": {
                            "original_filename": os.path.basename(file_path),
                            "section_header": f"{header} (Paragraph {i+1})",
                        },
                    }


def process_documents(input_dir: str, output_file: str, max_chunk_words: int) -> None:
    """
    Processes all Markdown files in a directory, chunks them, and saves them to a JSONL file.

    Args:
        input_dir: The directory containing the .md files.
        output_file: The path to the output .jsonl file.
        max_chunk_words: The maximum word count for a chunk.
    """
    # Find all Markdown files in the input directory.
    md_files = glob.glob(os.path.join(input_dir, "*.md"))

    print(f"Found {len(md_files)} Markdown files to process.")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_path in md_files:
            print(f"Processing {file_path}...")
            try:
                with open(file_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()

                for chunk in chunk_by_section(content, max_chunk_words, file_path):
                    # Convert the chunk dictionary to a JSON string and write it to the file,
                    # followed by a newline to create the JSONL format.
                    f_out.write(json.dumps(chunk) + "\n")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Chunking complete. Output saved to {output_file}")


def main():
    """
    Main function to parse command-line arguments and start the chunking process.
    """
    parser = argparse.ArgumentParser(
        description="Chunk scientific articles in Markdown format for LLM applications."
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
        default="chunks.jsonl",
        help="The path to the output JSONL file. Defaults to 'chunks.jsonl'.",
    )
    parser.add_argument(
        "--max_chunk_words",
        type=int,
        default=500,
        help="The maximum number of words for a chunk. Sections exceeding this will be split by paragraph. Defaults to 500.",
    )

    args = parser.parse_args()

    process_documents(args.input_dir, args.output_file, args.max_chunk_words)


if __name__ == "__main__":
    main()
