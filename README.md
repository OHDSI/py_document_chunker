# py_document_chunker

This repository contains a state-of-the-art, open-source Python package for advanced text segmentation (chunking). It is designed to be a critical component in Retrieval-Augmented Generation (RAG) systems and advanced Natural Language Processing (NLP) data pipelines.

The package transforms large, unstructured or semi-structured documents into optimally sized segments (chunks) that maximize semantic coherence while adhering to token constraints.

## Core Features

- **Multiple Chunking Strategies**: From simple fixed-size and recursive splitting to advanced, structure-aware, and semantic-based strategies.
- **Rich Metadata**: Every chunk is enriched with detailed metadata, including its start/end position, a unique ID, sequence number, hierarchical context, and even the exact content of the overlap with its neighbors.
- **Framework Integration**: Seamless integration with popular RAG frameworks like LangChain and LlamaIndex.
- **Highly Configurable**: All strategies are hyper-parameterized with sensible, research-backed defaults.
- **Extensible Architecture**: The modular design makes it easy to implement, customize, and combine strategies.

## Installation

You can install the core package and its dependencies using pip. The package is structured with optional extras for strategies that require heavy dependencies.

```bash
# Install the core package
pip install py_document_chunker

# To include support for sentence splitting (via NLTK or Spacy)
pip install py_document_chunker[nlp,spacy]

# To include support for Markdown and HTML splitting
pip install py_document_chunker[markdown,html]

# To include support for semantic splitting (requires numpy)
pip install py_document_chunker[semantic]

# To include support for code splitting (requires tree-sitter)
pip install py_document_chunker[code]

# To include support for token-based length functions (via tiktoken)
pip install py_document_chunker[tokenizers]

# To install framework integrations
pip install py_document_chunker[langchain,llamaindex]

# To install everything for development
pip install py_document_chunker[dev,nlp,markdown,html,semantic,code,tokenizers,langchain,llamaindex]
```

## Global Configuration

All splitter classes inherit from a common `TextSplitter` base class and share a set of powerful configuration options for preprocessing and chunk management.

- `chunk_size` (int, default: `1024`): The target maximum size of each chunk. The unit (characters or tokens) is determined by the `length_function`.
- `chunk_overlap` (int, default: `200`): The amount of overlap between consecutive chunks, measured in the same units as `chunk_size`.
- `length_function` (Callable[[str], int], default: `len`): The function used to measure the size of a text segment. This is the key to token-aware chunking. By default, it measures characters, but you can plug in any tokenizer function.
- `normalize_whitespace` (bool, default: `False`): If `True`, collapses all consecutive whitespace characters (spaces, newlines, tabs) into a single space.
- `unicode_normalize` (str, default: `None`): Specifies a Unicode normalization form to apply (e.g., `'NFC'`, `'NFKC'`).
- `minimum_chunk_size` (int, default: `0`): If set, the splitter will attempt to handle chunks smaller than this size.
- `min_chunk_merge_strategy` (str, default: `'merge_with_previous'`): Defines how to handle small chunks.
    - `'merge_with_previous'`: Merges a small chunk with the one that came before it.
    - `'merge_with_next'`: Merges a small chunk with the one that comes after it.
    - `'discard'`: Simply removes any chunk smaller than `minimum_chunk_size`.

## Quick Start & Rich Metadata

The simplest way to get started is with the `RecursiveCharacterSplitter`. All splitters return a list of `Chunk` objects, which are enriched with comprehensive metadata as required by the FRD.

```python
from py_document_chunker import RecursiveCharacterSplitter

text = "This is a long document.   It has multiple sentences and paragraphs.\n\nWe want to split it into smaller chunks. Some chunks are small."

# Initialize the splitter
splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)

# Split the text
chunks = splitter.split_text(text)

# Each chunk is a rich data object
first_chunk = chunks[0]

print(f"--- Chunk Content ---\n{first_chunk.content}\n")
# --- Chunk Content ---
# This is a long document.   It has multiple

print(f"--- Chunk Metadata ---")
# The to_dict() method provides a full view of the metadata
# Note: Some fields like 'hierarchical_context' are populated by specific strategies.
import json
print(json.dumps(first_chunk.to_dict(), indent=2))
```

Example output of `first_chunk.to_dict()`:
```json
{
  "content": "This is a long document.   It has multiple",
  "start_index": 0,
  "end_index": 42,
  "sequence_number": 0,
  "chunk_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "source_document_id": null,
  "hierarchical_context": {},
  "overlap_content_previous": null,
  "overlap_content_next": " sentences",
  "chunking_strategy_used": "recursive_character",
  "metadata": {}
}
```

## Strategies

### FixedSizeSplitter
The most basic strategy. Splits text into chunks of a fixed character size.

```python
from py_document_chunker import FixedSizeSplitter
splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(my_text)
```

### RecursiveCharacterSplitter
Recursively splits text based on a prioritized list of separators (e.g., `["\n\n", "\n", ". ", " "]`). This is often the recommended starting point.

```python
from py_document_chunker import RecursiveCharacterSplitter
splitter = RecursiveCharacterSplitter(chunk_size=1024, chunk_overlap=200)
chunks = splitter.split_text(my_text)
```

### SentenceSplitter
Splits text based on sentence boundaries using NLTK, then aggregates sentences into chunks. Requires the `[nlp]` extra.

```python
from py_document_chunker import SentenceSplitter
# Ensure you have run: python -c "import nltk; nltk.download('punkt')"
splitter = SentenceSplitter(chunk_size=1024, overlap_sentences=1)
chunks = splitter.split_text(my_prose_text)
```

### SpacySentenceSplitter
A more advanced sentence splitter that uses `spacy` for higher accuracy sentence boundary detection. It functions similarly to the NLTK-based splitter but often provides better results for complex texts. Requires the `[spacy]` extra.

```python
from py_document_chunker import SpacySentenceSplitter
# Ensure you have run: pip install py_document_chunker[spacy]
# And downloaded the model: python -m spacy download en_core_web_sm
splitter = SpacySentenceSplitter(chunk_size=1024, overlap_sentences=1)
chunks = splitter.split_text(my_prose_text)
```

### MarkdownSplitter
A structure-aware splitter that uses Markdown headers (H1-H6), paragraphs, and other elements as boundaries. Requires the `[markdown]` extra.

```python
from py_document_chunker import MarkdownSplitter
splitter = MarkdownSplitter(chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_markdown_text)
# Chunks will have `hierarchical_context` metadata populated.
print(chunks[0].hierarchical_context)
# Output: {'H1': 'Main Title', 'H2': 'Section 1'}
```

### HTMLSplitter
A structure-aware splitter for HTML documents. Requires the `[html]` extra.

```python
from py_document_chunker import HTMLSplitter
splitter = HTMLSplitter(chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_html_text)
```

### SemanticSplitter
Splits text by finding semantic breakpoints between sentences using an embedding model. It identifies points where the cosine similarity between adjacent sentence embeddings drops significantly. This is a powerful way to create topically coherent chunks. Requires the `[semantic]` extra.

The `breakpoint_method` and `breakpoint_threshold` parameters control how a breakpoint is determined:
- `breakpoint_method='percentile'` (default): A split occurs if the similarity is below the `X`-th percentile of all similarities in the document. `breakpoint_threshold` is the percentile (e.g., `95`).
- `breakpoint_method='std_dev'`: A split occurs if the similarity is more than `X` standard deviations below the mean similarity. `breakpoint_threshold` is the number of standard deviations (e.g., `1.5`).
- `breakpoint_method='absolute'`: A split occurs if the similarity is below a fixed value. `breakpoint_threshold` is the similarity value (e.g., `0.85`).

```python
from py_document_chunker import SemanticSplitter
# You must provide your own embedding function.
# e.g., from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_function = model.encode

# Example using standard deviation to find breakpoints
splitter = SemanticSplitter(
    embedding_function=my_embedding_function,
    breakpoint_method="std_dev",
    breakpoint_threshold=1.2
)
chunks = splitter.split_text(my_text)

# If a resulting semantic chunk is larger than chunk_size, it is automatically
# split further by the RecursiveCharacterSplitter.
```

### CodeSplitter
A syntax-aware splitter for source code. Requires the `[code]` extra.

```python
from py_document_chunker import CodeSplitter
splitter = CodeSplitter(language="python", chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_python_code)
```

## Token-Aware Chunking

A core design principle of this package is its tokenization awareness (FRD R-1.3.3). While the default behavior is to count characters, you can supply any tokenizer as a `length_function` to control chunk size based on token count. This is critical for RAG applications where LLM context windows are constrained by tokens.

### Using `tiktoken` for OpenAI Models

The package provides an optional utility to create a length function from OpenAI's `tiktoken` library. First, ensure you have the necessary dependency installed:

```bash
pip install py_document_chunker[tokenizers]
```

Then, you can create a token-based length function and pass it to any splitter.

```python
from py_document_chunker import RecursiveCharacterSplitter
from py_document_chunker.tokenizers import from_tiktoken

# A long text about the history of AI...
text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and other animals. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."

# Create a length function for the gpt-4 encoding
# This function will count tokens instead of characters
length_function = from_tiktoken("cl100k_base")

# Initialize the splitter with a token-based chunk size
token_splitter = RecursiveCharacterSplitter(
    chunk_size=30,          # Max 30 tokens per chunk
    chunk_overlap=5,        # 5-token overlap
    length_function=length_function
)

chunks = token_splitter.split_text(text)

# Verify that each chunk respects the token limit
for i, chunk in enumerate(chunks):
    token_count = length_function(chunk.content)
    print(f"Chunk {i+1}: {token_count} tokens")
    # assert token_count <= 30
```

## Framework Integrations

### LangChain
Use any splitter in a LangChain pipeline. Requires the `[langchain]` extra.
```python
from py_document_chunker import RecursiveCharacterSplitter
from py_document_chunker.integrations.langchain import LangChainWrapper

# 1. Create a splitter instance from this package
ats_splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)

# 2. Wrap it for LangChain
langchain_splitter = LangChainWrapper(ats_splitter)

# 3. Use it like any other LangChain TextSplitter
from langchain_core.documents import Document
docs = [Document(page_content="Some long text...")]
split_docs = langchain_splitter.split_documents(docs)
print(split_docs[0].metadata)
```

### LlamaIndex
Use any splitter as a LlamaIndex `NodeParser`. Requires the `[llamaindex]` extra.
```python
from py_document_chunker import SentenceSplitter
from py_document_chunker.integrations.llamaindex import LlamaIndexWrapper

# 1. Create a splitter instance
ats_splitter = SentenceSplitter(chunk_size=512, overlap_sentences=1)

# 2. Create the LlamaIndex-compatible NodeParser
node_parser = LlamaIndexWrapper(ats_splitter)

# 3. Use it in your LlamaIndex pipeline
from llama_index.core.schema import Document
nodes = node_parser.get_nodes_from_documents([Document(text="Some long text...")])
print(nodes[0].metadata)
```
