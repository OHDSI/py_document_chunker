# py_document_chunker Package: Technical Documentation

This document provides a detailed technical documentation of the py_document_chunker package, mapping its features and implementation directly to the Functional Requirements Document (FRD).

## 1.0 Introduction and Scope

This section outlines the high-level purpose, objectives, and design principles of the package, corresponding to Section 1.0 of the FRD.

### 1.1 Purpose (FRD 1.1)

The **py_document_chunker** package is a specialized Python library designed to be a foundational component in modern Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) systems. Its primary purpose is to ingest large, unstructured, or semi-structured documents and intelligently divide them into smaller, more manageable segments, or "chunks."

### 1.2 Objectives (FRD 1.2)

The core objective is to transform documents into optimally sized chunks that are both semantically coherent and compliant with the token-based constraints of Large Language Models (LLMs) and embedding models. By creating high-quality chunks, the package aims to significantly enhance the relevance and accuracy of information retrieval in RAG pipelines.

### 1.3 Core Design Principles (FRD 1.3)

The package's architecture is built upon five core principles, which are reflected throughout the codebase.

*   **Hyper-Parameterization**: Every chunking strategy is highly configurable. The base `TextSplitter` class (`src/py_document_chunker/base.py`) establishes a common set of parameters (`chunk_size`, `chunk_overlap`, `minimum_chunk_size`, etc.) with sensible defaults, allowing users to fine-tune the chunking process to their specific needs.

*   **Modularity and Extensibility**: The architecture is fundamentally modular. It is built around a central abstract base class, `TextSplitter`, which defines a common interface for all strategies. Each chunking strategy (e.g., `RecursiveCharacterSplitter`, `MarkdownSplitter`) is implemented as a separate, self-contained class, making it easy to extend the package with new, custom strategies.

*   **Tokenization Awareness**: The package is designed to be fully aware of token-based length calculations, which are critical for LLM applications. The `length_function` parameter in all splitters allows users to plug in any tokenizer function (e.g., `tiktoken` for OpenAI models, or a Hugging Face tokenizer). The `src/py_document_chunker/tokenizers.py` module provides helpers for this purpose. This directly fulfills the requirement for the package to accurately measure chunk size using pluggable tokenization methods.

*   **Ecosystem Compatibility**: The package ensures seamless integration with the most popular NLP and RAG frameworks. The `src/py_document_chunker/integrations/` directory contains dedicated wrapper classes:
    *   `LangChainWrapper` inherits from LangChain's `TextSplitter`, allowing any splitter from this package to be used in a LangChain pipeline.
    *   `LlamaIndexWrapper` inherits from LlamaIndex's `NodeParser`, providing the same drop-in compatibility for the LlamaIndex ecosystem.

*   **Reliability and Stability**: The core package maintains a minimal dependency footprint, as defined in `pyproject.toml`. The base library has zero external dependencies, ensuring maximum reliability. Advanced features that require heavy or specialized libraries (like `nltk`, `spacy`, `numpy`, or `tree-sitter`) are implemented as optional extras (e.g., `pip install .[nlp]`), ensuring that users only install what they need.

## 2.0 Input Handling and Preprocessing

This section details how the package ingests and preprocesses text before segmentation, corresponding to Section 2.0 of the FRD. These features are primarily implemented in the base `TextSplitter` class (`src/py_document_chunker/base.py`).

### 2.1 Data Ingestion

*   **R-2.1.1 (Standard String Input)**: The primary input format for all splitter objects is a standard Python string (`str`). The main method for splitting, `split_text(text: str)`, is consistent across all strategies.

*   **R-2.1.2 (UTF-8 Encoding)**: The package implicitly assumes and operates on UTF-8 encoded strings, which is the standard for Python 3. The `CodeSplitter`, for instance, explicitly encodes text to `utf-8` when interfacing with the `tree-sitter` library.

*   **R-2.1.3 & R-2.1.4 (Structured Format Handling)**: The package provides dedicated, structure-aware handlers for Markdown and HTML:
    *   The `MarkdownSplitter` (`strategies/structure/markdown.py`) parses CommonMark-compliant Markdown.
    *   The `HTMLSplitter` (`strategies/structure/html.py`) parses HTML content.

    Both of these splitters are designed to extract the plain text from the document while preserving the structural hierarchy as metadata (see `hierarchical_context` in Section 5.2).

### 2.2 Preprocessing

The `TextSplitter` base class provides optional, configurable preprocessing steps that are applied to the text before any chunking logic begins.

*   **R-2.2.1 (Whitespace Normalization)**: This is controlled by the `normalize_whitespace: bool` parameter in the constructor of any splitter. When set to `True`, it collapses all consecutive whitespace characters (including spaces, tabs, and newlines) into a single space and trims leading/trailing whitespace from the entire text.

*   **R-2.2.2 (Character Handling)**: The package offers two parameters for robust character handling:
    *   `unicode_normalize: str`: This parameter accepts a standard Unicode normalization form (`'NFC'`, `'NFKC'`, `'NFD'`, or `'NFKD'`) to resolve character inconsistencies.
    *   `strip_control_chars: bool`: When set to `True`, this removes all Unicode control characters (e.g., null bytes `\x00`, byte-order marks) which can cause issues in downstream processing.

## 3.0 Chunking Strategies

This section provides a detailed description of each chunking strategy implemented in the package, corresponding to Section 3.0 of the FRD. All strategies inherit from the `TextSplitter` base class and share the common configuration parameters detailed in Section 4.0.

---

### 3.1 Fixed-Size Chunking (R-3.1)

*   **Concept**: This is the most basic approach to chunking. It splits the text into segments of a predetermined maximum size, with a defined overlap between them. It does not consider any semantic or structural boundaries.
*   **Implementation**: `py_document_chunker.strategies.fixed_size.FixedSizeSplitter`
*   **How it Works**: This class is cleverly implemented as a subclass of `RecursiveCharacterSplitter`. It forces the recursive splitter to use an empty string (`""`) as its only separator. This has the effect of splitting the text into individual characters, which are then merged back together into chunks that strictly adhere to the `chunk_size`, as measured by the `length_function`. This ensures that even with complex tokenizers, the chunk size limit is never violated.

**Code Example:**
```python
from py_document_chunker import FixedSizeSplitter

text = "This is a simple text that will be split into fixed-size chunks."
splitter = FixedSizeSplitter(chunk_size=20, chunk_overlap=5)
chunks = splitter.split_text(text)
# Result: ['This is a simple tex', 'simple text that wil', 'hat will be split in', 'split into fixed-siz', 'fixed-size chunks.']
```

---

### 3.2 Recursive Character Splitting (R-3.2)

*   **Concept**: This is a highly versatile and commonly used strategy. It attempts to split text based on a prioritized, ordered list of separators. It starts with the highest-priority separator (e.g., double newlines for paragraphs) and recursively moves to lower-priority separators (e.g., newlines, periods, spaces) until the resulting segments are smaller than the `chunk_size`.
*   **Implementation**: `py_document_chunker.strategies.recursive.RecursiveCharacterSplitter`
*   **How it Works**: The implementation uses a robust two-stage "split-then-merge" algorithm.
    1.  **Split**: The `_recursive_split` method breaks the text down into the smallest possible fragments based on the separators, crucially preserving the `start_index` of each fragment relative to the original document.
    2.  **Merge**: The main `split_text` method then intelligently merges these fragments back together, creating chunks that respect the `chunk_size` and calculating the correct `chunk_overlap`. If a segment remains too large after all separators have been tried, it performs a hard character split as a fallback (R-3.2.4).

**Code Example:**
```python
from py_document_chunker import RecursiveCharacterSplitter

text = "First paragraph.\n\nSecond paragraph. It has two sentences.\nThird paragraph."
# Default separators are ["\n\n", "\n", ". ", " ", ""]
splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=10)
chunks = splitter.split_text(text)
# Result: ['First paragraph.', 'Second paragraph. It has two sentences.', 'Third paragraph.']
```

---

### 3.3 Syntactic Chunking (Sentence Splitting) (R-3.3)

*   **Concept**: This strategy splits a document based on sentence boundaries, aiming to keep whole sentences intact within chunks. It then aggregates these sentences into chunks that do not exceed the `chunk_size`. This is ideal for prose and text where grammatical completeness is important.
*   **Implementation**:
    *   `py_document_chunker.strategies.sentence.SentenceSplitter` (uses NLTK)
    *   `py_document_chunker.strategies.spacy_sentence.SpacySentenceSplitter` (uses spaCy for potentially higher accuracy)
*   **How it Works**: The splitter first uses an NLP library (NLTK or spaCy) to perform Sentence Boundary Detection (SBD), identifying the exact start and end character indices of each sentence. It then groups these sentences into a chunk until adding the next sentence would exceed `chunk_size`. Overlap is controlled by `overlap_sentences`, which specifies how many sentences from the end of one chunk should be repeated at the beginning of the next (R-3.3.3). If a single sentence is longer than `chunk_size`, it is broken down further using a fallback `RecursiveCharacterSplitter`.

**Code Example (NLTK):**
```python
from py_document_chunker import SentenceSplitter

# Requires: pip install .[nlp]
# And: python -c "import nltk; nltk.download('punkt')"
text = "This is the first sentence. This is the second. A third one follows. And a fourth."
splitter = SentenceSplitter(chunk_size=50, overlap_sentences=1)
chunks = splitter.split_text(text)
# Result: ['This is the first sentence. This is the second.', 'This is the second. A third one follows.', 'A third one follows. And a fourth.']
```

---

### 3.4 Structure-Aware Chunking (R-3.4)

*   **Concept**: This strategy leverages the explicit structure of semi-structured documents like Markdown and HTML to create highly logical chunks. It prioritizes splitting at major structural boundaries (e.g., between sections marked by headers) rather than just at a certain size.
*   **Implementation**:
    *   `py_document_chunker.strategies.structure.markdown.MarkdownSplitter`
    *   `py_document_chunker.strategies.structure.html.HTMLSplitter`
*   **How it Works**: The `MarkdownSplitter` uses a sophisticated multi-pass algorithm:
    1.  **Parse & Extract**: It uses `markdown-it-py` to parse the document into a syntax tree. It traverses the tree to extract a flat list of "blocks" (headings, paragraphs, lists, etc.), attaching the current header hierarchy (e.g., `{'H1': 'Title', 'H2': 'Section'}`) to each block as `hierarchical_context` metadata (R-3.4.4).
    2.  **Group**: It iterates through the blocks and groups them into chunks, splitting primarily when a major header boundary is crossed (R-3.4.3) or when the block type changes.
    3.  **Fallback**: If a resulting chunk (often a single large block) is still larger than `chunk_size`, it is split further by a `RecursiveCharacterSplitter`.

**Code Example (Markdown):**
```python
from py_document_chunker import MarkdownSplitter

# Requires: pip install .[markdown]
markdown_text = "# Title\n\nThis is a paragraph.\n\n## Section 1\n\nThis content is under section 1."
splitter = MarkdownSplitter(chunk_size=50)
chunks = splitter.split_text(markdown_text)
# chunk[0].content -> "# Title\n\nThis is a paragraph."
# chunk[0].hierarchical_context -> {'H1': 'Title'}
# chunk[1].content -> "## Section 1\n\nThis content is under section 1."
# chunk[1].hierarchical_context -> {'H1': 'Title', 'H2': 'Section 1'}
```

---

### 3.5 Semantic Chunking (R-3.5)

*   **Concept**: This is an advanced strategy that splits text based on semantic meaning rather than fixed rules. It identifies points in the text where the topic or context appears to shift, creating chunks that are topically coherent.
*   **Implementation**: `py_document_chunker.strategies.semantic.SemanticSplitter`
*   **How it Works**: The process involves several steps:
    1.  **Pluggable Embeddings (R-3.5.1)**: The user MUST provide an `embedding_function` that can convert text into numerical vectors (embeddings).
    2.  **Sentence Splitting**: The text is first divided into individual sentences.
    3.  **Breakpoint Detection (R-3.5.2)**: Each sentence is embedded. The cosine similarity between adjacent sentence embeddings is calculated. A "breakpoint" is identified wherever this similarity score drops below a certain threshold.
    4.  **Breakpoint Configuration (R-3.5.3)**: The threshold can be determined in three ways, configured via `breakpoint_method`:
        *   `percentile`: The threshold is a percentile of all similarity scores.
        *   `std_dev`: The threshold is a number of standard deviations below the mean similarity.
        *   `absolute`: The threshold is a fixed similarity value.
    5.  **Fallback (R-3.5.4)**: Sentences between breakpoints are grouped into a chunk. If this chunk exceeds `chunk_size`, it is split further by a `RecursiveCharacterSplitter`.

**Code Example:**
```python
from py_document_chunker import SemanticSplitter
# Requires: pip install .[semantic]
# Assume 'my_embedding_function' is a function you provide, e.g., from sentence-transformers.
# def my_embedding_function(texts: list[str]) -> list[list[float]]: ...

text = "AI is transforming the world. Meanwhile, quantum computing is an emerging field."
splitter = SemanticSplitter(
    embedding_function=my_embedding_function,
    breakpoint_method="absolute",
    breakpoint_threshold=0.85 # Split if similarity is below 0.85
)
chunks = splitter.split_text(text)
# Expected result (if similarity is low enough): Two chunks, one for each sentence.
```

---

### 3.6 Specialized Chunking (Code) (R-3.6)

*   **Concept**: This strategy is specifically designed for source code. It uses a proper code parser (`tree-sitter`) to split code along syntactic boundaries, such as functions, classes, or methods. This ensures that chunks are syntactically complete and logically self-contained.
*   **Implementation**: `py_document_chunker.strategies.code.CodeSplitter`
*   **How it Works**: The splitter uses the `tree-sitter` library to parse the source code into a Concrete Syntax Tree (CST).
    1.  **Language Support (R-3.6.1, R-3.6.2)**: The user specifies the `language` (e.g., `'python'`), and the splitter loads the appropriate grammar.
    2.  **Hierarchical Splitting (R-3.6.3)**: It traverses the syntax tree to find the highest-level "chunkable" nodes (defined per-language in `LANGUAGE_QUERIES`). For example, it will identify an entire class definition rather than the individual methods inside it. This ensures that the logical structure of the code is preserved.
    3.  **Fallback**: If a high-level syntactic block (like a very long function) is still larger than `chunk_size`, it is split further by a `RecursiveCharacterSplitter`.

**Code Example (Python):**
```python
from py_document_chunker import CodeSplitter

# Requires: pip install .[code]
python_code = "class MyClass:\n    def method_one(self):\n        pass\n\ndef my_function():\n    return True"
splitter = CodeSplitter(language="python", chunk_size=1000)
chunks = splitter.split_text(python_code)
# Result: Two chunks, one for the class and one for the function.
# chunk[0].content -> "class MyClass:\n    def method_one(self):\n        pass"
# chunk[1].content -> "def my_function():\n    return True"
```

## 4.0 Configuration and Optimization

This section details the core configuration and optimization parameters available for all text splitters, corresponding to Section 4.0 of the FRD. These parameters are part of the base `TextSplitter` class (`src/py_document_chunker/base.py`) and can be passed to the constructor of any splitter.

### 4.1 Core Parameters

These are the fundamental parameters that control the chunking process.

*   **`chunk_size: int` (Default: `1024`)**: Implements **R-4.1.2**. This defines the target maximum size for any generated chunk. The unit of measurement (characters or tokens) is determined by the `length_function`.

*   **`chunk_overlap: int` (Default: `200`)**: Implements **R-4.1.3**. This defines the amount of overlap between consecutive chunks, measured in the same units as `chunk_size`.

*   **`length_function: Callable[[str], int]` (Default: `len`)**: Implements **R-4.1.1**. This is a callable function used to measure the size of a text string. By default, it counts characters using Python's built-in `len`. However, it can be replaced with any custom function, most notably a tokenizer, to enable token-aware chunking.

**Example: Token-Aware Chunking with `tiktoken`**
```python
from py_document_chunker import RecursiveCharacterSplitter
from py_document_chunker.tokenizers import from_tiktoken

# Requires: pip install .[tokenizers]
text = "A long document about Large Language Models and their token limits."

# Create a length function for the gpt-4 encoding (counts tokens)
length_function = from_tiktoken("cl100k_base")

token_splitter = RecursiveCharacterSplitter(
    chunk_size=10,  # Max 10 tokens per chunk
    chunk_overlap=2, # 2-token overlap
    length_function=length_function
)

chunks = token_splitter.split_text(text)
# Each chunk's content will have a token count <= 10.
for chunk in chunks:
    print(f"Content: '{chunk.content}', Tokens: {length_function(chunk.content)}")
```

### 4.2 Optimization Parameters

These parameters control how the splitter handles "runt" chunks—segments that are undesirably small. The logic is implemented in the `_enforce_minimum_chunk_size` method in the `TextSplitter` base class.

*   **`minimum_chunk_size: int` (Default: `0`)**: Implements **R-4.2.1**. This sets a threshold below which a generated chunk is considered a "runt." If a chunk's size is less than this value, it will be handled according to the `min_chunk_merge_strategy`.

*   **`min_chunk_merge_strategy: str` (Default: `'merge_with_previous'`)**: Implements **R-4.2.2**. This defines the action to take when a runt chunk is found:
    *   `'merge_with_previous'`: The small chunk is merged with the chunk that came before it (if possible without exceeding `chunk_size`).
    *   `'merge_with_next'`: The small chunk is merged with the chunk that comes after it.
    *   `'discard'`: The small chunk is removed entirely.

## 5.0 Output Schema and Metadata Enrichment

This section details the standardized output format of all splitting operations, corresponding to Section 5.0 of the FRD. A primary goal of this package is to enrich every generated chunk with comprehensive and useful metadata.

### 5.1 Output Structure

*   **R-5.1.1 (Standardized Object)**: The output of any `splitter.split_text()` operation is a list of `Chunk` objects (`List[Chunk]`). The `Chunk` is a dedicated data class defined in `src/py_document_chunker/core.py` that holds both the content and all associated metadata.

*   **R-5.1.2 (Framework Compatibility)**: When a splitter is used via one of the integration classes (see Section 6.2), the `Chunk` objects are automatically and losslessly converted into the appropriate schema for that framework:
    *   **LangChain**: `Chunk` is converted to a `langchain_core.documents.Document`. The `content` becomes `page_content`, and all other chunk attributes are placed in the `metadata` dictionary.
    *   **LlamaIndex**: `Chunk` is converted to a `llama_index.core.schema.TextNode`. The `content` becomes the `text`, and all other chunk attributes are placed in the `metadata` dictionary.

### 5.2 Metadata Enrichment

Every `Chunk` object is enriched with the following metadata fields, directly corresponding to the requirements in FRD Section 5.2.

*   **`content: str`**: The actual text content of the chunk.

*   **`chunk_id: str` (R-5.2.1)**: A unique identifier (UUID v4) for the chunk, automatically generated upon creation.

*   **`source_document_id: Optional[str]` (R-5.2.2)**: An optional identifier for the original document from which the chunk was derived. This can be passed as an argument to the `split_text` method.

*   **`start_index: int` (R-5.2.3)**: The exact character start position (0-indexed integer) of the chunk's content relative to the start of the original, preprocessed source document.

*   **`end_index: int` (R-5.2.4)**: The exact character end position of the chunk's content relative to the start of the original, preprocessed source document.

*   **`sequence_number: int` (R-5.2.5)**: The ordinal position (0-indexed integer) of the chunk within the sequence of chunks generated from a single document.

*   **`hierarchical_context: Dict[str, Any]` (R-5.2.6)**: A structured dictionary representing the logical section the chunk belongs to. This is primarily populated by the structure-aware strategies (`MarkdownSplitter`, `HTMLSplitter`) and contains the hierarchy of headers above the chunk (e.g., `{'H1': 'Introduction', 'H2': 'Methods'}`).

*   **`overlap_content_previous: Optional[str]` (R-5.2.7)**: The exact text segment that this chunk shares with the immediately preceding chunk. This is `None` for the first chunk. This metadata is populated by the `_populate_overlap_metadata` utility function.

*   **`overlap_content_next: Optional[str]` (R-5.2.8)**: The exact text segment that this chunk shares with the immediately subsequent chunk. This is `None` for the last chunk.

*   **`chunking_strategy_used: Optional[str]` (R-5.2.9)**: The name of the strategy that generated this specific chunk (e.g., `'recursive_character'`, `'markdown'`, `'code-fallback'`). This is useful for diagnostics and for understanding how a particular chunk was created.

*   **`metadata: Dict[str, Any]`**: A flexible dictionary for any additional, unstructured metadata that might be passed through from framework integrations.

## 6.0 Integration and Architecture

This final section describes the package's high-level software architecture, its extensibility model, and its deep integration with popular external frameworks, corresponding to Section 6.0 of the FRD.

### 6.1 Architecture and Extensibility

*   **R-6.1.1 (Modular Architecture)**: The package is built on a modular and extensible foundation. The core of this design is the `TextSplitter` abstract base class (ABC) located in `src/py_document_chunker/base.py`. This ABC defines a common interface (the `split_text` method) and a shared set of configuration parameters that all concrete chunking strategies must adhere to. This ensures consistency and allows for any splitter to be used interchangeably.

*   **R-6.1.2 (Framework-Agnostic Core)**: The core implementation of every chunking strategy is framework-agnostic. The strategies operate on standard Python strings and produce a `List[Chunk]`, without any knowledge of LangChain or LlamaIndex. This clean separation of concerns makes the core logic highly portable, stable, and easy to test in isolation.

### 6.2 Framework Integrations

The package provides dedicated, high-quality wrapper classes to ensure seamless, drop-in compatibility with major RAG frameworks. These are located in the `src/py_document_chunker/integrations/` directory.

*   **R-6.2.1 (LangChain Compatibility)**: The `LangChainWrapper` class (`integrations/langchain.py`) serves as an adapter to the LangChain ecosystem.
    *   It inherits from `langchain_core.text_splitter.TextSplitter`.
    *   It wraps an instance of any splitter from this package.
    *   When its `split_documents` method is called, it uses the wrapped splitter to generate `Chunk` objects and then losslessly converts them into LangChain `Document` objects, preserving all the rich metadata as described in Section 5.1.

*   **R-6.2.2 (LlamaIndex Compatibility)**: The `LlamaIndexWrapper` class (`integrations/llamaindex.py`) provides the same level of integration for LlamaIndex.
    *   It inherits from `llama_index.core.node_parser.NodeParser`.
    *   It also wraps a splitter from this package.
    *   When its `get_nodes_from_documents` method is called, it uses the wrapped splitter to generate `Chunk` objects and converts them into LlamaIndex `TextNode` objects, preserving all metadata and correctly setting the source node relationships.

### 6.3 Dependency Management

The package is designed to be as lightweight or as powerful as the user requires, following a minimal-dependency philosophy.

*   **R-6.3.1 (Minimal Core Dependencies)**: The core package, as defined in `pyproject.toml`, has **zero** external dependencies. This makes it highly reliable, easy to install, and conflict-free for basic use cases like `FixedSizeSplitter` and `RecursiveCharacterSplitter`.

*   **R-6.3.2 (Optional Extras)**: All features requiring specialized or heavy libraries are implemented as optional extras. This allows users to install only the dependencies they need for the specific strategies they intend to use. The available extras are:
    *   `nlp`: For `SentenceSplitter` (installs `nltk`).
    *   `spacy`: For `SpacySentenceSplitter` (installs `spacy`).
    *   `semantic`: For `SemanticSplitter` (installs `numpy`).
    *   `code`: For `CodeSplitter` (installs `tree-sitter` and `tree-sitter-language-pack`).
    *   `markdown`: For `MarkdownSplitter` (installs `markdown-it-py`).
    *   `html`: For `HTMLSplitter` (installs `beautifulsoup4`, etc.).
    *   `tokenizers`: For token-based length functions (installs `tiktoken`).
    *   `langchain`: For the LangChain integration.
    *   `llamaindex`: For the LlamaIndex integration.
