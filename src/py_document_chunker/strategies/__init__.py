"""
The `strategies` module contains the different chunking strategies available
in the package.

Each strategy is implemented as a class that inherits from the base `TextSplitter`.
"""

from .code import CodeSplitter
from .fixed_size import FixedSizeSplitter
from .recursive import RecursiveCharacterSplitter
from .semantic import SemanticSplitter
from .sentence import SentenceSplitter
from .spacy_sentence import SpacySentenceSplitter
from .structure.html import HTMLSplitter
from .structure.markdown import MarkdownSplitter

__all__ = [
    "CodeSplitter",
    "FixedSizeSplitter",
    "RecursiveCharacterSplitter",
    "SemanticSplitter",
    "SentenceSplitter",
    "SpacySentenceSplitter",
    "HTMLSplitter",
    "MarkdownSplitter",
]
