"""
Text Chunker
Splits long text into overlapping windows for embedding.
Preserves sentence boundaries where possible to avoid cutting mid-thought.
"""

import re
from typing import List

from core.config import settings


class TextChunker:
    """
    Splits text into overlapping chunks suitable for embedding.

    Design choices:
    - Respects sentence boundaries to avoid severing geological measurements mid-value.
    - Uses character-level chunking with overlap for context continuity.
    - Skips chunks that are too short to be meaningful (< 50 chars).
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        Tries to break at sentence or paragraph boundaries.
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Prefer splitting at paragraph breaks, then sentence ends
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Single sentence longer than chunk_size — force split
            if sentence_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Carry over overlap
                    overlap_text = " ".join(current_chunk)[-self.chunk_overlap:]
                    current_chunk = [overlap_text, sentence]
                    current_len = len(overlap_text) + sentence_len
                else:
                    # Hard split the sentence itself
                    for hard_chunk in self._hard_split(sentence):
                        chunks.append(hard_chunk)
                continue

            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap — keep last N chars
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sentences + [sentence]
                current_len = overlap_len + sentence_len
            else:
                current_chunk.append(sentence)
                current_len += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c for c in chunks if len(c.strip()) >= 50]

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving geological notation.
        Avoids splitting at decimal points in measurements (e.g. '2.5 g/t Au').
        """
        # Split on paragraph breaks first
        paragraphs = re.split(r"\n\s*\n", text)
        sentences = []
        for para in paragraphs:
            # Split on sentence-ending punctuation, but not decimal numbers
            para_sentences = re.split(r"(?<!\d)(?<=[.!?])\s+(?=[A-Z])", para)
            sentences.extend([s.strip() for s in para_sentences if s.strip()])
        return sentences

    def _hard_split(self, text: str) -> List[str]:
        """Force-split text at character boundaries when no natural break exists."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
