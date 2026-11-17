"""Chunk course materials into retrieval units."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Iterable

from coursemate.documents import Document


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with source metadata."""

    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    text: str
    start_word: int
    end_word: int


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 110,
    overlap: int = 25,
) -> list[Chunk]:
    """Split documents into overlapping word chunks.

    The overlap preserves local context across chunk boundaries while keeping
    passages compact enough to fit into small local-model prompts.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    ids = count(1)
    for document in documents:
        words = document.text.split()
        if not words:
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            if not chunk_words:
                continue
            chunk_number = next(ids)
            chunks.append(
                Chunk(
                    chunk_id=f"chunk-{chunk_number:04d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    source_path=document.source_path,
                    text=" ".join(chunk_words),
                    start_word=start,
                    end_word=end,
                )
            )
            if end == len(words):
                break
    return chunks
