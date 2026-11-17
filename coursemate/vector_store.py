"""Small local TF-IDF vector store used by the CourseMate RAG pipeline."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from coursemate.chunking import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class TfidfVectorStore:
    """A dependency-free vector store suitable for a small course corpus."""

    backend_name = "tfidf"

    def __init__(self, chunks: list[Chunk], idf: dict[str, float], vectors: list[dict[str, float]]):
        self.chunks = chunks
        self.idf = idf
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "TfidfVectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store without chunks")

        tokenized = [_tokenize(chunk.text) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        total_docs = len(chunks)
        idf = {
            term: math.log((1 + total_docs) / (1 + df)) + 1
            for term, df in document_frequency.items()
        }
        vectors = [_normalize(_tfidf(tokens, idf)) for tokens in tokenized]
        return cls(chunks=chunks, idf=idf, vectors=vectors)

    def search(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = _normalize(_tfidf(_tokenize(query), self.idf))
        scored = [
            SearchResult(chunk=chunk, score=_dot(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [item for item in scored[:top_k] if item.score > 0]

    def save(self, path: str | Path) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "idf": self.idf,
            "vectors": self.vectors,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TfidfVectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [Chunk(**chunk) for chunk in payload["chunks"]]
        return cls(chunks=chunks, idf=payload["idf"], vectors=payload["vectors"])


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    ]


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in tokens if token in idf)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {term: (count / total) * idf[term] for term, count in counts.items()}


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {term: value / norm for term, value in vector.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())
