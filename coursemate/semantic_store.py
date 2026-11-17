"""Semantic retrieval with Sentence Transformers and ChromaDB."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from coursemate.chunking import Chunk
from coursemate.vector_store import SearchResult


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticStoreError(RuntimeError):
    """Raised when semantic retrieval dependencies or operations fail."""


class ChromaSemanticVectorStore:
    """A ChromaDB-backed semantic vector store for uploaded course documents."""

    backend_name = "sentence-transformers-chromadb"

    def __init__(
        self,
        *,
        persist_dir: str | Path,
        collection_name: str = "coursemate_documents",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SemanticStoreError("Install chromadb for semantic retrieval.") from exc

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        if reset and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SemanticStoreError("Install chromadb and sentence-transformers for retrieval.") from exc

        try:
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        except Exception as exc:
            raise SemanticStoreError(
                f"Could not load embedding model '{embedding_model}'. "
                "Check network/model cache availability or choose a local model."
            ) from exc
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Chunk],
        *,
        persist_dir: str | Path,
        collection_name: str = "coursemate_documents",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reset: bool = True,
    ) -> "ChromaSemanticVectorStore":
        store = cls(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            reset=reset,
        )
        store.add_chunks(list(chunks))
        return store

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "chunk_id": chunk.chunk_id,
                    "start_word": chunk.start_word,
                    "end_word": chunk.end_word,
                }
                for chunk in chunks
            ],
        )

    def search(self, query: str, *, top_k: int = 4) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.collection.count() == 0:
            return []

        response = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            chunk = Chunk(
                chunk_id=str(metadata["chunk_id"]),
                doc_id=str(metadata["doc_id"]),
                title=str(metadata["title"]),
                source_path=str(metadata["source_path"]),
                text=text,
                start_word=int(metadata["start_word"]),
                end_word=int(metadata["end_word"]),
            )
            # Chroma cosine distance is lower-is-better. Convert to a similarity-like score.
            score = max(0.0, 1.0 - float(distance))
            results.append(SearchResult(chunk=chunk, score=score))
        return results

    def count(self) -> int:
        return self.collection.count()
