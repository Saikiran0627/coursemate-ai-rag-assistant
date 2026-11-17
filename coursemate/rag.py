"""CourseMate RAG orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coursemate.chunking import chunk_documents
from coursemate.documents import load_documents
from coursemate.llm import LanguageModel, build_prompt
from coursemate.vector_store import SearchResult, TfidfVectorStore


@dataclass(frozen=True)
class Answer:
    """Assistant answer with retrieval metadata."""

    question: str
    answer: str
    mode: str
    retrieved: list[SearchResult]
    prompt: str
    model_name: str


class CourseMateAssistant:
    """RAG assistant that can answer with or without retrieval."""

    def __init__(self, vector_store: TfidfVectorStore, llm: LanguageModel):
        self.vector_store = vector_store
        self.llm = llm

    @classmethod
    def from_directory(
        cls,
        data_dir: str | Path,
        llm: LanguageModel,
        *,
        chunk_size: int = 110,
        overlap: int = 25,
    ) -> "CourseMateAssistant":
        documents = load_documents(data_dir)
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        return cls(TfidfVectorStore.build(chunks), llm)

    def answer(
        self,
        question: str,
        *,
        top_k: int = 4,
        use_rag: bool = True,
        chat_history: list[dict[str, str]] | None = None,
    ) -> Answer:
        retrieved = self.vector_store.search(question, top_k=top_k) if use_rag else []
        context_blocks = [_format_context(result) for result in retrieved]
        prompt = build_prompt(
            question,
            context_blocks,
            chat_history=chat_history,
            require_context=use_rag,
        )
        response = self.llm.generate(prompt)
        return Answer(
            question=question,
            answer=response,
            mode="rag" if use_rag else "no_rag",
            retrieved=retrieved,
            prompt=prompt,
            model_name=self.llm.name,
        )

    def compare(
        self,
        question: str,
        *,
        top_k: int = 4,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Answer]:
        """Return no-RAG and RAG answers for the same question."""

        return {
            "no_rag": self.answer(
                question,
                top_k=top_k,
                use_rag=False,
                chat_history=chat_history,
            ),
            "rag": self.answer(
                question,
                top_k=top_k,
                use_rag=True,
                chat_history=chat_history,
            ),
        }


def _format_context(result: SearchResult) -> str:
    chunk = result.chunk
    return (
        f"[Source: {chunk.title}; chunk={chunk.chunk_id}; score={result.score:.3f}]\n"
        f"{chunk.text}"
    )
