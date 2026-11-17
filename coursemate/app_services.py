"""Service helpers used by the Streamlit CourseMate app."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from coursemate.chunking import Chunk, chunk_documents
from coursemate.documents import Document
from coursemate.evaluation import EvaluationCase, EvaluationResult
from coursemate.ingestion import UploadedDocument
from coursemate.llm import LanguageModel
from coursemate.rag import Answer, CourseMateAssistant
from coursemate.semantic_store import ChromaSemanticVectorStore, DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True)
class CorpusBuildResult:
    assistant: CourseMateAssistant
    chunks: list[Chunk]
    document_count: int


@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    latency_seconds: float


def build_semantic_assistant(
    uploaded_documents: list[UploadedDocument],
    llm: LanguageModel,
    *,
    persist_dir: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 160,
    overlap: int = 35,
    relevance_threshold: float = 0.30,
) -> CorpusBuildResult:
    """Build a Chroma-backed assistant from uploaded documents."""

    documents: list[Document] = [item.document for item in uploaded_documents]
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    store = ChromaSemanticVectorStore.from_chunks(
        chunks,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        reset=True,
    )
    return CorpusBuildResult(
        assistant=CourseMateAssistant(store, llm, relevance_threshold=relevance_threshold),
        chunks=chunks,
        document_count=len(documents),
    )


def timed_answer(
    assistant: CourseMateAssistant,
    question: str,
    *,
    top_k: int,
    use_rag: bool,
    chat_history: list[dict[str, str]] | None = None,
    relevance_threshold: float | None = None,
) -> TimedAnswer:
    started = time.perf_counter()
    answer = assistant.answer(
        question,
        top_k=top_k,
        use_rag=use_rag,
        chat_history=chat_history,
        relevance_threshold=relevance_threshold,
    )
    return TimedAnswer(answer=answer, latency_seconds=round(time.perf_counter() - started, 3))


def evaluate_questions(
    assistant: CourseMateAssistant,
    questions: list[EvaluationCase],
    *,
    top_k: int,
) -> list[EvaluationResult]:
    """Evaluate RAG answers for dashboard display."""

    results: list[EvaluationResult] = []
    from coursemate.evaluation import evaluate

    results.extend(evaluate(assistant, questions, top_k=top_k))
    return results


def hallucination_risk(result: EvaluationResult) -> str:
    """A simple, inspectable hallucination-risk heuristic for the dashboard."""

    if result.mode == "no_rag":
        return "High: answer was produced without retrieved course evidence."
    if result.source_hit < 1.0:
        return "Medium: expected source was not fully retrieved."
    if result.keyword_recall < 0.5:
        return "Medium: answer missed many expected concepts."
    return "Low: retrieved expected source and covered expected concepts."
