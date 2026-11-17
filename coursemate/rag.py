"""CourseMate RAG orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from coursemate.chunking import chunk_documents
from coursemate.documents import load_documents
from coursemate.intent import build_retrieval_query, conversational_response
from coursemate.llm import LanguageModel, build_prompt
from coursemate.vector_store import SearchResult, TfidfVectorStore


SEMANTIC_RELEVANCE_THRESHOLD = 0.29
TFIDF_RELEVANCE_THRESHOLD = 0.18
DEFAULT_MAX_EVIDENCE_CHUNKS = 4


@dataclass(frozen=True)
class Answer:
    """Assistant answer with retrieval metadata."""

    question: str
    answer: str
    mode: str
    retrieved: list[SearchResult]
    prompt: str
    model_name: str
    retrieval_query: str = ""
    evidence_status: str = "not_checked"
    retrieval_backend: str = "unknown"


class CourseMateAssistant:
    """RAG assistant that can answer with or without retrieval."""

    def __init__(
        self,
        vector_store: TfidfVectorStore,
        llm: LanguageModel,
        *,
        relevance_threshold: float = SEMANTIC_RELEVANCE_THRESHOLD,
        max_evidence_chunks: int = DEFAULT_MAX_EVIDENCE_CHUNKS,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.max_evidence_chunks = max_evidence_chunks
        self.retrieval_backend = getattr(vector_store, "backend_name", "unknown")

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
        return cls(
            TfidfVectorStore.build(chunks),
            llm,
            relevance_threshold=TFIDF_RELEVANCE_THRESHOLD,
        )

    def answer(
        self,
        question: str,
        *,
        top_k: int = 4,
        use_rag: bool = True,
        chat_history: list[dict[str, str]] | None = None,
        relevance_threshold: float | None = None,
    ) -> Answer:
        direct_response = conversational_response(question)
        if direct_response is not None:
            return Answer(
                question=question,
                answer=direct_response,
                mode="conversation",
                retrieved=[],
                prompt="",
                model_name="coursemate-intent-handler",
                retrieval_query="",
                evidence_status="not_needed",
                retrieval_backend=self.retrieval_backend,
            )

        retrieval_query = build_retrieval_query(question, chat_history)
        retrieval_k = max(top_k * 3, top_k) if use_rag else top_k
        retrieved = self.vector_store.search(retrieval_query, top_k=retrieval_k) if use_rag else []
        if use_rag:
            retrieved = _filter_relevant(
                retrieved,
                query=retrieval_query,
                threshold=self.relevance_threshold if relevance_threshold is None else relevance_threshold,
                max_chunks=self.max_evidence_chunks,
            )
            if not retrieved:
                return Answer(
                    question=question,
                    answer=(
                        "I could not find enough relevant evidence in the uploaded course materials "
                        "to answer that reliably. Try uploading the relevant document or rephrasing "
                        "the question with more course-specific details."
                    ),
                    mode="rag",
                    retrieved=[],
                    prompt="",
                    model_name=self.llm.name,
                    retrieval_query=retrieval_query,
                    evidence_status="insufficient",
                    retrieval_backend=self.retrieval_backend,
                )
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
            retrieval_query=retrieval_query if use_rag else "",
            evidence_status="sufficient" if use_rag else "not_checked",
            retrieval_backend=self.retrieval_backend,
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
    location = _chunk_location(chunk.text)
    location_part = f"; location={location}" if location else ""
    return (
        f"[Source: {chunk.title}; chunk={chunk.chunk_id}{location_part}; score={result.score:.3f}]\n"
        f"{chunk.text}"
    )


def _filter_relevant(
    results: list[SearchResult],
    *,
    query: str,
    threshold: float,
    max_chunks: int,
) -> list[SearchResult]:
    if not results:
        return []
    top_score = results[0].score
    if top_score < threshold:
        return []
    filtered: list[SearchResult] = []
    for result in results:
        if result.score < threshold:
            continue
        # Very strong semantic matches can pass without keyword overlap; borderline
        # chunks need lexical support so citations stay tied to the user's question.
        if result.score >= threshold + 0.20 or _lexical_overlap(query, result.chunk.text) >= 2:
            filtered.append(result)
    return _diversify_sources(filtered, max_chunks=max_chunks)


def _diversify_sources(results: list[SearchResult], *, max_chunks: int) -> list[SearchResult]:
    selected: list[SearchResult] = []
    selected_titles: set[str] = set()
    for result in results:
        if result.chunk.title in selected_titles:
            continue
        selected.append(result)
        selected_titles.add(result.chunk.title)
        if len(selected) == max_chunks:
            return selected
    for result in results:
        if result in selected:
            continue
        selected.append(result)
        if len(selected) == max_chunks:
            break
    return selected


def _chunk_location(text: str) -> str:
    match = re.search(r"\b(Page|Slide)\s+(\d+)\b", text)
    return f"{match.group(1)} {match.group(2)}" if match else ""


def _lexical_overlap(query: str, text: str) -> int:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }
    query_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", query)
        if term.lower() not in stopwords
    }
    text_terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text)}
    return len(query_terms & text_terms)
