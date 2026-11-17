import pytest
from pathlib import Path

from coursemate.chunking import chunk_documents
from coursemate.app_services import build_semantic_assistant
from coursemate.documents import Document, load_documents
from coursemate.evaluation import load_cases
from coursemate.ingestion import UploadedDocument
from coursemate.llm import DeterministicCourseLLM
from coursemate.semantic_store import ChromaSemanticVectorStore, SemanticStoreError


def test_chroma_semantic_store_retrieves_relevant_chunk(tmp_path):
    documents = [
        Document(
            doc_id="rag",
            title="RAG Notes",
            source_path="rag.txt",
            text="RAG systems retrieve course documents before generating answers.",
        ),
        Document(
            doc_id="policy",
            title="Policy Notes",
            source_path="policy.txt",
            text="Final project submissions include source code and a demonstration.",
        ),
    ]
    chunks = chunk_documents(documents, chunk_size=20, overlap=0)

    try:
        store = ChromaSemanticVectorStore.from_chunks(chunks, persist_dir=tmp_path)
    except SemanticStoreError as exc:
        pytest.skip(str(exc))

    results = store.search("How does retrieval help answer from course documents?", top_k=1)

    assert results
    assert results[0].chunk.title == "RAG Notes"


def test_semantic_app_backend_uses_chroma_thresholding(tmp_path):
    documents = [
        Document(
            doc_id="rag",
            title="RAG Notes",
            source_path="rag.txt",
            text=(
                "A standard RAG pipeline collects documents, splits them into chunks, "
                "computes embeddings, stores vectors, retrieves relevant chunks, and "
                "sends retrieved context to the LLM."
            ),
        ),
        Document(
            doc_id="policy",
            title="Policy Notes",
            source_path="policy.txt",
            text="Final project submissions include source code, slides, and a demonstration.",
        ),
    ]
    uploads = [
        UploadedDocument(
            filename=Path(document.source_path).name,
            content_hash=document.doc_id,
            document=document,
        )
        for document in documents
    ]

    try:
        result = build_semantic_assistant(
            uploads,
            DeterministicCourseLLM(),
            persist_dir=tmp_path,
            relevance_threshold=0.30,
        )
    except SemanticStoreError as exc:
        pytest.skip(str(exc))

    assert result.assistant.retrieval_backend == "sentence-transformers-chromadb"

    greeting = result.assistant.answer("Hello", use_rag=True)
    assert greeting.mode == "conversation"
    assert greeting.retrieved == []

    valid = result.assistant.answer("What are the steps in a RAG pipeline?", use_rag=True)
    assert valid.evidence_status == "sufficient"
    assert valid.retrieval_backend == "sentence-transformers-chromadb"
    assert {item.chunk.title for item in valid.retrieved} == {"RAG Notes"}

    unanswerable = result.assistant.answer("What is tomorrow's cafeteria menu?", use_rag=True)
    assert unanswerable.evidence_status == "insufficient"
    assert unanswerable.retrieved == []


def test_realistic_evaluation_dataset_has_required_coverage():
    cases = load_cases("data/evaluation/realistic_questions.jsonl")

    assert len(cases) == 30
    categories = {case.category for case in cases}
    assert {
        "factual",
        "conceptual",
        "paraphrased",
        "multi_document",
        "follow_up",
        "unanswerable",
    }.issubset(categories)
    assert all(case.reference_answer for case in cases)


def test_realistic_course_materials_retrieve_across_documents(tmp_path):
    documents = load_documents("data/realistic_course_materials")
    uploads = [
        UploadedDocument(
            filename=Path(document.source_path).name,
            content_hash=document.doc_id,
            document=document,
        )
        for document in documents
    ]

    try:
        result = build_semantic_assistant(
            uploads,
            DeterministicCourseLLM(),
            persist_dir=tmp_path,
            relevance_threshold=0.30,
        )
    except SemanticStoreError as exc:
        pytest.skip(str(exc))

    answer = result.assistant.answer(
        "How do semantic embeddings and RAG work together in CourseMate?",
        use_rag=True,
        top_k=6,
    )
    titles = {item.chunk.title for item in answer.retrieved}

    assert answer.evidence_status == "sufficient"
    assert "Embedding and Vector Store Lab" in titles
    assert "Lecture 2: Retrieval-Augmented Generation" in titles
