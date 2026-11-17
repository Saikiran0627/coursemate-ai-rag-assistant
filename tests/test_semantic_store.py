import pytest

from coursemate.chunking import chunk_documents
from coursemate.documents import Document
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
