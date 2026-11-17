"""Streamlit web application for CourseMate."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from coursemate.app_services import (
    build_semantic_assistant,
    evaluate_questions,
    hallucination_risk,
    timed_answer,
)
from coursemate.documents import load_documents
from coursemate.evaluation import EvaluationCase, load_cases, summarize
from coursemate.ingestion import DocumentIngestionError, UploadedDocument, parse_uploaded_file
from coursemate.llm import (
    DeterministicCourseLLM,
    OllamaLLM,
    list_ollama_models,
    ollama_generate_smoke_test,
)
from coursemate.rag import SEMANTIC_RELEVANCE_THRESHOLD


APP_TITLE = "CourseMate"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_EVAL_FILE = Path("data/evaluation/questions.jsonl")
REALISTIC_EVAL_FILE = Path("data/evaluation/realistic_questions.jsonl")
DEFAULT_SAMPLE_DIR = Path("data/course_materials")
PAGES = [
    "Home",
    "AI Chat",
    "My Documents",
    "RAG Comparison",
    "Evaluation",
    "About CourseMate",
]


st.set_page_config(page_title="CourseMate", page_icon="🎓", layout="wide")


def main() -> None:
    init_state()
    inject_theme()
    page, llm = render_sidebar()
    st.session_state["active_llm"] = llm
    attach_llm(llm)

    if page == "Home":
        render_home_page()
    elif page == "AI Chat":
        render_chat_tab()
    elif page == "My Documents":
        render_documents_tab()
    elif page == "RAG Comparison":
        render_comparison_tab()
    elif page == "Evaluation":
        render_evaluation_tab()
    elif page == "About CourseMate":
        render_about_page()


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("uploaded_docs", [])
    st.session_state.setdefault("assistant", None)
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_index_signature", None)
    st.session_state.setdefault("page", "Home")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cm-navy: #0b1f33;
            --cm-teal: #00a7a5;
            --cm-teal-dark: #087f7d;
            --cm-bg: #f6f8fb;
            --cm-card: #ffffff;
            --cm-border: #dbe5ef;
            --cm-muted: #5d6b7a;
        }
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #f5f7fb 48%, #eef4f8 100%);
            color: var(--cm-navy);
        }
        section[data-testid="stSidebar"] {
            background: #0b1f33;
        }
        section[data-testid="stSidebar"] * {
            color: #eef7fb;
        }
        section[data-testid="stSidebar"] .stButton button {
            border: 1px solid rgba(255,255,255,.22);
            background: rgba(255,255,255,.08);
            color: #ffffff;
        }
        .cm-hero {
            border: 1px solid var(--cm-border);
            border-radius: 28px;
            padding: 2.2rem;
            background:
                radial-gradient(circle at top right, rgba(0,167,165,.20), transparent 34%),
                linear-gradient(135deg, #0b1f33 0%, #123a54 55%, #0e5b62 100%);
            color: white;
            box-shadow: 0 24px 70px rgba(11,31,51,.16);
        }
        .cm-hero h1 {
            margin: 0 0 .75rem 0;
            font-size: clamp(2.2rem, 5vw, 4rem);
            letter-spacing: -0.05em;
        }
        .cm-hero p {
            color: rgba(255,255,255,.86);
            max-width: 780px;
            font-size: 1.08rem;
        }
        .cm-card {
            border: 1px solid var(--cm-border);
            border-radius: 22px;
            padding: 1.25rem;
            background: rgba(255,255,255,.92);
            box-shadow: 0 12px 34px rgba(11,31,51,.08);
            min-height: 100%;
        }
        .cm-card h3 {
            margin-top: 0;
            color: var(--cm-navy);
        }
        .cm-muted {
            color: var(--cm-muted);
        }
        .cm-pill {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: rgba(0,167,165,.12);
            color: #075e61;
            border: 1px solid rgba(0,167,165,.28);
            font-weight: 650;
            font-size: .85rem;
            margin-right: .35rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.9);
            border: 1px solid var(--cm-border);
            border-radius: 18px;
            padding: .85rem;
        }
        .stButton button[kind="primary"] {
            background: var(--cm-teal);
            border: 1px solid var(--cm-teal-dark);
            color: white;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## CourseMate")
        st.caption("AI course material assistant")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")

        st.divider()
        st.markdown("### Model")
        backend = st.radio("LLM backend", ["Ollama", "Deterministic demo"], index=0)
        model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)

        if backend == "Ollama":
            models = list_ollama_models()
            if models:
                st.success("Ollama is reachable.")
                if model_name in models:
                    st.info(f"Model available: {model_name}")
                else:
                    st.warning(
                        f"{model_name} is not installed. Available models: {', '.join(models)}"
                    )
                if st.button("Run Ollama response test"):
                    ok, message = ollama_generate_smoke_test(model=model_name)
                    if ok:
                        st.success(f"Ollama responded: {message}")
                    else:
                        st.error(message)
            else:
                st.error("Ollama is not reachable. Start Ollama and pull llama3.2:3b.")
            llm = OllamaLLM(model=model_name)
        else:
            st.warning("Using deterministic demo mode, not a real local LLM.")
            llm = DeterministicCourseLLM()

        with st.expander("Advanced retrieval settings", expanded=False):
            top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
            st.session_state["top_k"] = top_k
            relevance_threshold = st.slider(
                "Evidence relevance threshold",
                min_value=0.0,
                max_value=0.8,
                value=SEMANTIC_RELEVANCE_THRESHOLD,
                step=0.01,
                help=(
                    "Chunks below this calibrated similarity score are not used as evidence. "
                    "Raise it to be stricter; lower it if valid uploaded material is missed."
                ),
            )
            st.session_state["relevance_threshold"] = relevance_threshold

        doc_count = len(st.session_state["uploaded_docs"])
        chunk_count = len(st.session_state["chunks"])
        st.divider()
        st.markdown("### App status")
        st.metric("Documents", doc_count)
        st.metric("Indexed chunks", chunk_count)
        assistant = st.session_state.get("assistant")
        if assistant is not None:
            st.caption(f"Retrieval backend: `{getattr(assistant, 'retrieval_backend', 'unknown')}`")
        if st.session_state["assistant"] is None:
            st.info("Add documents and build the semantic index before chatting.")
        return page, llm


def add_uploaded_files(uploaded_files) -> None:
    existing_hashes = {item.content_hash for item in st.session_state["uploaded_docs"]}
    added = 0
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        try:
            parsed = parse_uploaded_file(uploaded_file.name, content)
        except DocumentIngestionError as exc:
            st.error(f"{uploaded_file.name}: {exc}")
            continue
        if parsed.content_hash in existing_hashes:
            st.info(f"Skipped duplicate file: {uploaded_file.name}")
            continue
        st.session_state["uploaded_docs"].append(parsed)
        existing_hashes.add(parsed.content_hash)
        added += 1
    if added:
        st.success(f"Added {added} document(s). Click 'Build / refresh semantic index'.")


def render_home_page() -> None:
    st.markdown(
        """
        <div class="cm-hero">
            <span class="cm-pill">Semantic RAG</span>
            <span class="cm-pill">Source-grounded answers</span>
            <span class="cm-pill">Local Ollama model</span>
            <h1>CourseMate</h1>
            <p>
                A focused AI assistant for course materials. Upload lecture notes,
                slides, policy documents, or handouts; build a semantic index; and
                ask questions with transparent source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    left, middle, right = st.columns(3)
    with left:
        st.markdown(
            """
            <div class="cm-card">
                <h3>1. Add documents</h3>
                <p class="cm-muted">Upload PDF, PPTX, DOCX, TXT, or Markdown course materials.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with middle:
        st.markdown(
            """
            <div class="cm-card">
                <h3>2. Build the index</h3>
                <p class="cm-muted">CourseMate chunks documents, embeds them, and stores vectors in ChromaDB.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="cm-card">
                <h3>3. Ask with citations</h3>
                <p class="cm-muted">Get grounded answers, inspect retrieved chunks, and compare RAG vs no-RAG.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    status_cols = st.columns(4)
    status_cols[0].metric("Documents", len(st.session_state["uploaded_docs"]))
    status_cols[1].metric("Indexed chunks", len(st.session_state["chunks"]))
    status_cols[2].metric("Backend", getattr(st.session_state.get("assistant"), "retrieval_backend", "Not indexed"))
    status_cols[3].metric("Evaluation cases", len(load_cases(REALISTIC_EVAL_FILE)))

    cta_left, cta_right = st.columns([1, 1])
    with cta_left:
        if st.button("Open My Documents", type="primary", use_container_width=True):
            st.session_state["page"] = "My Documents"
            st.rerun()
    with cta_right:
        if st.button("Start AI Chat", use_container_width=True):
            st.session_state["page"] = "AI Chat"
            st.rerun()


def load_sample_documents() -> None:
    existing_hashes = {item.content_hash for item in st.session_state["uploaded_docs"]}
    added = 0
    for document in load_documents(DEFAULT_SAMPLE_DIR):
        content_hash = f"sample-{document.doc_id}"
        if content_hash in existing_hashes:
            continue
        st.session_state["uploaded_docs"].append(
            UploadedDocument(
                filename=Path(document.source_path).name,
                content_hash=content_hash,
                document=document,
            )
        )
        added += 1
    st.success(f"Loaded {added} sample document(s).")


def build_index(llm) -> None:
    docs = st.session_state["uploaded_docs"]
    if not docs:
        st.error("Upload or load documents before building the index.")
        return

    persist_dir = Path(tempfile.gettempdir()) / "coursemate_chroma" / st.session_state["session_id"]
    with st.spinner("Embedding documents and building ChromaDB index..."):
        try:
            result = _build_index_with_retry(docs, llm, persist_dir)
        except Exception as exc:
            st.error(f"Could not build semantic index: {exc}")
            return
    st.session_state["assistant"] = result.assistant
    st.session_state["chunks"] = result.chunks
    st.session_state["last_index_signature"] = tuple(item.content_hash for item in docs)
    st.success(f"Indexed {result.document_count} document(s) into {len(result.chunks)} chunks.")


def attach_llm(llm) -> None:
    assistant = st.session_state.get("assistant")
    if assistant is not None:
        assistant.llm = llm


def _build_index_with_retry(docs, llm, persist_dir: Path):
    try:
        return build_semantic_assistant(
            docs,
            llm,
            persist_dir=persist_dir,
            relevance_threshold=st.session_state["relevance_threshold"],
        )
    except Exception as exc:
        message = str(exc).lower()
        recoverable_chroma_error = (
            "no such table: tenants" in message
            or "readonly database" in message
            or "attempt to write a readonly database" in message
        )
        if not recoverable_chroma_error:
            raise
        _reset_index_state(delete_store=True)
        persist_dir = Path(tempfile.gettempdir()) / "coursemate_chroma" / st.session_state["session_id"]
        return build_semantic_assistant(
            docs,
            llm,
            persist_dir=persist_dir,
            relevance_threshold=st.session_state["relevance_threshold"],
        )


def _reset_index_state(*, delete_store: bool) -> None:
    if delete_store:
        persist_dir = Path(tempfile.gettempdir()) / "coursemate_chroma" / st.session_state["session_id"]
        shutil.rmtree(persist_dir, ignore_errors=True)
    st.session_state["assistant"] = None
    st.session_state["chunks"] = []
    st.session_state["last_index_signature"] = None
    st.session_state["session_id"] = uuid.uuid4().hex


def render_chat_tab() -> None:
    st.title("AI Chat")
    st.caption("Ask questions about indexed course materials and inspect citations.")
    assistant = st.session_state.get("assistant")
    if assistant is None:
        st.info("Upload documents and build the semantic index to start chatting.")
        return

    if st.button("Clear conversation"):
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about your course materials")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {
            "role": item["role"],
            "content": item["content"],
            "evidence_status": item.get("evidence_status", ""),
        }
        for item in st.session_state["messages"][:-1]
        if item["role"] in {"user", "assistant"}
    ]
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            try:
                timed = timed_answer(
                    assistant,
                    question,
                    top_k=st.session_state["top_k"],
                    use_rag=True,
                    chat_history=history,
                    relevance_threshold=st.session_state["relevance_threshold"],
                )
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                return
        st.markdown(timed.answer.answer)
        st.caption(f"Response time: {timed.latency_seconds}s")
        if timed.answer.retrieved:
            render_sources(timed.answer.retrieved)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": timed.answer.answer,
            "sources": timed.answer.retrieved,
            "evidence_status": timed.answer.evidence_status,
        }
    )


def render_comparison_tab() -> None:
    st.title("RAG Comparison")
    st.caption("Compare the same Ollama model with and without retrieved course context.")
    assistant = st.session_state.get("assistant")
    if assistant is None:
        st.info("Build an index before comparing RAG and no-RAG answers.")
        return

    question = st.text_area("Comparison question", height=90)
    if st.button("Compare answers", type="primary") and question.strip():
        history = [
            {
                "role": item["role"],
                "content": item["content"],
                "evidence_status": item.get("evidence_status", ""),
            }
            for item in st.session_state["messages"]
            if item["role"] in {"user", "assistant"}
        ]
        left, right = st.columns(2)
        with left:
            st.subheader("Without RAG")
            try:
                no_rag = timed_answer(
                    assistant,
                    question,
                    top_k=st.session_state["top_k"],
                    use_rag=False,
                    chat_history=history,
                    relevance_threshold=st.session_state["relevance_threshold"],
                )
                st.markdown(no_rag.answer.answer)
                st.caption(f"Response time: {no_rag.latency_seconds}s")
            except Exception as exc:
                st.error(f"No-RAG answer failed: {exc}")
        with right:
            st.subheader("With RAG")
            try:
                rag = timed_answer(
                    assistant,
                    question,
                    top_k=st.session_state["top_k"],
                    use_rag=True,
                    chat_history=history,
                    relevance_threshold=st.session_state["relevance_threshold"],
                )
                st.markdown(rag.answer.answer)
                st.caption(f"Response time: {rag.latency_seconds}s")
                if rag.answer.retrieved:
                    render_sources(rag.answer.retrieved)
            except Exception as exc:
                st.error(f"RAG answer failed: {exc}")


def render_evaluation_tab() -> None:
    st.title("Evaluation")
    st.caption("Review automated estimates separately from human judgment.")
    assistant = st.session_state.get("assistant")
    if assistant is None:
        st.info("Build an index before running evaluation.")
        return

    st.write("Evaluate answer quality, retrieval relevance, hallucination risk, and latency.")
    eval_dataset = st.selectbox(
        "Evaluation dataset",
        [
            "Original quick checks",
            "Realistic 30-question course evaluation",
        ],
        index=1,
    )
    selected_eval_file = (
        REALISTIC_EVAL_FILE
        if eval_dataset == "Realistic 30-question course evaluation"
        else DEFAULT_EVAL_FILE
    )
    use_default_cases = st.checkbox("Use selected built-in test cases", value=True)
    custom_question = st.text_input("Optional custom test question")
    custom_keywords = st.text_input("Expected keywords for custom question, comma-separated")
    custom_sources = st.text_input("Expected source titles for custom question, comma-separated")

    if st.button("Run evaluation", type="primary"):
        cases: list[EvaluationCase] = []
        if use_default_cases:
            cases.extend(load_cases(selected_eval_file))
        if custom_question.strip():
            cases.append(
                EvaluationCase(
                    id="custom",
                    question=custom_question.strip(),
                    expected_keywords=_split_csv(custom_keywords),
                    expected_sources=_split_csv(custom_sources),
                )
            )
        if not cases:
            st.error("Select built-in cases or enter a custom question.")
            return

        with st.spinner("Running evaluation..."):
            try:
                results = evaluate_questions(
                    assistant,
                    cases,
                    top_k=st.session_state["top_k"],
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return

        summary = summarize(results)
        st.subheader("Summary")
        st.dataframe(pd.DataFrame(summary).T, use_container_width=True)

        rows = []
        case_lookup = {case.id: case for case in cases}
        for result in results:
            case = case_lookup.get(result.case_id)
            rows.append(
                {
                    "case_id": result.case_id,
                    "category": case.category if case else "",
                    "mode": result.mode,
                    "keyword_recall": result.keyword_recall,
                    "source_hit": result.source_hit if result.source_hit_applicable else "N/A",
                    "correct_refusal": (
                        result.correct_refusal
                        if result.correct_refusal is not None
                        else "N/A"
                    ),
                    "latency_seconds": result.latency_seconds,
                    "retrieved_sources": ", ".join(result.retrieved_sources),
                    "hallucination_risk": hallucination_risk(result),
                    "human_factual_correctness": result.human_factual_correctness,
                    "human_citation_support": result.human_citation_support,
                    "human_completeness": result.human_completeness,
                    "human_notes": result.human_notes,
                    "reference_answer": case.reference_answer if case else "",
                    "answer": result.answer,
                }
            )
        st.subheader("Detailed Results")
        st.caption(
            "Automated keyword/source metrics are estimates. Keep them separate from "
            "human review of faithfulness, citation support, and usefulness."
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.subheader("Latency")
        st.bar_chart(pd.DataFrame(rows), x="case_id", y="latency_seconds", color="mode")


def render_documents_tab() -> None:
    st.title("My Documents")
    st.caption("Upload course materials, build the semantic index, and manage indexed files.")
    uploaded_files = st.file_uploader(
        "Upload PDF, PPTX, DOCX, TXT, or Markdown files",
        type=["pdf", "pptx", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    doc_actions = st.columns([1, 1, 2])
    with doc_actions[0]:
        if uploaded_files and st.button("Add uploaded files", type="primary", use_container_width=True):
            add_uploaded_files(uploaded_files)
    with doc_actions[1]:
        if st.button("Load sample materials", use_container_width=True):
            load_sample_documents()
    with doc_actions[2]:
        if st.button("Build / refresh semantic index", use_container_width=True):
            build_index(st.session_state.get("active_llm", DeterministicCourseLLM()))

    st.divider()
    docs: list[UploadedDocument] = st.session_state["uploaded_docs"]
    if not docs:
        st.info("No documents loaded yet.")
        return

    st.subheader("Loaded documents")
    for index, item in enumerate(list(docs)):
        with st.expander(f"{item.filename} ({len(item.document.text.split())} words)"):
            st.caption(f"Hash: {item.content_hash}")
            st.text_area("Preview", item.document.text[:3000], height=180, key=f"preview-{index}")
            if st.button("Remove document", key=f"remove-{index}"):
                st.session_state["uploaded_docs"].pop(index)
                _reset_index_state(delete_store=True)
                st.rerun()

    if st.button("Clear all documents"):
        st.session_state["uploaded_docs"] = []
        _reset_index_state(delete_store=True)
        st.session_state["messages"] = []
        st.rerun()

    st.subheader("Indexed chunks")
    chunks = st.session_state.get("chunks", [])
    if chunks:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "source": chunk.source_path,
                        "words": chunk.end_word - chunk.start_word,
                    }
                    for chunk in chunks
                ]
            ),
            use_container_width=True,
        )
    else:
        st.info("No chunks indexed yet. Build the semantic index after loading documents.")


def render_about_page() -> None:
    st.title("About CourseMate")
    st.markdown(
        """
        CourseMate is a retrieval-augmented generation application for course
        materials. It helps students ask questions about uploaded documents while
        keeping answers grounded in retrieved source passages.
        """
    )
    st.subheader("RAG architecture")
    st.code("Documents -> Chunking -> Embeddings -> Vector Store -> Retrieval -> LLM -> Response")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="cm-card">
                <h3>Core technologies</h3>
                <p>Python, Streamlit, Sentence Transformers, all-MiniLM-L6-v2,
                ChromaDB, and Ollama with llama3.2:3b.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="cm-card">
                <h3>Supported documents</h3>
                <p>PDF, PowerPoint, Word, plain text, and Markdown files can be
                uploaded and indexed for semantic retrieval.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.subheader("Capabilities")
    st.markdown(
        """
        - Course-material question answering with citations.
        - Conversation history and follow-up handling.
        - RAG vs no-RAG comparison using the same model.
        - Evaluation dashboard with automated metrics and human-review fields.
        - Insufficient-evidence responses for unsupported questions.
        """
    )
    st.subheader("Limitations")
    st.markdown(
        """
        - Automated metrics estimate quality but do not replace human review.
        - Retrieval quality depends on document clarity, chunking, and embedding similarity.
        - Local Ollama response time depends on available hardware.
        - Important policy answers should be verified against official course materials.
        """
    )


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("Sources and retrieved chunks", expanded=False):
        for item in sources:
            location = _chunk_location(item.chunk.text)
            location_part = f" · {location}" if location else ""
            st.markdown(
                f"**{item.chunk.title}**{location_part} · `{item.chunk.chunk_id}` · "
                f"score `{item.score:.3f}`"
            )
            st.caption(item.chunk.source_path)
            st.write(item.chunk.text)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _chunk_location(text: str) -> str:
    match = re.search(r"\b(Page|Slide)\s+(\d+)\b", text)
    return f"{match.group(1)} {match.group(2)}" if match else ""


if __name__ == "__main__":
    main()
