# CourseMate - AI Course Material Assistant

CourseMate is an interactive Retrieval-Augmented Generation (RAG) web
application for course materials. It lets students upload documents, ask
follow-up questions in a ChatGPT-style interface, compare answers with and
without RAG, and inspect evaluation results with retrieval relevance,
hallucination-risk, and latency signals.

Team members:

1. Manish Chowdary Kalluri
2. Sai Kiran Sikilammetla
3. Venkata Naga Kiran Katari

## Project Topic

Selected topic: **Retrieval-Augmented Generation (RAG) System**

Demonstrated pipeline:

```text
Documents -> Chunking -> Vector Store -> Retrieval -> LLM -> Response
```

The web app uses free and open-source technologies:

- Streamlit for the browser interface.
- Sentence Transformers for semantic embeddings.
- ChromaDB as the local vector database.
- Ollama for a local LLM such as Llama 3.2 3B.
- PDF, PPTX, DOCX, TXT, and Markdown upload parsing.

The Streamlit web application uses Sentence Transformers embeddings and ChromaDB
vector retrieval by default. The original TF-IDF implementation remains
available only as a lightweight CLI/test fallback with a separately calibrated
threshold.

## Repository Structure

```text
coursemate/                  Python package
  app_services.py            web app service layer
  chunking.py                document chunking
  documents.py               course material loading
  evaluation.py              evaluation metrics and result writer
  ingestion.py               upload parsing for PDF/PPTX/DOCX/TXT/MD
  llm.py                     LLM adapters and prompt template
  rag.py                     RAG orchestration
  semantic_store.py          Sentence Transformers + ChromaDB retrieval
  vector_store.py            TF-IDF vector store
app.py                       Streamlit web application
data/course_materials/       sample course corpus
data/evaluation/             labeled evaluation questions
tests/                       unit tests
```

## Quick Start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the Web App

Start Ollama and install the preferred local model:

```bash
ollama pull llama3.2:3b
```

Launch CourseMate:

```bash
streamlit run app.py
```

In the browser:

1. Confirm Ollama is reachable and `llama3.2:3b` is available.
2. Upload PDF, PPTX, DOCX, TXT, or Markdown course materials.
3. Click **Build / refresh semantic index**.
4. Ask questions in the chat tab.
5. Use **RAG Comparison** to view no-RAG and RAG answers side by side.
6. Use **Evaluation** to inspect test cases, answer quality, source relevance,
   hallucination-risk analysis, and response times.

CourseMate handles simple greetings and thanks directly without retrieval, so
messages such as "Hello" do not produce irrelevant citations. The sidebar also
includes an **Evidence relevance threshold** control. Chunks below that
similarity threshold are not used as evidence, and CourseMate will say when the
uploaded materials do not contain enough support for an answer.

If Ollama is unavailable, deterministic demo mode can be used to verify
retrieval and UI behavior, but it is not a substitute for testing real local LLM
responses.

## CLI Utilities

Ask a question with retrieval:

```bash
python3 -m coursemate.cli ask "What should the final project submission include?"
```

Compare no-RAG vs. RAG:

```bash
python3 -m coursemate.cli compare "What are the main steps in a RAG pipeline?"
```

Run evaluation:

```bash
python3 -m coursemate.cli evaluate
```

Run tests:

```bash
pytest
```

## Using a Local Open LLM with Ollama

Install and start Ollama, then pull a model:

```bash
ollama pull llama3.2:3b
```

Run CourseMate with the Ollama backend:

```bash
python3 -m coursemate.cli --llm ollama --model llama3.2:3b ask \
  "Why should a course assistant use retrieval?"
```

## Using an OpenAI-Compatible API

Do not commit API keys. Set them in your shell:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
python3 -m coursemate.cli --llm openai ask "What metrics evaluate RAG?"
```

## Evaluation Design

The evaluation compares the same questions in two modes:

1. No-RAG baseline: the model receives no course context.
2. RAG system: the retriever supplies top-k course chunks.

Metrics:

- keyword recall against expected answer concepts;
- source hit rate against expected course documents;
- latency per response;
- qualitative review of successful and failed examples.

This directly addresses the final-project question: why does the system work,
when does it fail, and how do we know whether it is good?

## Responsible AI and Security Notes

- Do not include API keys, passwords, credentials, or sensitive student data.
- Answers should cite retrieved course materials.
- If evidence is missing, the assistant should say so instead of inventing.
- Hosted model use should follow course and institutional privacy rules.
