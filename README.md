# CourseMate - AI Course Material Assistant

CourseMate is a final-project implementation of a Retrieval-Augmented
Generation (RAG) assistant for course materials. It answers questions from local
lecture notes and policy documents, compares responses with and without RAG, and
includes an evaluation dataset plus paper/slide sources.

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

The implementation is intentionally lightweight. It uses a pure-Python TF-IDF
vector store so the project can run without paid APIs or heavyweight services.
For generation, it supports:

- `deterministic` - reproducible local generator for tests/evaluation;
- `ollama` - local open models such as `llama3.2:3b`;
- `openai` - OpenAI-compatible chat API via environment variables.

## Repository Structure

```text
coursemate/                  Python package
  chunking.py                document chunking
  documents.py               course material loading
  evaluation.py              evaluation metrics and result writer
  llm.py                     LLM adapters and prompt template
  rag.py                     RAG orchestration
  vector_store.py            TF-IDF vector store
data/course_materials/       sample course corpus
data/evaluation/             labeled evaluation questions
docs/final_paper.md          technical paper source
docs/presentation_slides.md  slide deck source
scripts/build_artifacts.py   standard-library PDF generator
tests/                       unit tests
```

## Quick Start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

Build PDF artifacts:

```bash
python3 scripts/build_artifacts.py
```

Generated files are written to `artifacts/`.

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
