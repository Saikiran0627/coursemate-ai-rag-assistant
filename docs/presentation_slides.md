# CourseMate: AI Course Material Assistant

Team: Manish Chowdary Kalluri, Sai Kiran Sikilammetla, Venkata Naga Kiran Katari

---

# Problem

- Students need answers from lecture notes, project instructions, and policies.
- General LLMs may not know the current course materials.
- Unsupported answers can mislead students.
- Goal: answer questions with evidence from course documents.

---

# Selected Project Topic

- Topic 1: Retrieval-Augmented Generation System.
- Required flow: Documents -> Chunking -> Embeddings/Vector Store -> Retrieval -> LLM -> Response.
- Required comparison: answers with RAG vs. answers without RAG.

---

# System Architecture

```text
Course documents
  -> loader
  -> overlapping chunks
  -> TF-IDF vectors
  -> vector store
  -> top-k retrieval
  -> grounded prompt
  -> LLM backend
  -> cited answer
```

---

# Data

- Module 1: LLM basics and hallucination.
- Module 2: RAG pipeline.
- Module 3: agents and tool use.
- Module 4: evaluation methodology.
- Course policy summary.

---

# Prompt Design

- Role: CourseMate, a course-material assistant.
- Context: retrieved chunks with source titles.
- Rules:
  - answer only from provided course context;
  - cite sources;
  - do not invent definitions, deadlines, formulas, or policies;
  - say when evidence is missing.

---

# Model Backends

- Local open model path: Ollama, for example llama3.2:3b.
- Optional OpenAI-compatible backend through environment variables.
- Deterministic local generator for reproducible evaluation.
- Same retrieval pipeline is used regardless of backend.

---

# Evaluation Methodology

- Six labeled test questions.
- Each question has expected source documents and expected keywords.
- Metrics:
  - keyword recall;
  - source hit rate;
  - latency;
  - qualitative success/failure review.
- Baseline: same assistant without retrieval.

---

# Example: RAG Works

Question: What are the main steps in a RAG pipeline?

Retrieved source: Module 2: Retrieval-Augmented Generation.

Expected answer: documents are chunked, represented as vectors, stored, retrieved for a query, and passed with the question to the LLM.

---

# Example: No-RAG Baseline

Without retrieval, the assistant has no course context.

Expected safe behavior: say there is not enough evidence instead of inventing a policy or lecture detail.

This demonstrates why retrieval matters for course-specific QA.

---

# Failure Modes

- TF-IDF can miss paraphrases with little keyword overlap.
- Retrieved chunks may be incomplete for multi-hop questions.
- A real LLM may ignore instructions and hallucinate.
- Source citations must be checked because citation presence does not guarantee support.

---

# Responsible AI and Security

- Do not commit API keys or credentials.
- Use environment variables for hosted model access.
- Avoid sending private student data to unapproved services.
- Show citations and encourage verification.
- Future work: prompt-injection filtering and access control.

---

# Demo Commands

```bash
python3 -m coursemate.cli compare "What are the main steps in a RAG pipeline?"
python3 -m coursemate.cli ask "What should the final project submission include?"
python3 -m coursemate.cli evaluate
```

---

# Conclusion

- CourseMate implements a complete RAG pipeline.
- Retrieval improves grounding over the no-RAG baseline.
- The system includes code, tests, evaluation data, documentation, paper source, and slide source.
- Future improvements: neural embeddings, PDF ingestion, web UI, and multi-model evaluation.
