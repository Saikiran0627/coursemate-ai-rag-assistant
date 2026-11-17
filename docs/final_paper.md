# CourseMate: An AI Course Material Assistant Using Retrieval-Augmented Generation

Team members: Manish Chowdary Kalluri, Sai Kiran Sikilammetla, Venkata Naga Kiran Katari

## Abstract

CourseMate is an LLM-powered course material assistant that answers student
questions using a retrieval-augmented generation (RAG) architecture. The system
loads course documents, chunks them into passages, creates local TF-IDF vector
representations, retrieves relevant passages for a user question, and sends the
retrieved context to a language-model generation layer. CourseMate can run with
a local Ollama model for a fully LLM-backed demonstration and includes a
deterministic local generator for reproducible evaluation. We evaluate the
system against a no-retrieval baseline using six representative course
questions. Results show that RAG substantially improves source grounding and
keyword coverage because the assistant has access to current course materials
instead of relying only on model parameters.

## 1. Introduction and Problem Statement

Students often need quick answers about lectures, assignments, project
requirements, policies, and technical concepts. Searching through slides and
documents manually can be slow, especially when course materials are spread
across multiple files. A general-purpose LLM can explain common concepts, but it
may not know the exact content of a specific course and may hallucinate project
requirements or deadlines.

The objective of CourseMate is to build a question-answering assistant that
grounds answers in course materials. RAG is appropriate because it separates
course knowledge from the language model: the model does not need to memorize
every document, and the corpus can be updated without retraining.

## 2. Background and Related Work

Large language models generate text from learned statistical patterns. They are
useful for explanation and synthesis, but they can produce unsupported answers
when asked about private or recently updated information. Prompt engineering can
reduce this risk by instructing the model to answer only from evidence, but a
prompt alone cannot provide missing course-specific facts.

Retrieval-augmented generation addresses this limitation by retrieving external
context before generation. The common RAG workflow is documents -> chunking ->
embeddings or vectors -> vector store -> retrieval -> LLM -> response.
CourseMate implements this workflow with a local TF-IDF vector store. TF-IDF is
not as semantically rich as neural embeddings, but it is transparent,
lightweight, reproducible, and sufficient for a small educational corpus. The
generation layer supports Ollama, allowing open local models such as Llama 3.2,
Mistral, or Phi to be used without commercial API access.

Relevant references include Lewis et al. (2020) on retrieval-augmented
generation, the Ollama documentation for local model serving, and standard
information retrieval concepts such as term frequency-inverse document
frequency and cosine similarity.

## 3. System Architecture and Design

CourseMate contains five major components:

- Document loader: reads Markdown and text course materials.
- Chunker: splits documents into overlapping word chunks.
- Vector store: computes TF-IDF vectors and cosine similarity scores.
- Retriever: selects the top-k chunks for a user question.
- Generation layer: builds a grounded prompt and calls an LLM backend.

Architecture diagram:

```text
User Question
     |
     v
CourseMate CLI / App Interface
     |
     v
TF-IDF Query Vector
     |
     v
Vector Store Search
     |
     v
Top-k Course Chunks + Source Metadata
     |
     v
Grounded Prompt Template
     |
     v
LLM Backend (Ollama, OpenAI-compatible, or deterministic evaluator)
     |
     v
Answer with Citations
```

The assistant also supports a no-RAG mode. In no-RAG mode, the same generation
template is used but no retrieved course context is supplied. This creates a
controlled baseline for evaluating the effect of retrieval.

## 4. Data and Methodology

The demonstration corpus consists of five course-material documents:

- Module 1: Large Language Model Basics
- Module 2: Retrieval-Augmented Generation
- Module 3: Agents and Tool Use
- Module 4: LLM Evaluation
- Course Policy Summary

These files represent lecture notes and project policy content. They were
written as realistic sample materials for the final project demonstration. The
system preprocesses the documents by loading Markdown, extracting titles,
splitting text into overlapping chunks of 110 words with 25 words of overlap,
tokenizing text, removing common stopwords, and computing TF-IDF vectors.

Prompt design:

- The system role tells the model it is CourseMate, a course-material assistant.
- The prompt includes retrieved context with source titles and similarity
  scores.
- The instructions require concise student-friendly answers.
- The prompt explicitly says not to invent definitions, deadlines, formulas, or
  policies when evidence is missing.

Model configuration:

- Recommended demo backend: Ollama with an open local model such as
  llama3.2:3b.
- Reproducible evaluation backend: deterministic-course-generator.
- Optional hosted backend: OpenAI-compatible chat completions through
  environment variables, without committing API keys.

## 5. Implementation

CourseMate is implemented in Python with no required third-party dependencies.
The project structure is:

- `coursemate/documents.py`: document loading and title extraction.
- `coursemate/chunking.py`: overlapping chunk creation.
- `coursemate/vector_store.py`: TF-IDF vector construction and cosine search.
- `coursemate/llm.py`: generation adapters and grounded prompt construction.
- `coursemate/rag.py`: orchestration of retrieval and answering.
- `coursemate/evaluation.py`: evaluation cases, metrics, and result summaries.
- `coursemate/cli.py`: command-line interface for ask, compare, and evaluate.

Important technical decisions:

1. Local retrieval was implemented without external services so the system can
   run in limited student environments.
2. The LLM interface is abstracted so the same RAG system can use a local open
   model, a hosted compatible model, or the deterministic evaluator.
3. Source metadata is preserved in every chunk so answers can cite the material
   used.
4. The evaluation compares RAG and no-RAG using the same questions and model
   backend, isolating retrieval as the experimental variable.

Challenges included balancing simplicity with realism. A production system
would likely use neural embeddings and a persistent vector database. For this
project, TF-IDF gives a clear, inspectable implementation while preserving the
core RAG concepts.

## 6. Evaluation and Results

The evaluation set contains six questions covering RAG architecture, course
policy, hallucination prevention, evaluation metrics, and tool-use safety. Each
case includes expected keywords and expected source documents.

Metrics:

- Keyword recall: fraction of expected keywords appearing in the answer.
- Source hit rate: fraction of expected source documents retrieved.
- Latency: time to answer each question.
- Qualitative review: inspection of successful and failed responses.

Example test case:

- Question: What are the main steps in a RAG pipeline?
- Expected source: Module 2: Retrieval-Augmented Generation
- Expected keywords: documents, chunks, vector, retrieve, LLM

Expected result pattern:

- No-RAG baseline: refuses or gives an incomplete answer because no context is
  supplied.
- RAG system: retrieves Module 2 and answers with the document-to-chunk-to-vector
  to-retrieval-to-LLM workflow.

The reproducible evaluation can be run with:

```bash
python3 -m coursemate.cli evaluate
```

In local testing with the deterministic backend, RAG improves both source hit
rate and keyword recall over the no-RAG baseline. Exact results are written to
`artifacts/evaluation_results.json` when the evaluation command is run.

## 7. Successful and Failed Responses

Successful response example:

- Question: What should the final project submission include?
- Retrieval: Course Policy Summary.
- Result: The assistant identifies the paper, presentation slides, source code,
  and demonstration requirements.

Another successful response:

- Question: What metrics are useful for evaluating a RAG assistant?
- Retrieval: Module 4: LLM Evaluation.
- Result: The assistant mentions retrieval source accuracy, keyword recall,
  citation correctness, refusal accuracy, and latency.

Failure case:

- Question: Ask for a detail not present in the materials, such as the exact
  classroom number for the final presentation.
- Expected behavior: The assistant should refuse or say the materials do not
  contain enough evidence.
- Possible failure: A less constrained LLM may invent a room number. The prompt
  and evaluation procedure are designed to detect this type of hallucination.

Retrieval failure mode:

- If the user asks with vocabulary that does not overlap the course documents,
  TF-IDF may retrieve weak context. Neural embeddings could improve semantic
  matching for paraphrased questions.

## 8. Discussion

CourseMate demonstrates why RAG is useful for course-specific question
answering. The model does not need to know the course policy from training; it
uses retrieved evidence. This makes the system more maintainable and easier to
audit. The citations help students verify answers rather than blindly trusting a
chatbot.

The project also shows limitations. TF-IDF retrieval works well when questions
share vocabulary with the documents, but it is weaker for synonyms and
multi-hop questions. The deterministic generator is useful for evaluation but
less fluent than a real LLM. Local open models may vary in speed and answer
quality depending on hardware.

## 9. Responsible AI and Security

CourseMate should not be treated as an authoritative replacement for the
instructor. It should show sources and encourage verification. The system avoids
committing secrets and expects API keys to be provided through environment
variables when hosted models are used. If real student data were added, privacy
controls would be required, including access control, data minimization, audit
logs, and removal of sensitive information from prompts.

Risks and safeguards:

- Hallucination: mitigated by retrieval, source citations, and refusal
  instructions.
- Privacy leakage: avoid uploading private course or student data to unapproved
  hosted models.
- Prompt injection: future versions should filter retrieved content and separate
  trusted system instructions from untrusted documents.
- Overreliance: the interface should tell students to verify important policy
  answers against official course materials.

## 10. Conclusion and Future Work

CourseMate is a working RAG application that answers course-material questions,
compares RAG with a no-RAG baseline, and includes evaluation code and test
cases. The main finding is that retrieval improves grounding because answers
are based on the current course corpus rather than only the language model's
parameters.

Future improvements include replacing TF-IDF with neural embeddings, adding a
web interface, supporting PDF slide ingestion, adding instructor-managed
document uploads, evaluating multiple open local models, and implementing
stronger prompt-injection defenses.

## 11. References

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
  Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., and Kiela, D.
  Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS,
  2020.
- Ollama documentation. https://ollama.com/
- Manning, C. D., Raghavan, P., and Schutze, H. Introduction to Information
  Retrieval. Cambridge University Press, 2008.
- OpenAI API documentation. https://platform.openai.com/docs/
- Course final project prompt supplied for this assignment.
