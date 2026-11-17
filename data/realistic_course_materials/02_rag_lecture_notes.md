# Lecture 2: Retrieval-Augmented Generation

Page 1

Retrieval augmented generation, or RAG, combines information retrieval with
language model generation. A RAG system first retrieves relevant passages from a
document collection and then places those passages into the model prompt. The
model should answer from the retrieved context instead of relying only on its
training data.

Page 2

A typical RAG pipeline has six stages: collect documents, split documents into
overlapping chunks, compute semantic embeddings, store vectors in a vector
database, retrieve the most relevant chunks for a question, and generate an
answer with source citations. Chunk overlap helps preserve meaning across
section boundaries.

Page 3

RAG is especially useful for course assistants because syllabi, slides,
assignments, and policies change during a semester. Retrieval lets the assistant
use current course materials without retraining the model. However, retrieval
can fail when the query is ambiguous, when the document is missing, or when the
retrieved chunks are only weakly related to the question.

Page 4

CourseMate should use a semantic vector database as its primary retrieval
system. It should apply a relevance threshold to avoid treating weak matches as
evidence. If no sufficiently relevant chunks are found, CourseMate should say
that it does not have enough evidence in the uploaded materials.
