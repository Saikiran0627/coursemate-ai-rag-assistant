# Module 2: Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, combines search with language-model
generation. Instead of asking a model to answer only from its parameters, the
system first retrieves relevant documents and places those passages into the
prompt. The model then answers using the retrieved context.

A standard RAG pipeline contains these steps: collect documents, split them into
chunks, compute embeddings or vector representations, store the vectors in a
vector store, retrieve the most relevant chunks for a user question, and send
the question plus retrieved context to the LLM. The final answer should include
citations or source information so the user can verify the response.

Chunk size matters. Chunks that are too small may lose important context, while
chunks that are too large may include irrelevant information and use too much
prompt space. Overlap between chunks helps preserve continuity across section
boundaries.

RAG is useful for course assistants because lecture notes, slides, syllabi, and
assignments change faster than a model's training data. Retrieval lets the
assistant use the current course materials without retraining the model.
