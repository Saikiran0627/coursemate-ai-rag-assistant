# Lecture 4: Evaluation and Failure Analysis

Slide 1

Evaluation should compare the complete RAG system against a baseline that uses
the same model without retrieved context. This isolates the value of retrieval.
Useful measurements include answer correctness, retrieval relevance, citation
accuracy, hallucination rate, and response latency.

Slide 2

Automated metrics are estimates. Keyword recall checks whether expected concepts
appear in an answer. Source hit rate checks whether expected documents were
retrieved. Latency measures response time. These metrics should be kept separate
from human review because they cannot fully judge faithfulness or usefulness.

Slide 3

Human review should inspect successful answers, failed answers, missing
evidence, and unsupported claims. A hallucination occurs when the assistant
states information that is not supported by the uploaded materials. For
unanswerable questions, the safest behavior is to say that the material does
not contain enough evidence.

Slide 4

Evaluation sets should include factual questions, conceptual questions,
paraphrased questions, multi-document questions, follow-up questions, and
unanswerable questions. The evaluation should record the reference answer,
expected sources, actual retrieved sources, model answer, and reviewer notes.
