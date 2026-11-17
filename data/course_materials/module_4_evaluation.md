# Module 4: LLM Evaluation

Evaluation measures whether an LLM application works for its intended users. A
good evaluation set contains realistic inputs, expected behaviors, and clear
metrics. For a RAG assistant, useful metrics include retrieval source accuracy,
answer keyword recall, citation correctness, refusal accuracy when evidence is
missing, and response latency.

Evaluation should compare the complete system against a baseline. In a RAG
project, a common baseline is the same model answering without retrieved
documents. If RAG is working, it should improve factual grounding and source
coverage compared with the no-retrieval baseline.

Successful examples show where the system retrieves the right material and
answers correctly. Failure analysis is equally important. The team should record
cases where retrieval misses the right source, the model ignores context, the
answer is incomplete, or the system refuses when it could have answered.

Metrics do not replace human review. Human inspection is needed to judge
faithfulness, clarity, usefulness, and whether citations actually support the
answer.
