# Module 1: Large Language Model Basics

Large language models are neural networks trained to predict text. During
pretraining, the model observes a very large corpus and learns statistical
patterns about language, facts, code, and reasoning traces. During instruction
tuning, the model is further trained to follow user requests and produce safer,
more helpful responses.

Prompt design is the process of giving the model clear task instructions,
context, constraints, and examples. A strong prompt tells the model what role it
is playing, what information it can use, and what output format is expected.
For factual course assistants, prompts should explicitly tell the model not to
invent answers when evidence is missing.

LLMs can fail by hallucinating facts, following ambiguous instructions, or
overgeneralizing from training data. Reliable systems reduce these failures by
grounding generation in trusted context, validating outputs, and evaluating
behavior on representative test cases.

Temperature controls randomness during decoding. Low temperature values make
responses more consistent and focused, while higher values can make responses
more diverse but less predictable. For educational question answering, a low
temperature is usually preferred.
