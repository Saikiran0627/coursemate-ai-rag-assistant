"""LLM adapters for CourseMate.

The project can run with a local Ollama model for a true LLM-backed demo. The
deterministic fallback keeps tests and evaluation reproducible in environments
where no model server is available.
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class LanguageModel(Protocol):
    """Interface implemented by generation backends."""

    name: str

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt."""


@dataclass
class DeterministicCourseLLM:
    """A local extractive generator used for repeatable offline demos."""

    name: str = "deterministic-course-generator"

    def generate(self, prompt: str) -> str:
        question = _extract_section(prompt, "QUESTION")
        context = _extract_section(prompt, "CONTEXT")
        if not context:
            return (
                "I do not have enough course-material context to answer reliably. "
                "Please provide course notes or enable retrieval."
            )

        answer_context = _remove_source_lines(context)
        sentences = _sentences(answer_context)
        ranked = sorted(
            sentences,
            key=lambda sentence: _overlap_score(question, sentence),
            reverse=True,
        )
        selected = [sentence for sentence in ranked[:3] if sentence.strip()]
        if not selected:
            return (
                "I found course context, but it does not directly answer the question. "
                "The safest next step is to review the cited material."
            )

        answer = " ".join(selected)
        citations = _extract_citations(context)
        if citations:
            answer += "\n\nSources: " + ", ".join(citations[:3])
        return answer


@dataclass
class OllamaLLM:
    """Adapter for a locally running Ollama model."""

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    name: str = "ollama"

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {url}. Start Ollama or use --llm deterministic."
            ) from exc
        return body.get("response", "").strip()


@dataclass
class OpenAICompatibleLLM:
    """Adapter for OpenAI-compatible chat completion APIs."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    name: str = "openai-compatible"

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def build_llm(kind: str, *, model: str | None = None) -> LanguageModel:
    """Construct a language-model backend from a CLI/config name."""

    normalized = kind.lower()
    if normalized == "deterministic":
        return DeterministicCourseLLM()
    if normalized == "ollama":
        return OllamaLLM(model=model or os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    if normalized in {"openai", "openai-compatible"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI-compatible backend")
        return OpenAICompatibleLLM(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    raise ValueError(f"Unknown LLM backend: {kind}")


def build_prompt(question: str, context_blocks: list[str]) -> str:
    """Create the prompt used by both RAG and no-RAG modes."""

    context = "\n\n".join(context_blocks).strip()
    return textwrap.dedent(
        f"""
        You are CourseMate, an AI course-material assistant.
        Answer only from the provided course context. If the context is missing
        or insufficient, say that you do not have enough evidence.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        RESPONSE REQUIREMENTS:
        - Give a concise answer in student-friendly language.
        - Cite the course source titles when context is available.
        - Do not invent policies, deadlines, formulas, or definitions.
        """
    ).strip()


def _extract_section(prompt: str, marker: str) -> str:
    label = f"{marker}:"
    if label not in prompt:
        return ""
    after = prompt.split(label, 1)[1]
    for next_label in ("CONTEXT:", "QUESTION:", "RESPONSE REQUIREMENTS:"):
        if next_label != label and next_label in after:
            after = after.split(next_label, 1)[0]
    return after.strip()


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    pieces: list[str] = []
    current: list[str] = []
    for token in normalized.split(" "):
        current.append(token)
        if token.endswith((".", "?", "!")):
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def _remove_source_lines(context: str) -> str:
    return "\n".join(line for line in context.splitlines() if not line.startswith("[Source:"))


def _overlap_score(question: str, sentence: str) -> int:
    question_terms = {term.lower().strip(".,:;!?()[]") for term in question.split()}
    sentence_terms = {term.lower().strip(".,:;!?()[]") for term in sentence.split()}
    return len((question_terms - {"the", "and", "what", "how", "why"}) & sentence_terms)


def _extract_citations(context: str) -> list[str]:
    citations: list[str] = []
    for line in context.splitlines():
        if line.startswith("[Source:") and line.endswith("]"):
            citation = line.removeprefix("[Source:").removesuffix("]").strip()
            if citation not in citations:
                citations.append(citation)
    return citations
