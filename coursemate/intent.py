"""Conversation intent and retrieval-query helpers."""

from __future__ import annotations

import re


GREETING_RESPONSES = {
    "greeting": "Hello! I'm CourseMate, your AI course material assistant. How can I help you today?",
    "thanks": "You're welcome! Ask me anytime if you want help with your course materials.",
}

_GREETING_RE = re.compile(r"^(hi|hello|hey|hello there|hi there|hey there)[!. ]*$", re.I)
_THANKS_RE = re.compile(r"^(thanks|thank you|thx|ty|thanks coursemate)[!. ]*$", re.I)
_FOLLOW_UP_RE = re.compile(
    r"\b(it|that|this|they|them|those|these|he|she|there|above|previous|same)\b",
    re.I,
)


def conversational_response(message: str) -> str | None:
    """Return a direct response for simple non-retrieval conversation."""

    normalized = " ".join(message.strip().split())
    if not normalized:
        return "Ask me a question about your course materials when you're ready."
    if _GREETING_RE.match(normalized):
        return GREETING_RESPONSES["greeting"]
    if _THANKS_RE.match(normalized):
        return GREETING_RESPONSES["thanks"]
    return None


def build_retrieval_query(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Rewrite context-dependent follow-ups into a standalone retrieval query.

    This lightweight rewrite avoids another LLM call for latency. It prepends the
    latest user question when the current question contains pronouns or other
    follow-up markers.
    """

    cleaned = question.strip()
    if not chat_history or not _FOLLOW_UP_RE.search(cleaned):
        return cleaned

    previous_user_turns = _supported_previous_user_turns(chat_history)
    if not previous_user_turns:
        return cleaned

    return f"{previous_user_turns[-1]} Follow-up: {cleaned}"


def _supported_previous_user_turns(chat_history: list[dict[str, str]]) -> list[str]:
    supported_turns: list[str] = []
    latest_user: str | None = None
    for item in chat_history:
        role = item.get("role")
        content = item.get("content", "").strip()
        if role == "user" and content:
            latest_user = content
        elif role == "assistant" and latest_user:
            evidence_status = item.get("evidence_status", "")
            unsupported = (
                evidence_status == "insufficient"
                or "could not find enough relevant evidence" in content.lower()
            )
            if not unsupported:
                supported_turns.append(latest_user)
            latest_user = None
    if latest_user:
        supported_turns.append(latest_user)
    return supported_turns
