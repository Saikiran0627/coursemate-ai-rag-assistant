"""Evaluation utilities for CourseMate."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from coursemate.rag import CourseMateAssistant


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    reference_answer: str = ""
    category: str = "general"
    notes: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    mode: str
    question: str
    answer: str
    keyword_recall: float
    source_hit: float | None
    source_hit_applicable: bool
    retrieved_sources: list[str]
    latency_seconds: float
    correct_refusal: bool | None = None
    human_factual_correctness: str = "not_reviewed"
    human_citation_support: str = "not_reviewed"
    human_completeness: str = "not_reviewed"
    human_notes: str = ""


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            cases.append(EvaluationCase(**payload))
        except TypeError as exc:
            raise ValueError(f"Invalid evaluation case on line {line_number}") from exc
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def evaluate(
    assistant: CourseMateAssistant,
    cases: list[EvaluationCase],
    *,
    top_k: int = 4,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for case in cases:
        for use_rag in (False, True):
            started = time.perf_counter()
            answer = assistant.answer(case.question, top_k=top_k, use_rag=use_rag)
            latency = time.perf_counter() - started
            retrieved_sources = sorted({item.chunk.title for item in answer.retrieved})
            results.append(
                EvaluationResult(
                    case_id=case.id,
                    mode=answer.mode,
                    question=case.question,
                    answer=answer.answer,
                    keyword_recall=_keyword_recall(answer.answer, case.expected_keywords),
                    source_hit=(
                        _source_hit(retrieved_sources, case.expected_sources)
                        if answer.mode == "rag"
                        else None
                    ),
                    source_hit_applicable=answer.mode == "rag",
                    retrieved_sources=retrieved_sources,
                    latency_seconds=round(latency, 4),
                    correct_refusal=(
                        _is_refusal(answer.answer)
                        if case.category == "unanswerable"
                        else None
                    ),
                )
            )
    return results


def summarize(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    summary = {}
    for mode, items in sorted(by_mode.items()):
        applicable_source_hits = [
            item.source_hit for item in items if item.source_hit_applicable and item.source_hit is not None
        ]
        refusal_items = [item for item in items if item.correct_refusal is not None]
        summary[mode] = {
            "cases": float(len(items)),
            "avg_keyword_recall": round(mean(item.keyword_recall for item in items), 3),
            "avg_source_hit": (
                round(mean(applicable_source_hits), 3)
                if applicable_source_hits
                else None
            ),
            "avg_latency_seconds": round(mean(item.latency_seconds for item in items), 4),
        }
        if refusal_items:
            summary[mode]["correct_refusal_rate"] = round(
                mean(1.0 if item.correct_refusal else 0.0 for item in refusal_items),
                3,
            )
    return summary


def write_results(results: list[EvaluationResult], path: str | Path) -> None:
    payload = [asdict(result) for result in results]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    normalized = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in normalized)
    return round(hits / len(expected_keywords), 3)


def _source_hit(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 1.0
    retrieved = {source.lower() for source in retrieved_sources}
    expected = {source.lower() for source in expected_sources}
    return round(len(retrieved & expected) / len(expected), 3)


def _is_refusal(answer: str) -> bool:
    normalized = answer.lower()
    refusal_markers = [
        "could not find enough relevant evidence",
        "not enough evidence",
        "do not have enough evidence",
        "does not contain",
        "not specify",
        "cannot determine",
    ]
    return any(marker in normalized for marker in refusal_markers)
