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
    notes: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    mode: str
    question: str
    answer: str
    keyword_recall: float
    source_hit: float
    retrieved_sources: list[str]
    latency_seconds: float


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
                    source_hit=_source_hit(retrieved_sources, case.expected_sources),
                    retrieved_sources=retrieved_sources,
                    latency_seconds=round(latency, 4),
                )
            )
    return results


def summarize(results: list[EvaluationResult]) -> dict[str, dict[str, float]]:
    by_mode: dict[str, list[EvaluationResult]] = {}
    for result in results:
        by_mode.setdefault(result.mode, []).append(result)

    return {
        mode: {
            "cases": float(len(items)),
            "avg_keyword_recall": round(mean(item.keyword_recall for item in items), 3),
            "avg_source_hit": round(mean(item.source_hit for item in items), 3),
            "avg_latency_seconds": round(mean(item.latency_seconds for item in items), 4),
        }
        for mode, items in sorted(by_mode.items())
    }


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
