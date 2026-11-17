from pathlib import Path

from coursemate.evaluation import evaluate, load_cases, summarize
from coursemate.llm import DeterministicCourseLLM
from coursemate.rag import CourseMateAssistant


ROOT = Path(__file__).resolve().parents[1]


def test_rag_retrieves_expected_source_for_pipeline_question():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What are the main steps in a RAG pipeline?", use_rag=True)

    assert answer.retrieved
    assert answer.retrieved[0].chunk.title == "Module 2: Retrieval-Augmented Generation"
    assert "vector" in answer.answer.lower()


def test_no_rag_refuses_without_context():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What should the final project submission include?", use_rag=False)

    assert "do not have enough" in answer.answer.lower()
    assert answer.retrieved == []


def test_evaluation_shows_rag_improves_source_hit():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )
    cases = load_cases(ROOT / "data" / "evaluation" / "questions.jsonl")

    summary = summarize(evaluate(assistant, cases))

    assert summary["rag"]["avg_source_hit"] > summary["no_rag"]["avg_source_hit"]
    assert summary["rag"]["avg_keyword_recall"] > summary["no_rag"]["avg_keyword_recall"]
