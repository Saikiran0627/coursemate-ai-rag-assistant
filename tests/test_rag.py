from pathlib import Path

from coursemate.evaluation import evaluate, load_cases, summarize
from coursemate.llm import DeterministicCourseLLM
from coursemate.rag import CourseMateAssistant


ROOT = Path(__file__).resolve().parents[1]


class CountingLLM(DeterministicCourseLLM):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return super().generate(prompt)


def test_rag_retrieves_expected_source_for_pipeline_question():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What are the main steps in a RAG pipeline?", use_rag=True)

    assert answer.retrieved
    assert answer.retrieved[0].chunk.title == "Module 2: Retrieval-Augmented Generation"
    assert "vector" in answer.answer.lower()


def test_greeting_does_not_retrieve_or_call_llm():
    llm = CountingLLM()
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        llm,
    )

    answer = assistant.answer("Hello", use_rag=True)

    assert answer.mode == "conversation"
    assert "Hello" in answer.answer
    assert answer.retrieved == []
    assert llm.calls == 0


def test_thanks_does_not_show_citations():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("Thank you", use_rag=True)

    assert answer.evidence_status == "not_needed"
    assert answer.retrieved == []
    assert "Sources:" not in answer.answer


def test_no_rag_refuses_without_context():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What should the final project submission include?", use_rag=False)

    assert "do not have enough" in answer.answer.lower()
    assert answer.retrieved == []


def test_unanswerable_low_relevance_question_refuses_without_sources():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What is tomorrow's cafeteria menu?", use_rag=True)

    assert answer.evidence_status == "insufficient"
    assert "could not find enough relevant evidence" in answer.answer.lower()
    assert answer.retrieved == []


def test_follow_up_question_uses_previous_user_turn_for_retrieval():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer(
        "Why is it useful for course assistants?",
        use_rag=True,
        chat_history=[
            {"role": "user", "content": "What is retrieval augmented generation?"},
            {"role": "assistant", "content": "It combines retrieval with generation."},
        ],
    )

    assert "retrieval augmented generation" in answer.retrieval_query.lower()
    assert answer.retrieved
    assert any("Retrieval-Augmented Generation" in item.chunk.title for item in answer.retrieved)


def test_follow_up_skips_previous_unsupported_turn():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer(
        "Why is it useful for course assistants?",
        use_rag=True,
        chat_history=[
            {"role": "user", "content": "What are the main steps in a RAG pipeline?"},
            {"role": "assistant", "content": "It uses retrieved context.", "evidence_status": "sufficient"},
            {"role": "user", "content": "What is tomorrow's cafeteria menu?"},
            {
                "role": "assistant",
                "content": "I could not find enough relevant evidence.",
                "evidence_status": "insufficient",
            },
        ],
    )

    assert "cafeteria" not in answer.retrieval_query.lower()
    assert "rag pipeline" in answer.retrieval_query.lower()


def test_multi_document_question_can_retrieve_multiple_sources():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
        chunk_size=180,
        overlap=20,
    )

    answer = assistant.answer(
        "What should final submissions include and what metrics evaluate a RAG assistant?",
        use_rag=True,
        top_k=6,
        relevance_threshold=0.12,
    )

    titles = {item.chunk.title for item in answer.retrieved}
    assert "Course Policy Summary" in titles
    assert "Module 4: LLM Evaluation" in titles


def test_citations_match_filtered_retrieved_sources():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    answer = assistant.answer("What are the main steps in a RAG pipeline?", use_rag=True)

    retrieved_titles = {item.chunk.title for item in answer.retrieved}
    assert retrieved_titles == {"Module 2: Retrieval-Augmented Generation"}
    assert "Module 2: Retrieval-Augmented Generation" in answer.answer
    assert "Course Policy Summary" not in answer.answer


def test_rag_vs_no_rag_comparison_keeps_retrieval_separate():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )

    comparison = assistant.compare("What are the main steps in a RAG pipeline?")

    assert comparison["no_rag"].retrieved == []
    assert comparison["rag"].retrieved
    assert comparison["rag"].evidence_status == "sufficient"


def test_evaluation_shows_rag_improves_source_hit():
    assistant = CourseMateAssistant.from_directory(
        ROOT / "data" / "course_materials",
        DeterministicCourseLLM(),
    )
    cases = load_cases(ROOT / "data" / "evaluation" / "questions.jsonl")

    summary = summarize(evaluate(assistant, cases))

    assert summary["rag"]["avg_source_hit"] > summary["no_rag"]["avg_source_hit"]
    assert summary["rag"]["avg_keyword_recall"] > summary["no_rag"]["avg_keyword_recall"]
