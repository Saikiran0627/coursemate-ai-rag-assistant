"""Command line interface for the CourseMate RAG assistant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coursemate.evaluation import evaluate, load_cases, summarize, write_results
from coursemate.llm import build_llm
from coursemate.rag import CourseMateAssistant


DEFAULT_DATA_DIR = Path("data/course_materials")
DEFAULT_EVAL_FILE = Path("data/evaluation/questions.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="CourseMate AI course-material assistant")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory of course documents")
    parser.add_argument("--llm", default="deterministic", choices=["deterministic", "ollama", "openai"])
    parser.add_argument("--model", default=None, help="Model name for Ollama or OpenAI-compatible backends")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask a single question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=4)
    ask.add_argument("--no-rag", action="store_true", help="Disable retrieval for comparison")
    ask.add_argument("--show-prompt", action="store_true")

    compare = subparsers.add_parser("compare", help="Show no-RAG and RAG answers")
    compare.add_argument("question")
    compare.add_argument("--top-k", type=int, default=4)

    eval_parser = subparsers.add_parser("evaluate", help="Run the evaluation dataset")
    eval_parser.add_argument("--cases", default=DEFAULT_EVAL_FILE)
    eval_parser.add_argument("--top-k", type=int, default=4)
    eval_parser.add_argument("--output", default="artifacts/evaluation_results.json")

    args = parser.parse_args()
    llm = build_llm(args.llm, model=args.model)
    assistant = CourseMateAssistant.from_directory(args.data_dir, llm)

    if args.command == "ask":
        answer = assistant.answer(args.question, top_k=args.top_k, use_rag=not args.no_rag)
        _print_answer(answer)
        if args.show_prompt:
            print("\n--- Prompt ---")
            print(answer.prompt)
    elif args.command == "compare":
        comparison = assistant.compare(args.question, top_k=args.top_k)
        for label in ("no_rag", "rag"):
            print(f"\n=== {label.upper()} ===")
            _print_answer(comparison[label])
    elif args.command == "evaluate":
        cases = load_cases(args.cases)
        results = evaluate(assistant, cases, top_k=args.top_k)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_results(results, output_path)
        print(json.dumps(summarize(results), indent=2))
        print(f"\nDetailed results written to {output_path}")


def _print_answer(answer) -> None:
    print(answer.answer)
    if answer.retrieved:
        print("\nRetrieved context:")
        for item in answer.retrieved:
            print(f"- {item.chunk.title} ({item.chunk.chunk_id}, score={item.score:.3f})")
    else:
        print("\nRetrieved context: none")


if __name__ == "__main__":
    main()
