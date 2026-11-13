"""Command-line entry point for the car-manual assistant."""

import argparse
import sys

from car_rag.assistant import CarManualAssistant
from car_rag.config import Settings, load_local_env
from car_rag.documents import DocumentError, load_manuals
from car_rag.model import ModelError, OpenRouterChatModel
from car_rag.retrieval import BM25Retriever


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions grounded in your car's owner manual."
    )
    parser.add_argument(
        "--manual",
        nargs="+",
        required=True,
        help="One or more .html, .htm, .txt, or .md manual files.",
    )
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument(
        "--top-k", type=int, default=4, help="Number of relevant chunks (default: 4)."
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print matching passages without calling OpenRouter (useful for demos).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("--question must not be empty.")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    try:
        chunks = load_manuals(args.manual)
        retriever = BM25Retriever(chunks)
        if args.retrieve_only:
            results = retriever.search(args.question, top_k=args.top_k)
            if not results:
                print("No matching passages found in the supplied manual.")
                return 0
            for result in results:
                print(f"[{result.chunk.source}, chunk {result.chunk.index}]")
                print(result.chunk.text)
                print()
            return 0

        load_local_env()
        assistant = CarManualAssistant(
            retriever,
            OpenRouterChatModel(Settings.from_env()),
            top_k=args.top_k,
        )
        response = assistant.ask(args.question)
        print(response.answer)
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(f"- {source.source}, chunk {source.index}")
        return 0
    except (DocumentError, ModelError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
