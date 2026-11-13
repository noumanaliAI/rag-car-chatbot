"""Grounded question answering over retrieved car-manual passages."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from car_rag.documents import ManualChunk
from car_rag.retrieval import BM25Retriever, SearchResult


class ChatModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class AssistantResponse:
    answer: str
    sources: tuple[ManualChunk, ...]


SYSTEM_PROMPT = """You answer questions about a car using only the supplied owner-manual excerpts.
Do not add specifications, procedures, or warnings that are not supported by those excerpts.
If the excerpts do not answer the question, say that the manual excerpts do not provide the answer.
For safety-critical or urgent problems, advise the driver to stop safely and consult the vehicle manual
or a qualified service professional. Cite supporting excerpts with their [source, chunk] labels."""

NOT_FOUND = "I couldn't find information about that in the supplied manual."


class CarManualAssistant:
    def __init__(
        self,
        retriever: BM25Retriever,
        model: ChatModel,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._retriever = retriever
        self._model = model
        self._top_k = top_k

    def ask(self, question: str) -> AssistantResponse:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        results = self._retriever.search(question, top_k=self._top_k)
        if not results:
            return AssistantResponse(answer=NOT_FOUND, sources=())

        context = _format_context(results)
        answer = self._model.generate(
            SYSTEM_PROMPT,
            f"Manual excerpts:\n{context}\n\nQuestion: {question.strip()}",
        )
        return AssistantResponse(
            answer=answer,
            sources=tuple(result.chunk for result in results),
        )


def _format_context(results: Sequence[SearchResult]) -> str:
    return "\n\n".join(
        f"[{result.chunk.source}, chunk {result.chunk.index}]\n{result.chunk.text}"
        for result in results
    )
