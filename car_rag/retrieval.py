"""Small, dependency-free BM25 retrieval for manual text."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

from car_rag.documents import ManualChunk

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "if", "in", "is", "it", "me", "mean", "means", "of",
    "on", "or", "should", "the", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "you",
}
_TOKEN_ALIASES = {
    "tire": "tyre",
    "tires": "tyre",
    "tyres": "tyre",
    "light": "lamp",
    "lights": "lamp",
    "lamps": "lamp",
}


@dataclass(frozen=True)
class SearchResult:
    chunk: ManualChunk
    score: float


def _tokens(text: str) -> list[str]:
    return [
        _TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS
    ]


class BM25Retriever:
    """Rank chunks using BM25 so exact manual terms remain easy to find."""

    def __init__(
        self,
        chunks: Sequence[ManualChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("At least one manual chunk is required.")
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one.")

        self._chunks = tuple(chunks)
        self._tokens = tuple(_tokens(chunk.text) for chunk in chunks)
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )
        self._average_length = sum(map(len, self._tokens)) / len(self._tokens)
        self._k1 = k1
        self._b = b

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_terms = list(dict.fromkeys(_tokens(query)))
        if not query_terms:
            return []

        document_count = len(self._chunks)
        results: list[SearchResult] = []
        for chunk, terms, frequencies in zip(
            self._chunks, self._tokens, self._term_frequencies
        ):
            length = len(terms)
            score = 0.0
            matched_terms = 0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                matched_terms += 1
                document_frequency = self._document_frequencies[term]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                score += inverse_frequency * (
                    frequency * (self._k1 + 1) / denominator
                )
            if score > 0 and matched_terms >= min(2, len(query_terms)):
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: (-result.score, result.chunk.index))
        return results[:top_k]