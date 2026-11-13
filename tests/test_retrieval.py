import unittest

from car_rag.documents import ManualChunk
from car_rag.retrieval import BM25Retriever


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            ManualChunk(
                "The diesel particulate filter is full. Drive at a steady speed "
                "to allow regeneration.",
                "manual.html",
                1,
            ),
            ManualChunk("Check tyre pressure before a long journey.", "manual.html", 2),
            ManualChunk("The battery warning indicates a charging system issue.", "other.txt", 1),
        ]
        self.retriever = BM25Retriever(self.chunks)

    def test_relevant_passage_ranks_first_and_unrelated_query_has_no_hits(self) -> None:
        results = self.retriever.search("diesel particulate filter full", top_k=2)

        self.assertEqual(self.chunks[0], results[0].chunk)
        self.assertEqual([], self.retriever.search("transmission oil", top_k=3))

    def test_generic_single_term_match_does_not_rank_as_relevant(self) -> None:
        results = self.retriever.search("battery warning mean", top_k=3)

        self.assertEqual([self.chunks[2]], [result.chunk for result in results])

    def test_common_tire_spelling_and_warning_light_terms_match(self) -> None:
        results = self.retriever.search(
            "How do I reset the tire pressure warning light?", top_k=3
        )

        self.assertEqual([self.chunks[1]], [result.chunk for result in results])

    def test_query_and_limit_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.retriever.search("   ")
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.search("battery", top_k=0)


if __name__ == "__main__":
    unittest.main()