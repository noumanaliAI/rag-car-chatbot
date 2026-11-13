import unittest

from car_rag.assistant import CarManualAssistant, NOT_FOUND
from car_rag.documents import ManualChunk
from car_rag.retrieval import BM25Retriever


class FakeModel:
    def __init__(self, answer: str = "Charge the battery.") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.answer


class AssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        chunks = [
            ManualChunk(
                "The battery warning indicates a charging system issue. "
                "Have the vehicle checked by a qualified technician.",
                "manual.txt",
                1,
            )
        ]
        self.model = FakeModel()
        self.assistant = CarManualAssistant(BM25Retriever(chunks), self.model)

    def test_answers_with_source_and_passes_retrieved_context_to_model(self) -> None:
        response = self.assistant.ask("What does the battery warning mean?")

        self.assertEqual("Charge the battery.", response.answer)
        self.assertEqual(["manual.txt"], [source.source for source in response.sources])
        self.assertIn("charging system issue", self.model.calls[0][1])
        self.assertIn("[manual.txt, chunk 1]", self.model.calls[0][1])

    def test_does_not_call_model_when_manual_has_no_relevant_passage(self) -> None:
        response = self.assistant.ask("How do I pair a phone? ")

        self.assertEqual(NOT_FOUND, response.answer)
        self.assertEqual((), response.sources)
        self.assertEqual([], self.model.calls)

    def test_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "Question must not be empty"):
            self.assistant.ask(" ")


if __name__ == "__main__":
    unittest.main()