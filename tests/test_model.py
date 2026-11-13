import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.error import HTTPError

from car_rag.config import Settings
from car_rag.model import ModelError, OpenRouterChatModel


class OpenRouterModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            api_key="test-key",
            model="anthropic/claude-haiku-5.5",
            base_url="https://openrouter.ai/api/v1",
        )

    @patch("car_rag.model.urlopen")
    def test_sends_openrouter_request_and_returns_answer(self, mocked_urlopen) -> None:
        response = mocked_urlopen.return_value.__enter__.return_value
        response.read.return_value = json.dumps(
            {
                "model": "anthropic/claude-haiku-5.5",
                "choices": [{"message": {"content": "Use the manual."}}],
            }
        ).encode("utf-8")

        output = io.StringIO()
        with redirect_stdout(output):
            answer = OpenRouterChatModel(self.settings).generate("system", "question")

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual("https://openrouter.ai/api/v1/chat/completions", request.full_url)
        self.assertEqual("Bearer test-key", request.get_header("Authorization"))
        self.assertEqual("Use the manual.", answer)
        self.assertEqual("anthropic/claude-haiku-5.5", json.loads(request.data)["model"])
        self.assertIn("OpenRouter model: anthropic/claude-haiku-5.5", output.getvalue())

    @patch("car_rag.model.urlopen")
    def test_surfaces_provider_http_errors(self, mocked_urlopen) -> None:
        mocked_urlopen.side_effect = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"rate limit reached"}}'),
        )

        with self.assertRaisesRegex(ModelError, "HTTP 429.*rate limit reached"):
            OpenRouterChatModel(self.settings).generate("system", "question")

    @patch("car_rag.model.urlopen")
    def test_requires_api_key_without_making_a_request(self, mocked_urlopen) -> None:
        settings = Settings(api_key=None, model=self.settings.model, base_url=self.settings.base_url)

        with self.assertRaisesRegex(ModelError, "OPENROUTER_API_KEY"):
            OpenRouterChatModel(settings).generate("system", "question")
        mocked_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()