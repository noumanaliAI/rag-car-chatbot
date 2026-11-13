"""OpenRouter chat-completions client using only the Python standard library."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from car_rag.config import Settings


class ModelError(Exception):
    """Raised when OpenRouter cannot return a usable response."""


class OpenRouterChatModel:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_key = (self._settings.api_key or "").strip()
        if not api_key:
            raise ModelError(
                "OPENROUTER_API_KEY is not set. Add it to .env or your environment."
            )
        if not self._settings.model.strip():
            raise ModelError("OPENROUTER_MODEL must not be empty.")

        payload = json.dumps(
            {
                "model": self._settings.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": self._settings.site_name,
        }
        if self._settings.site_url:
            headers["HTTP-Referer"] = self._settings.site_url

        request = Request(
            f"{self._settings.base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._settings.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                if isinstance(response_data, dict):
                    print("OpenRouter model:", response_data.get("model"))
        except HTTPError as exc:
            detail = _provider_error_detail(exc.read())
            raise ModelError(
                f"OpenRouter request failed (HTTP {exc.code}): {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise ModelError(f"Could not reach OpenRouter: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelError("OpenRouter returned an invalid JSON response.") from exc

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("OpenRouter response did not contain a chat answer.") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelError("OpenRouter returned an empty chat answer.")
        return content.strip()


def _provider_error_detail(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "The provider did not return a readable error message."
    if not isinstance(data, dict):
        return "The provider did not return a readable error message."
    error = data.get("error", {})
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return "The provider did not return a readable error message."