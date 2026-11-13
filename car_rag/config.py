"""Environment-backed configuration for the chatbot."""

from dataclasses import dataclass
import os
from pathlib import Path
import re


DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without overwriting process environment."""
    env_path = Path(path)
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Could not read environment file '{env_path}': {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[len("export "):].lstrip()
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME.fullmatch(name):
            raise ValueError(
                f"Invalid environment entry in '{env_path}' on line {line_number}."
            )

        value = value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(
                    f"Unclosed quote in '{env_path}' on line {line_number}."
                )
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    base_url: str
    timeout_seconds: float = 60.0
    site_url: str | None = None
    site_name: str = "Car Manual RAG"

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
        timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60")
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("OPENROUTER_TIMEOUT_SECONDS must be a number.") from exc
        if timeout <= 0:
            raise ValueError("OPENROUTER_TIMEOUT_SECONDS must be greater than zero.")
        if not model:
            raise ValueError("OPENROUTER_MODEL must not be empty.")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OPENROUTER_BASE_URL must be an HTTP(S) URL.")

        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model,
            base_url=base_url,
            timeout_seconds=timeout,
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME", "Car Manual RAG"),
        )