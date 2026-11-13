"""Load supported manual files and split them into readable chunks."""

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable


class DocumentError(Exception):
    """Raised when a manual cannot be loaded or chunked."""


@dataclass(frozen=True)
class ManualChunk:
    text: str
    source: str
    index: int


class _ManualHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li",
        "main", "ol", "p", "section", "table", "td", "th", "tr", "ul",
    }
    _IGNORED_TAGS = {"script", "style", "noscript", "svg"}
    _VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed_tags: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if self.suppressed_tags:
            if tag not in self._VOID_TAGS:
                self.suppressed_tags.append(tag)
            return
        attrs = {name.lower(): value for name, value in _attrs}
        style = (attrs.get("style") or "").replace(" ", "").lower()
        is_hidden = (
            "hidden" in attrs
            or attrs.get("aria-hidden", "").lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        )
        if tag in self._IGNORED_TAGS or is_hidden:
            if tag not in self._VOID_TAGS:
                self.suppressed_tags.append(tag)
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.suppressed_tags:
            for position in range(len(self.suppressed_tags) - 1, -1, -1):
                if self.suppressed_tags[position] == tag:
                    del self.suppressed_tags[position:]
                    break
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.suppressed_tags:
            self.parts.append(data)


def _read_manual(path: Path) -> str:
    if not path.exists():
        raise DocumentError(f"Manual file does not exist: {path}")
    if not path.is_file():
        raise DocumentError(f"Manual path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".html", ".htm", ".txt", ".md"}:
        raise DocumentError(
            f"Unsupported manual format '{suffix or '(no extension)'}'. "
            "Use HTML, TXT, or Markdown."
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DocumentError(f"Could not read manual '{path}': {exc}") from exc

    if suffix in {".html", ".htm"}:
        parser = _ManualHTMLParser()
        parser.feed(raw_text)
        raw_text = "".join(parser.parts)

    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise DocumentError(f"Manual contains no readable text: {path}")
    return text


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 160,
) -> list[ManualChunk]:
    """Split text at paragraph/sentence boundaries with a bounded overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    normalized_text = re.sub(r"[ \t]+", " ", text)
    normalized_text = re.sub(r" *\n *", "\n", normalized_text).strip()
    if not normalized_text:
        return []

    chunks: list[ManualChunk] = []
    start = 0
    while start < len(normalized_text):
        hard_end = min(start + chunk_size, len(normalized_text))
        end = hard_end
        if hard_end < len(normalized_text):
            floor = start + int(chunk_size * 0.6)
            for delimiter in ("\n\n", ". ", " "):
                boundary = normalized_text.rfind(delimiter, floor, hard_end)
                if boundary >= floor:
                    end = boundary + len(delimiter)
                    break

        content = normalized_text[start:end].strip()
        if content:
            chunks.append(ManualChunk(content, source, len(chunks) + 1))
        if end == len(normalized_text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start < end and not normalized_text[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return chunks


def load_manuals(
    paths: Iterable[str | Path],
    chunk_size: int = 1000,
    overlap: int = 160,
) -> list[ManualChunk]:
    chunks: list[ManualChunk] = []
    for item in paths:
        path = Path(item)
        text = _read_manual(path)
        chunks.extend(
            chunk_text(text, path.name, chunk_size=chunk_size, overlap=overlap)
        )
    if not chunks:
        raise DocumentError("No manual files were supplied.")
    return chunks