import tempfile
import unittest
from pathlib import Path
import re

from car_rag.documents import DocumentError, chunk_text, load_manuals


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size_and_preserve_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(80))
        chunks = chunk_text(text, "manual.txt", chunk_size=80, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))
        self.assertEqual([1, 2], [chunk.index for chunk in chunks[:2]])
        all_tokens = {
            token
            for chunk in chunks
            for token in re.findall(r"\w+", chunk.text)
        }
        self.assertEqual(set(re.findall(r"\w+", text)), all_tokens)

    def test_html_loader_keeps_headings_and_omits_script_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manual = Path(directory) / "manual.html"
            manual.write_text(
                "<h1>Warning lights</h1><p>Stop safely.</p>"
                "<script>ignore this</script>"
                "<div hidden>ignore this <div>nested hidden text</div></div>",
                encoding="utf-8",
            )

            chunks = load_manuals([manual])

        combined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Warning lights", combined)
        self.assertIn("Stop safely.", combined)
        self.assertNotIn("ignore this", combined)
        self.assertNotIn("nested hidden text", combined)

    def test_missing_and_unsupported_files_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(DocumentError, "does not exist"):
            load_manuals(["missing-manual.txt"])

        with tempfile.TemporaryDirectory() as directory:
            manual = Path(directory) / "manual.pdf"
            manual.write_bytes(b"not supported")
            with self.assertRaisesRegex(DocumentError, "Unsupported manual format"):
                load_manuals([manual])


if __name__ == "__main__":
    unittest.main()