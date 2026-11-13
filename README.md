# Car Manual RAG Chatbot

A command-line assistant that searches one or more car manuals and answers questions
from the matching passages. Chat responses use the OpenRouter API, with
`anthropic/claude-haiku-5.5` as the default model. Retrieval and its tests run locally
without an API key.

## Features

- Loads UTF-8 HTML, plain text, and Markdown manuals; removes scripts and styles from HTML.
- Splits text into bounded, overlapping chunks, preferring paragraph and sentence boundaries.
- Ranks passages locally with BM25. Exact manual terms (warning names, components, codes)
  receive strong matches without sending the manual to an embedding service.
- Supplies only retrieved passages to the model, includes source/chunk labels, and avoids
  making a model call when no passage matches.
- Reports missing/unsupported manuals, invalid input, missing API keys, and OpenRouter errors.
- Includes an original, fictional sample manual for a no-key retrieval demonstration.

## Architecture

```text
Manual files
  -> car_rag.documents: validate, extract text, chunk with overlap
  -> car_rag.retrieval: local BM25 search
  -> car_rag.assistant: ground the question in ranked passages
  -> car_rag.model: OpenRouter chat-completions API
  -> answer + source labels
```

| Module | Responsibility |
| --- | --- |
| `car_rag/documents.py` | File validation, HTML text extraction, chunking |
| `car_rag/retrieval.py` | BM25 indexing and ranked search |
| `car_rag/assistant.py` | Retrieval-to-answer orchestration and grounding prompt |
| `car_rag/model.py` | OpenRouter HTTP client and response/error validation |
| `car_rag/config.py` | Environment configuration and defaults |
| `car_rag/cli.py` | Command-line interface |
| `tests/` | Offline tests for ingestion, retrieval, and answer behavior |

BM25 is a lightweight lexical retriever, which is especially useful for exact terms
found in manuals. It normalizes common `tire`/`tyre` and `light`/`lamp` wording but
does not infer general synonyms as an embedding search might; if your manual uses
different terminology, include that wording in the question.

## Technologies

- Python 3.10+
- OpenRouter's OpenAI-compatible chat-completions endpoint
- Claude Haiku 5.5 by default (`anthropic/claude-haiku-5.5`)
- Python standard library for HTML parsing, chunking, ranking, and HTTP

## Setup

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, then set your OpenRouter key:

   ```text
   OPENROUTER_API_KEY=your-openrouter-api-key
   OPENROUTER_MODEL=anthropic/claude-haiku-5.5
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   ```

   The key is read from `.env` or the process environment. Never commit your real key.
   The model and base URL can be overridden with `OPENROUTER_MODEL` and
   `OPENROUTER_BASE_URL`; an optional site URL/name can be set using the variables
   shown in `.env.example`.

## Usage

Ask a question against one or multiple manuals:

```powershell
python main.py --manual "C:\manuals\vehicle.html" --question "What does the battery warning mean?"
```

Supported input formats: `.html`, `.htm`, `.txt`, `.md`. Use repeated paths after
`--manual` for multiple documents:

```powershell
python -m car_rag --manual manual-one.html manual-two.txt --question "How should I check tyre pressure?"
```

Tune the number of retrieved passages with `--top-k` (default: 4). To preview the
retrieved passages without a key or network request:

```powershell
python main.py --manual demo\sample-car-manual.txt --question "What does the battery warning mean?" --retrieve-only
```

The demo file is deliberately fictional. Its sample text is not an official vehicle
manual and is not service or operating advice. For real vehicles, use the correct
manufacturer manual and qualified service professionals.

## Demo

Run the retrieval-only command above. It should return the sample's **Battery warning**
passage, including the statements that the charging system may not be charging the
battery correctly and that a qualified service technician should check it. No API key
is used in this mode.

For an actual generated answer, omit `--retrieve-only` after configuring the key:

```powershell
python main.py --manual demo\sample-car-manual.txt --question "What should I do if the engine temperature warning appears?"
```

The answer includes the manual filename and chunk number used as sources. The chatbot
is informational only; treat dashboard warnings and urgent vehicle problems according
to the official manual and appropriate professional guidance.

## Tests

Tests use the Python standard library and do not call OpenRouter:

```powershell
python -m unittest discover -v
```