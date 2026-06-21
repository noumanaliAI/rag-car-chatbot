# MG ZS Car Manual RAG Chatbot

**Author:** [@noumanaliai](https://github.com/noumanaliai)

> A RAG-powered Q&A chatbot that reads your MG ZS owner's manual and answers car warning questions using LangChain + Anthropic Claude.

## Demo

Ask a question about your car warning, get an answer pulled directly from the manual — no searching needed.

## How It Works

```
Car Manual (HTML)
  → UnstructuredHTMLLoader
  → Text Chunks (1000 chars, 200 overlap)
  → HuggingFace all-MiniLM-L6-v2 embeddings (free)
  → ChromaDB Vector Store

User Question
  → Embed Query
  → Retrieve Top-4 Relevant Chunks
  → Claude Sonnet 4 (temperature=0)
  → Answer
```

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **Anthropic Claude** — claude-sonnet-4-6 (LLM)
- **HuggingFace** — all-MiniLM-L6-v2 embeddings (free, no API key needed)
- **ChromaDB** — vector store / similarity search
- **Datacamp Workspace** — run environment

## Setup

### 1. Prerequisites

- Python 3.10+
- Anthropic API key (`ANTHROPIC_API_KEY`) — get one at [console.anthropic.com](https://console.anthropic.com)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Then in your notebook or script:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

### 4. Upload Your Manual

Place your MG ZS HTML manual (e.g. `mg-zs-warning-messages.html`) in your Datacamp Workspace or project directory.

### 5. Run

Paste `main.py` into a notebook cell (after your setup cell with loader) and run it.

## Usage

```python
query = "The Gasoline Particular Filter Full warning has appeared. What should I do?"
answer = rag_chain.invoke(query)
print(answer.content)
```

Change the `query` variable to ask anything from the manual.

## Project Structure

```
rag-car-chatbot/
├── main.py              # Main RAG pipeline
├── requirements.txt     # Dependencies
├── .env.example          # Env template (copy to .env)
├── .gitignore            # Ignores .env, __pycache__, chroma_db/
└── README.md             # This file
```

## License

MIT — feel free to use and modify.
