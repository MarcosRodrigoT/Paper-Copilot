# Step 1: Project Setup

This document explains how the project is set up, what uv is, and how to get Ollama running.

## What is uv?

**uv** is a fast Python package manager written in Rust (by the same team that created `ruff`). It replaces `pip`, `pip-tools`, `virtualenv`, and parts of `poetry`/`pipenv` — all in one tool that runs 10-100x faster.

### Key concepts

| Concept | uv equivalent | What it does |
|---------|--------------|--------------|
| Create a project | `uv init` | Creates `pyproject.toml`, `.python-version`, and a git repo |
| Add a dependency | `uv add <package>` | Adds to `pyproject.toml` and installs it |
| Install everything | `uv sync` | Installs all dependencies from the lock file |
| Run a script | `uv run python script.py` | Runs using the project's virtual environment |
| Virtual environment | `.venv/` (automatic) | Created automatically by `uv sync`, no manual activation needed |

### Why uv instead of pip?

- **Speed:** uv resolves and installs packages much faster than pip
- **Reproducibility:** `uv.lock` pins exact versions (like `package-lock.json` in Node.js)
- **No manual venv:** `uv run` automatically uses the project's `.venv`
- **Single tool:** No need to juggle pip, virtualenv, pip-tools separately

### How we used it

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize the project (creates pyproject.toml, .python-version, .git)
uv init --name paper-copilot

# Install all dependencies
uv sync
```

## pyproject.toml

This is the project's configuration file. Here's what ours looks like:

```toml
[project]
name = "paper-copilot"
version = "0.1.0"
description = "A LangChain agent that summarizes research papers into Notion-ready markdown"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.3",
    "langchain-ollama>=0.3",
    "pymupdf>=1.25",
    "matplotlib>=3.10",
    "streamlit>=1.42",
]
```

### What each dependency does

| Package | Purpose |
|---------|---------|
| `langchain` | The agent framework — provides tool definitions, agent loops, chains |
| `langchain-ollama` | LangChain integration with Ollama for local LLM inference |
| `pymupdf` | PDF parsing library — extracts text, images, and metadata from PDFs |
| `matplotlib` | Chart generation — we use it to create the references pie chart |
| `streamlit` | Web UI framework — creates the drag-and-drop interface |

## What is Ollama?

**Ollama** is a tool for running LLMs locally on your machine. It handles model downloading, quantization, GPU memory management, and exposes an API that applications can call.

### Installing Ollama

```bash
# On Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify it's running
ollama --version
```

### Pulling a model

```bash
# Pull a model that supports tool calling
ollama pull mistral-small

# Other good options:
# ollama pull mistral-small3.2
# ollama pull gpt-oss:20b
```

### How Ollama works with this project

```
┌─────────────┐     HTTP API      ┌─────────────┐
│  Our Agent   │ ──────────────►  │   Ollama     │
│  (Python)    │ ◄──────────────  │   Server     │
│              │  JSON responses  │  (GPU)       │
└─────────────┘                   └─────────────┘
```

1. Ollama runs as a local server on `http://localhost:11434`
2. Our Python code uses `langchain-ollama` to send requests to it
3. The model runs on your GPUs — no data leaves your machine
4. Tool calling is handled through Ollama's native function-calling support

### Checking available models

```bash
# List installed models
ollama list

# Test a model interactively
ollama run mistral-small "Hello, what can you do?"
```

## Project directory structure

After setup, the project looks like this:

```
Summarize-Papers/
├── .git/               # Git repository (created by uv init)
├── .venv/              # Virtual environment (created by uv sync)
├── pyproject.toml      # Project config
├── uv.lock             # Locked dependency versions
├── .python-version     # Python version pin (3.12)
├── src/
│   ├── __init__.py
│   ├── config.py       # Centralized settings
│   └── tools/
│       └── __init__.py
├── input/              # Drop PDFs here
├── output/             # Generated summaries go here
└── docs/               # You are reading this
```

## config.py

All configurable values live in `src/config.py`:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

OLLAMA_MODEL = "mistral-small"       # Change to any Ollama model
OLLAMA_BASE_URL = "http://localhost:11434"
```

To switch models, just edit `OLLAMA_MODEL`. No other code changes needed.

## Verification

To verify the setup works:

```bash
# Check uv can see the project
uv run python -c "from src.config import OLLAMA_MODEL; print(f'Model: {OLLAMA_MODEL}')"
```

## Next step

With the scaffolding in place, the next step is building the first tool: **PDF text extraction** (`docs/02-pdf-extraction.md`).
