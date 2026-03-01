# Research Paper Copilot — Project Overview

## What Is This?

A local AI agent that reads a research paper (PDF) and produces a structured markdown summary you can paste directly into Notion. Built as a learning project for **LangChain tool calling** with **local models via Ollama**.

## The Problem

Reading research papers is time-consuming. You often need to extract the same information: what's the contribution, how did they do it, how did they evaluate it, what are the key results. This agent automates that extraction into a consistent, readable format.

## How It Works

```
PDF → Agent → Markdown Summary
```

The agent is a LangChain-based AI that has access to five tools:

1. **extract_text** — Pulls text from the PDF, organized by sections
2. **extract_images** — Extracts figures from the PDF with their captions
3. **parse_references** — Parses the references section into structured data
4. **generate_chart** — Creates a pie chart showing journal distribution in references
5. **save_markdown** — Assembles everything into the final markdown file

The agent decides which tools to call and in what order. This is the key concept: instead of a hardcoded pipeline, the LLM reasons about what to do next.

## What the Output Looks Like

The generated markdown contains:

- **Metadata** — title, authors, venue, year
- **Overview** — what the paper is about (3-5 sentences)
- **Contribution** — what the authors propose and why it matters
- **State of the Art** — summary of related work
- **Methodology** — overview + detailed method walkthrough (including formulae)
- **Evaluation** — datasets, metrics, baselines
- **Key Results** — main findings as bullet points
- **References Analysis** — pie chart of journal/venue distribution
- **Relevant figures** — images from the paper selected by the agent

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Package manager | uv |
| Agent framework | LangChain |
| Local LLM | Ollama (model configurable) |
| PDF parsing | PyMuPDF |
| Charts | matplotlib |
| Web UI | Streamlit |

## Project Structure

```
Summarize-Papers/
├── pyproject.toml          # Project config and dependencies
├── src/
│   ├── agent.py            # The LangChain agent
│   ├── config.py           # Settings (model name, paths)
│   ├── prompts.py          # System prompt templates
│   └── tools/
│       ├── extract_text.py
│       ├── extract_images.py
│       ├── parse_references.py
│       ├── generate_chart.py
│       └── save_markdown.py
├── app.py                  # Streamlit web UI
├── input/                  # Drop PDFs here
├── output/                 # Generated summaries
└── docs/                   # Explanatory docs (you are here)
```

## Implementation Plan

The project is built step by step, with each step producing working code and an explanatory document:

| Step | What | Doc |
|------|------|-----|
| 1 | Project scaffolding with uv | `01-setup.md` |
| 2 | PDF text extraction tool | `02-pdf-extraction.md` |
| 3 | PDF image extraction tool | (included in 02) |
| 4 | Reference parsing tool | (included in 03) |
| 5 | Chart generation tool | (included in 03) |
| 6 | Markdown assembly tool | `03-tools.md` |
| 7 | Agent with tool bindings | `04-agent.md` |
| 8 | Streamlit UI | `05-ui.md` |
| 9 | Testing and refinement | — |

## Requirements

- **Hardware:** Machine with 2x RTX 4090 GPUs (for running local LLMs)
- **Software:** Python 3.12+, uv, Ollama
- **Input:** English-language research papers as PDFs
- **Output:** Markdown files ready for Notion
