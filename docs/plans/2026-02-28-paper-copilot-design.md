# Research Paper Copilot — Design Document

**Date:** 2026-02-28

## Context & Motivation

This project is a personal learning project built alongside an AI agents course. The goal is to get hands-on experience with **LangChain tool calling** using **local models via Ollama** on a machine with 2x RTX 4090 GPUs.

The chosen project: a **Research Paper Copilot** — an agent that takes a research paper (PDF) and produces a structured, Notion-ready markdown summary.

## Requirements

### Input
- A research paper as a PDF file
- Provided either through a **Streamlit drag-and-drop UI** or by placing the file in an `input/` folder
- English-language papers only

### Output
A markdown file (`.md`) ready to paste into Notion, containing:

1. **Metadata** — title, authors, publication venue, year, DOI/link
2. **Overview** — 3-5 sentence summary of the paper's context
3. **Contribution** — what the authors propose and why it matters (4-6 sentences)
4. **State of the Art** — summary of related work and how this paper positions itself (1-2 paragraphs)
5. **Methodology** — overview paragraph + detailed method walkthrough including formulae, architectural choices, loss functions. Concise but retaining all important details
6. **Evaluation** — datasets, metrics, baselines, experimental setup
7. **Key Results** — main findings as bullet points with numbers
8. **References Analysis** — pie chart (PNG) showing journal/venue distribution in the paper's references
9. **Relevant figures** — images extracted from the PDF, agent-selected for relevance, with captions

### Tech Stack
- **Python** with `uv` for dependency management
- **LangChain** for the agent and tool framework
- **Ollama** for local model inference (model configurable, no default chosen yet)
- **Streamlit** for the web UI
- **matplotlib** for chart generation
- **PyMuPDF (fzitz)** or similar for PDF parsing and image extraction

### Non-functional
- Fully local — no API calls to external services
- Educational — each component accompanied by an explanatory markdown document
- Step-by-step implementation to learn along the way

## Architecture

### Approach: Multi-tool LangChain Agent

The agent is a LangChain `AgentExecutor` (or equivalent) with a set of bound tools. The LLM decides which tools to call and in what order to process the paper.

### Tools

| Tool | Input | Output |
|------|-------|--------|
| `extract_text` | PDF path | Structured text sections from the paper |
| `extract_images` | PDF path | Image files saved to disk + caption mapping |
| `parse_references` | PDF path or text | Structured list of references with journal/venue names |
| `generate_chart` | Reference data | Pie chart PNG saved to disk |
| `save_markdown` | All generated content | Final `.md` file in `output/` |

### Data Flow

```
User drops PDF (UI or folder)
        │
        ▼
    Agent receives task
        │
        ├──► extract_text ──► paper text by sections
        ├──► extract_images ──► image files + captions
        └──► parse_references ──► reference list
                │
                ▼
        Agent generates summaries for each section
        using the extracted text
                │
                ├──► generate_chart ──► pie chart PNG
                └──► selects relevant images per section
                        │
                        ▼
                save_markdown ──► final .md + images in output/
```

### Project Structure

```
Summarize-Papers/
├── pyproject.toml
├── .python-version
├── src/
│   ├── agent.py                # LangChain agent with tool bindings
│   ├── tools/
│   │   ├── extract_text.py
│   │   ├── extract_images.py
│   │   ├── parse_references.py
│   │   ├── generate_chart.py
│   │   └── save_markdown.py
│   ├── prompts.py              # System/user prompt templates
│   └── config.py               # Model name, paths, settings
├── app.py                      # Streamlit UI
├── input/                      # PDF drop folder
├── output/                     # Generated summaries
└── docs/
    ├── 00-project-overview.md
    ├── 01-setup.md
    ├── 02-pdf-extraction.md
    ├── 03-tools.md
    ├── 04-agent.md
    └── 05-ui.md
```

### Output Format

```markdown
# Paper Summary: <Paper Title>

**Authors:** <Author list>
**Published:** <Year, venue/journal>
**DOI/Link:** <if available>

---

## Overview
<3-5 sentence summary of the paper's context and what it addresses>

## Contribution
<What the authors propose — 4-6 sentences covering the novelty and
why it matters>

## State of the Art
<Summary of the related work landscape as described in the paper —
what approaches exist, their limitations, how this work positions
itself among them. ~1-2 paragraphs>

## Methodology

<Overview paragraph: high-level description of the method/approach,
how the pieces fit together>

### Method Details
<Detailed walkthrough of the method including key formulations,
loss functions, architectural choices, algorithms. Formulae included
where central to understanding. Concise but nothing important omitted
— only truly minor implementation details skipped.>

![<caption>](images/<figure_file>)
*<caption text>*

## Evaluation
<Datasets used, evaluation metrics, baselines compared against,
experimental setup>

## Key Results
<Main quantitative and qualitative findings — bullet points with
numbers where available>

![<caption>](images/<figure_file>)
*<caption text>*

## References Analysis
![Reference Distribution by Journal](images/references_piechart.png)

<Top 5 most-cited journals/venues listed>

---
*Generated by Research Paper Copilot on <date>*
```

## Model Considerations

The model is kept configurable. Candidates for local Ollama deployment on 2x RTX 4090:
- **mistral-small** — best documented for tool calling + local use
- **mistral-small3.2** — 24B, improved function calling, vision capable
- **gpt-oss:20b** — built for function calling and structured outputs
- **devstral-small-2** — best for code-centric tasks

The model choice will be finalized during setup.
