# Step 2: PDF Text Extraction

This document explains how the `extract_text` tool works — our first LangChain tool.

## What is a LangChain tool?

A **tool** is a Python function that an LLM agent can call. Instead of the agent doing everything through text generation, it can invoke tools to perform specific actions (read files, query databases, make calculations, etc.).

In LangChain, you create a tool by decorating a function with `@tool`:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Description the LLM reads to decide when to use this tool."""
    return do_something(arg)
```

The key parts:
1. **Type hints** — the agent uses these to know what arguments to pass
2. **Docstring** — the agent reads this to decide *when* to call the tool
3. **Return type** — should be a string (the agent reads the result as text)

## How PDF text extraction works

### The library: PyMuPDF

PyMuPDF (imported as `fitz`) is a fast PDF parsing library. It can extract:
- Text with font information (size, bold, italic)
- Images embedded in the PDF
- Page structure and layout

### Our strategy for detecting sections

Research papers have a consistent visual structure: **headings are in a larger font** than body text. We exploit this:

```
┌─────────────────────────────────┐
│  "Abstract"       fontsize=14   │ ← heading (larger than body)
│  "This paper..."  fontsize=10   │ ← body text
│  "Introduction"   fontsize=14   │ ← heading
│  "We propose..."  fontsize=10   │ ← body text
└─────────────────────────────────┘
```

The algorithm:

1. **First pass** — Read every text span from the PDF with its font size
2. **Find body size** — The most common font size (by character count) is the body text
3. **Heading threshold** — Anything 1+ points larger than body size is a heading
4. **Second pass** — Walk through spans, starting a new section each time a heading is found

### Output format

The tool returns a JSON array of sections:

```json
[
  {
    "heading": "Abstract",
    "content": "This paper presents a method for...",
    "page": 1
  },
  {
    "heading": "Introduction",
    "content": "Recent advances in...",
    "page": 1
  }
]
```

### Limitations

- **Two-column layouts:** PyMuPDF reads text in visual order, which can interleave columns. Most papers still produce reasonable results because headings span full width.
- **Papers without clear font size hierarchy:** Some papers use the same font size for headings (just bold). Our heading detection would miss these. A future improvement could also check for bold flags.
- **Very long papers:** The full text is returned as one JSON string. If a paper is very long, this could be a lot of text for the LLM to process in one go.

## Code walkthrough

See [src/tools/extract_text.py](../src/tools/extract_text.py):

- `_extract_sections(pdf_path)` — The internal function that does the actual work. It's kept separate from the tool wrapper so it can be tested and reused independently.
- `extract_text(pdf_path)` — The `@tool`-decorated function the agent calls. It calls `_extract_sections` and returns the result as a JSON string.

## Testing

```bash
# Quick test with a PDF
uv run python -c "
from src.tools.extract_text import _extract_sections
import json
sections = _extract_sections('input/test_paper.pdf')
print(json.dumps(sections, indent=2))
"
```

## Next step

The next tool extracts **images** from the PDF: [Step 3](./02-pdf-extraction.md#next-step) continues with `extract_images`.
