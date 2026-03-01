# Steps 4-6: All the Tools

This document covers the remaining three tools (reference parsing, chart generation, markdown assembly) and how all five tools fit together.

## Quick recap: what is a tool?

A tool is a function the agent can call. The agent reads the tool's **name** and **docstring** to decide when to use it, then passes arguments matching the function's **type hints**.

All five tools in this project follow the same pattern:

```python
@tool
def tool_name(arg: str) -> str:
    """Description the agent reads."""
    # do work
    return result_as_string
```

## Tool 4: parse_references

**File:** `src/tools/parse_references.py`

**Purpose:** Takes the raw text of a paper's references section and extracts structured data — specifically the **venue/journal name** and **year** for each reference.

### How it works

Reference formats vary wildly (IEEE, ACM, APA, numbered, etc.), so this uses a cascade of heuristics:

**Step 1 — Split into individual references:**
```
[1] Vaswani et al. Attention is all you need. In NeurIPS, 2017.
[2] Devlin et al. BERT: Pre-training of... In NAACL, 2019.
```
We try splitting by `[N]` markers first, then `N.` numbering, then blank lines, then individual lines.

**Step 2 — Extract year:**
Simple regex for 4-digit years in the 1900-2099 range.

**Step 3 — Extract venue:**
A cascade of patterns:
1. Look for `"In <Venue>"` pattern (very common in CS papers)
2. Look for known conference abbreviations (IEEE, NeurIPS, CVPR, etc.)
3. Look for generic venue words (Journal, Transactions, Proceedings, etc.)
4. Fall back to "Unknown"

### Limitations

- Non-English references may not parse correctly
- Venue extraction is heuristic — unusual formats will return "Unknown"
- The tool doesn't extract individual author names or paper titles (just the raw text)

## Tool 5: generate_chart

**File:** `src/tools/generate_chart.py`

**Purpose:** Takes the parsed references (JSON from `parse_references`) and creates a pie chart PNG showing venue distribution.

### How it works

```
Parsed references → Count venues → Top 8 + "Other" → matplotlib pie chart → PNG
```

Key design choices:
- **Max 8 slices** — more than that makes a pie chart unreadable. Everything else goes into "Other".
- **Legend instead of labels** — venue names are long; a side legend with counts is cleaner than trying to label each wedge.
- **Percentages only for slices > 5%** — avoids cluttering tiny slices with unreadable text.
- **`Agg` backend** — matplotlib is set to the non-interactive backend since we never show charts on screen, only save to files.

### The output

A PNG image at 150 DPI. Example:

```
┌────────────────────────────────────┐
│                                    │
│    [Pie chart]     Venue           │
│                    ● CVPR (3)      │
│                    ● NeurIPS (2)   │
│                    ● ICML (1)      │
│                    ● Other (2)     │
│                                    │
│   "Reference Distribution by       │
│    Venue"                          │
└────────────────────────────────────┘
```

## Tool 6: save_markdown

**File:** `src/tools/save_markdown.py`

**Purpose:** Takes all the content the agent has generated (summaries, figure selections, chart path) and assembles the final markdown file.

### Input

The agent passes a single JSON string with all fields:

```json
{
  "title": "Attention Is All You Need",
  "authors": "Vaswani et al.",
  "published": "2017, NeurIPS",
  "doi": "...",
  "overview": "This paper introduces...",
  "contribution": "The authors propose...",
  "state_of_the_art": "Prior work on sequence...",
  "methodology_overview": "The Transformer model...",
  "methodology_details": "The model uses scaled dot-product attention...",
  "evaluation": "The model was evaluated on...",
  "key_results": "- BLEU score of 28.4 on...\n- ...",
  "references_summary": "Top venues: NeurIPS (5), ICML (3), ...",
  "chart_source_path": "/path/to/chart.png",
  "image_source_dir": "/path/to/extracted/images",
  "output_dir": "/path/to/output/paper_name",
  "figures": [
    {"filename": "figure_01.png", "caption": "Figure 1: ...", "section": "methodology"},
    {"filename": "figure_03.png", "caption": "Figure 3: ...", "section": "key results"}
  ]
}
```

### How figures get placed

Each figure in the `figures` list has a `"section"` field. The markdown builder inserts figures at the end of the matching section. The agent decides which figures go with which sections when it selects them.

### Output directory layout

```
output/<paper_name>/
├── summary.md          # The final markdown
└── images/
    ├── figure_01.png   # Copied from extraction
    ├── figure_03.png
    └── references_piechart.png
```

Images are **copied** (not moved) so the originals in the temp extraction directory are preserved.

## How all 5 tools work together

```
                    ┌──────────────────────────┐
                    │         Agent             │
                    │  (decides tool order)      │
                    └──────┬───────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   extract_text     extract_images    parse_references
   (PDF → sections) (PDF → PNGs)     (text → venues)
          │                │                │
          ▼                │                ▼
   Agent summarizes        │         generate_chart
   each section            │         (venues → pie PNG)
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    save_markdown
                    (everything → .md + images/)
```

The agent calls the first three tools to gather data, then uses the LLM to generate summaries for each section, then calls the last two tools to produce the final output.

## Next step

With all tools built, the next step is wiring them into the **LangChain agent** — see `docs/04-agent.md`.
