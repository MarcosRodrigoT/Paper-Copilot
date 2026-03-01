"""
The Research Paper Copilot agent.

This module orchestrates the paper summarization pipeline. It uses a
hybrid approach:

- **Deterministic steps** (code): PDF extraction, image extraction,
  reference parsing, chart generation, and markdown assembly are called
  directly — they don't need LLM reasoning.
- **LLM steps**: The model writes section summaries one at a time.
  Each call gets only the relevant paper sections, keeping the context
  small enough for local models to handle reliably.

Usage:
    python -m src.agent path/to/paper.pdf
"""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from langchain_ollama import ChatOllama

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OUTPUT_DIR
from src.prompts import (
    CONTRIBUTION_PROMPT,
    EVALUATION_PROMPT,
    FIGURE_SELECTION_PROMPT,
    KEY_RESULTS_PROMPT,
    METADATA_PROMPT,
    METHODOLOGY_DETAILS_PROMPT,
    METHODOLOGY_OVERVIEW_PROMPT,
    OVERVIEW_PROMPT,
    SECTION_MAPPING,
    STATE_OF_ART_PROMPT,
)
from src.tools.extract_images import _extract_images_from_pdf
from src.tools.extract_text import _extract_sections
from src.tools.generate_chart import _generate_pie_chart
from src.tools.parse_references import _parse_reference_list
from src.tools.save_markdown import _build_markdown, _save_output


def _create_llm(model_name: str | None = None) -> ChatOllama:
    """Create a ChatOllama instance."""
    return ChatOllama(
        model=model_name or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _find_references_section(sections: list[dict]) -> str:
    """Find the references/bibliography section from extracted sections."""
    ref_keywords = ["references", "bibliography", "works cited"]
    for section in sections:
        if any(kw in section["heading"].lower() for kw in ref_keywords):
            return section["content"]

    # Fallback: check the last section (references are usually last)
    if sections and len(sections) > 2:
        last = sections[-1]
        if len(last["content"]) > 200:
            return last["content"]

    return ""


def _get_relevant_text(sections: list[dict], section_type: str) -> str:
    """
    Get the text from paper sections that are relevant to a given summary section.

    Uses SECTION_MAPPING to match paper headings to output section types.
    Falls back to abstract + introduction if no specific match is found.
    """
    keywords = SECTION_MAPPING.get(section_type, [])
    matched = []

    for section in sections:
        heading_lower = section["heading"].lower()
        if any(kw in heading_lower for kw in keywords):
            matched.append(f"## {section['heading']}\n{section['content']}")

    if matched:
        return "\n\n".join(matched)

    # Fallback: use abstract + introduction
    fallback = []
    for section in sections[:3]:  # First few sections
        fallback.append(f"## {section['heading']}\n{section['content']}")
    return "\n\n".join(fallback)


def _clean_llm_output(text: str) -> str:
    """Clean LLM output: strip thinking tags and normalize LaTeX delimiters."""
    import re

    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Convert \[...\] to $$...$$ (display math)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # Convert \(...\) to $...$ (inline math)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)

    return text


def _llm_call(llm: ChatOllama, prompt: str, label: str) -> str:
    """Make an LLM call with logging."""
    print(f"      Calling LLM for: {label}...")
    response = llm.invoke(prompt)
    content = _clean_llm_output(response.content)
    print(f"      Got {len(content)} chars")
    return content


def _parse_metadata(text: str) -> dict:
    """Parse the metadata response into title, authors, published, doi."""
    result = {"title": "", "authors": "", "published": "", "doi": ""}
    for line in text.split("\n"):
        line = line.strip()
        lower = line.lower()
        if lower.startswith("title:"):
            result["title"] = line[6:].strip().strip("*")
        elif lower.startswith("authors:"):
            result["authors"] = line[8:].strip()
        elif lower.startswith("published"):
            # Handle "Published:" or "Published (year and venue):"
            idx = line.find(":")
            if idx != -1:
                result["published"] = line[idx + 1 :].strip()
        elif lower.startswith("doi"):
            idx = line.find(":")
            if idx != -1:
                val = line[idx + 1 :].strip()
                if val.lower() != "none":
                    result["doi"] = val
    return result


def process_paper(pdf_path: str, model_name: str | None = None) -> str:
    """
    Process a single paper end-to-end.

    Pipeline:
    1. Extract text sections from PDF (code)
    2. Extract images from PDF (code)
    3. Parse references and generate chart (code)
    4. LLM generates each summary section individually
    5. LLM selects relevant figures
    6. Assemble and save markdown (code)
    """
    pdf_path = str(Path(pdf_path).resolve())
    paper_name = Path(pdf_path).stem

    # Create output directories
    paper_output_dir = OUTPUT_DIR / paper_name
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = paper_output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    chart_path = str(images_dir / "references_piechart.png")

    # Temp directory for extracted images
    tmp_images = tempfile.mkdtemp(prefix="paper_copilot_")

    llm = _create_llm(model_name)

    print(f"Processing: {pdf_path}")
    print(f"Output: {paper_output_dir}")
    print(f"Model: {model_name or OLLAMA_MODEL}")
    print("-" * 60)

    # ── Step 1: Extract text ──────────────────────────────────────
    print("[1/6] Extracting text from PDF...")
    sections = _extract_sections(pdf_path)
    print(f"      Found {len(sections)} sections")

    # ── Step 2: Extract images ────────────────────────────────────
    print("[2/6] Extracting images from PDF...")
    images = _extract_images_from_pdf(pdf_path, tmp_images)
    print(f"      Found {len(images)} images")

    # ── Step 3: Parse references & generate chart ─────────────────
    print("[3/6] Parsing references...")
    refs_text = _find_references_section(sections)
    parsed_refs = _parse_reference_list(refs_text) if refs_text else []
    print(f"      Parsed {len(parsed_refs)} references")

    chart_generated = False
    if parsed_refs:
        print("      Generating references chart...")
        _generate_pie_chart(json.dumps(parsed_refs), chart_path)
        chart_generated = Path(chart_path).exists()

    # ── Step 4: LLM generates summaries (one section at a time) ──
    print("[4/6] Generating summaries section by section...")

    # 4a: Metadata — always include the first section (title + authors live there)
    meta_parts = []
    if sections:
        first = sections[0]
        meta_parts.append(f"## {first['heading']}\n{first['content']}")
    meta_extra = _get_relevant_text(sections, "metadata")
    if meta_extra and meta_extra not in meta_parts:
        meta_parts.append(meta_extra)
    meta_text = "\n\n".join(meta_parts)
    meta_raw = _llm_call(llm, METADATA_PROMPT.format(text=meta_text), "metadata")
    metadata = _parse_metadata(meta_raw)

    # 4b: Overview
    overview_text = _get_relevant_text(sections, "overview")
    overview = _llm_call(llm, OVERVIEW_PROMPT.format(text=overview_text), "overview")

    # 4c: Contribution
    contrib_text = _get_relevant_text(sections, "contribution")
    contribution = _llm_call(
        llm, CONTRIBUTION_PROMPT.format(text=contrib_text), "contribution"
    )

    # 4d: State of the Art
    sota_text = _get_relevant_text(sections, "state_of_the_art")
    state_of_the_art = _llm_call(
        llm, STATE_OF_ART_PROMPT.format(text=sota_text), "state of the art"
    )

    # 4e: Methodology overview
    method_text = _get_relevant_text(sections, "methodology_overview")
    methodology_overview = _llm_call(
        llm,
        METHODOLOGY_OVERVIEW_PROMPT.format(text=method_text),
        "methodology overview",
    )

    # 4f: Methodology details
    methodology_details = _llm_call(
        llm,
        METHODOLOGY_DETAILS_PROMPT.format(text=method_text),
        "methodology details",
    )

    # 4g: Evaluation
    eval_text = _get_relevant_text(sections, "evaluation")
    evaluation = _llm_call(
        llm, EVALUATION_PROMPT.format(text=eval_text), "evaluation"
    )

    # 4h: Key results
    results_text = _get_relevant_text(sections, "key_results")
    key_results = _llm_call(
        llm, KEY_RESULTS_PROMPT.format(text=results_text), "key results"
    )

    # Build references summary
    venue_counts = Counter(r.get("venue", "Unknown") for r in parsed_refs)
    top_venues = venue_counts.most_common(5)
    refs_summary = (
        "**Top referenced venues:** "
        + ", ".join(f"{v} ({c})" for v, c in top_venues)
        if top_venues
        else "No references parsed."
    )

    # ── Step 5: LLM selects figures ──────────────────────────────
    selected_figures = []
    if images:
        print("[5/6] Selecting relevant figures...")
        images_description = json.dumps(images, indent=2)
        brief_summary = overview[:500] if overview else "Research paper"
        figure_message = FIGURE_SELECTION_PROMPT.format(
            images_description=images_description,
            summary=brief_summary,
        )
        fig_response = llm.invoke(figure_message)
        selected_figures = _parse_figure_selection(fig_response.content, images)
        print(f"      Selected {len(selected_figures)} figures")
    else:
        print("[5/6] No images found, skipping figure selection.")

    # ── Step 6: Assemble markdown ────────────────────────────────
    print("[6/6] Assembling markdown...")
    chart_filename = Path(chart_path).name if chart_generated else ""

    md_content = _build_markdown(
        title=metadata.get("title", paper_name),
        authors=metadata.get("authors", "Unknown"),
        published=metadata.get("published", "Unknown"),
        doi=metadata.get("doi", ""),
        overview=overview,
        contribution=contribution,
        state_of_the_art=state_of_the_art,
        methodology_overview=methodology_overview,
        methodology_details=methodology_details,
        evaluation=evaluation,
        key_results=key_results,
        references_summary=refs_summary,
        chart_filename=chart_filename,
        figures=selected_figures,
    )

    figure_filenames = [f["filename"] for f in selected_figures]
    summary_path = _save_output(
        markdown_content=md_content,
        output_dir=str(paper_output_dir),
        image_source_dir=tmp_images,
        chart_source_path=chart_path if chart_generated else "",
        figure_filenames=figure_filenames,
    )

    print("-" * 60)
    print(f"Done! Summary saved to: {summary_path}")
    return summary_path


def _parse_figure_selection(text: str, available_images: list[dict]) -> list[dict]:
    """
    Parse the LLM's figure selection response.

    The LLM is asked to output lines like:
        figure_01.png -> methodology
        figure_03.png -> key results
    """
    selected = []
    available_filenames = {img["filename"] for img in available_images}
    caption_map = {img["filename"]: img.get("caption", "") for img in available_images}

    for line in text.split("\n"):
        line = line.strip().lower()
        if "->" not in line:
            continue
        parts = line.split("->")
        if len(parts) != 2:
            continue

        filename_part = parts[0].strip()
        section_part = parts[1].strip()

        # Find matching filename
        matched_file = None
        for fname in available_filenames:
            if fname.lower() in filename_part or filename_part in fname.lower():
                matched_file = fname
                break

        if matched_file:
            selected.append({
                "filename": matched_file,
                "caption": caption_map.get(matched_file, ""),
                "section": section_part,
            })

    # If LLM didn't follow the format, just take first few images
    if not selected and available_images:
        for img in available_images[:3]:
            selected.append({
                "filename": img["filename"],
                "caption": img.get("caption", ""),
                "section": "methodology",
            })

    return selected


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent <path-to-pdf> [model-name]")
        print("Example: python -m src.agent input/paper.pdf gpt-oss")
        sys.exit(1)

    pdf = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(pdf).exists():
        print(f"Error: file not found: {pdf}")
        sys.exit(1)

    process_paper(pdf, model)
