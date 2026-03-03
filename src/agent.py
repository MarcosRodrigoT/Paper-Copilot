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

import gc
import json
import logging
import sys
import tempfile

# Suppress noisy INFO/DEBUG logs from Docling, RapidOCR, and friends.
# Must be set before those libraries are imported.
for _logger_name in ("docling", "docling_core", "rapidocr", "RapidOCR", "deepsearch_glm"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
from collections import Counter
from collections.abc import Callable
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
from src.tools.parse_references import (
    _classify_unknown_venues,
    _normalize_venue,
    _parse_reference_list,
)
from src.tools.save_markdown import _build_markdown, _save_output


def _convert_pdf(pdf_path: str):
    """Convert a PDF once using Docling, returning the conversion result.

    Enables picture image extraction so both text and image extractors
    can share the same result without converting twice.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter.convert(pdf_path)


def _free_gpu_memory():
    """Force-free GPU memory held by Docling's ML models.

    Removes cached docling modules so their model objects become
    unreachable, then runs garbage collection and clears the CUDA cache.
    """
    # Drop cached docling modules so model objects can be GC'd.
    to_remove = [k for k in sys.modules if k.startswith(("docling", "docling_core"))]
    for key in to_remove:
        del sys.modules[key]

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def _llm_call(
    llm: ChatOllama,
    prompt: str,
    label: str,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Make an LLM call with logging."""
    msg = f"Calling LLM for: {label}..."
    if on_progress:
        on_progress(msg)
    else:
        print(f"      {msg}")
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


def process_paper(
    pdf_path: str,
    model_name: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> str:
    """
    Process a single paper end-to-end.

    Args:
        pdf_path: Path to the PDF file.
        model_name: Optional Ollama model name override.
        on_progress: Optional callback(step_label, detail_message) for UI updates.

    Pipeline:
    1. Extract text sections from PDF (code)
    2. Extract images from PDF (code)
    3. Parse references and generate chart (code)
    4. LLM generates each summary section individually
    5. LLM selects relevant figures
    6. Assemble and save markdown (code)
    """

    def _progress(step: str, detail: str = ""):
        print(f"{step} {detail}".strip())
        if on_progress:
            on_progress(step, detail)

    def _llm_with_progress(prompt: str, label: str) -> str:
        return _llm_call(llm, prompt, label, on_progress=lambda msg: _progress("", msg))

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

    _progress(f"Processing: {pdf_path}")
    _progress(f"Model: {model_name or OLLAMA_MODEL}")

    # ── Step 0: Convert PDF with Docling ─────────────────────────
    _progress("[0/6] Converting PDF with Docling...")
    conv_result = _convert_pdf(pdf_path)

    # ── Step 1: Extract text ──────────────────────────────────────
    _progress("[1/6] Extracting text sections...")
    sections = _extract_sections(conv_result)
    _progress("[1/6]", f"Found {len(sections)} sections")

    # ── Step 2: Extract images ────────────────────────────────────
    _progress("[2/6] Extracting images...")
    images = _extract_images_from_pdf(conv_result, tmp_images)
    _progress("[2/6]", f"Found {len(images)} images")

    # Free Docling models and GPU memory now that extraction is done.
    del conv_result
    _free_gpu_memory()

    # ── Step 3: Parse references & generate chart ─────────────────
    _progress("[3/6] Parsing references...")
    refs_text = _find_references_section(sections)
    parsed_refs = _parse_reference_list(refs_text) if refs_text else []
    _progress("[3/6]", f"Parsed {len(parsed_refs)} references")

    # Classify unknown/vague venues using LLM
    from src.tools.parse_references import _is_vague_venue

    needs_llm = sum(
        1 for r in parsed_refs
        if r.get("venue") == "Unknown" or _is_vague_venue(r.get("venue", ""))
    )
    if needs_llm > 0:
        _progress("[3/6]", f"{needs_llm} references need LLM venue classification...")
        parsed_refs = _classify_unknown_venues(
            parsed_refs, _llm_with_progress
        )
        remaining = sum(1 for r in parsed_refs if r.get("venue") == "Unknown")
        _progress("[3/6]", f"After LLM: {remaining} still unknown")

    # Normalize all venue names for consistent chart labels
    for ref in parsed_refs:
        ref["venue"] = _normalize_venue(ref.get("venue", "Unknown"))

    chart_generated = False
    if parsed_refs:
        _progress("[3/6]", "Generating references chart...")
        _generate_pie_chart(json.dumps(parsed_refs), chart_path)
        chart_generated = Path(chart_path).exists()

    # ── Step 4: LLM generates summaries (one section at a time) ──
    _progress("[4/6] Generating summaries section by section...")

    # 4a: Metadata — always include the first section (title + authors live there)
    meta_parts = []
    if sections:
        first = sections[0]
        meta_parts.append(f"## {first['heading']}\n{first['content']}")
    meta_extra = _get_relevant_text(sections, "metadata")
    if meta_extra and meta_extra not in meta_parts:
        meta_parts.append(meta_extra)
    meta_text = "\n\n".join(meta_parts)
    meta_raw = _llm_with_progress(
        METADATA_PROMPT.format(text=meta_text), "metadata"
    )
    metadata = _parse_metadata(meta_raw)

    # 4b: Overview
    overview_text = _get_relevant_text(sections, "overview")
    overview = _llm_with_progress(
        OVERVIEW_PROMPT.format(text=overview_text), "overview"
    )

    # 4c: Contribution
    contrib_text = _get_relevant_text(sections, "contribution")
    contribution = _llm_with_progress(
        CONTRIBUTION_PROMPT.format(text=contrib_text), "contribution"
    )

    # 4d: State of the Art
    sota_text = _get_relevant_text(sections, "state_of_the_art")
    state_of_the_art = _llm_with_progress(
        STATE_OF_ART_PROMPT.format(text=sota_text), "state of the art"
    )

    # 4e: Methodology overview
    method_text = _get_relevant_text(sections, "methodology_overview")
    methodology_overview = _llm_with_progress(
        METHODOLOGY_OVERVIEW_PROMPT.format(text=method_text),
        "methodology overview",
    )

    # 4f: Methodology details
    methodology_details = _llm_with_progress(
        METHODOLOGY_DETAILS_PROMPT.format(text=method_text),
        "methodology details",
    )

    # 4g: Evaluation
    eval_text = _get_relevant_text(sections, "evaluation")
    evaluation = _llm_with_progress(
        EVALUATION_PROMPT.format(text=eval_text), "evaluation"
    )

    # 4h: Key results
    results_text = _get_relevant_text(sections, "key_results")
    key_results = _llm_with_progress(
        KEY_RESULTS_PROMPT.format(text=results_text), "key results"
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
        _progress("[5/6] Selecting relevant figures...")
        images_description = json.dumps(images, indent=2)
        brief_summary = overview[:500] if overview else "Research paper"
        figure_message = FIGURE_SELECTION_PROMPT.format(
            images_description=images_description,
            summary=brief_summary,
        )
        fig_response = llm.invoke(figure_message)
        selected_figures = _parse_figure_selection(fig_response.content, images)
        _progress("[5/6]", f"Selected {len(selected_figures)} figures")
    else:
        _progress("[5/6] No images found, skipping figure selection.")

    # ── Step 6: Assemble markdown ────────────────────────────────
    _progress("[6/6] Assembling markdown...")
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

    _progress("Done!", f"Summary saved to: {summary_path}")
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
