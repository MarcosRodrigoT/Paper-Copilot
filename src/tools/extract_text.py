"""
Tool: extract_text

Extracts text from a PDF and organizes it by sections.
Uses Docling to parse the document structure and detect section headings.
"""

import json
import re

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TextItem
from langchain_core.tools import tool

# Pattern for numbered section headings (e.g. "1. Introduction", "I. INTRODUCTION", "2 Methods")
_NUMBERED_HEADING = re.compile(
    r"^(?:[0-9]+\.?\s|[IVXLC]+\.?\s|[A-Z]\.?\s)", re.IGNORECASE
)


def _is_numbered_heading(text: str) -> bool:
    """Check if a heading looks like a numbered section (not the paper title)."""
    return bool(_NUMBERED_HEADING.match(text.strip()))


def _extract_sections(conv_result) -> list[dict]:
    """
    Extract text sections from a Docling conversion result.

    Iterates the document's items in reading order, grouping body text
    under section headers detected by Docling's layout analysis.

    The first section_header on page 1 that appears before any numbered
    section heading is treated as the paper title and included in the
    Preamble content (so metadata extraction can find it).

    Args:
        conv_result: A Docling ConversionResult from DocumentConverter.convert().

    Returns a list of dicts: [{"heading": "...", "content": "...", "page": N}, ...]
    """
    doc = conv_result.document

    sections = []
    current_heading = "Preamble"
    current_content: list[str] = []
    current_page = 1
    seen_numbered_heading = False

    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue

        text = item.text.strip()
        if not text:
            continue

        label = str(item.label).lower()
        page = item.prov[0].page_no if item.prov else current_page

        if "section_header" in label or "title" in label:
            is_numbered = _is_numbered_heading(text)

            # The first non-numbered heading on page 1 before any numbered
            # sections is the paper title — fold it into Preamble content.
            if (
                not seen_numbered_heading
                and not is_numbered
                and page == 1
                and current_heading == "Preamble"
            ):
                current_content.append(text)
                continue

            if is_numbered:
                seen_numbered_heading = True

            # Save previous section
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n\n".join(current_content),
                        "page": current_page,
                    }
                )
            current_heading = text
            current_content = []
            current_page = page
        else:
            current_content.append(text)

    # Don't forget the last section
    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n\n".join(current_content),
                "page": current_page,
            }
        )

    if not sections:
        return [{"heading": "Full Text", "content": "", "page": 1}]

    return sections


def _convert_pdf_for_text(pdf_path: str):
    """Standalone Docling conversion for the @tool entry point."""
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions())
        }
    )
    return converter.convert(pdf_path)


@tool
def extract_text(pdf_path: str) -> str:
    """Extract text from a research paper PDF, organized by sections.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        JSON string with a list of sections, each having a heading,
        content, and page number.
    """
    conv_result = _convert_pdf_for_text(pdf_path)
    sections = _extract_sections(conv_result)
    return json.dumps(sections, ensure_ascii=False)
