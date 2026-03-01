"""
Tool: extract_text

Extracts text from a PDF and organizes it by sections.
Uses PyMuPDF (fitz) to read the PDF and detect section headings
based on font size differences.
"""

import json

import fitz  # PyMuPDF
from langchain_core.tools import tool


def _extract_sections(pdf_path: str) -> list[dict]:
    """
    Parse a PDF and split its text into sections based on heading detection.

    Strategy:
    1. Read all text blocks with their font sizes from each page.
    2. Identify headings by finding text spans that are significantly larger
       than the body text (the most common font size).
    3. Group consecutive body text under each heading.

    Returns a list of dicts: [{"heading": "...", "content": "...", "page": N}, ...]
    """
    doc = fitz.open(pdf_path)

    # First pass: collect all text spans with font info
    spans = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        spans.append(
                            {
                                "text": text,
                                "size": round(span["size"], 1),
                                "flags": span["flags"],
                                "page": page_num + 1,
                            }
                        )

    if not spans:
        doc.close()
        return [{"heading": "Full Text", "content": "", "page": 1}]

    # Find the most common font size (= body text size)
    size_counts: dict[float, int] = {}
    for span in spans:
        size_counts[span["size"]] = size_counts.get(span["size"], 0) + len(
            span["text"]
        )
    body_size = max(size_counts, key=size_counts.get)

    # A span is a "heading" if its font size is noticeably larger than body text
    heading_threshold = body_size + 1.0

    # Second pass: build sections
    sections = []
    current_heading = "Preamble"
    current_content: list[str] = []
    current_page = 1

    for span in spans:
        is_heading = span["size"] >= heading_threshold and len(span["text"]) < 200

        if is_heading:
            # Save previous section
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": " ".join(current_content),
                        "page": current_page,
                    }
                )
            current_heading = span["text"]
            current_content = []
            current_page = span["page"]
        else:
            current_content.append(span["text"])

    # Don't forget the last section
    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": " ".join(current_content),
                "page": current_page,
            }
        )

    doc.close()
    return sections


@tool
def extract_text(pdf_path: str) -> str:
    """Extract text from a research paper PDF, organized by sections.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        JSON string with a list of sections, each having a heading,
        content, and page number.
    """
    sections = _extract_sections(pdf_path)
    return json.dumps(sections, ensure_ascii=False)
