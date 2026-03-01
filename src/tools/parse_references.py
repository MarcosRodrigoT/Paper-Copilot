"""
Tool: parse_references

Parses the references/bibliography section from extracted paper text
and extracts structured information (title, authors, venue, year).

This uses heuristics — reference formats vary widely across papers
(IEEE, ACM, APA, numbered, etc.), so we do best-effort extraction.
"""

import json
import re

from langchain_core.tools import tool


def _split_references(references_text: str) -> list[str]:
    """
    Split a block of references text into individual references.

    Handles common formats:
    - Numbered: [1] Author... or 1. Author...
    - Unnumbered: separated by blank lines or newlines with author patterns
    """
    # Try numbered format: [1], [2], ...
    # Allow optional newline/space before the bracket (PDF extraction often
    # joins everything into a single line with spaces)
    numbered = re.split(r"\s*\[(\d+)\]\s*", references_text)
    if len(numbered) > 3:
        # numbered[0] is text before first ref, then alternating: number, text
        refs = [numbered[i].strip() for i in range(2, len(numbered), 2)]
        return [r for r in refs if len(r) > 20]

    # Try "1." style numbering
    dot_numbered = re.split(r"(?:\n|^)\s*(\d+)\.\s+", references_text)
    if len(dot_numbered) > 3:
        refs = [dot_numbered[i].strip() for i in range(2, len(dot_numbered), 2)]
        return [r for r in refs if len(r) > 20]

    # Fallback: split on double newlines
    refs = [r.strip() for r in references_text.split("\n\n") if len(r.strip()) > 20]
    if refs:
        return refs

    # Last resort: each line is a reference
    return [r.strip() for r in references_text.split("\n") if len(r.strip()) > 20]


def _extract_year(ref_text: str) -> str:
    """Extract a 4-digit year (1900-2099) from a reference string."""
    match = re.search(r"\b((?:19|20)\d{2})\b", ref_text)
    return match.group(1) if match else ""


def _extract_venue(ref_text: str) -> str:
    """
    Best-effort extraction of the venue/journal from a reference.

    Strategy: try patterns from most specific to least specific.
    """
    # arXiv preprints (very common in ML/AI papers)
    if re.search(r"arXiv preprint|arXiv:\d{4}\.\d+", ref_text, re.IGNORECASE):
        return "arXiv"

    # CoRR (another common preprint label)
    if re.search(r"\bCoRR\b", ref_text):
        return "arXiv"

    # "In <Venue>" pattern (common in CS papers)
    in_match = re.search(
        r"\bIn\s+([A-Z][\w\s:&\-]+?)(?:,\s*\d{4}|\.\s|\s*pp\.|\s*$)", ref_text
    )
    if in_match:
        return in_match.group(1).strip().rstrip(",.")

    # Known venue abbreviations
    venue_patterns = [
        r"((?:IEEE|ACM|AAAI|NeurIPS|NIPS|ICML|CVPR|ICCV|ECCV|ICLR|NAACL|ACL|"
        r"EMNLP|SIGIR|KDD|WWW|IJCAI|AISTATS|ICRA|IROS|RSS|CoRL|INTERSPEECH|"
        r"EACL|COLING|TACL)[\w\s\-]*)",
        r"((?:Journal|Trans(?:actions)?\.?|Proceedings|Conference|Workshop|Symposium)"
        r"\s+[\w\s\-:&]+?)(?:,\s*\d|\.\s|\s*pp\.|\s*\d+\s*[\(:]])",
        r"((?:Nature|Science|PNAS|JMLR|PAMI|TIP|TNNLS|Artificial Intelligence|"
        r"Machine Learning|Neural Networks|Pattern Recognition|Computational "
        r"Linguistics|Neural Computation)[\w\s\-]*?)"
        r"(?:,\s*\d|\.\s|\s*pp\.)",
    ]
    for pattern in venue_patterns:
        match = re.search(pattern, ref_text)
        if match:
            return match.group(1).strip().rstrip(",.")

    return "Unknown"


def _parse_reference_list(references_text: str) -> list[dict]:
    """
    Parse a references section into structured entries.

    Returns a list of dicts with: raw_text, year, venue.
    """
    individual_refs = _split_references(references_text)
    parsed = []

    for ref_text in individual_refs:
        parsed.append(
            {
                "raw_text": ref_text[:300],  # truncate very long refs
                "year": _extract_year(ref_text),
                "venue": _extract_venue(ref_text),
            }
        )

    return parsed


@tool
def parse_references(references_text: str) -> str:
    """Parse the references section text from a research paper.

    Args:
        references_text: The raw text of the references/bibliography section.

    Returns:
        JSON string with a list of parsed references, each having
        raw_text, year, and venue fields.
    """
    refs = _parse_reference_list(references_text)
    return json.dumps(refs, ensure_ascii=False)
