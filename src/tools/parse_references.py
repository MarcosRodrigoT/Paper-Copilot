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


def _normalize_venue(venue: str) -> str:
    """Normalize venue names: strip years, unify common variations."""
    # Remove trailing year: "NeurIPS 2020" -> "NeurIPS"
    venue = re.sub(r"\s+\d{4}\s*$", "", venue).strip()

    # Full name -> abbreviation mappings
    _FULL_TO_SHORT = {
        "Advances in Neural Information Processing Systems": "NeurIPS",
        "Neural Information Processing Systems": "NeurIPS",
        "International Conference on Machine Learning": "ICML",
        "Computer Vision and Pattern Recognition": "CVPR",
        "International Conference on Computer Vision": "ICCV",
        "European Conference on Computer Vision": "ECCV",
        "International Conference on Learning Representations": "ICLR",
        "Association for Computational Linguistics": "ACL",
        "Empirical Methods in Natural Language Processing": "EMNLP",
        "North American Chapter of the Association for Computational Linguistics": "NAACL",
        "Association for the Advancement of Artificial Intelligence": "AAAI",
        "International Joint Conference on Artificial Intelligence": "IJCAI",
        "Knowledge Discovery and Data Mining": "KDD",
        "Journal of Machine Learning Research": "JMLR",
        "Computational Linguistics": "CL",
        # IEEE-specific full names
        "IEEE Transactions on Pattern Analysis and Machine Intelligence": "IEEE TPAMI",
        "IEEE Transactions on Image Processing": "IEEE TIP",
        "IEEE Transactions on Circuits and Systems for Video Technology": "IEEE TCSVT",
        "IEEE Transactions on Neural Networks and Learning Systems": "IEEE TNNLS",
        "IEEE Transactions on Multimedia": "IEEE TMM",
        "IEEE Transactions on Signal Processing": "IEEE TSP",
        "IEEE Transactions on Information Forensics and Security": "IEEE TIFS",
        "IEEE Signal Processing Letters": "IEEE SPL",
        "International Conference on Acoustics, Speech and Signal Processing": "ICASSP",
        "International Conference on Multimedia and Expo": "ICME",
        "ACM International Conference on Multimedia": "ACM MM",
        "ACM Multimedia": "ACM MM",
        "Winter Conference on Applications of Computer Vision": "WACV",
        "British Machine Vision Conference": "BMVC",
    }

    # Check if venue contains a known full name (longest match first)
    venue_lower = venue.lower()
    for full_name, short in sorted(
        _FULL_TO_SHORT.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if full_name.lower() in venue_lower:
            return short

    # IEEE Trans. abbreviation patterns (e.g. "IEEE Trans. Circuits Syst. Video Technol.")
    _IEEE_TRANS_PATTERNS = {
        r"circuits\s+.*syst.*video": "IEEE TCSVT",
        r"pattern\s+anal.*mach.*intell": "IEEE TPAMI",
        r"image\s+process": "IEEE TIP",
        r"neural\s+net.*learn": "IEEE TNNLS",
        r"multimedia": "IEEE TMM",
        r"signal\s+process(?!.*lett)": "IEEE TSP",
        r"signal\s+process.*lett": "IEEE SPL",
        r"info.*forensic": "IEEE TIFS",
        r"comput.*vision": "IEEE TCSVT",
        r"knowledge.*data\s+eng": "IEEE TKDE",
        r"cybernetics": "IEEE TSMC",
        r"affective": "IEEE TAFFC",
    }
    if "ieee" in venue_lower and ("trans" in venue_lower or "t." in venue_lower):
        for pattern, short_name in _IEEE_TRANS_PATTERNS.items():
            if re.search(pattern, venue_lower):
                return short_name

    # Direct abbreviation normalizations
    normalizations = {
        "NIPS": "NeurIPS",
        "Proc": "Other",
    }
    if venue in normalizations:
        return normalizations[venue]

    # Check if venue contains a known abbreviation
    known_abbrevs = [
        "NeurIPS", "NIPS", "ICML", "CVPR", "ICCV", "ECCV", "ICLR",
        "ACL", "EMNLP", "NAACL", "AAAI", "IJCAI", "KDD", "SIGIR",
        "WWW", "AISTATS", "ICRA", "IROS", "CoRL", "RSS",
        "INTERSPEECH", "EACL", "COLING", "TACL", "JMLR",
        "ICASSP", "ICME", "WACV", "BMVC",
    ]
    for abbrev in known_abbrevs:
        if abbrev in venue:
            return normalizations.get(abbrev, abbrev)

    # IEEE-specific abbreviations in venue text
    ieee_abbrevs = {
        "TPAMI": "IEEE TPAMI", "TIP": "IEEE TIP", "TCSVT": "IEEE TCSVT",
        "TNNLS": "IEEE TNNLS", "TMM": "IEEE TMM", "TSP": "IEEE TSP",
        "TKDE": "IEEE TKDE", "TAFFC": "IEEE TAFFC",
    }
    for abbrev, full in ieee_abbrevs.items():
        if abbrev in venue:
            return full

    return venue


# Venue labels that are too vague and should be sent to the LLM for refinement
_VAGUE_VENUES = {
    "ieee", "ieee trans", "ieee transactions", "ieee conf", "ieee conference",
    "ieee proc", "ieee proceedings", "acm", "acm conf", "acm proceedings",
    "springer", "elsevier", "proceedings", "conference", "journal", "workshop",
    "symposium", "transactions",
}


def _is_vague_venue(venue: str) -> bool:
    """Check if a venue label is too generic to be useful."""
    return venue.lower().strip().rstrip(".") in _VAGUE_VENUES


def _classify_unknown_venues(
    parsed_refs: list[dict], llm_call_fn
) -> list[dict]:
    """
    Use an LLM to classify venues for references where regex failed or
    returned a vague/generic label (e.g. bare "IEEE", "IEEE Trans").

    Args:
        parsed_refs: List of parsed reference dicts (with 'venue' and 'raw_text').
        llm_call_fn: Callable(prompt, label) -> str. Used to call the LLM.

    Returns:
        The same list with 'venue' fields updated where the LLM identified a venue.
    """
    from src.prompts import VENUE_CLASSIFICATION_PROMPT

    needs_llm = [
        (i, r) for i, r in enumerate(parsed_refs)
        if r.get("venue") == "Unknown" or _is_vague_venue(r.get("venue", ""))
    ]
    if not needs_llm:
        return parsed_refs

    BATCH_SIZE = 20
    for batch_start in range(0, len(needs_llm), BATCH_SIZE):
        batch = needs_llm[batch_start : batch_start + BATCH_SIZE]
        refs_text = "\n".join(
            f"{j + 1}: {r['raw_text']}" for j, (_, r) in enumerate(batch)
        )
        prompt = VENUE_CLASSIFICATION_PROMPT.format(references=refs_text)
        response = llm_call_fn(prompt, "venue classification")

        for line in response.split("\n"):
            m = re.match(r"^(\d+)\s*[:\.]\s*(.+)$", line.strip())
            if m:
                idx_in_batch = int(m.group(1)) - 1  # 1-indexed to 0-indexed
                venue = m.group(2).strip().rstrip(",.")
                if 0 <= idx_in_batch < len(batch) and venue.lower() != "unknown":
                    original_idx = batch[idx_in_batch][0]
                    parsed_refs[original_idx]["venue"] = venue

    return parsed_refs


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
