"""
Tool: extract_images

Extracts images from a PDF and matches each image to its caption
using document-wide caption scanning with ordinal matching.
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.tools import tool

# Minimum dimensions (pixels) to filter out tiny decorative images
MIN_WIDTH = 100
MIN_HEIGHT = 100

# Regex to detect caption starts: Figure 1:, Fig. 2., Table 3:, Algorithm 1:
_CAPTION_START = re.compile(
    r"(Fig(?:ure)?|Table|Alg(?:orithm)?)\s*\.?\s*(\d+)\s*[.:]",
    re.IGNORECASE,
)

# Patterns that signal the end of a caption block
_CAPTION_END = re.compile(
    r"(?:Fig(?:ure)?|Table|Alg(?:orithm)?)\s*\.?\s*\d+\s*[.:]"  # next caption
    r"|\n\s*\n",  # double newline (paragraph break)
)

# Sentence-ending period: a period followed by whitespace and an uppercase letter,
# but NOT after common abbreviations (Fig., Sec., Eq., etc.)
_SENTENCE_END = re.compile(
    r"(?<!Fig)(?<!Sec)(?<!Eq)(?<!Tab)(?<!Alg)(?<!Fig)(?<!cf)(?<!etc)"
    r"(?<!i\.e)(?<!e\.g)"
    r"\.\s",
)

# Max caption length — most real captions are 1-3 sentences, under 250 chars
_MAX_CAPTION_LEN = 280


def _extract_all_captions(doc: fitz.Document) -> dict[int, str]:
    """
    Scan the entire document for figure/table captions and return them
    keyed by their number.

    Returns:
        Dict mapping figure/table number to its full caption text.
        Example: {1: "Figure 1: Overview of our architecture...", 2: "Figure 2: ..."}
    """
    captions: dict[int, str] = {}

    for page in doc:
        text = page.get_text("text")
        for match in _CAPTION_START.finditer(text):
            fig_num = int(match.group(2))
            if fig_num in captions:
                continue  # keep the first occurrence

            # Extract caption text starting from the match
            start = match.start()
            remaining = text[start:]

            # Find the end of this caption: next caption label or paragraph break
            # (skip the current match itself)
            end_match = _CAPTION_END.search(remaining, pos=len(match.group(0)))
            if end_match:
                caption_text = remaining[: end_match.start()]
            else:
                caption_text = remaining

            # Clean up: collapse whitespace, strip
            caption_text = re.sub(r"\s+", " ", caption_text).strip()

            # Cap length at a sentence boundary
            if len(caption_text) > _MAX_CAPTION_LEN:
                # Find the last sentence-ending period within the limit
                best_end = 0
                for m in _SENTENCE_END.finditer(caption_text):
                    if m.start() > _MAX_CAPTION_LEN:
                        break
                    if m.start() > 30:  # skip very short matches
                        best_end = m.start()
                if best_end > 0:
                    caption_text = caption_text[: best_end + 1]
                else:
                    caption_text = caption_text[:_MAX_CAPTION_LEN]

            captions[fig_num] = caption_text

    return captions


def _find_caption(page: fitz.Page, image_rect: fitz.Rect) -> str:
    """
    Fallback: look for a caption below or above an image on the page.

    Used only when document-wide ordinal matching finds no caption.
    """
    page_rect = page.rect

    for search_rect in [
        # Below the image (most common)
        fitz.Rect(
            page_rect.x0,
            image_rect.y1,
            page_rect.x1,
            min(image_rect.y1 + 120, page_rect.y1),
        ),
        # Above the image
        fitz.Rect(
            page_rect.x0,
            max(image_rect.y0 - 120, 0),
            page_rect.x1,
            image_rect.y0,
        ),
    ]:
        text = page.get_text("text", clip=search_rect).strip()
        match = _CAPTION_START.search(text)
        if match:
            caption = text[match.start():]
            caption = re.sub(r"\s+", " ", caption).strip()
            if len(caption) > _MAX_CAPTION_LEN:
                best_end = 0
                for m in _SENTENCE_END.finditer(caption):
                    if m.start() > _MAX_CAPTION_LEN:
                        break
                    if m.start() > 30:
                        best_end = m.start()
                if best_end > 0:
                    caption = caption[: best_end + 1]
                else:
                    caption = caption[:_MAX_CAPTION_LEN]
            return caption

    return ""


def _extract_images_from_pdf(
    pdf_path: str, output_dir: str
) -> list[dict]:
    """
    Extract all significant images from a PDF and save them as PNG files.

    Uses a two-pass approach for caption matching:
    1. Scan the entire document for all captions (keyed by figure number)
    2. Match each extracted image to its caption by ordinal position
    3. Fall back to spatial proximity search if ordinal matching fails

    Returns a list of dicts with:
        - filename: the saved image file name
        - caption: detected caption text (may be empty)
        - page: page number where the image was found
        - figure_number: ordinal position of the image
    """
    doc = fitz.open(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First pass: extract all captions from the document
    all_captions = _extract_all_captions(doc)

    results = []
    image_counter = 0

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]

            # Extract the image
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image:
                continue

            width = base_image["width"]
            height = base_image["height"]

            # Skip tiny images (logos, icons, decorative elements)
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            image_counter += 1
            ext = base_image["ext"]
            filename = f"figure_{image_counter:02d}.{ext}"
            filepath = output_path / filename

            # Save image to disk
            with open(filepath, "wb") as f:
                f.write(base_image["image"])

            # Try ordinal matching first, then fall back to spatial search
            caption = all_captions.get(image_counter, "")
            if not caption:
                try:
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        caption = _find_caption(page, img_rects[0])
                except Exception:
                    pass

            results.append(
                {
                    "filename": filename,
                    "caption": caption,
                    "page": page_num + 1,
                    "width": width,
                    "height": height,
                    "figure_number": image_counter,
                }
            )

    doc.close()
    return results


@tool
def extract_images(pdf_path: str, output_dir: str) -> str:
    """Extract images from a research paper PDF and save them as files.

    Args:
        pdf_path: Absolute path to the PDF file.
        output_dir: Directory where extracted images will be saved.

    Returns:
        JSON string with a list of extracted images, each having
        filename, caption, page number, and dimensions.
    """
    images = _extract_images_from_pdf(pdf_path, output_dir)
    return json.dumps(images, ensure_ascii=False)
