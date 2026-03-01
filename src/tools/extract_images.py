"""
Tool: extract_images

Extracts images from a PDF and attempts to match each image
to its caption by looking at nearby text.
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.tools import tool

# Minimum dimensions (pixels) to filter out tiny decorative images
MIN_WIDTH = 100
MIN_HEIGHT = 100


def _find_caption(page: fitz.Page, image_rect: fitz.Rect) -> str:
    """
    Look for a caption below or above an image on the page.

    Strategy: search for text starting with "Fig", "Figure", or "Table"
    in the area immediately below (or above) the image bounding box.
    """
    page_rect = page.rect

    # Search below the image first (most common caption placement)
    below_rect = fitz.Rect(
        image_rect.x0,
        image_rect.y1,
        image_rect.x1,
        min(image_rect.y1 + 80, page_rect.y1),
    )
    text_below = page.get_text("text", clip=below_rect).strip()

    # Check if it looks like a caption
    caption_pattern = re.compile(
        r"^(Fig\.?|Figure|Table|Alg\.?|Algorithm)\s*\d*", re.IGNORECASE
    )
    if text_below and caption_pattern.match(text_below):
        # Take just the first few lines (captions are usually 1-3 lines)
        lines = text_below.split("\n")
        return " ".join(lines[:4]).strip()

    # Try above the image
    above_rect = fitz.Rect(
        image_rect.x0,
        max(image_rect.y0 - 80, 0),
        image_rect.x1,
        image_rect.y0,
    )
    text_above = page.get_text("text", clip=above_rect).strip()
    if text_above and caption_pattern.match(text_above):
        lines = text_above.split("\n")
        return " ".join(lines[:4]).strip()

    return ""


def _extract_images_from_pdf(
    pdf_path: str, output_dir: str
) -> list[dict]:
    """
    Extract all significant images from a PDF and save them as PNG files.

    Returns a list of dicts with:
        - filename: the saved image file name
        - caption: detected caption text (may be empty)
        - page: page number where the image was found
    """
    doc = fitz.open(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

            # Try to find the image's bounding box on the page for caption detection
            caption = ""
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
