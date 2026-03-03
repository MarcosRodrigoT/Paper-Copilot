"""
Tool: extract_images

Extracts figures from a PDF using Docling's layout analysis,
which structurally associates images with their captions.
"""

import json
import re
from pathlib import Path

from langchain_core.tools import tool

# Minimum dimensions (pixels) to filter out tiny decorative images and headshots
MIN_WIDTH = 200
MIN_HEIGHT = 200

# Pattern matching caption text (e.g. "Figure 1.", "Fig. 3:", "TABLE II.")
_CAPTION_START = re.compile(
    r"^(fig(?:ure)?|table)\s*\.?\s*\d+", re.IGNORECASE
)


def _build_caption_map(doc) -> dict[int, list[str]]:
    """
    Build a page-indexed map of caption texts found in the document.

    Scans all TextItems whose label contains 'caption' or whose text
    starts with 'Figure'/'Table'. Returns {page_no: [caption_text, ...]}.
    """
    from docling_core.types.doc import TextItem

    captions: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        if not isinstance(item, TextItem):
            continue
        text = item.text.strip()
        if not text:
            continue
        label = str(item.label).lower()
        is_caption = "caption" in label or bool(_CAPTION_START.match(text))
        if is_caption:
            page = item.prov[0].page_no if item.prov else 0
            captions.setdefault(page, []).append(text)
    return captions


def _find_caption_for_image(
    page: int,
    image_number: int,
    caption_map: dict[int, list[str]],
    used_captions: set[str],
) -> str:
    """
    Find a matching caption for an image by its ordinal number and page.

    Looks for captions on the same page that mention the figure number
    (e.g. "Figure 2" for image_number=2). Falls back to any unused
    caption on the same page.
    """
    page_captions = caption_map.get(page, [])
    if not page_captions:
        return ""

    # Try to match by figure number mentioned in caption text
    for cap in page_captions:
        if cap in used_captions:
            continue
        # Check if this caption mentions the right figure number
        match = re.search(r"(?:fig(?:ure)?|table)\s*\.?\s*(\d+)", cap, re.IGNORECASE)
        if match and int(match.group(1)) == image_number:
            used_captions.add(cap)
            return cap

    # Fallback: first unused caption on the same page
    for cap in page_captions:
        if cap not in used_captions:
            used_captions.add(cap)
            return cap

    return ""


def _extract_images_from_pdf(
    conv_result, output_dir: str
) -> list[dict]:
    """
    Extract all significant figures from a Docling conversion result.

    Docling's layout model identifies figures and structurally associates
    them with their captions. For multi-part figures where Docling only
    attaches the caption to the last sub-image, a fallback scans the
    document for caption-labeled text items matching the figure number.

    Args:
        conv_result: A Docling ConversionResult (with generate_picture_images=True).
        output_dir: Directory where extracted images will be saved as PNGs.

    Returns a list of dicts with:
        - filename: the saved image file name
        - caption: detected caption text (may be empty)
        - page: page number where the image was found
        - width, height: image dimensions
        - figure_number: ordinal position of the image
    """
    doc = conv_result.document
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First pass: collect all caption texts by page for fallback matching
    caption_map = _build_caption_map(doc)
    used_captions: set[str] = set()

    results = []
    image_counter = 0

    from docling_core.types.doc import PictureItem

    for element, _level in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue

        image = element.get_image(doc)
        if image is None:
            continue

        # Skip tiny images (logos, icons, decorative elements)
        if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
            continue

        image_counter += 1
        filename = f"figure_{image_counter:02d}.png"
        filepath = output_path / filename

        # Save image to disk
        image.save(filepath, "PNG")

        # Get caption via Docling's structural caption references
        caption = element.caption_text(doc=doc) or ""

        # Get page number from provenance
        page = element.prov[0].page_no if element.prov else 0

        if caption:
            used_captions.add(caption)
        else:
            # Fallback: find a caption by figure number or page proximity
            caption = _find_caption_for_image(
                page, image_counter, caption_map, used_captions
            )

        results.append(
            {
                "filename": filename,
                "caption": caption,
                "page": page,
                "width": image.width,
                "height": image.height,
                "figure_number": image_counter,
            }
        )

    return results


def _convert_pdf_for_images(pdf_path: str):
    """Standalone Docling conversion for the @tool entry point."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    ).convert(pdf_path)


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
    conv_result = _convert_pdf_for_images(pdf_path)
    images = _extract_images_from_pdf(conv_result, output_dir)
    return json.dumps(images, ensure_ascii=False)
