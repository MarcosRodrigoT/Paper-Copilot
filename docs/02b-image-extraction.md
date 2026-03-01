# Step 3: PDF Image Extraction

This document explains how the `extract_images` tool works — our second LangChain tool.

## What this tool does

Given a PDF path and an output directory, it:

1. Scans every page for embedded images
2. Filters out tiny images (icons, logos, decorative elements)
3. Saves each significant image as a file (PNG, JPEG, etc.)
4. Attempts to find a caption for each image by looking at nearby text

## How image extraction works with PyMuPDF

PDFs embed images as internal objects, each with an **xref** (cross-reference number). PyMuPDF gives us two key methods:

```python
# Get a list of all images on a page
page.get_images(full=True)
# Returns: [(xref, smask, width, height, bpc, colorspace, ...), ...]

# Extract the actual image bytes
doc.extract_image(xref)
# Returns: {"ext": "png", "width": 400, "height": 300, "image": b"..."}
```

### Filtering small images

Research papers contain many tiny images — bullet points, journal logos, separator lines. We skip anything smaller than 100x100 pixels:

```python
MIN_WIDTH = 100
MIN_HEIGHT = 100

if width < MIN_WIDTH or height < MIN_HEIGHT:
    continue  # skip decorative images
```

## Caption detection

Figures in papers almost always have a caption starting with "Fig.", "Figure", "Table", etc. Our strategy:

```
┌─────────────────────────────────┐
│                                 │
│         [IMAGE]                 │
│                                 │
├─────────────────────────────────┤ ← image bottom edge
│ "Figure 3: Our proposed..."     │ ← search here first (below)
└─────────────────────────────────┘
```

1. **Get the image's bounding box** on the page using `page.get_image_rects(xref)`
2. **Look below** the image (up to 80 pixels) for text matching `^(Fig|Figure|Table|Algorithm)\s*\d*`
3. **If not found, look above** — some papers place captions on top
4. **Take the first 4 lines** of matching text (captions are usually 1-3 lines)

```python
below_rect = fitz.Rect(
    image_rect.x0,                          # same left edge
    image_rect.y1,                          # start at image bottom
    image_rect.x1,                          # same right edge
    min(image_rect.y1 + 80, page_rect.y1),  # up to 80px below
)
text_below = page.get_text("text", clip=below_rect).strip()
```

### Limitations

- **Captions not starting with "Figure":** Some papers use non-standard labels. These would be missed.
- **Multi-part figures:** A single logical figure made of multiple sub-images may be extracted as separate images.
- **Vector graphics:** Diagrams drawn with PDF vector commands (lines, curves) are not "images" — they won't be extracted. Only raster images embedded in the PDF are captured.

## Output format

The tool returns a JSON array:

```json
[
  {
    "filename": "figure_01.png",
    "caption": "Figure 1: Overview of our proposed architecture.",
    "page": 3,
    "width": 800,
    "height": 450
  },
  {
    "filename": "figure_02.jpeg",
    "caption": "Figure 2: Comparison of results on the test set.",
    "page": 7,
    "width": 600,
    "height": 400
  }
]
```

Images are saved to the specified output directory with sequential names (`figure_01`, `figure_02`, ...).

## Code location

See [src/tools/extract_images.py](../src/tools/extract_images.py):

- `_find_caption(page, image_rect)` — Searches for caption text near an image
- `_extract_images_from_pdf(pdf_path, output_dir)` — Main logic: iterate pages, extract images, match captions
- `extract_images(pdf_path, output_dir)` — The `@tool`-decorated wrapper

## Next step

The next tools handle **reference parsing** and **chart generation** — see `docs/03-tools.md`.
