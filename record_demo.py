"""
Record a smooth demo GIF of the Paper Copilot Streamlit UI.

Features:
- High frame rate capture (~10 fps)
- Large rendered mouse cursor overlay
- Smooth mouse movement animations with easing
- Drag-and-drop file upload animation (cursor exits right, returns with file)
- Uses the Attention paper for the demo

Usage: .venv/bin/python record_demo.py
Requires: playwright, Pillow
"""

import io
import math
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

STREAMLIT_URL = "http://localhost:8501"
PDF_PATH = Path(__file__).parent / "input" / "Attention.pdf"
FRAMES_DIR = Path(__file__).parent / "demo_frames"
FRAMES_DIR.mkdir(exist_ok=True)

# Viewport
WIDTH, HEIGHT = 1280, 800

# Cursor state
_cursor_x, _cursor_y = WIDTH // 2, HEIGHT // 2
_frame_idx = 0
_dragging = False       # Whether we're currently dragging a file
_cursor_visible = True  # Whether cursor is visible (hidden when "off screen")


# ── Drawing helpers ───────────────────────────────────────────


def _draw_cursor(img: Image.Image, x: int, y: int) -> Image.Image:
    """Draw a pointer cursor on the image. Bigger than default."""
    if not _cursor_visible:
        if _dragging:
            # Still draw the file icon entering from edge
            img = img.copy()
            draw = ImageDraw.Draw(img)
            s = 2
            cx, cy = int(x * s), int(y * s)
            _draw_dragged_file(draw, cx, cy, s)
            return img
        return img

    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Scale: 2x device pixel ratio, then 1.6x bigger cursor
    s = 2
    sc = 1.6  # cursor scale multiplier
    cx, cy = int(x * s), int(y * s)

    # Arrow pointer shape — scaled up
    arrow = [
        (cx, cy),
        (cx, cy + int(28 * sc)),
        (cx + int(6 * sc), cy + int(22 * sc)),
        (cx + int(12 * sc), cy + int(32 * sc)),
        (cx + int(16 * sc), cy + int(30 * sc)),
        (cx + int(10 * sc), cy + int(20 * sc)),
        (cx + int(18 * sc), cy + int(18 * sc)),
        (cx, cy),
    ]
    draw.polygon(arrow, fill="black")
    inner = [
        (cx + 3, cy + 5),
        (cx + 3, cy + int(24 * sc)),
        (cx + int(6 * sc) + 1, cy + int(19 * sc)),
        (cx + int(11 * sc), cy + int(28 * sc)),
        (cx + int(14 * sc), cy + int(27 * sc)),
        (cx + int(9 * sc), cy + int(18 * sc)),
        (cx + int(15 * sc), cy + int(17 * sc)),
        (cx + 3, cy + 5),
    ]
    draw.polygon(inner, fill="white")

    if _dragging:
        _draw_dragged_file(draw, cx, cy, s)

    return img


def _draw_dragged_file(draw: ImageDraw.Draw, cx: int, cy: int, s: int) -> None:
    """Draw a large file icon being dragged next to the cursor."""
    # Much larger PDF icon
    fx = cx + 30
    fy = cy + 15
    fw, fh = 80, 100  # large file dimensions
    corner = 16

    # Shadow
    for off in (4, 3, 2):
        draw.rectangle(
            [fx + off, fy + off, fx + fw + off, fy + fh + off],
            fill=(180, 180, 180),
        )

    # File body
    body = [
        (fx, fy),
        (fx + fw - corner, fy),
        (fx + fw, fy + corner),
        (fx + fw, fy + fh),
        (fx, fy + fh),
    ]
    draw.polygon(body, fill="white", outline=(80, 80, 80), width=2)

    # Folded corner
    draw.polygon(
        [
            (fx + fw - corner, fy),
            (fx + fw - corner, fy + corner),
            (fx + fw, fy + corner),
        ],
        fill=(230, 230, 230),
        outline=(80, 80, 80),
    )

    # Red PDF badge
    badge_x, badge_y = fx + 8, fy + fh // 2 - 16
    badge_w, badge_h = fw - 16, 32
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=4,
        fill=(220, 50, 50),
    )
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((badge_x + 10, badge_y + 5), "PDF", fill="white", font=font)

    # Filename
    try:
        name_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
    except Exception:
        name_font = ImageFont.load_default()
    draw.text(
        (fx + 6, fy + fh // 2 + 22), "Attention.pdf", fill=(60, 60, 60),
        font=name_font,
    )


# ── Frame capture ─────────────────────────────────────────────


def _capture(page) -> None:
    """Capture a single frame with cursor overlay."""
    global _frame_idx
    raw = page.screenshot()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = _draw_cursor(img, _cursor_x, _cursor_y)
    img.save(FRAMES_DIR / f"{_frame_idx:05d}.png")
    _frame_idx += 1


def _hold(page, seconds: float, fps: float = 10) -> None:
    """Hold the current view, capturing frames at the given fps."""
    n_frames = max(1, int(seconds * fps))
    interval = seconds / n_frames
    for _ in range(n_frames):
        _capture(page)
        time.sleep(interval)


# ── Movement helpers ──────────────────────────────────────────


def _smooth_move(page, x1: int, y1: int, x2: int, y2: int,
                 duration: float = 0.6, fps: float = 15) -> None:
    """Smoothly move cursor from (x1,y1) to (x2,y2) with eased animation."""
    global _cursor_x, _cursor_y
    n_frames = max(2, int(duration * fps))
    for i in range(n_frames + 1):
        t = i / n_frames
        if t < 0.5:
            ease = 4 * t * t * t
        else:
            ease = 1 - (-2 * t + 2) ** 3 / 2

        _cursor_x = int(x1 + (x2 - x1) * ease)
        _cursor_y = int(y1 + (y2 - y1) * ease)
        page.mouse.move(_cursor_x, _cursor_y)
        _capture(page)
        time.sleep(duration / n_frames)


def _move_to(page, x: int, y: int, duration: float = 0.6) -> None:
    """Move cursor to (x, y) from current position."""
    _smooth_move(page, _cursor_x, _cursor_y, x, y, duration)


def _smooth_scroll(page, pixels: int, duration: float = 0.8, fps: float = 12) -> None:
    """Smooth scroll by pixel amount."""
    n_frames = max(2, int(duration * fps))
    step = pixels / n_frames
    for i in range(n_frames):
        page.mouse.wheel(0, step)
        time.sleep(duration / n_frames)
        _capture(page)


def _get_button_center(page, name: str) -> tuple[int, int]:
    """Get center of a button by its visible text."""
    btn = page.get_by_role("button", name=name).first
    box = btn.bounding_box()
    if box:
        return int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)
    raise ValueError(f"Button not found: {name}")


def _get_tab_center(page, name: str) -> tuple[int, int]:
    """Get center of a tab by its visible text."""
    tab = page.get_by_role("tab", name=name).first
    box = tab.bounding_box()
    if box:
        return int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)
    raise ValueError(f"Tab not found: {name}")


def _get_element_box(page, selector: str) -> dict:
    """Get bounding box of an element."""
    box = page.locator(selector).first.bounding_box()
    if box:
        return box
    raise ValueError(f"Element not found: {selector}")


# ── Drag and drop ─────────────────────────────────────────────


def _drag_and_drop_file(page) -> None:
    """Cursor exits right side, returns with a file, drops on uploader."""
    global _dragging, _cursor_visible

    # Get the drop zone position
    try:
        dropzone = _get_element_box(page, '[data-testid="stFileUploaderDropzone"]')
        drop_x = int(dropzone["x"] + dropzone["width"] / 2)
        drop_y = int(dropzone["y"] + dropzone["height"] / 2)
    except Exception:
        drop_x, drop_y = 155, 290

    # Step 1: Move cursor to right edge and "exit" the screen
    _move_to(page, WIDTH - 50, HEIGHT // 2, 0.5)
    _hold(page, 0.2)
    _smooth_move(page, WIDTH - 50, HEIGHT // 2, WIDTH + 40, HEIGHT // 2, 0.3, 15)
    _cursor_visible = False
    _hold(page, 1.0)  # Pause with cursor off-screen (picking up file)

    # Step 2: Cursor re-enters from right edge, now dragging the file
    _dragging = True
    _cursor_x = WIDTH + 40
    _cursor_y = HEIGHT // 2 - 50

    # Re-enter from right edge
    _smooth_move(
        page, WIDTH + 40, HEIGHT // 2 - 50, WIDTH - 100, HEIGHT // 2 - 30, 0.4, 15
    )
    _cursor_visible = True  # Now visible with file

    # Trigger drag highlight on the dropzone
    page.evaluate("""() => {
        const dt = new DataTransfer();
        dt.items.add(new File([''], 'Attention.pdf', {type: 'application/pdf'}));
        const dz = document.querySelector('[data-testid="stFileUploaderDropzone"]');
        if (dz) {
            dz.dispatchEvent(new DragEvent('dragenter', {
                bubbles: true, dataTransfer: dt
            }));
            dz.dispatchEvent(new DragEvent('dragover', {
                bubbles: true, dataTransfer: dt
            }));
        }
    }""")

    # Smooth arc towards drop zone
    _smooth_move(
        page, WIDTH - 100, HEIGHT // 2 - 30, drop_x, drop_y, duration=0.8, fps=15
    )
    _hold(page, 0.3)

    # Drop the file
    _dragging = False

    file_input = page.locator(
        '[data-testid="stFileUploaderDropzone"] input[type="file"]'
    )
    file_input.set_input_files(str(PDF_PATH))

    page.evaluate("""() => {
        const dz = document.querySelector('[data-testid="stFileUploaderDropzone"]');
        if (dz) dz.dispatchEvent(new DragEvent('dragleave', {bubbles: true}));
    }""")

    page.wait_for_timeout(1500)


# ── Main recording flow ──────────────────────────────────────


def main():
    global _cursor_x, _cursor_y

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
        )
        page = context.new_page()

        # ── 1. Landing page ──────────────────────────────────
        print("[1] Landing page")
        page.goto(STREAMLIT_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)

        _cursor_x, _cursor_y = WIDTH // 2, HEIGHT // 3
        _hold(page, 2.0)

        _move_to(page, 100, 200, 0.6)
        _hold(page, 0.8)

        _move_to(page, WIDTH // 2, HEIGHT // 2, 0.5)
        _hold(page, 1.0)

        # ── 2. Drag-and-drop PDF ─────────────────────────────
        print("[2] Drag-and-drop upload")

        _drag_and_drop_file(page)

        _move_to(page, 130, 400, 0.4)
        _hold(page, 1.5)

        _move_to(page, WIDTH // 2, 200, 0.5)
        _hold(page, 1.5)

        # ── 3. Click "Generate Summary" ──────────────────────
        print("[3] Clicking Generate Summary")

        try:
            gx, gy = _get_button_center(page, "Generate Summary")
        except Exception:
            gx, gy = 320, 242
        _move_to(page, gx, gy, 0.6)
        _hold(page, 0.6)

        page.mouse.click(gx, gy)
        _capture(page)
        _hold(page, 1.0)

        # ── 4. Monitor progress ──────────────────────────────
        print("[4] Processing...")

        max_wait = 600
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(1.0)
            elapsed += 1.0

            wobble_x = int(WIDTH // 2 + 40 * math.sin(elapsed * 0.3))
            wobble_y = int(260 + 10 * math.cos(elapsed * 0.5))
            _cursor_x, _cursor_y = wobble_x, wobble_y
            _capture(page)

            if page.locator("text=Done!").count() > 0:
                print(f"   Completed after ~{int(elapsed)}s")
                _hold(page, 1.5)
                break

            if int(elapsed) % 15 == 0:
                print(f"   Still processing... ({int(elapsed)}s)")

        # ── 5. Browse results ────────────────────────────────
        print("[5] Browsing results")
        page.wait_for_timeout(1000)

        # Scroll down ONCE so paper title is at the top (hide button + progress bar)
        _move_to(page, WIDTH // 2, HEIGHT // 2, 0.3)
        _smooth_scroll(page, 460, 0.6)
        _hold(page, 1.0)

        # Remember this scroll position — we'll return to it for every tab
        # (except References which gets a small extra scroll)
        TABS_SCROLL_Y = 460  # the scrollTo value that keeps title at top

        # Tabs that get a quick scroll-down-and-back peek
        PEEK_TABS = {"Methodology", "Evaluation", "Key Results"}

        all_tabs = [
            ("Overview", 1.8),
            ("Contribution", 1.3),
            ("State of the Art", 1.3),
            ("Methodology", 1.5),
            ("Evaluation", 1.3),
            ("Key Results", 1.5),
            ("References", 2.0),
        ]

        for tab_name, view_time in all_tabs:
            try:
                # Reset scroll to the fixed position (title at top)
                page.evaluate(f"window.scrollTo(0, {TABS_SCROLL_Y})")
                page.wait_for_timeout(100)

                tx, ty = _get_tab_center(page, tab_name)
                _move_to(page, tx, ty, 0.25)
                page.mouse.click(tx, ty)
                _capture(page)
                page.wait_for_timeout(250)

                _hold(page, view_time)

                # Quick peek scroll for content-heavy tabs
                if tab_name in PEEK_TABS:
                    _smooth_scroll(page, 250, 0.5)
                    _hold(page, 0.6)
                    _smooth_scroll(page, -250, 0.5)
                    _hold(page, 0.4)

                # Scroll for References to center the pie chart, then end
                if tab_name == "References":
                    _smooth_scroll(page, 180, 0.5)
                    _hold(page, 2.0)

            except Exception as e:
                print(f"   Could not interact with tab '{tab_name}': {e}")

        context.close()
        browser.close()

    print(f"\nDone! {_frame_idx} frames saved to {FRAMES_DIR}")


if __name__ == "__main__":
    main()
