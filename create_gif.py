"""
Create an optimized GIF from the demo screenshots.

Curates frames, applies appropriate timing, and optimizes for README display.
"""

from pathlib import Path

from PIL import Image

FRAMES_DIR = Path(__file__).parent / "demo_frames"
OUTPUT_GIF = Path(__file__).parent / "docs" / "demo.gif"

# Define the frames to include and their display duration (ms).
# We skip redundant progress frames to keep the GIF concise.
FRAME_SEQUENCE = [
    # Landing page
    ("0000_landing.png", 3000),
    # File uploaded + ready to process
    ("0002_uploaded.png", 3000),
    # Processing starts
    ("0004_processing_start.png", 1500),
    # A few key progress snapshots (skip most to keep it short)
    ("0006_progress_6s.png", 1000),
    ("0010_progress_18s.png", 1000),
    ("0014_progress_30s.png", 1000),
    ("0018_progress_42s.png", 1000),
    ("0022_progress_54s.png", 1000),
    # Done!
    ("0026_done.png", 2500),
    # Results - Overview tab
    ("0028_tab_overview.png", 4000),
    # Contribution tab
    ("0030_tab_contribution.png", 3000),
    # State of the Art
    ("0031_tab_state_of_the_art.png", 3000),
    # Methodology tab
    ("0033_tab_methodology.png", 4000),
    # Evaluation tab
    ("0036_tab_evaluation.png", 3000),
    # Key Results tab
    ("0037_tab_key_results.png", 4000),
    # References with pie chart
    ("0039_tab_references.png", 4000),
    # Download buttons
    ("0040_download_buttons.png", 3000),
]


def main():
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    durations = []

    for filename, duration in FRAME_SEQUENCE:
        path = FRAMES_DIR / filename
        if not path.exists():
            print(f"WARNING: Missing frame {filename}, skipping")
            continue

        img = Image.open(path).convert("RGBA")

        # Downscale from 2x retina to 1x for reasonable GIF size
        # Target: 640x400 (half of 1280x800)
        target_w, target_h = 800, 500
        img = img.resize((target_w, target_h), Image.LANCZOS)

        # Convert to RGB with white background (GIF doesn't support alpha)
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)

        frames.append(bg)
        durations.append(duration)

    if not frames:
        print("No frames found!")
        return

    # Save as GIF with optimized palette
    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,  # Loop forever
        optimize=True,
    )

    size_mb = OUTPUT_GIF.stat().st_size / (1024 * 1024)
    print(f"GIF saved: {OUTPUT_GIF} ({size_mb:.1f} MB, {len(frames)} frames)")

    # If too large, we can further optimize with ffmpeg
    if size_mb > 10:
        print("GIF is large. Consider using ffmpeg for further optimization.")


if __name__ == "__main__":
    main()
