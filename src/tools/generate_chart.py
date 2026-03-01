"""
Tool: generate_chart

Takes parsed reference data and generates a pie chart showing
the distribution of venues/journals in the paper's bibliography.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from langchain_core.tools import tool

# How many slices to show individually before grouping the rest as "Other"
MAX_SLICES = 8


def _generate_pie_chart(
    references_json: str, output_path: str
) -> str:
    """
    Generate a pie chart PNG from reference data.

    Args:
        references_json: JSON string — list of dicts with "venue" field.
        output_path: Where to save the PNG file.

    Returns:
        The path to the saved PNG.
    """
    refs = json.loads(references_json)

    # Count venues
    venue_counts = Counter(ref.get("venue", "Unknown") for ref in refs)

    # Sort by count descending
    sorted_venues = venue_counts.most_common()

    # Group small slices into "Other"
    if len(sorted_venues) > MAX_SLICES:
        top = sorted_venues[:MAX_SLICES]
        other_count = sum(count for _, count in sorted_venues[MAX_SLICES:])
        labels = [v for v, _ in top] + ["Other"]
        sizes = [c for _, c in top] + [other_count]
    else:
        labels = [v for v, _ in sorted_venues]
        sizes = [c for _, c in sorted_venues]

    # Generate the chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set3.colors[: len(labels)]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # We use a legend instead for readability
        autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
        colors=colors,
        startangle=90,
        pctdistance=0.8,
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color("black")

    ax.legend(
        wedges,
        [f"{label} ({size})" for label, size in zip(labels, sizes)],
        title="Venue",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
    )

    ax.set_title("Reference Distribution by Venue", fontsize=13, fontweight="bold")
    fig.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


@tool
def generate_chart(references_json: str, output_path: str) -> str:
    """Generate a pie chart showing venue distribution from parsed references.

    Args:
        references_json: JSON string of parsed references (from parse_references tool).
        output_path: Absolute path where the pie chart PNG will be saved.

    Returns:
        The path to the saved pie chart image.
    """
    return _generate_pie_chart(references_json, output_path)
