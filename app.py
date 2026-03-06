"""
Streamlit UI for the Research Paper Copilot.

Provides a drag-and-drop interface to upload PDFs, view
generated markdown summaries, and browse previously processed papers.

Run with: uv run streamlit run app.py
"""

import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import streamlit as st

from src.config import INPUT_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL, OUTPUT_DIR


@st.cache_data(ttl=30)
def _get_tool_capable_models() -> list[str]:
    """Query Ollama for locally installed models that support tool calling."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception:
        return [OLLAMA_MODEL]

    capable = []
    for m in models:
        name = m["name"]
        try:
            detail = requests.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"name": name},
                timeout=5,
            )
            template = detail.json().get("template", "")
            if ".Tools" in template or "tool" in template.lower():
                capable.append(name)
        except Exception:
            continue

    return capable if capable else [OLLAMA_MODEL]


def _get_processed_papers() -> list[dict]:
    """Scan output directory for previously processed papers."""
    papers = []
    if not OUTPUT_DIR.exists():
        return papers
    for paper_dir in sorted(OUTPUT_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        summary = paper_dir / "summary.md"
        if summary.exists():
            # Try to extract title from first line of summary
            first_line = summary.read_text(encoding="utf-8").split("\n", 1)[0]
            title = first_line.lstrip("# ").replace("Paper Summary: ", "")
            display_name = title if title else paper_dir.name
            papers.append({
                "name": paper_dir.name,
                "title": display_name,
                "summary_path": summary,
                "output_dir": paper_dir,
            })
    return papers


# Known top-level sections in the order they appear in the summary.
_SECTION_HEADINGS = [
    "Overview",
    "Contribution",
    "State of the Art",
    "Methodology",
    "Evaluation",
    "Key Results",
    "References Analysis",
]

# Tab labels with icons for each section.
_SECTION_TAB_LABELS = [
    "🔎 Overview",
    "💡 Contribution",
    "📊 State of the Art",
    "⚙️ Methodology",
    "🧪 Evaluation",
    "📈 Key Results",
    "📚 References",
]

# Regex for image tags: ![alt](images/file.png) with optional italic caption.
_IMAGE_RE = re.compile(
    r"!\[(.*?)\]\(images/([^)]+)\)\n?"
    r"(?:\*([^*]+)\*\n?)?",
)


def _split_sections(md_content: str) -> tuple[str, dict[str, str]]:
    """Split a summary markdown into a header and a dict of section contents.

    Returns (header, {section_name: body}).
    """
    # Build a pattern that matches any of our known section headings as H2.
    heading_pattern = re.compile(
        r"^## (" + "|".join(re.escape(h) for h in _SECTION_HEADINGS) + r")\s*$",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(md_content))
    if not matches:
        return md_content, {}

    header = md_content[: matches[0].start()].strip()
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_content)
        sections[name] = md_content[start:end].strip()

    return header, sections


def _render_markdown_chunk(text: str, images_dir: Path):
    """Render a markdown chunk, replacing image tags with centered st.image."""
    last_end = 0
    for match in _IMAGE_RE.finditer(text):
        text_before = text[last_end : match.start()].strip()
        if text_before:
            st.markdown(text_before)

        alt_text = match.group(1)
        img_filename = match.group(2)
        italic_caption = match.group(3)
        img_path = images_dir / img_filename

        caption = italic_caption or alt_text or ""
        if img_path.exists():
            _, col_img, _ = st.columns([1, 3, 1])
            with col_img:
                st.image(str(img_path), caption=caption if caption else None)

        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        st.markdown(remaining)


def _render_summary(summary_path: Path, images_dir: Path):
    """Render a markdown summary with section tabs and centered images."""
    md_content = summary_path.read_text(encoding="utf-8")
    header, sections = _split_sections(md_content)

    # Render the header (title, authors, metadata).
    if header:
        st.markdown(header)

    if not sections:
        # Fallback: render everything as one block.
        _render_markdown_chunk(md_content, images_dir)
        return

    # Build tabs only for sections that exist in this summary.
    tab_names = []
    tab_labels = []
    for name, label in zip(_SECTION_HEADINGS, _SECTION_TAB_LABELS):
        if name in sections:
            tab_names.append(name)
            tab_labels.append(label)

    tabs = st.tabs(tab_labels)
    for tab, name in zip(tabs, tab_names):
        with tab:
            _render_markdown_chunk(sections[name], images_dir)


def _render_download_buttons(paper_name: str, summary_path: Path, images_dir: Path):
    """Render download buttons for markdown and zip."""
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            label="📥 Download Markdown",
            data=summary_path.read_text(encoding="utf-8"),
            file_name=f"{paper_name}_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_dl2:
        if images_dir.exists() and any(images_dir.iterdir()):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(summary_path, "summary.md")
                for img_file in images_dir.iterdir():
                    zf.write(img_file, f"images/{img_file.name}")

            st.download_button(
                label="📦 Download All (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{paper_name}_summary.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Copilot",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    available_models = _get_tool_capable_models()
    if OLLAMA_MODEL in available_models:
        available_models.remove(OLLAMA_MODEL)
        available_models.insert(0, OLLAMA_MODEL)

    model_name = st.selectbox(
        "Ollama model",
        options=available_models,
        index=0,
        help="Only locally installed models with tool-calling support are shown.",
    )

    st.divider()

    # Previously processed papers
    st.header("📚 Papers")
    processed = _get_processed_papers()

    if processed:
        paper_options = {p["name"]: p["title"] for p in processed}
        selected_paper = st.selectbox(
            "Processed papers",
            options=["(none)"] + list(paper_options.keys()),
            format_func=lambda x: paper_options.get(x, "Select a paper..."),
            index=0,
            label_visibility="collapsed",
        )
    else:
        selected_paper = "(none)"
        st.caption("No papers processed yet.")

    uploaded_file = st.file_uploader(
        "Upload new paper",
        type=["pdf"],
        help="Upload a PDF of a research paper.",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. 📄 Upload a PDF\n"
        "2. 🔍 The agent extracts text, images, and references\n"
        "3. ✍️ It generates section summaries\n"
        "4. 📋 You get a markdown file ready for Notion"
    )


# ── Styling ──────────────────────────────────────────────────
# Center display-mode LaTeX formulas ($$...$$) rendered by Streamlit/KaTeX.
st.markdown(
    """
    <style>
    /* Hide Streamlit toolbar/header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    /* Center display-mode KaTeX blocks */
    .katex-display {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Main content ─────────────────────────────────────────────
st.title("📄 Research Paper Copilot")
st.markdown(
    "Upload a research paper PDF and get a structured, "
    "Notion-ready markdown summary."
)

# Total pipeline steps for progress tracking
_TOTAL_STEPS = 6

# --- Show a previously selected processed paper ---
if selected_paper != "(none)":
    # Dropdown selection takes priority (file_uploader retains its state
    # across reruns, so we check the dropdown first).
    paper_info = next(p for p in processed if p["name"] == selected_paper)
    summary_path = paper_info["summary_path"]
    paper_output_dir = paper_info["output_dir"]
    images_dir = paper_output_dir / "images"

    _render_summary(summary_path, images_dir)
    st.divider()
    _render_download_buttons(selected_paper, summary_path, images_dir)

# --- Handle uploaded file ---
elif uploaded_file is not None:
    input_path = INPUT_DIR / uploaded_file.name
    input_path.write_bytes(uploaded_file.getvalue())

    paper_name = Path(uploaded_file.name).stem
    paper_output_dir = OUTPUT_DIR / paper_name
    summary_path = paper_output_dir / "summary.md"

    st.info(f"Ready to process: **{uploaded_file.name}**")

    btn_col, info_col = st.columns([1, 2])
    with btn_col:
        run_agent = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True,
        )
    with info_col:
        if summary_path.exists():
            st.caption(
                "A summary already exists for this paper. "
                "Click **Generate Summary** to regenerate."
            )

    if run_agent:
        progress_bar = st.progress(0, text="Starting...")
        _step_re = re.compile(r"\[(\d+)/(\d+)\]")
        _last_frac = 0.0

        # Run the agent in a subprocess so all GPU memory is freed on exit.
        cmd = [sys.executable, "-u", "-m", "src.agent", str(input_path)]
        if model_name and model_name != OLLAMA_MODEL:
            cmd.append(model_name)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                # Skip library log lines (e.g. [INFO], [DEBUG], WARNING:)
                stripped = re.sub(r"\x1b\[[0-9;]*m", "", line)  # strip ANSI codes
                if re.match(r"\[?(INFO|DEBUG|WARNING|ERROR)\]?", stripped):
                    continue
                match = _step_re.match(line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    label = line.split("]", 1)[-1].strip()
                    _last_frac = current / total
                    progress_bar.progress(_last_frac, text=label)
                else:
                    # Sub-step detail — update label, keep bar position.
                    progress_bar.progress(_last_frac, text=line)

            proc.wait()

            if proc.returncode == 0 and summary_path.exists():
                progress_bar.progress(1.0, text="✅ Done!")
            else:
                st.warning("Agent finished but no summary was created.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display results for the uploaded paper
    if summary_path.exists():
        st.divider()
        images_dir = paper_output_dir / "images"
        _render_summary(summary_path, images_dir)
        st.divider()
        _render_download_buttons(paper_name, summary_path, images_dir)

else:
    st.markdown(
        """
        ---
        ### 👋 Getting started

        1. 🖥️ Make sure Ollama is running: `ollama serve`
        2. 🤖 Pull a model: `ollama pull qwen3:32b`
        3. 📄 Upload a PDF using the sidebar

        You can also process papers from the command line:
        ```bash
        uv run python -m src.agent input/paper.pdf
        ```
        """
    )
