"""
Streamlit UI for the Research Paper Copilot.

Provides a drag-and-drop interface to upload PDFs and view
the generated markdown summaries.

Run with: uv run streamlit run app.py
"""

import shutil
from pathlib import Path

import streamlit as st

from src.agent import process_paper
from src.config import INPUT_DIR, OLLAMA_MODEL, OUTPUT_DIR

st.set_page_config(
    page_title="Research Paper Copilot",
    page_icon="📄",
    layout="wide",
)

st.title("Research Paper Copilot")
st.markdown(
    "Upload a research paper PDF and get a structured, "
    "Notion-ready markdown summary."
)

# --- Sidebar: settings ---
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input(
        "Ollama model",
        value=OLLAMA_MODEL,
        help="Any Ollama model that supports tool calling.",
    )
    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Upload a PDF\n"
        "2. The agent extracts text, images, and references\n"
        "3. It generates section summaries\n"
        "4. You get a markdown file ready for Notion"
    )

# --- File upload ---
uploaded_file = st.file_uploader(
    "Drop a research paper here",
    type=["pdf"],
    help="Upload a PDF of a research paper.",
)

if uploaded_file is not None:
    # Save uploaded file to the input directory
    input_path = INPUT_DIR / uploaded_file.name
    input_path.write_bytes(uploaded_file.getvalue())
    st.success(f"Uploaded: **{uploaded_file.name}**")

    # Check if we already have output for this paper
    paper_name = Path(uploaded_file.name).stem
    paper_output_dir = OUTPUT_DIR / paper_name
    summary_path = paper_output_dir / "summary.md"

    col1, col2 = st.columns([1, 1])

    with col1:
        run_agent = st.button(
            "Generate Summary",
            type="primary",
            use_container_width=True,
        )

    with col2:
        if summary_path.exists():
            st.info("A summary already exists for this paper. "
                    "Click 'Generate Summary' to regenerate.")

    if run_agent:
        with st.status("Agent is working...", expanded=True) as status:
            st.write("Extracting text and images from PDF...")
            st.write(f"Using model: **{model_name}**")

            try:
                result_path = process_paper(
                    str(input_path),
                    model_name=model_name if model_name != OLLAMA_MODEL else None,
                )

                if result_path and Path(result_path).exists():
                    status.update(label="Summary generated!", state="complete")
                else:
                    status.update(
                        label="Agent finished but no summary was created.",
                        state="error",
                    )
            except Exception as e:
                status.update(label="Error during processing", state="error")
                st.error(f"An error occurred: {e}")
                result_path = ""

    # --- Display results ---
    if summary_path.exists():
        st.divider()
        st.header("Generated Summary")

        # Read and display the markdown
        md_content = summary_path.read_text(encoding="utf-8")

        # Replace relative image paths with absolute paths for Streamlit display
        images_dir = paper_output_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                relative = f"images/{img_file.name}"
                if relative in md_content:
                    md_content = md_content.replace(
                        f"![]({relative})",
                        "",
                    )
                    md_content = md_content.replace(
                        relative,
                        str(img_file),
                    )

        st.markdown(md_content, unsafe_allow_html=True)

        # Display images that couldn't be inlined
        if images_dir.exists():
            image_files = sorted(images_dir.glob("*.png")) + sorted(
                images_dir.glob("*.jpg")
            )
            if image_files:
                st.subheader("Figures")
                for img_path in image_files:
                    st.image(str(img_path), caption=img_path.stem)

        # Download buttons
        st.divider()
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.download_button(
                label="Download Markdown",
                data=summary_path.read_text(encoding="utf-8"),
                file_name=f"{paper_name}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_dl2:
            # Create a zip of the full output (md + images)
            if images_dir.exists() and any(images_dir.iterdir()):
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(summary_path, "summary.md")
                    for img_file in images_dir.iterdir():
                        zf.write(img_file, f"images/{img_file.name}")

                st.download_button(
                    label="Download All (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"{paper_name}_summary.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
else:
    # Show placeholder when no file is uploaded
    st.markdown(
        """
        ---
        ### Getting started

        1. Make sure Ollama is running: `ollama serve`
        2. Pull a model: `ollama pull mistral-small`
        3. Upload a PDF above

        You can also process papers from the command line:
        ```bash
        uv run python -m src.agent input/paper.pdf
        ```
        """
    )
