"""
Configuration for the Research Paper Copilot.

All settings are centralized here so you can change the model,
paths, or behavior from a single place.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Model ---
# Change this to any Ollama model that supports tool calling.
# Good options: "qwen3:32b", "llama3.3", "mistral-small", "gpt-oss"
OLLAMA_MODEL = "ministral-3:14b"

# Ollama server URL (default local)
OLLAMA_BASE_URL = "http://localhost:11434"
