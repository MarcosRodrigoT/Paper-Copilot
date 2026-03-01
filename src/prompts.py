"""
Prompt templates for the Research Paper Copilot.

Instead of one massive prompt, we use focused per-section prompts.
Each prompt gets only the relevant paper sections as input, keeping
the context manageable for local models.
"""

METADATA_PROMPT = """\
Extract the following metadata from this paper text. Answer with ONLY \
the requested information, nothing else.

Title:
Authors:
Published (year and venue):
DOI or link (if found, otherwise write "none"):

Paper text (first section):
{text}
"""

OVERVIEW_PROMPT = """\
Write a 3-5 sentence overview of the following research paper. \
What is it about? What problem does it address? What is the context?

Be concise but informative.

Paper text:
{text}
"""

CONTRIBUTION_PROMPT = """\
Based on the following paper text, describe the authors' main contribution \
in 4-6 sentences. What do they propose? Why does it matter? What is novel?

Paper text:
{text}
"""

STATE_OF_ART_PROMPT = """\
Based on the following related work / background section, write 1-2 paragraphs \
summarizing the state of the art. What existing approaches are discussed? \
What are their limitations? How does this work position itself among them?

Paper text:
{text}
"""

METHODOLOGY_OVERVIEW_PROMPT = """\
Write one paragraph giving a high-level overview of the method proposed in \
this paper. How do the pieces fit together?

Paper text:
{text}
"""

METHODOLOGY_DETAILS_PROMPT = """\
Write a detailed walkthrough of the method described below. Include:
- Key formulations and mathematical expressions
- Loss functions if any
- Architectural choices and design decisions
- Algorithms or training procedures

IMPORTANT formatting rules for math:
- Use $...$ for inline math (e.g., $x^2$, $d_{{model}} = 512$)
- Use $$...$$ for display/block equations (e.g., $$L = -\\sum \\log p(y)$$)
- Do NOT use \\( \\) or \\[ \\] delimiters — only $ and $$

Be concise but do NOT leave out important details. Only skip truly minor \
implementation specifics. This is the most important section.

Paper text:
{text}
"""

EVALUATION_PROMPT = """\
Write one paragraph describing the evaluation setup from this paper. \
What datasets were used? What metrics? What baselines were compared against?

Paper text:
{text}
"""

KEY_RESULTS_PROMPT = """\
List the main results from this paper as bullet points. Include specific \
numbers where available (e.g., "- Achieved 28.4 BLEU on WMT 2014 EN-DE").

Paper text:
{text}
"""

FIGURE_SELECTION_PROMPT = """\
Below is a list of figures extracted from a research paper, and a brief \
summary of what the paper is about.

Your task: select the most relevant figures for the methodology, evaluation, \
and key results sections. For each selected figure, indicate which section \
it belongs to.

Output format — one figure per line, nothing else:
figure_01.png -> methodology
figure_03.png -> evaluation

Only select relevant figures. Select at most 5 figures total.

Available figures:
{images_description}

Paper summary:
{summary}
"""

# Mapping from section names to the paper headings likely to contain
# the relevant information. Used to select which extracted sections
# to feed into each prompt.
SECTION_MAPPING = {
    "metadata": ["preamble", "abstract", "title"],
    "overview": ["abstract", "introduction", "preamble"],
    "contribution": ["abstract", "introduction"],
    "state_of_the_art": ["related work", "background", "prior work", "literature review"],
    "methodology_overview": ["method", "approach", "model", "architecture", "framework", "proposed"],
    "methodology_details": ["method", "approach", "model", "architecture", "framework", "proposed"],
    "evaluation": ["experiment", "evaluation", "setup", "training", "implementation"],
    "key_results": ["result", "experiment", "evaluation", "ablation"],
}
