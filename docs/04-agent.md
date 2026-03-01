# Step 7: The Agent

This is the core of the project — where the LLM, the tools, and the agent loop come together.

## What is an agent?

A **chain** is a fixed sequence: input → step A → step B → output. You decide the order.

An **agent** is different: the LLM decides what to do next. It looks at the current state, picks a tool to call (or decides it's done), processes the result, and loops.

```
┌─────────────────────────────────────────────────┐
│                  Agent Loop                      │
│                                                  │
│   User message                                   │
│       │                                          │
│       ▼                                          │
│   ┌─────────┐   "call extract_text"   ┌──────┐  │
│   │   LLM   │ ────────────────────►   │ Tool │  │
│   │         │ ◄────────────────────   │      │  │
│   │ decides │   tool result           └──────┘  │
│   │  next   │                                    │
│   │ action  │   "call parse_refs"     ┌──────┐  │
│   │         │ ────────────────────►   │ Tool │  │
│   │         │ ◄────────────────────   │      │  │
│   │         │   tool result           └──────┘  │
│   │         │                                    │
│   │         │   "I'm done, here's the summary"   │
│   └────┬────┘                                    │
│        │                                         │
│        ▼                                         │
│   Final response                                 │
└─────────────────────────────────────────────────┘
```

This is called a **ReAct** pattern (Reason + Act): the LLM reasons about what to do, acts by calling a tool, observes the result, then reasons again.

## Key concepts in the code

### 1. ChatOllama

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="mistral-small",
    base_url="http://localhost:11434",
    temperature=0.1,
)
```

This creates a connection to your local Ollama server. The `temperature=0.1` keeps outputs factual and consistent (less randomness) — good for summarization.

### 2. Binding tools to the model

When we create the agent with `create_react_agent(model=llm, tools=TOOLS)`, the tools get **bound** to the model. This means:

- The model receives a description of each tool (name + docstring + argument types)
- When the model wants to use a tool, it outputs a special **tool call** message instead of regular text
- The agent framework intercepts this, runs the tool, and feeds the result back

The model never runs the Python code itself — it just says "I want to call `extract_text` with argument `pdf_path='/path/to/file.pdf'`" and the framework does the rest.

### 3. create_react_agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)
```

This is a LangGraph utility that builds a complete ReAct agent. Under the hood it creates a graph with:
- A **model node** that calls the LLM
- A **tools node** that executes tool calls
- An **edge** that routes back to the model after each tool result
- A **condition** that stops the loop when the model produces a final text response (no more tool calls)

### 4. The system prompt

The system prompt (in `src/prompts.py`) is critical — it tells the model:
- What tools are available and what each does
- The exact workflow to follow (extract → parse → chart → summarize → save)
- Guidelines for each section (length, detail level, formulae preservation)
- Rules (don't invent information, note missing sections)

The agent doesn't *have* to follow this order — it could call tools in any sequence — but the prompt guides it toward the right workflow.

### 5. Invoking the agent

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": user_message}]},
)
```

The agent takes a list of messages (like a chat API) and returns the full message history including all tool calls and results. The last message is the agent's final response.

## The full flow for one paper

```
1. User: "Summarize this paper at /path/to/paper.pdf"

2. Agent thinks: "I should extract the text first"
   → Calls extract_text(pdf_path="/path/to/paper.pdf")
   ← Gets back JSON with sections

3. Agent thinks: "Now I need the images"
   → Calls extract_images(pdf_path="...", output_dir="...")
   ← Gets back JSON with image list

4. Agent thinks: "I need to parse the references"
   → Calls parse_references(references_text="[1] Vaswani...")
   ← Gets back JSON with venue/year data

5. Agent thinks: "Generate the chart"
   → Calls generate_chart(references_json="...", output_path="...")
   ← Gets back the chart file path

6. Agent generates all section summaries internally (no tool needed)

7. Agent thinks: "Save everything"
   → Calls save_markdown(content_json="{title: ..., overview: ...}")
   ← Gets back the path to summary.md

8. Agent: "Done! Summary saved to output/paper_name/summary.md"
```

## Running from the command line

```bash
# Basic usage
uv run python -m src.agent input/paper.pdf

# With a specific model
uv run python -m src.agent input/paper.pdf mistral-small3.2
```

## What can go wrong

- **Ollama not running:** Start it with `ollama serve`
- **Model not pulled:** Run `ollama pull mistral-small` first
- **Tool call errors:** If the model passes wrong arguments, the error goes back to the model and it can retry
- **Context window overflow:** Very long papers may exceed the model's context. The text extraction returns all sections, which can be a lot of text for a 30+ page paper

## Next step

The agent works from the command line. The next step adds a **Streamlit web UI** for drag-and-drop — see `docs/05-ui.md`.
