# Deep Research Agent

A multi-agent research assistant built with [LangGraph](https://github.com/langchain-ai/langgraph) and a [Streamlit](https://streamlit.io/) UI.

The agent takes a user request, clarifies scope if needed, writes a research brief, delegates parallel research tasks through a supervisor, and synthesizes everything into a final report.

**Pipeline:** `Clarification → Brief → Supervisor → Research → Report`

## Features

- Interactive Streamlit chat UI with live streaming of each pipeline stage
- Pluggable model providers (Anthropic, Gemini, OpenAI, Groq) — pick a different model per role (research, compression, final report, summarization)
- Parallel sub-researchers coordinated by a supervisor agent, each with web search via [Tavily](https://tavily.com/)
- Configurable clarification behavior, concurrency limits, and iteration/tool-call caps
- Pipeline graph tab (Mermaid) visualizing the compiled LangGraph, including subgraphs
- Downloadable final report (Markdown)

## Project Structure

```
.
├── app.py                        # Streamlit UI entry point
├── deep_research/
│   ├── config.py                 # Default models & limits
│   ├── state.py                  # LangGraph state schemas & pydantic models
│   ├── prompt.py                 # System/task prompt templates
│   ├── utils.py                  # Tavily search, summarization, think_tool
│   └── agents/
│       ├── scoping_agent.py      # Clarification + research brief nodes
│       ├── research_agent.py     # Single researcher subgraph (search loop + compression)
│       ├── supervisor_agent.py   # Supervisor subgraph (delegates to researchers)
│       └── edge_agent.py         # Top-level graph wiring all stages together
├── requirements.txt
├── .env.example
└── .gitignore
```

## Prerequisites

- Python 3.11+
- API keys for the providers you plan to use:
  - **Anthropic** (default provider for all agent roles)
  - **Tavily** (required — used for all web search)
  - Optionally: OpenAI, Google (Gemini), Groq — if you select those providers in the sidebar

## Setup

1. **Clone the repo and enter the project directory**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create and activate a virtual environment**

   ```bash
   uv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

4. **Configure environment variables**

   Copy the example file and fill in your keys:

   ```bash
   cp .env.example --> .env
   ```

   At minimum, set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`.

## Running

```bash
streamlit run app.py
```

This opens the UI in your browser at `http://localhost:8501`.

- Use the sidebar to choose a provider and models for each agent role, and to tune clarification behavior, concurrency, and iteration/tool-call limits.
- Type a research request in the chat box. If clarification is enabled, the agent may ask a follow-up question before starting research.
- Each pipeline stage streams live into a collapsible section; the final report is shown in full and can be downloaded as Markdown.
- The **Pipeline Graph** tab shows the compiled LangGraph structure, including the researcher and supervisor subgraphs.

## Configuration Reference

Defaults live in `deep_research/config.py` and can be overridden per-run from the Streamlit sidebar:

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_RESEARCH_MODEL` | `anthropic:claude-haiku-4-5-20251001` | Used for clarification, brief writing, supervisor logic, and researchers |
| `DEFAULT_FINAL_REPORT_MODEL` | `anthropic:claude-haiku-4-5-20251001` | Used to write the final synthesis report |
| `DEFAULT_COMPRESSION_MODEL` | `anthropic:claude-haiku-4-5-20251001` | Used to compress each researcher's findings |
| `DEFAULT_SUMMARIZATION_MODEL` | `groq:llama-3.3-70b-versatile` | Used to summarize individual Tavily search results |
| `MAX_RESEARCHER_ITERATIONS` | `6` | Max supervisor planning rounds |
| `MAX_CONCURRENT_RESEARCHERS` | `3` | Max parallel researchers per supervisor round |

## Notes

- Model strings follow the `provider:model-name` format expected by LangChain's `init_chat_model`.
- Only providers with a valid API key configured in `.env` will work when selected in the sidebar.
