# shoal

A minimal ReAct (Reasoning + Acting) agent served over a FastAPI HTTP interface. The agent iteratively reasons about a query, calls tools, observes their output, and repeats until it can produce a final answer — all backed by a local LLM via a llama.cpp OpenAI-compatible endpoint.

## How it works

Each request to the `/agent` endpoint triggers a ReAct loop:

1. The LLM receives a system prompt describing available tools and the expected output format.
2. It produces a `Thought` + `Action` + `Action Input`.
3. The agent executes the named tool and appends the result as an `Observation`.
4. Steps 2–3 repeat until the LLM emits a `Final Answer` or the iteration limit is hit.

## Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), and a running llama.cpp server on `http://localhost:8080`.

```bash
uv sync
uv run uvicorn main:app --reload
```

The server starts on `http://localhost:8000`.

## Configuration

Edit `config.py` to change LLM settings:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | llama.cpp server URL |
| `LLM_MODEL` | `qwen2.5-7b-instruct` | Model name passed to the API |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `REACT_MAX_ITERATIONS` | `6` | Max reasoning steps before giving up |

## API

### `POST /agent`

Run the agent on a query.

**Request:**
```json
{ "query": "What is 2 to the power of 10?" }
```

**Response:**
```json
{
  "query": "What is 2 to the power of 10?",
  "answer": "2 to the power of 10 is 1024.",
  "steps": [...],
  "total_tokens": 312,
  "iterations": 2,
  "success": true,
  "error": null
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Tools

| Tool | Description |
|---|---|
| `calculator` | Evaluates a Python math expression (`abs`, `round`, `min`, `max`, `pow` allowed) |
| `search` | Looks up factual information (mock implementation — swap in SerpAPI/Tavily/Brave) |
| `datetime` | Returns the current UTC date and time |

### Adding a tool

1. Create `tools/your_tool.py` subclassing `BaseTool` and implementing `execute(input_str) -> str`.
2. Add a `ToolName` enum value in `models.py`.
3. Register the tool in `tools/__init__.py`.

## Project structure

```
shoal/
├── main.py          # FastAPI app + /agent endpoint
├── agent.py         # ReAct loop, LLM calls, output parsing
├── prompts.py       # System prompt builder
├── models.py        # Pydantic models and ToolName enum
├── config.py        # LLM + agent configuration
├── logger.py        # Structured logging setup
└── tools/
    ├── base.py          # BaseTool abstract class
    ├── calculator.py
    ├── datetime_tool.py
    └── search.py
```
