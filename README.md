# shoal

A multi-agent ensemble system that runs K role-differentiated ReAct agents in parallel, evaluates their outputs, and compiles the best answer. Served over a FastAPI HTTP interface, backed by configurable LLM endpoints (local Ollama, Ollama Cloud, or OpenAI).

## How it works

Each request to the `/agent` endpoint triggers an ensemble pipeline:

1. **Role Selection** — K agent configs are built from a catalog of distinct roles (methodical, creative, skeptical, research-focused, concise), each with its own prompt addendum and decoding parameters.
2. **Planner** — Each agent decomposes the query into numbered sub-tasks using the compiler model.
3. **Executor** — A ReAct loop executes each sub-task: the LLM reasons, calls tools, observes results, and repeats until it produces a final answer.
4. **Evaluator** — Each sub-task output is evaluated (PASS/FAIL). Failed tasks retry with feedback.
5. **Compiler** — All agent results (with their PASS/FAIL evaluations) are compiled into a single best answer by the compiler model.

## Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
uv run uvicorn main:app --reload
```

The server starts on `http://localhost:8000`.

Copy `.env.example` to `.env` and set your API keys for cloud deployment modes.

## Configuration

Edit `config.py` to change settings:

| Variable | Default | Description |
|---|---|---|
| `DEPLOYMENT_MODE` | `ollama_cloud` | `local`, `ollama_cloud`, or `openai` |
| `WORKER_MODEL` | `gemma3:4b` | Model for executor + evaluator |
| `COMPILER_MODEL` | `gemma3:12b` | Model for planner + compiler |
| `ENSEMBLE_K` | `5` | Number of parallel agent loops |
| `COMPILER_TEMPERATURE` | `0.8` | Temperature for planner/compiler |
| `REACT_MAX_ITERATIONS` | `4` | Max tool loops per executor run |
| `MAX_PLAN_STEPS` | `5` | Max sub-tasks per plan |
| `MAX_EVALUATOR_RETRIES` | `2` | Retries per sub-task on failure |
| `LLM_MAX_TOKENS` | `4096` | Max tokens per LLM call |

## Agent Roles

Each ensemble member adopts a distinct role defined in `roles.py`:

| Role | Strategy | Temperature |
|---|---|---|
| methodical | Systematic step-by-step, tool-heavy verification | 0.10 |
| creative | Lateral thinking, alternative framings | 0.70 |
| skeptical | Cross-checks results, verifies claims | 0.15 |
| research_focused | Extensive search, multiple data points | 0.20 |
| concise | Minimal steps, direct answers | 0.10 |

## API

### `POST /agent`

Run the ensemble agent on a query.

**Request:**
```json
{ "query": "What is 2 to the power of 10?" }
```

**Response:**
```json
{
  "query": "What is 2 to the power of 10?",
  "plan": ["K=5 parallel agent loops"],
  "sub_task_results": [...],
  "answer": "2 to the power of 10 is 1024.",
  "total_tokens": 3120,
  "iterations": 10,
  "success": true
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Tools

| Tool | Description |
|---|---|
| `calculator` | Evaluates math expressions via AST-validated safe evaluator |
| `search` | Web search via DuckDuckGo (async, non-blocking) |
| `datetime` | Returns current UTC date and time |

### Adding a tool

1. Create `tools/your_tool.py` subclassing `BaseTool` and implementing `async execute(input_str) -> str`.
2. Add a `ToolName` enum value in `models.py`.
3. Register the tool in `tools/__init__.py`.

## Project structure

```
shoal/
├── main.py            # FastAPI app + /agent endpoint
├── agent_hybrid.py    # Ensemble orchestrator (Planner→Executor↔Evaluator→Compiler)
├── roles.py           # Agent role definitions and selection
├── prompts.py         # Prompt builders (planner, executor, evaluator, compiler)
├── models.py          # Pydantic models and ToolName enum
├── config.py          # LLM + agent configuration
├── llm_client.py      # Unified LLM client (local/cloud/OpenAI)
├── env_config.py      # Environment-based deployment config
├── logger.py          # Structured logging setup
├── tests/             # Pytest test suite
└── tools/
    ├── base.py            # BaseTool abstract class (async)
    ├── calculator.py      # AST-safe math evaluator
    ├── datetime_tool.py
    └── search.py          # Async DuckDuckGo search
```

## Testing

```bash
uv run pytest tests/
```
