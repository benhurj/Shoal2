# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shoal is a multi-agent ensemble system that runs N role-differentiated ReAct agents in parallel, evaluates their outputs, and compiles the best answer. Served via FastAPI, backed by configurable LLM endpoints (local Ollama, Ollama Cloud, OpenAI, Together.ai, or Modal with full logits access).

## Commands

```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn main:app --reload

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_calculator_safety.py -v

# Run a specific test by name
uv run pytest tests/ -k test_name
```

No linter or formatter is currently configured.

## Architecture

### Two Architectures Supported

**1. Legacy Role-Based Ensemble (`/agent` endpoint):**
```
Query → generate_roles(N) → N independent agent loops → compile → Answer
```
Each agent loop: **Planner → (Executor ↔ Evaluator) per sub-task**

**2. Multi-Sample Ensemble (`/agent/multi-sample` endpoint):**
```
Query → 1 shared plan → N sampled executions per sub-task → best-of-N → compile → Answer
```
- Single forward pass through planner
- N sampled executions share KV cache prefix (branched generation)
- Best-of-N selection per sub-task based on evaluator scores
- Much more efficient for smaller models

### Two-tier model system

- **Compiler model** (`COMPILER_MODEL`, default `Qwen/Qwen3.5-9B`): Used by Planner and Compiler stages
- **Worker model** (`WORKER_MODEL`, default `Qwen/Qwen3.5-0.8B`): Used by Executor and Evaluator stages

### Deployment Modes

Set `DEPLOYMENT_MODE` in `config.py`:

| Mode | Description | Models |
|------|-------------|--------|
| `local` | Local Ollama instances | Any Ollama model |
| `ollama_cloud` | Ollama Cloud API | Cloud-hosted models |
| `openai` | OpenAI API | GPT-4, etc. |
| `together` | Together.ai API (with logprobs) | Various open models |
| `modal` | Custom Modal deployment (full logits, branched generation) | Qwen3.5-0.8B, Qwen3.5-9B |

**Modal deployment** enables:
- `call_llm_branched()` - K samples from single forward pass (shared KV cache)
- `call_llm_with_logits()` - raw pre-softmax logits for custom sampling
- `get_next_token_logits()` - step-by-step custom sampling loops

### Key modules

- `agent_hybrid.py` — Core orchestrator with two modes:
  - `run_agent()` — Legacy role-based ensemble
  - `run_agent_multi_sample()` — New multi-sample architecture (more efficient)
  - `run_planner_multi_sample()` — N plans from 1 forward pass
  - `run_executor_multi_sample()` — N ReAct loops from shared KV cache
  - `run_evaluator_multi_sample()` — N evaluations in parallel
  
- `llm_client.py` — Unified async LLM client:
  - `call_llm()` — Standard text generation
  - `call_llm_branched()` — Branched generation (Modal only)
  - `call_llm_with_logits()` — Returns raw logits (Modal only)
  - `get_next_token_logits()` — Next token logits for custom sampling (Modal only)
  
- `modal_server.py` — Modal deployment for Qwen 3.5 models:
  - Deploys both models to single L4 GPU
  - Handles Qwen 3.5's hybrid cache: KV + conv_states + ssm_states
  - Endpoints: `/generate`, `/generate_branched`, `/next_token_logits`
  
- `roles.py` — LLM-driven role generation with 5 static fallbacks

- `config.py` / `env_config.py` — Deployment mode routing

### Concurrency patterns

- `asyncio.gather()` parallelizes N agent loops / samples
- `asyncio.Semaphore(3)` rate-limits concurrent LLM calls
- Branched generation: 1 forward pass + N sampling strategies (Modal only)

### Adding a tool

1. Create `tools/your_tool.py` subclassing `BaseTool` with `async execute(input_str) -> str`
2. Add a `ToolName` enum value in `models.py`
3. Register in `tools/__init__.py`

## Testing

Tests use pytest + pytest-asyncio. Key test files cover the LLM output parser, calculator AST safety (attack vector rejection), and role selection logic. Async tests use `@pytest.mark.asyncio` with `unittest.mock.AsyncMock`.

## Configuration

Copy `.env.example` to `.env` and fill in keys for your deployment mode.