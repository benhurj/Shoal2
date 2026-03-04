# agent.py
import asyncio
import random
import re
import httpx
import structlog
from config import (
    WORKER_MODEL, WORKER_PORTS, COMPILER_MODEL, COMPILER_PORTS,
    LLM_MAX_TOKENS, REACT_MAX_ITERATIONS, MAX_PLAN_STEPS,
    MAX_EVALUATOR_RETRIES, ENSEMBLE_K,
    TEMPERATURE_MEAN, TEMPERATURE_STD, TOP_P_MEAN, TOP_P_STD,
    COMPILER_TEMPERATURE, COMPILER_TOP_P,
)
from models import (
    ToolCall, ToolName, AgentStep, AgentResponse,
    EvaluationResult, SubTaskResult, EnsembleMemberResult,
)
from tools import TOOL_REGISTRY
from prompts import (
    build_planner_prompt, build_executor_prompt,
    build_evaluator_prompt, build_synthesizer_prompt,
    build_compiler_prompt,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def sample_ensemble_configs(k: int) -> list[dict]:
    """Sample K stochastic configs from normal distributions, clamped to valid ranges."""
    configs = []
    for i in range(k):
        temp = max(0.01, min(2.0, random.gauss(TEMPERATURE_MEAN, TEMPERATURE_STD)))
        top_p = max(0.1, min(1.0, random.gauss(TOP_P_MEAN, TOP_P_STD)))
        configs.append({
            "label": f"agent_{i+1}",
            "temperature": round(temp, 3),
            "top_p": round(top_p, 3),
            "port": WORKER_PORTS[i % len(WORKER_PORTS)],
        })
    return configs


# ---------------------------------------------------------------------------
# LLM caller — routes to a specific endpoint + model
# ---------------------------------------------------------------------------
async def call_llm(
    messages: list[dict],
    port: int,
    model: str,
    stop: list[str] | None = None,
    **overrides,
) -> tuple[str, int]:
    """Call an Ollama instance on a specific port."""
    # For qwen3 models, disable thinking mode to get direct output
    if "qwen3" in model.lower():
        messages = [dict(m) for m in messages]  # shallow copy
        for m in reversed(messages):
            if m["role"] == "user":
                m["content"] = "/no_think\n" + m["content"]
                break

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": overrides.get("max_tokens", LLM_MAX_TOKENS),
        "temperature": overrides.get("temperature", 0.5),
    }
    if "top_p" in overrides:
        payload["top_p"] = overrides["top_p"]
    if stop:
        payload["stop"] = stop

    url = f"{_base_url(port)}/chat/completions"
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError(f"LLM call timed out (port {port}, model {model})")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM returned HTTP {e.response.status_code} (port {port})")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Strip <think>...</think> blocks (qwen3 thinking mode artifacts)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_llm_output(text: str) -> AgentStep:
    """Parse the LLM's output into a structured step."""
    text = text.strip()

    # Check for Action FIRST
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text, re.DOTALL)

    if action_match and input_match:
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        tool_name_raw = action_match.group(1).strip().lower()
        try:
            tool_name = ToolName(tool_name_raw)
            tool_call = ToolCall(tool=tool_name, input=input_match.group(1).strip())
            return AgentStep(iteration=0, thought=thought, action=tool_call)
        except ValueError:
            logger.warning("invalid_tool_name", raw=tool_name_raw)

    # Only then check for Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        thought_match = re.search(r"Thought:\s*(.+?)(?=Final Answer:)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        return AgentStep(
            iteration=0, thought=thought,
            is_final=True, final_answer=final_match.group(1).strip(),
        )

    # Fallback
    return AgentStep(iteration=0, thought=text)


# ---------------------------------------------------------------------------
# Complexity check
# ---------------------------------------------------------------------------
def should_plan(query: str) -> bool:
    """Heuristic: decide if a query needs multi-step planning."""
    indicators = ["then", "after that", "next", "finally", "also", "and then", "step"]
    query_lower = query.lower()
    hits = sum(1 for ind in indicators if ind in query_lower)
    return hits >= 1 or len(query.split()) > 20


# ---------------------------------------------------------------------------
# Worker stages (all route to a specific worker port)
# ---------------------------------------------------------------------------
async def run_planner(query: str, port: int, model_override: str | None = None, **overrides) -> tuple[list[dict], int]:
    """Decompose a query into numbered sub-tasks with tool hints."""
    model = model_override or WORKER_MODEL
    logger.info("planner_started", query=query, port=port, model=model)
    messages = [
        {"role": "system", "content": build_planner_prompt()},
        {"role": "user", "content": query},
    ]
    raw_output, tokens = await call_llm(messages, port=port, model=model, **overrides)
    logger.info("planner_raw_output", output=raw_output[:300])

    # Parse "1. [tool] description" format
    pattern = r'^\s*\d+[\.\)]\s*(?:\[(\w+)\]\s*)?(.+)'
    steps = []
    for match in re.finditer(pattern, raw_output, re.MULTILINE):
        tool_hint = match.group(1) or "none"
        description = match.group(2).strip()
        steps.append({"task": description, "tool_hint": tool_hint.lower()})

    # Fallback: treat each non-empty line as a task with no tool hint
    if not steps:
        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if line:
                steps.append({"task": line, "tool_hint": "none"})

    steps = steps[:MAX_PLAN_STEPS]
    logger.info("planner_completed", port=port, num_steps=len(steps), steps=steps)
    return steps, tokens


async def run_executor(
    task: str,
    port: int,
    history: str = "",
    feedback: str = "",
    **overrides,
) -> tuple[list[AgentStep], str, int]:
    """Run the ReAct tool loop for a single sub-task with optional LLM param overrides."""
    logger.info("executor_started", task=task[:60], port=port)
    system_prompt = build_executor_prompt(task=task, history=history, feedback=feedback)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    steps: list[AgentStep] = []
    total_tokens = 0
    previous_raw = ""

    for i in range(REACT_MAX_ITERATIONS):
        try:
            raw_output, tokens = await call_llm(
                messages, port=port, model=WORKER_MODEL,
                stop=["Observation:"], **overrides,
            )
            total_tokens += tokens
        except Exception as e:
            logger.error("executor_llm_failed", error=str(e))
            return steps, f"LLM call failed: {e}", total_tokens

        if raw_output.strip() == previous_raw:
            logger.warning("executor_repetitive_loop", iteration=i + 1)
            return steps, previous_raw, total_tokens
        previous_raw = raw_output.strip()

        step = parse_llm_output(raw_output)
        step.iteration = i + 1

        if step.is_final:
            steps.append(step)
            return steps, step.final_answer or "", total_tokens

        if step.action:
            tool = TOOL_REGISTRY.get(step.action.tool)
            observation = tool.execute(step.action.input) if tool else f"Error: Unknown tool '{step.action.tool}'"
            step.observation = observation
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": "You must use a tool (Action + Action Input) or provide a Final Answer."})

        steps.append(step)

    logger.warning("executor_max_iterations", task=task[:60])
    return steps, previous_raw, total_tokens


async def run_evaluator(task: str, worker_output: str, port: int, **overrides) -> tuple[EvaluationResult, int]:
    """Single-turn LLM call to evaluate if the Executor completed the sub-task."""
    logger.info("evaluator_checking", task=task[:60], port=port)
    messages = [
        {"role": "system", "content": build_evaluator_prompt(task, worker_output)},
        {"role": "user", "content": "Evaluate the worker output above."},
    ]
    raw_output, tokens = await call_llm(messages, port=port, model=WORKER_MODEL, **overrides)
    raw_output = raw_output.strip()

    if raw_output.upper().startswith("PASS"):
        result = EvaluationResult(passed=True)
    elif raw_output.upper().startswith("FAIL"):
        reason = raw_output.split(":", 1)[1].strip() if ":" in raw_output else raw_output
        result = EvaluationResult(passed=False, feedback=reason)
    else:
        logger.warning("evaluator_parse_failed", raw=raw_output[:100])
        result = EvaluationResult(passed=True, feedback="(unparseable, default pass)")

    return result, tokens


# ---------------------------------------------------------------------------
# Full agent loop (runs complete Planner→Executor↔Evaluator on ONE worker)
# ---------------------------------------------------------------------------
async def run_full_agent_loop(
    query: str,
    port: int,
    label: str,
    **overrides,
) -> EnsembleMemberResult:
    """Run a complete Planner→Executor↔Evaluator pipeline on a single worker instance."""
    logger.info("agent_loop_started", label=label, port=port, overrides=overrides)
    total_tokens = 0
    all_steps: list[AgentStep] = []

    # Plan — use the smarter compiler model for planning
    try:
        plan, tokens = await run_planner(query, port=COMPILER_PORTS[0], model_override=COMPILER_MODEL)
        total_tokens += tokens
    except Exception as e:
        logger.error("agent_loop_planner_failed", label=label, error=repr(e), error_type=type(e).__name__)
        return EnsembleMemberResult(
            config_label=label, llm_params=overrides,
            final_answer=f"Planner failed: {type(e).__name__}: {repr(e)}", tokens_used=total_tokens,
            evaluation=EvaluationResult(passed=False, feedback=repr(e)),
        )

    if not plan:
        plan = [{"task": query, "tool_hint": "none"}]  # Fallback

    # Execute + Evaluate each sub-task
    history = ""
    for i, step_info in enumerate(plan):
        sub_task = step_info["task"]
        tool_hint = step_info.get("tool_hint", "none")
        # Enrich the task description with the tool hint
        enriched_task = f"{sub_task} (suggested tool: {tool_hint})" if tool_hint != "none" else sub_task

        feedback = ""
        for attempt in range(1, MAX_EVALUATOR_RETRIES + 1):
            exec_steps, exec_answer, tokens = await run_executor(
                task=enriched_task, port=port, history=history, feedback=feedback, **overrides,
            )
            total_tokens += tokens
            all_steps.extend(exec_steps)

            try:
                eval_result, tokens = await run_evaluator(sub_task, exec_answer, port=port, **overrides)
                total_tokens += tokens
            except Exception as e:
                eval_result = EvaluationResult(passed=True, feedback=f"Evaluator failed: {e}")

            if eval_result.passed:
                history += f"- Step {i+1} [{tool_hint}] ({sub_task}): {exec_answer}\n"
                break
            else:
                feedback = eval_result.feedback

    # The final answer is the full accumulated history
    final_answer = history.strip() or exec_answer

    logger.info("agent_loop_completed", label=label, port=port)
    return EnsembleMemberResult(
        config_label=label,
        llm_params=overrides,
        executor_steps=all_steps,
        final_answer=final_answer,
        evaluation=EvaluationResult(passed=True),
        tokens_used=total_tokens,
    )


# ---------------------------------------------------------------------------
# Compiler (routes to COMPILER model)
# ---------------------------------------------------------------------------
async def run_compiler(
    query: str,
    ensemble_results: list[EnsembleMemberResult],
) -> tuple[str, int]:
    """Compile K agent loop results using the better compiler model."""
    logger.info("compiler_started", num_results=len(ensemble_results))
    port = COMPILER_PORTS[0]

    answers_text = "\n\n".join(
        f"Agent '{r.config_label}' (temp={r.llm_params.get('temperature', '?')}, top_p={r.llm_params.get('top_p', '?')}):\n"
        f"  Answer: {r.final_answer}\n"
        f"  Evaluation: {'PASS' if r.evaluation and r.evaluation.passed else 'FAIL'}"
        f"{' — ' + r.evaluation.feedback if r.evaluation and r.evaluation.feedback else ''}"
        for r in ensemble_results
    )

    messages = [
        {"role": "system", "content": build_compiler_prompt(query, answers_text)},
        {"role": "user", "content": "Compile the best answer now."},
    ]
    raw_output, tokens = await call_llm(
        messages, port=port, model=COMPILER_MODEL,
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
    )
    logger.info("compiler_completed", answer=raw_output.strip()[:100])
    return raw_output.strip(), tokens


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_agent(query: str) -> AgentResponse:
    """Run K full agent loops in parallel, then compile results."""
    total_tokens = 0

    # ── Always use ensemble: K parallel agent loops ──
    # Sample K stochastic configs
    configs = sample_ensemble_configs(ENSEMBLE_K)
    logger.info("ensemble_configs_sampled", configs=configs)

    # Fire K full agent loops in parallel (asyncio.gather)
    ensemble_results = await asyncio.gather(*[
        run_full_agent_loop(
            query=query,
            port=cfg["port"],
            label=cfg["label"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        for cfg in configs
    ])
    ensemble_results = list(ensemble_results)
    total_tokens += sum(r.tokens_used for r in ensemble_results)

    # Compile using the better model
    try:
        compiled_answer, tokens = await run_compiler(query, ensemble_results)
        total_tokens += tokens
    except Exception as e:
        logger.error("compiler_failed", error=str(e))
        compiled_answer = ensemble_results[0].final_answer if ensemble_results else "(no result)"

    # Build response
    sub_task_results = [
        SubTaskResult(
            sub_task=query,
            ensemble_results=ensemble_results,
            compiled_answer=compiled_answer,
            passed=any(r.evaluation and r.evaluation.passed for r in ensemble_results),
        )
    ]

    return AgentResponse(
        query=query,
        plan=[f"K={ENSEMBLE_K} parallel agent loops"],
        sub_task_results=sub_task_results,
        answer=compiled_answer,
        total_tokens=total_tokens,
        iterations=sum(len(r.executor_steps) for r in ensemble_results),
        success=True,
    )