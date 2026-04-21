# agent_hybrid.py
import asyncio
import re
import time
import structlog
from config import (
    WORKER_MODEL, WORKER_PORTS, COMPILER_MODEL, COMPILER_PORTS,
    REACT_MAX_ITERATIONS, MAX_PLAN_STEPS,
    MAX_EVALUATOR_RETRIES, ENSEMBLE_K,
    COMPILER_TEMPERATURE, COMPILER_TOP_P,
)
from models import (
    ToolCall, ToolName, AgentStep, AgentResponse,
    EvaluationResult, SubTaskResult, EnsembleMemberResult,
)
from tools import TOOL_REGISTRY
from prompts import (
    build_planner_prompt, build_executor_prompt,
    build_evaluator_prompt, build_compiler_prompt,
)
from roles import select_roles
from llm_client import llm_client

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_ensemble_configs(k: int) -> list[dict]:
    """Build K role-based ensemble configs. Uses defined roles first, stochastic overflow."""
    return select_roles(k)

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
# Worker stages (using unified LLM client)
# ---------------------------------------------------------------------------
async def run_planner(query: str, role_addendum: str = "") -> tuple[list[dict], int]:
    """Decompose a query into numbered sub-tasks with tool hints."""
    logger.info("planner_started", query=query)
    messages = [
        {"role": "system", "content": build_planner_prompt(role_addendum=role_addendum)},
        {"role": "user", "content": query},
    ]

    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        port=COMPILER_PORTS[0], temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
    )
    logger.info("planner_raw_output", output=raw_output[:300])

    # Parse "1. [tool] description" format (numbered)
    pattern = r'^\s*\d+[\.\)]\s*(?:\[(\w+)\]\s*)?(.+)'
    steps = []
    for match in re.finditer(pattern, raw_output, re.MULTILINE):
        tool_hint = match.group(1) or "none"
        description = match.group(2).strip()
        steps.append({"task": description, "tool_hint": tool_hint.lower()})

    # Fallback: parse "[tool] description" format (non-numbered)
    if not steps:
        unnumbered = r'^\s*\[(\w+)\]\s*(.+)'
        for match in re.finditer(unnumbered, raw_output, re.MULTILINE):
            tool_hint = match.group(1) or "none"
            description = match.group(2).strip()
            steps.append({"task": description, "tool_hint": tool_hint.lower()})

    # Last resort: treat each non-empty line as a task with no tool hint
    if not steps:
        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if line:
                steps.append({"task": line, "tool_hint": "none"})

    steps = steps[:MAX_PLAN_STEPS]
    logger.info("planner_completed", num_steps=len(steps), steps=steps)
    return steps, tokens

async def run_executor(
    task: str,
    history: str = "",
    feedback: str = "",
    role_addendum: str = "",
    **overrides,
) -> tuple[list[AgentStep], str, int]:
    """Run the ReAct tool loop for a single sub-task."""
    logger.info("executor_started", task=task[:60])
    system_prompt = build_executor_prompt(task=task, history=history, feedback=feedback, role_addendum=role_addendum)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    steps: list[AgentStep] = []
    total_tokens = 0
    previous_raw = ""

    for i in range(REACT_MAX_ITERATIONS):
        try:
            raw_output, tokens = await llm_client.call_llm(
                messages, WORKER_MODEL,
                stop=["Observation:"],
                port=WORKER_PORTS[0],
                **{k: v for k, v in overrides.items() if k not in ('stop', 'port')}
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
            observation = (await tool.execute(step.action.input)) if tool else f"Error: Unknown tool '{step.action.tool}'"
            step.observation = observation
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": "You must use a tool (Action + Action Input) or provide a Final Answer."})

        steps.append(step)

    logger.warning("executor_max_iterations", task=task[:60])
    return steps, previous_raw, total_tokens

async def run_evaluator(task: str, worker_output: str, **overrides) -> tuple[EvaluationResult, int]:
    """Single-turn LLM call to evaluate if the Executor completed the sub-task."""
    logger.info("evaluator_checking", task=task[:60])
    messages = [
        {"role": "system", "content": build_evaluator_prompt(task, worker_output)},
        {"role": "user", "content": "Evaluate the worker output above."},
    ]
    raw_output, tokens = await llm_client.call_llm(messages, WORKER_MODEL, port=WORKER_PORTS[0], **{k: v for k, v in overrides.items() if k != 'port'})
    raw_output = raw_output.strip()

    if raw_output.upper().startswith("PASS"):
        result = EvaluationResult(passed=True)
    elif raw_output.upper().startswith("FAIL"):
        reason = raw_output.split(":", 1)[1].strip() if ":" in raw_output else raw_output
        result = EvaluationResult(passed=False, feedback=reason)
    else:
        logger.warning("evaluator_parse_failed", raw=raw_output[:100])
        result = EvaluationResult(passed=True, feedback="(evaluator output unparseable, default PASS to match lenient prompt)")

    return result, tokens

# ---------------------------------------------------------------------------
# Full agent loop (runs complete Planner→Executor↔Evaluator)
# ---------------------------------------------------------------------------
async def run_full_agent_loop(
    query: str,
    label: str,
    planner_addendum: str = "",
    executor_addendum: str = "",
    **overrides,
) -> EnsembleMemberResult:
    """Run a complete Planner→Executor↔Evaluator pipeline with role-specific behavior."""
    t0 = time.monotonic()
    logger.info("agent_loop_started", label=label, overrides=overrides)
    total_tokens = 0
    all_steps: list[AgentStep] = []

    # Plan (uses COMPILER_TEMPERATURE, role shapes the planning approach)
    try:
        plan, tokens = await run_planner(query, role_addendum=planner_addendum)
        total_tokens += tokens
    except Exception as e:
        logger.error("agent_loop_planner_failed", label=label, error=repr(e))
        return EnsembleMemberResult(
            config_label=label, role=label, llm_params=overrides,
            final_answer=f"Planner failed: {type(e).__name__}: {repr(e)}", tokens_used=total_tokens,
            evaluation=EvaluationResult(passed=False, feedback=repr(e)),
        )

    if not plan:
        plan = [{"task": query, "tool_hint": "none"}]  # Fallback

    # Execute + Evaluate each sub-task
    history = ""
    exec_answer = ""
    eval_results: list[EvaluationResult] = []
    for i, step_info in enumerate(plan):
        sub_task = step_info["task"]
        tool_hint = step_info.get("tool_hint", "none")
        enriched_task = f"{sub_task} (suggested tool: {tool_hint})" if tool_hint != "none" else sub_task

        feedback = ""
        eval_result = EvaluationResult(passed=False, feedback="never evaluated")
        for attempt in range(1, MAX_EVALUATOR_RETRIES + 1):
            exec_steps, exec_answer, tokens = await run_executor(
                task=enriched_task, history=history, feedback=feedback,
                role_addendum=executor_addendum, **overrides,
            )
            total_tokens += tokens
            all_steps.extend(exec_steps)

            try:
                eval_result, tokens = await run_evaluator(sub_task, exec_answer, **overrides)
                total_tokens += tokens
            except Exception as e:
                eval_result = EvaluationResult(passed=False, feedback=f"Evaluator failed: {e}")

            if eval_result.passed:
                history += f"- Step {i+1} [{tool_hint}] ({sub_task}): {exec_answer}\n"
                break
            else:
                feedback = eval_result.feedback
        eval_results.append(eval_result)

    # The final answer is the full accumulated history
    final_answer = history.strip() or exec_answer

    overall_passed = all(e.passed for e in eval_results) if eval_results else False
    failed_feedback = "; ".join(e.feedback for e in eval_results if not e.passed and e.feedback)

    elapsed = time.monotonic() - t0
    logger.info("agent_loop_completed", label=label, elapsed_s=round(elapsed, 2))
    return EnsembleMemberResult(
        config_label=label,
        role=label,
        llm_params=overrides,
        executor_steps=all_steps,
        final_answer=final_answer,
        evaluation=EvaluationResult(passed=overall_passed, feedback=failed_feedback),
        tokens_used=total_tokens,
    )

# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
async def run_compiler(
    query: str,
    ensemble_results: list[EnsembleMemberResult],
) -> tuple[str, int]:
    """Compile K agent loop results using the better compiler model."""
    logger.info("compiler_started", num_results=len(ensemble_results))

    answers_text = "\n\n".join(
        f"Agent '{r.config_label}' (role={r.role}, temp={r.llm_params.get('temperature', '?')}):\n"
        f"  Answer: {r.final_answer}\n"
        f"  Evaluation: {'PASS' if r.evaluation and r.evaluation.passed else 'FAIL'}"
        f"{' — ' + r.evaluation.feedback if r.evaluation and r.evaluation.feedback else ''}"
        for r in ensemble_results
    )

    messages = [
        {"role": "system", "content": build_compiler_prompt(query, answers_text)},
        {"role": "user", "content": "Compile the best answer now."},
    ]
    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P, port=COMPILER_PORTS[0],
    )
    logger.info("compiler_completed", answer=raw_output.strip()[:100])
    return raw_output.strip(), tokens

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
async def run_agent(query: str) -> AgentResponse:
    """Run K full agent loops in parallel, then compile results."""
    total_tokens = 0

    # Build K role-based configs
    configs = build_ensemble_configs(ENSEMBLE_K)
    logger.info("ensemble_configs", configs=[c["label"] for c in configs])

    # Fire K full agent loops in parallel
    t_ensemble_start = time.monotonic()
    ensemble_results = await asyncio.gather(*[
        run_full_agent_loop(
            query=query,
            label=cfg["label"],
            planner_addendum=cfg.get("planner_addendum", ""),
            executor_addendum=cfg.get("executor_addendum", ""),
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        for cfg in configs
    ])
    ensemble_results = list(ensemble_results)
    t_ensemble_elapsed = time.monotonic() - t_ensemble_start
    logger.info("ensemble_completed", elapsed_s=round(t_ensemble_elapsed, 2), k=ENSEMBLE_K)
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
