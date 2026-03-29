# agent_hybrid.py
import asyncio
import re
import time
import structlog
from config import (
    WORKER_MODEL, WORKER_PORTS, COMPILER_MODEL, COMPILER_PORTS,
    REACT_MAX_ITERATIONS, MAX_PLAN_STEPS,
    PLANNER_MAX_TOKENS, EXECUTOR_MAX_TOKENS, SELF_CHECK_MAX_TOKENS,
    EVALUATOR_MAX_TOKENS, COMPILER_MAX_TOKENS,
    MAX_AGENT_RETRIES, ENSEMBLE_K,
    COMPILER_TEMPERATURE, COMPILER_TOP_P,
    SAMPLES_PER_ROLE, WORKER_POOL_SIZE,
)
from models import (
    ToolCall, ToolName, AgentStep, AgentResponse,
    EvaluationResult, SubTaskResult, EnsembleMemberResult,
)
from tools import TOOL_REGISTRY
from prompts import (
    build_planner_and_roles_prompt, build_plan_adjuster_prompt,
    build_self_check_prompt, build_full_evaluator_prompt,
    build_executor_prompt, build_compiler_prompt,
    build_unified_evaluator_prompt,
)
from roles import _parse_roles, _FALLBACK_ROLES
from llm_client import llm_client

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Parser (unchanged)
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
# Plan parsers
# ---------------------------------------------------------------------------
def _parse_plan_text(raw_output: str) -> list[dict]:
    """Parse planner LLM output into a list of step dicts."""
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

    return steps[:MAX_PLAN_STEPS]


def _parse_adjusted_plans(raw_output: str) -> list[list[dict]]:
    """Parse K adjusted plans from plan adjuster output.

    Expected format:
        AGENT: role_name
        1. [tool] description
        2. [tool] description
    """
    plans = []
    # Split on AGENT: headers (handle both leading and mid-text positions)
    blocks = re.split(r'(?:^|\n)(?=AGENT:)', raw_output.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if not re.search(r'^AGENT:\s*\S+', block, re.MULTILINE):
            continue
        # Strip the AGENT: header line, parse numbered steps
        step_text = re.sub(r'^AGENT:\s*\S+\s*\n?', '', block, count=1)
        steps = _parse_plan_text(step_text)
        if steps:
            plans.append(steps)
    return plans


# ---------------------------------------------------------------------------
# Stage 1: Planner + Role Designer (1× 9B)
# ---------------------------------------------------------------------------
async def run_planner_and_roles(query: str, k: int) -> tuple[list[dict], list[dict], int]:
    """Single 9B call: generate base plan + K agent roles."""
    logger.info("planner_and_roles_started", query=query[:60], k=k)
    messages = [
        {"role": "system", "content": build_planner_and_roles_prompt(query, k)},
        {"role": "user", "content": query},
    ]
    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        port=COMPILER_PORTS[0],
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
        max_tokens=PLANNER_MAX_TOKENS,
    )
    logger.info("planner_and_roles_raw", output=raw_output[:400])

    # Parse plan section (before first ROLE: block)
    plan_m = re.search(r'PLAN:\s*\n(.*?)(?=\nROLE:|\Z)', raw_output, re.DOTALL)
    if plan_m:
        plan_section = plan_m.group(1)
    else:
        role_start = raw_output.find("ROLE:")
        plan_section = raw_output[:role_start] if role_start > 0 else raw_output

    base_plan = _parse_plan_text(plan_section)
    if not base_plan:
        base_plan = [{"task": query, "tool_hint": "none"}]

    # Parse roles
    roles = _parse_roles(raw_output, k)
    if len(roles) < k:
        logger.warning("roles_parse_fallback", parsed=len(roles), needed=k)
        fallback = _FALLBACK_ROLES * ((k // len(_FALLBACK_ROLES)) + 1)
        roles = roles + fallback[len(roles):k]

    logger.info("planner_and_roles_completed",
                plan_steps=len(base_plan), roles=[r["label"] for r in roles])
    return base_plan, roles, tokens


# ---------------------------------------------------------------------------
# Stage 2: Plan Adjuster (1× 9B)
# ---------------------------------------------------------------------------
async def run_plan_adjuster(
    base_plan: list[dict], roles: list[dict], query: str
) -> tuple[list[list[dict]], int]:
    """Single 9B call: output K adjusted plans (one per role)."""
    logger.info("plan_adjuster_started", k=len(roles))
    messages = [
        {"role": "system", "content": build_plan_adjuster_prompt(base_plan, roles)},
        {"role": "user", "content": "Generate the adjusted plans now."},
    ]
    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        port=COMPILER_PORTS[0],
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
        max_tokens=PLANNER_MAX_TOKENS,
    )
    logger.info("plan_adjuster_raw", output=raw_output[:400])

    plans = _parse_adjusted_plans(raw_output)

    # Fill in missing plans with base_plan
    while len(plans) < len(roles):
        plans.append(list(base_plan))

    logger.info("plan_adjuster_completed", k=len(plans),
                step_counts=[len(p) for p in plans])
    return plans, tokens


# ---------------------------------------------------------------------------
# Executor: core ReAct loop (unchanged)
# ---------------------------------------------------------------------------
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
    last_obs = next((s.observation for s in reversed(steps) if s.observation), None)
    return steps, last_obs or previous_raw, total_tokens


# ---------------------------------------------------------------------------
# Stage 3a: Self-check (0.8B, max_tokens=16)
# ---------------------------------------------------------------------------
async def run_self_check(query: str, history: str, **params) -> tuple[bool, int]:
    """Ask the 0.8B worker: 'Have I fully addressed the query?' Returns (bool, tokens)."""
    messages = [
        {"role": "system", "content": build_self_check_prompt(query, history)},
        {"role": "user", "content": "Have you fully addressed the query? Answer YES or NO."},
    ]
    filtered = {k: v for k, v in params.items() if k in ("temperature", "top_p")}
    raw_output, tokens = await llm_client.call_llm(
        messages, WORKER_MODEL,
        port=WORKER_PORTS[0],
        max_tokens=SELF_CHECK_MAX_TOKENS,
        **filtered,
    )
    is_complete = bool(re.search(r'\bYES\b', raw_output.strip(), re.IGNORECASE))
    logger.info("self_check", is_complete=is_complete, raw=raw_output.strip()[:20])
    return is_complete, tokens


# ---------------------------------------------------------------------------
# Stage 3: Full agent executor (0.8B, per agent)
# ---------------------------------------------------------------------------
async def run_agent_executor(
    plan: list[dict],
    persona: str,
    query: str,
    feedback: str = "",
    **params,
) -> tuple[str, list[AgentStep], int]:
    """Execute all subtasks, then 0.8B self-check; one optional gap-fill pass."""
    history = ""
    all_steps: list[AgentStep] = []
    total_tokens = 0

    for i, step_info in enumerate(plan):
        sub_task = step_info["task"]
        tool_hint = step_info.get("tool_hint", "none")
        enriched_task = f"{sub_task} (suggested tool: {tool_hint})" if tool_hint != "none" else sub_task

        steps, answer, tokens = await run_executor(
            task=enriched_task,
            history=history,
            feedback=feedback if i == 0 else "",
            role_addendum=persona,
            max_tokens=EXECUTOR_MAX_TOKENS,
            **{k: v for k, v in params.items() if k != "max_tokens"},
        )
        total_tokens += tokens
        all_steps.extend(steps)

        clean_answer = answer
        if re.search(r"^Action:\s*\w+", clean_answer, re.MULTILINE):
            last_obs = next((s.observation for s in reversed(steps) if s.observation), None)
            clean_answer = last_obs or clean_answer
        history += f"- Step {i+1} [{tool_hint}] ({sub_task}): {clean_answer}\n"

    # 0.8B self-check
    is_complete, tokens = await run_self_check(query, history, **params)
    total_tokens += tokens

    if not is_complete:
        steps, answer, tokens = await run_executor(
            task=f"Address any remaining gaps in answering: {query}",
            history=history,
            feedback="Your previous work may have gaps. Address anything missing.",
            role_addendum=persona,
            max_tokens=EXECUTOR_MAX_TOKENS,
            **{k: v for k, v in params.items() if k != "max_tokens"},
        )
        total_tokens += tokens
        all_steps.extend(steps)

        clean_answer = answer
        if re.search(r"^Action:\s*\w+", clean_answer, re.MULTILINE):
            last_obs = next((s.observation for s in reversed(steps) if s.observation), None)
            clean_answer = last_obs or clean_answer
        history += f"- Follow-up: {clean_answer}\n"

    return history.strip(), all_steps, total_tokens


# ---------------------------------------------------------------------------
# Direct executor: call tools straight from the plan, no LLM reasoning loop
# ---------------------------------------------------------------------------
async def run_executor_direct(
    plan: list[dict],
    persona: str,
    query: str,
    **_params,
) -> tuple[str, list[AgentStep], int]:
    """Execute plan steps by calling tools directly — no LLM, no ReAct loop.

    Each step calls its tool_hint tool with the task description as input and
    records the raw observation. Returns a history string for the evaluator/compiler.
    """
    lines: list[str] = []
    all_steps: list[AgentStep] = []

    for step_info in plan:
        task = step_info["task"]
        tool_hint = step_info.get("tool_hint", "none")

        tool = None
        tool_name = None
        if tool_hint != "none":
            try:
                tool_name = ToolName(tool_hint)
                tool = TOOL_REGISTRY.get(tool_name)
            except ValueError:
                pass

        if tool:
            try:
                observation = await tool.execute(task)
            except Exception as e:
                observation = f"tool error: {e}"
            step = AgentStep(
                iteration=1, thought="", is_final=False,
                action=ToolCall(tool=tool_name, input=task),
                observation=observation,
            )
        else:
            observation = "(no tool assigned)"
            step = AgentStep(iteration=1, thought=task, is_final=False)

        all_steps.append(step)
        lines.append(f"[{tool_hint}] {task}:\n{observation}")

    return "\n\n".join(lines), all_steps, 0


# ---------------------------------------------------------------------------
# Stage 4: Holistic 9B evaluator (1 call per agent)
# ---------------------------------------------------------------------------
async def run_evaluator_full(query: str, agent_history: str) -> tuple[EvaluationResult, int]:
    """Holistic 9B evaluation of the full agent output."""
    logger.info("evaluator_full_started", query=query[:60])
    messages = [
        {"role": "system", "content": build_full_evaluator_prompt(query, agent_history)},
        {"role": "user", "content": "Evaluate the agent output above."},
    ]
    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        port=COMPILER_PORTS[0],
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
        max_tokens=EVALUATOR_MAX_TOKENS,
    )
    raw_output = raw_output.strip()

    # Strip any inline "Thinking Process:" prose that precedes the verdict
    # (model may output reasoning text before PASS/FAIL even outside <think> tags)
    search_text = re.sub(
        r'(?:Thinking Process:|Think(?:ing)?:).*?(?=\bPASS\b|\bFAIL\b)',
        '', raw_output, flags=re.DOTALL | re.IGNORECASE,
    ).strip() or raw_output

    fail_match = re.search(r"\bFAIL\b[:\s]*(.*)", search_text, re.IGNORECASE)
    pass_match = re.search(r"\bPASS\b", search_text, re.IGNORECASE)

    if fail_match:
        reason = fail_match.group(1).strip().splitlines()[0]
        result = EvaluationResult(passed=False, feedback=reason)
    elif pass_match:
        result = EvaluationResult(passed=True)
    else:
        logger.warning("evaluator_full_parse_failed", raw=raw_output[:100])
        result = EvaluationResult(passed=False, feedback="(evaluator output unparseable, default FAIL)")

    logger.info("evaluator_full_completed", passed=result.passed)
    return result, tokens


# ---------------------------------------------------------------------------
# Unified evaluator: 1× 9B call issues PASS/FAIL for every candidate
# ---------------------------------------------------------------------------
async def run_evaluator_unified(
    query: str,
    candidates: list[dict],
) -> tuple[list[dict], int]:
    """1× 9B call: PASS/FAIL verdict for all K×N candidates.

    candidates[i] must contain at least {"history": str, "role_label": str}.
    Returns (verdicts, tokens) where verdicts[i] = {"passed": bool, "reason": str}.
    """
    logger.info("evaluator_unified_started", n=len(candidates))
    messages = [
        {"role": "system", "content": build_unified_evaluator_prompt(query, candidates)},
        {"role": "user", "content": "Evaluate all candidates now."},
    ]
    raw_output, tokens = await llm_client.call_llm(
        messages, COMPILER_MODEL,
        port=COMPILER_PORTS[0],
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
        max_tokens=EVALUATOR_MAX_TOKENS,
    )

    verdicts: list[dict] = [{"passed": False, "reason": ""} for _ in candidates]
    for m in re.finditer(r"VERDICT\s+(\d+):\s*(PASS|FAIL)(?:\s*[-:]\s*(.+))?", raw_output, re.IGNORECASE):
        idx = int(m.group(1))
        if 0 <= idx < len(candidates):
            verdicts[idx] = {
                "passed": m.group(2).upper() == "PASS",
                "reason": (m.group(3) or "").strip(),
            }

    passed_count = sum(1 for v in verdicts if v["passed"])
    logger.info("evaluator_unified_completed", passed=passed_count, total=len(candidates))
    return verdicts, tokens


# ---------------------------------------------------------------------------
# Compiler (1× 9B)
# ---------------------------------------------------------------------------
async def run_compiler(
    query: str,
    ensemble_results: list[EnsembleMemberResult],
) -> tuple[str, int]:
    """Compile K agent results into one final answer."""
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
        temperature=COMPILER_TEMPERATURE, top_p=COMPILER_TOP_P,
        port=COMPILER_PORTS[0],
        max_tokens=COMPILER_MAX_TOKENS,
    )
    logger.info("compiler_completed", answer=raw_output.strip()[:100])
    return raw_output.strip(), tokens


# ---------------------------------------------------------------------------
# Orchestrator: pipelined execution with async evaluation
# ---------------------------------------------------------------------------
async def run_agent(query: str) -> AgentResponse:
    """New architecture: 2× 9B setup → K parallel 0.8B executors → K 9B evals (pipelined) → 1× 9B compile.

    Stage 1: Planner + Role Designer (1× 9B) — base plan + K roles
    Stage 2: Plan Adjuster (1× 9B) — K personalized plans
    Stage 3+4: K agents run concurrently (asyncio.gather):
               each: 0.8B executor loop → 0.8B self-check → 9B eval → optional retry
    Stage 5: Compiler (1× 9B) — compile K results
    """
    total_tokens = 0

    # Stage 1: Planner + Role Designer (1× 9B)
    try:
        base_plan, roles, tokens = await run_planner_and_roles(query, ENSEMBLE_K)
        total_tokens += tokens
    except Exception as e:
        logger.error("planner_and_roles_failed", error=repr(e))
        base_plan = [{"task": query, "tool_hint": "none"}]
        roles = _FALLBACK_ROLES[:ENSEMBLE_K]
        roles = roles + _FALLBACK_ROLES[:max(0, ENSEMBLE_K - len(roles))]

    # Stage 2: Plan Adjuster (1× 9B)
    try:
        plans, tokens = await run_plan_adjuster(base_plan, roles, query)
        total_tokens += tokens
    except Exception as e:
        logger.warning("plan_adjuster_failed_fallback", error=repr(e))
        plans = [list(base_plan) for _ in roles]

    # Stage 3+4: K agents execute + evaluate concurrently (pipelined)
    async def execute_and_evaluate(i: int, plan: list[dict], role: dict) -> EnsembleMemberResult:
        label = role["label"]
        persona = role.get("executor_addendum", "")
        temperature = role["temperature"]
        top_p = role["top_p"]
        agent_tokens = 0

        # Stage 3: Execute all subtasks + self-check
        history, steps, tokens = await run_agent_executor(
            plan, persona, query,
            temperature=temperature, top_p=top_p,
        )
        agent_tokens += tokens

        # Stage 4: Holistic 9B evaluation (fires as soon as this agent is done)
        eval_result, tokens = await run_evaluator_full(query, history)
        agent_tokens += tokens

        if not eval_result.passed:
            # One retry: re-execute with 9B feedback
            history, steps_retry, tokens = await run_agent_executor(
                plan, persona, query,
                feedback=eval_result.feedback,
                temperature=temperature, top_p=top_p,
            )
            agent_tokens += tokens
            steps = steps + steps_retry
            eval_result, tokens = await run_evaluator_full(query, history)
            agent_tokens += tokens

        logger.info("agent_done", label=label, passed=eval_result.passed,
                    tokens=agent_tokens)
        return EnsembleMemberResult(
            config_label=label,
            role=label,
            llm_params={"temperature": temperature, "top_p": top_p},
            executor_steps=steps,
            final_answer=history,
            evaluation=eval_result,
            tokens_used=agent_tokens,
        )

    t0 = time.monotonic()
    ensemble_results = list(await asyncio.gather(*[
        execute_and_evaluate(i, plans[i], roles[i])
        for i in range(ENSEMBLE_K)
    ]))
    elapsed = time.monotonic() - t0
    logger.info("ensemble_completed", elapsed_s=round(elapsed, 2), k=ENSEMBLE_K)
    total_tokens += sum(r.tokens_used for r in ensemble_results)

    # Stage 5: Compiler (1× 9B)
    try:
        compiled_answer, tokens = await run_compiler(query, ensemble_results)
        total_tokens += tokens
    except Exception as e:
        logger.error("compiler_failed", error=str(e))
        compiled_answer = ensemble_results[0].final_answer if ensemble_results else "(no result)"

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
        plan=[f"K={ENSEMBLE_K} pipelined agents → async 9B eval → compile"],
        sub_task_results=sub_task_results,
        answer=compiled_answer,
        total_tokens=total_tokens,
        iterations=sum(len(r.executor_steps) for r in ensemble_results),
        success=True,
    )


# ---------------------------------------------------------------------------
# Many-samples orchestrator: K×N cheap executors → best-of-N filter → compile
# ---------------------------------------------------------------------------
async def run_agent_many_samples(query: str) -> AgentResponse:
    """K×N 135M executor samples, 9B best-of-N per role, 9B compile.

    Stage 1: Planner + Role Designer (1× 9B)
    Stage 2: Plan Adjuster (1× 9B)
    Stage 3: K×N executor instances concurrently (135M worker, K CUDA streams)
    Stage 4: K best-of-N evaluators (1× 9B each)
    Stage 5: Compiler (1× 9B)
    """
    total_tokens = 0

    # Stage 1
    try:
        base_plan, roles, tokens = await run_planner_and_roles(query, ENSEMBLE_K)
        total_tokens += tokens
    except Exception as e:
        logger.error("planner_and_roles_failed", error=repr(e))
        base_plan = [{"task": query, "tool_hint": "none"}]
        roles = _FALLBACK_ROLES[:ENSEMBLE_K]
        roles = roles + _FALLBACK_ROLES[:max(0, ENSEMBLE_K - len(roles))]

    # Stage 2
    try:
        plans, tokens = await run_plan_adjuster(base_plan, roles, query)
        total_tokens += tokens
    except Exception as e:
        logger.warning("plan_adjuster_failed_fallback", error=repr(e))
        plans = [list(base_plan) for _ in roles]

    # Stage 3: K×N direct executions — tools called straight from the plan
    async def run_one_sample(role_idx: int, sample_idx: int, plan: list[dict], role: dict):
        persona = role.get("executor_addendum", "")
        history, steps, tokens = await run_executor_direct(plan, persona, query)
        return history, steps, tokens

    sample_tasks = [
        run_one_sample(i, j, plans[i], roles[i])
        for i in range(ENSEMBLE_K)
        for j in range(SAMPLES_PER_ROLE)
    ]

    t0 = time.monotonic()
    all_sample_results = list(await asyncio.gather(*sample_tasks))
    elapsed = time.monotonic() - t0
    logger.info("many_samples_executor_done", elapsed_s=round(elapsed, 2),
                k=ENSEMBLE_K, n=SAMPLES_PER_ROLE)
    total_tokens += sum(r[2] for r in all_sample_results)

    # Stage 4: single unified evaluator — PASS/FAIL for every candidate
    flat_candidates = [
        {
            "history":    all_sample_results[i * SAMPLES_PER_ROLE + j][0],
            "steps":      all_sample_results[i * SAMPLES_PER_ROLE + j][1],
            "role":       roles[i],
            "role_label": roles[i]["label"],
        }
        for i in range(ENSEMBLE_K)
        for j in range(SAMPLES_PER_ROLE)
    ]

    verdicts, tokens = await run_evaluator_unified(query, flat_candidates)
    total_tokens += tokens

    # Keep only candidates the evaluator accepted
    ensemble_results = [
        EnsembleMemberResult(
            config_label=c["role"]["label"],
            role=c["role"]["label"],
            llm_params={},
            executor_steps=c["steps"],
            final_answer=c["history"],
            evaluation=EvaluationResult(passed=v["passed"], feedback=v["reason"]),
            tokens_used=0,
        )
        for c, v in zip(flat_candidates, verdicts)
        if v["passed"]
    ]

    # Fallback: if nothing passed, send everything to the compiler
    if not ensemble_results:
        logger.warning("evaluator_unified_all_failed", total=len(flat_candidates))
        ensemble_results = [
            EnsembleMemberResult(
                config_label=c["role"]["label"],
                role=c["role"]["label"],
                llm_params={},
                executor_steps=c["steps"],
                final_answer=c["history"],
                evaluation=EvaluationResult(passed=False, feedback=verdicts[idx]["reason"]),
                tokens_used=0,
            )
            for idx, c in enumerate(flat_candidates)
        ]

    # Stage 5: Compiler
    try:
        compiled_answer, tokens = await run_compiler(query, ensemble_results)
        total_tokens += tokens
    except Exception as e:
        logger.error("compiler_failed", error=str(e))
        compiled_answer = ensemble_results[0].final_answer if ensemble_results else "(no result)"

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
        plan=[f"K={ENSEMBLE_K}×N={SAMPLES_PER_ROLE} many-samples → unified PASS/FAIL eval → compile"],
        sub_task_results=sub_task_results,
        answer=compiled_answer,
        total_tokens=total_tokens,
        iterations=sum(len(r.executor_steps) for r in ensemble_results),
        success=True,
    )
