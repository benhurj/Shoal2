# prompts.py
from datetime import date
from tools import TOOL_REGISTRY


def _tool_descriptions() -> str:
    return "\n".join(
        f"  - {t.name.value}: {t.description} (example: {t.usage_example})"
        for t in TOOL_REGISTRY.values()
    )


# ---------------------------------------------------------------------------
# Fast-path / legacy single-loop system prompt (kept for backwards compat)
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    return build_executor_prompt()


# ---------------------------------------------------------------------------
# Stage 1: Planner + Role Designer (merged, 1× 9B call)
# ---------------------------------------------------------------------------
def build_planner_and_roles_prompt(query: str, k: int) -> str:
    tool_descriptions = _tool_descriptions()
    tool_names = ", ".join(t.name.value for t in TOOL_REGISTRY.values())
    return f"""You are a task planner and agent role designer.

Given the user query, output:
1. A base plan (numbered list of sub-tasks with tool hints)
2. {k} distinct agent personas to tackle the query from different angles

Available tools:
{tool_descriptions}

Output format (EXACTLY as shown, no extra text):

PLAN:
1. [TOOL] description
2. [TOOL] description

ROLE: role_name
STYLE: precise
TEMPERATURE: 0.10
TOP_P: 0.80
PLANNER: One sentence: how this role should plan.
EXECUTOR: One sentence: how this role should execute.

(repeat ROLE block {k} times)

Rules:
- TOOL must be one of: {tool_names}, none
- For [calculator] steps the description MUST be a valid Python arithmetic expression (e.g. "17 * 23", "(144/12)*7"). Never write natural language like "calculate the result".
- For [search] steps the description is the search query string.
- For [datetime] steps write "now".
- Role names: 1-3 words, underscores, lowercase
- Each role must be genuinely different in approach
- Tailor roles to the nature of the query
- precise style: TEMPERATURE ~0.10, TOP_P ~0.80
- balanced style: TEMPERATURE ~0.35, TOP_P ~0.88
- exploratory style: TEMPERATURE ~0.70, TOP_P ~0.95
- Keep plan to 1-3 steps
"""


# ---------------------------------------------------------------------------
# Stage 2: Plan Adjuster (1× 9B call, outputs K adjusted plans)
# ---------------------------------------------------------------------------
def build_plan_adjuster_prompt(base_plan: list[dict], roles: list[dict]) -> str:
    tool_names = ", ".join(t.name.value for t in TOOL_REGISTRY.values())
    plan_text = "\n".join(
        f"{i+1}. [{s['tool_hint']}] {s['task']}"
        for i, s in enumerate(base_plan)
    )
    roles_text = "\n".join(
        f"ROLE: {r['label']} — {r.get('executor_addendum', '')}"
        for r in roles
    )
    k = len(roles)
    return f"""You are a task planner. Customize the base plan for each agent role.

Base plan:
{plan_text}

Agent roles:
{roles_text}

For each agent, output an adjusted version of the plan adapted to that role's approach.

Output format (EXACTLY as shown, one block per role):

AGENT: role_name
1. [TOOL] adapted description
2. [TOOL] adapted description

Rules:
- TOOL must be one of: {tool_names}, none
- Use calculator ONLY when the step description is an actual Python arithmetic expression (e.g. "330 * 0.9"). Do NOT assign calculator to text descriptions or data-gathering steps.
- Keep to 3 steps max per agent
- {k} AGENT blocks total, in the same order as the roles listed
- No extra text, no explanation
"""


# ---------------------------------------------------------------------------
# Stage 3: Follow-up decision (135M, max_tokens=64)
# ---------------------------------------------------------------------------
def build_followup_decision_prompt(
    task: str,
    tool_hint: str,
    observation: str,
    tool_descriptions: str | None = None,
) -> list[dict]:
    """Two-message prompt for SmolLM2-135M-Instruct.

    Returns a messages list ready for call_llm.
    The model must output either:
        DONE
    or:
        FOLLOWUP
        TOOL: <tool_name>
        INPUT: <query>

    tool_descriptions: if provided (from MCPClient), used instead of bare tool names.
    """
    if tool_descriptions:
        tool_info = f"Tools:\n{tool_descriptions}"
    else:
        tool_names = ", ".join(t.name.value for t in TOOL_REGISTRY.values())
        tool_info = f"Tools: {tool_names}"

    obs_truncated = observation[:600]
    return [
        {
            "role": "system",
            "content": (
                "You decide if a tool result answers the task. Reply with EXACTLY:\n"
                "  DONE\n"
                "or:\n"
                "  FOLLOWUP\n"
                "  TOOL: <tool_name>\n"
                "  INPUT: <value>\n\n"
                f"{tool_info}\n\n"
                "Use DONE only when the result directly contains the answer (a number, fact, or date).\n"
                "Use FOLLOWUP when:\n"
                "  - Result says 'not a Python expression' or 'skipped' → TOOL: calculator, INPUT: the numeric expression extracted from the task (digits and operators only, e.g. '17 * 23')\n"
                "  - Result is an error or empty → retry with a corrected input\n"
                "  - Result is a web page and a calculation is still needed\n"
                "For calculator INPUT use only digits and operators (+,-,*,/,**,%). No words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\n"
                f"Tool used: {tool_hint}\n"
                f"Result: {obs_truncated}"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Stage 3a: Self-check (0.8B, max_tokens=16)
# ---------------------------------------------------------------------------
def build_self_check_prompt(query: str, history: str) -> str:
    return f"""You are an agent reviewing your own work.

Original query: {query}

Your work so far:
{history}

Have you fully addressed every part of the original query?
Answer with exactly YES or NO."""


# ---------------------------------------------------------------------------
# Executor: the ReAct worker that uses tools (unchanged)
# ---------------------------------------------------------------------------
def build_executor_prompt(task: str = "", history: str = "", feedback: str = "", role_addendum: str = "") -> str:
    tool_descriptions = _tool_descriptions()

    context_block = ""
    if task:
        context_block += f"\n\nYOUR TASK: {task}\nYou MUST complete this exact task. Do NOT deviate from it."
    if history:
        context_block += f"\n\nPrevious results:\n{history}"
    if feedback:
        context_block += f"\n\nPrevious attempt FAILED: {feedback}. Try differently."
    if role_addendum:
        context_block += f"\n\nYour approach: {role_addendum}"

    tool_names = ", ".join(t.name.value for t in TOOL_REGISTRY.values())

    return f"""You are a task executor. You complete the given task using tools.
Today's date: {date.today().strftime('%B %d, %Y')}.
{context_block}

Available tools:
{tool_descriptions}

Format — to use a tool:
Action: <tool_name>
Action Input: <input>

where <tool_name> must be one of: {tool_names}

Format — when done:
Final Answer: <answer>

RULES:
- Only use the tools listed above. Do NOT invent tool names.
- Action Input must be relevant to YOUR TASK above. Nothing else.
- Do NOT make up answers. Use a tool first, then answer from the Observation."""


# ---------------------------------------------------------------------------
# Stage 4: Full evaluator (holistic 9B eval, max_tokens=128)
# ---------------------------------------------------------------------------
def build_full_evaluator_prompt(query: str, agent_history: str) -> str:
    return f"""You are a strict evaluator. An agent attempted to answer a user query by executing multiple sub-tasks.

Original query: {query}

Agent's complete work:
{agent_history}

Rules:
- PASS if the agent clearly addressed the full query with evidence from tool observations.
- FAIL if the answer is incomplete, vague, or unsupported by observations.
- FAIL if major parts of the query were not addressed.
- FAIL if the output contains raw tool Action/Input blocks rather than synthesized answers.

Your response MUST end with exactly one of these two lines (no other format accepted):
PASS
FAIL: <short reason>

Output the verdict line immediately. Do not write a preamble."""


# ---------------------------------------------------------------------------
# Synthesizer: combines sub-task results into one final answer
# ---------------------------------------------------------------------------
def build_synthesizer_prompt(query: str, results: str) -> str:
    return f"""You are a helpful assistant. The user asked a question and it was broken into sub-tasks. Here are the results of each sub-task:

{results}

Original user question: {query}

Using the sub-task results above, write a clear, concise final answer to the user's original question. Do not mention sub-tasks or internal processes.
"""


# ---------------------------------------------------------------------------
# Unified evaluator: 9B issues PASS/FAIL for every candidate (1 call total)
# ---------------------------------------------------------------------------
def build_unified_evaluator_prompt(query: str, candidates: list[dict]) -> str:
    n = len(candidates)

    def _fmt_history(history: str) -> str:
        """Per-step truncation: keep up to 800 chars per step, 3000 total."""
        steps = history.split("\n\n")
        parts = [s[:800] for s in steps]
        combined = "\n\n".join(parts)
        return combined[:3000]

    candidates_text = "\n\n".join(
        f"Candidate {i} (role: {c['role_label']}):\n{_fmt_history(c['history'])}"
        for i, c in enumerate(candidates)
    )
    return f"""You are an evaluator. Multiple agents collected information to answer a query.
Each candidate shows raw tool observations (search results, calculations, etc.).

Query: {query}

{candidates_text}

Rules:
- PASS if the tool observations contain specific, grounded data that covers the query (numbers, facts, comparisons as relevant).
- FAIL if key information needed to answer the query is clearly absent from the observations.
- FAIL if the observations show only errors or "(no tool assigned)" with no useful data.

Output one verdict per candidate, exactly in this format (no other text):
VERDICT 0: PASS|FAIL
VERDICT 1: PASS|FAIL
...
VERDICT {n - 1}: PASS|FAIL"""


# ---------------------------------------------------------------------------
# Compiler: synthesizes N ensemble results into one compiled answer
# ---------------------------------------------------------------------------
def build_compiler_prompt(task: str, ensemble_answers: str) -> str:
    return f"""You are a compiler. Multiple agents attempted the same task using different strategies. Each attempt has been evaluated as PASS or FAIL.

Task: {task}

Here are all the attempts and their evaluations:
{ensemble_answers}

Your job:
1. Weigh PASS results more heavily than FAIL results.
2. Identify the most accurate and complete answer across all attempts.
3. Compile a single, best answer that combines the strongest elements.

Output ONLY the compiled answer. Do not mention agents, attempts, or evaluations.
"""
