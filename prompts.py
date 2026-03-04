# prompts.py
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
# Planner: outputs a numbered list of sub-tasks
# ---------------------------------------------------------------------------
def build_planner_prompt() -> str:
    tool_descriptions = _tool_descriptions()
    return f"""You are a task planner. Break the user request into a numbered checklist of sub-tasks.

Available tools:
{tool_descriptions}

Rules:
- Output ONLY a numbered list. No extra text.
- Each item format: NUMBER. [TOOL] description
- TOOL must be one of: calculator, search, datetime, or none
- Use [none] if the sub-task needs no tool (e.g. reasoning).
- Keep the list short (1-4 items).

Examples:

User: What is 5 * 20 and is the result greater than 50?
1. [calculator] Calculate 5 * 20
2. [none] Check if the result is greater than 50

User: What is the weather in London?
1. [search] Search for current weather in London

User: What time is it and what is 100 / 4?
1. [datetime] Get the current date and time
2. [calculator] Calculate 100 divided by 4

User: What was Apple stock price yesterday?
1. [search] Search for Apple stock price yesterday

Now respond with ONLY a numbered list.
"""


# ---------------------------------------------------------------------------
# Executor: the ReAct worker that uses tools
# ---------------------------------------------------------------------------
def build_executor_prompt(task: str = "", history: str = "", feedback: str = "") -> str:
    tool_descriptions = _tool_descriptions()

    context_block = ""
    if task:
        context_block += f"\n\nYOUR TASK: {task}\nYou MUST complete this exact task. Do NOT search for anything else."
    if history:
        context_block += f"\n\nPrevious results:\n{history}"
    if feedback:
        context_block += f"\n\nPrevious attempt FAILED: {feedback}. Try differently."

    return f"""You are a task executor. You complete the given task using tools.
{context_block}

Tools:
{tool_descriptions}

Format — to use a tool:
Action: <tool_name>
Action Input: <input>

Format — when done:
Final Answer: <answer>

RULES:
- Action Input must be about YOUR TASK above. Nothing else.
- Do NOT make up answers. Use a tool first, then answer from the Observation."""


# ---------------------------------------------------------------------------
# Evaluator: checks if the Executor completed a sub-task
# ---------------------------------------------------------------------------
def build_evaluator_prompt(task: str, worker_output: str) -> str:
    return f"""You are a fair evaluator. You are given a sub-task instruction and the output produced by a worker.

Determine if the worker completed the sub-task to a reasonable standard. Be lenient: if the worker addressed the core intent, it passes.

Sub-task: {task}

Worker output: {worker_output}

Respond with EXACTLY one of:
- PASS
- FAIL: <short reason why it failed>

Do not output anything else.
"""


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