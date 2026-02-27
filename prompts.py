# prompts.py
from tools import TOOL_REGISTRY


def build_system_prompt() -> str:
    tool_descriptions = "\n".join(
        f"  - {t.name.value}: {t.description} (example: {t.usage_example})"
        for t in TOOL_REGISTRY.values()
    )

    return f"""You are a helpful AI assistant that answers questions by reasoning step-by-step and using tools.

Available tools:
{tool_descriptions}

You MUST follow this exact format for every response:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input to the tool>

After you receive an Observation (the tool's output), continue with another Thought.

When you have enough information to answer, respond with:

Thought: I now have enough information to answer.
Final Answer: <your complete answer to the user's question>

Rules:
- Always start with a Thought.
- Use exactly one tool per step.
- Never invent tool outputs. Wait for the Observation.
- If a tool returns an error, try a different approach.
- You MUST eventually produce a Final Answer within 6 steps.
"""