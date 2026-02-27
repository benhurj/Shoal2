# agent.py
import re
import httpx
import structlog
from config import LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, REACT_MAX_ITERATIONS
from models import ToolCall, ToolName, AgentStep, AgentResponse
from tools import TOOL_REGISTRY
from prompts import build_system_prompt

logger = structlog.get_logger()


def call_llm(messages: list[dict]) -> tuple[str, int]:
    """Call the llama.cpp OpenAI-compatible endpoint."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
                "stop": ["Observation:"],  # Stop before hallucinating observations
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return content, tokens


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
            iteration=0,
            thought=thought,
            is_final=True,
            final_answer=final_match.group(1).strip(),
        )

    # Fallback
    return AgentStep(iteration=0, thought=text)


def run_agent(query: str) -> AgentResponse:
    """Execute the full ReAct loop."""
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    steps: list[AgentStep] = []
    total_tokens = 0

    for i in range(REACT_MAX_ITERATIONS):
        logger.info("react_iteration", iteration=i + 1, query=query)

        # Call LLM
        try:
            raw_output, tokens = call_llm(messages)
            total_tokens += tokens
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            return AgentResponse(
                query=query, answer="", steps=steps,
                total_tokens=total_tokens, iterations=i + 1,
                success=False, error=f"LLM call failed: {e}",
            )

        # Parse
        step = parse_llm_output(raw_output)
        step.iteration = i + 1

        logger.info("parsed_step",
                    iteration=i + 1,
                    thought=step.thought[:100],
                    action=step.action.tool.value if step.action else None,
                    is_final=step.is_final,
                    )

        # Final answer reached
        if step.is_final:
            steps.append(step)
            return AgentResponse(
                query=query, answer=step.final_answer or "",
                steps=steps, total_tokens=total_tokens,
                iterations=i + 1, success=True,
            )

        # Execute tool
        if step.action:
            tool = TOOL_REGISTRY.get(step.action.tool)
            if tool:
                observation = tool.execute(step.action.input)
            else:
                observation = f"Error: Unknown tool '{step.action.tool}'"
            step.observation = observation

            # Append assistant message + observation to conversation
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No valid action parsed — nudge the model
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": "You must use a tool (Action + Action Input) or provide a Final Answer.",
            })

        steps.append(step)

    # Max iterations exhausted
    return AgentResponse(
        query=query, answer="Max iterations reached without a final answer.",
        steps=steps, total_tokens=total_tokens,
        iterations=REACT_MAX_ITERATIONS, success=False,
        error="Exceeded maximum iterations",
    )