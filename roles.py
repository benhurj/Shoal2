# roles.py
import re
import structlog
from config import COMPILER_MODEL, COMPILER_PORTS, COMPILER_TEMPERATURE, COMPILER_TOP_P
from prompts import build_role_designer_prompt

logger = structlog.get_logger()

# Style → (temperature, top_p)
_STYLE_PARAMS = {
    "precise":     (0.10, 0.80),
    "balanced":    (0.35, 0.88),
    "exploratory": (0.70, 0.95),
}

# Fallback roles used if LLM generation fails
_FALLBACK_ROLES = [
    {"label": "methodical",      "temperature": 0.10, "top_p": 0.80,
     "planner_addendum": "Break the task into small, precise steps.",
     "executor_addendum": "Work carefully. Verify each step before moving on."},
    {"label": "creative",        "temperature": 0.70, "top_p": 0.95,
     "planner_addendum": "Consider unconventional approaches.",
     "executor_addendum": "If the obvious approach fails, try a different angle."},
    {"label": "skeptical",       "temperature": 0.15, "top_p": 0.85,
     "planner_addendum": "Plan verification steps. Be suspicious of single-source answers.",
     "executor_addendum": "Cross-check information. Search for contradicting evidence."},
    {"label": "research_focused","temperature": 0.20, "top_p": 0.90,
     "planner_addendum": "Prioritize information gathering from multiple angles.",
     "executor_addendum": "Use search tools extensively. Gather multiple data points."},
    {"label": "concise",         "temperature": 0.10, "top_p": 0.80,
     "planner_addendum": "Use the minimum number of steps needed.",
     "executor_addendum": "Be efficient. Use the fewest tool calls necessary."},
]


def _parse_roles(text: str, k: int) -> list[dict]:
    """Parse LLM role blocks into config dicts."""
    blocks = re.split(r'\n(?=ROLE:)', text.strip())
    configs = []
    for block in blocks:
        role_m    = re.search(r'^ROLE:\s*(\S+)', block, re.MULTILINE)
        style_m   = re.search(r'^STYLE:\s*(\S+)', block, re.MULTILINE)
        planner_m = re.search(r'^PLANNER:\s*(.+)', block, re.MULTILINE)
        executor_m= re.search(r'^EXECUTOR:\s*(.+)', block, re.MULTILINE)

        if not (role_m and style_m and planner_m and executor_m):
            continue

        style = style_m.group(1).strip().lower()
        temp, top_p = _STYLE_PARAMS.get(style, _STYLE_PARAMS["balanced"])

        configs.append({
            "label":             role_m.group(1).strip().lower(),
            "temperature":       temp,
            "top_p":             top_p,
            "planner_addendum":  planner_m.group(1).strip(),
            "executor_addendum": executor_m.group(1).strip(),
        })

        if len(configs) == k:
            break

    return configs


async def generate_roles(query: str, k: int) -> tuple[list[dict], int]:
    """Ask the LLM to generate K query-tailored agent personas. Falls back to static roles."""
    from llm_client import llm_client  # local import to avoid circular

    logger.info("role_designer_started", k=k, query=query[:60])
    messages = [
        {"role": "system", "content": build_role_designer_prompt(query, k)},
        {"role": "user",   "content": "Generate the agent roles now."},
    ]

    try:
        raw, tokens = await llm_client.call_llm(
            messages, COMPILER_MODEL,
            port=COMPILER_PORTS[0],
            temperature=COMPILER_TEMPERATURE,
            top_p=COMPILER_TOP_P,
        )
        logger.info("role_designer_raw", output=raw[:300])
        configs = _parse_roles(raw, k)
    except Exception as e:
        logger.warning("role_designer_failed", error=str(e))
        configs = []
        tokens = 0

    if len(configs) < k:
        logger.warning("role_designer_fallback", parsed=len(configs), needed=k)
        fallback = _FALLBACK_ROLES * ((k // len(_FALLBACK_ROLES)) + 1)
        configs = configs + fallback[len(configs):k]

    logger.info("role_designer_completed", roles=[c["label"] for c in configs])
    return configs, tokens
