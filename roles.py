# roles.py
import re
import random
import structlog
from config import (
    TEMPERATURE_MEAN, TEMPERATURE_STD, TOP_P_MEAN, TOP_P_STD,
)

logger = structlog.get_logger()

# Style → (temperature, top_p)
_STYLE_PARAMS = {
    "precise":     (0.10, 0.80),
    "balanced":    (0.35, 0.88),
    "exploratory": (0.70, 0.95),
}

# Canonical defined agent roles (used by select_roles and as LLM fallback)
AGENT_ROLES = [
    {
        "name": "methodical",
        "temperature": 0.10, "top_p": 0.80,
        "planner_addendum": "Break the task into small, precise steps.",
        "executor_addendum": "Work carefully. Verify each step before moving on.",
    },
    {
        "name": "creative",
        "temperature": 0.70, "top_p": 0.95,
        "planner_addendum": "Consider unconventional approaches.",
        "executor_addendum": "If the obvious approach fails, try a different angle.",
    },
    {
        "name": "skeptical",
        "temperature": 0.15, "top_p": 0.85,
        "planner_addendum": "Plan verification steps. Be suspicious of single-source answers.",
        "executor_addendum": "Cross-check information. Search for contradicting evidence.",
    },
    {
        "name": "research_focused",
        "temperature": 0.20, "top_p": 0.90,
        "planner_addendum": "Prioritize information gathering from multiple angles.",
        "executor_addendum": "Use search tools extensively. Gather multiple data points.",
    },
    {
        "name": "concise",
        "temperature": 0.10, "top_p": 0.80,
        "planner_addendum": "Use the minimum number of steps needed.",
        "executor_addendum": "Be efficient. Use the fewest tool calls necessary.",
    },
]

# Backward-compatible alias (label-keyed dicts for agent_hybrid)
_FALLBACK_ROLES = [
    {
        "label": r["name"],
        "temperature": r["temperature"],
        "top_p": r["top_p"],
        "planner_addendum": r["planner_addendum"],
        "executor_addendum": r["executor_addendum"],
    }
    for r in AGENT_ROLES
]


def select_roles(k: int) -> list[dict]:
    """Return k role configs: defined roles first, stochastic overflow for extras."""
    configs = []
    for role in AGENT_ROLES[:k]:
        configs.append({
            "label": role["name"],
            "temperature": role["temperature"],
            "top_p": role["top_p"],
            "planner_addendum": role["planner_addendum"],
            "executor_addendum": role["executor_addendum"],
        })
    for i in range(len(AGENT_ROLES), k):
        temp = max(0.01, random.gauss(TEMPERATURE_MEAN, TEMPERATURE_STD))
        top_p = min(1.0, max(0.1, random.gauss(TOP_P_MEAN, TOP_P_STD)))
        configs.append({
            "label": f"stochastic_{i}",
            "temperature": temp,
            "top_p": top_p,
            "planner_addendum": "",
            "executor_addendum": "",
        })
    return configs


def _parse_roles(text: str, k: int) -> list[dict]:
    """Parse LLM role blocks into config dicts. Handles explicit TEMPERATURE/TOP_P or STYLE."""
    blocks = re.split(r'\n(?=ROLE:)', text.strip())
    configs = []
    for block in blocks:
        role_m     = re.search(r'^ROLE:\s*(\S+)', block, re.MULTILINE)
        style_m    = re.search(r'^STYLE:\s*(\S+)', block, re.MULTILINE)
        temp_m     = re.search(r'^TEMPERATURE:\s*([\d.]+)', block, re.MULTILINE)
        top_p_m    = re.search(r'^TOP_P:\s*([\d.]+)', block, re.MULTILINE)
        planner_m  = re.search(r'^PLANNER:\s*(.+)', block, re.MULTILINE)
        executor_m = re.search(r'^EXECUTOR:\s*(.+)', block, re.MULTILINE)

        if not (role_m and planner_m and executor_m):
            continue

        # Prefer explicit TEMPERATURE/TOP_P; fall back to STYLE-based mapping
        if temp_m and top_p_m:
            try:
                temp = float(temp_m.group(1))
                top_p = float(top_p_m.group(1))
            except ValueError:
                style = style_m.group(1).strip().lower() if style_m else "balanced"
                temp, top_p = _STYLE_PARAMS.get(style, _STYLE_PARAMS["balanced"])
        else:
            style = style_m.group(1).strip().lower() if style_m else "balanced"
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
