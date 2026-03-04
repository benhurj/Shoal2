# roles.py
import random
from config import TEMPERATURE_MEAN, TEMPERATURE_STD, TOP_P_MEAN, TOP_P_STD

AGENT_ROLES = [
    {
        "name": "methodical",
        "planner_addendum": "Break the task into small, precise steps. Prefer using tools for every factual claim. Be thorough.",
        "executor_addendum": "Work carefully and methodically. Verify each step before moving to the next. Prefer precision over speed.",
        "temperature": 0.10,
        "top_p": 0.80,
    },
    {
        "name": "creative",
        "planner_addendum": "Consider unconventional approaches. Look at the problem from multiple angles. Suggest alternative framings.",
        "executor_addendum": "Think creatively. If the obvious approach fails, try a different angle. Consider what information might be relevant that isn't immediately obvious.",
        "temperature": 0.70,
        "top_p": 0.95,
    },
    {
        "name": "skeptical",
        "planner_addendum": "Plan verification steps. Include cross-checking and validation in your plan. Be suspicious of single-source answers.",
        "executor_addendum": "Be critical of results. Cross-check information from tools. If a result seems wrong, search for contradicting evidence before accepting it.",
        "temperature": 0.15,
        "top_p": 0.85,
    },
    {
        "name": "research_focused",
        "planner_addendum": "Prioritize information gathering. Plan multiple search queries from different angles. Gather comprehensive evidence before concluding.",
        "executor_addendum": "Focus on thorough research. Use search tools extensively. Gather multiple data points before forming conclusions.",
        "temperature": 0.20,
        "top_p": 0.90,
    },
    {
        "name": "concise",
        "planner_addendum": "Use the minimum number of steps needed. Combine steps where possible. Be direct.",
        "executor_addendum": "Be efficient and direct. Use the fewest tool calls necessary. Give concise, focused answers.",
        "temperature": 0.10,
        "top_p": 0.80,
    },
]


def select_roles(k: int) -> list[dict]:
    """Select K agent role configs. Uses defined roles first, stochastic overflow for k > len(AGENT_ROLES)."""
    configs = []
    for i in range(min(k, len(AGENT_ROLES))):
        role = AGENT_ROLES[i]
        configs.append({
            "label": role["name"],
            "temperature": role["temperature"],
            "top_p": role["top_p"],
            "planner_addendum": role["planner_addendum"],
            "executor_addendum": role["executor_addendum"],
        })

    # Stochastic overflow agents beyond the defined roles
    for i in range(len(AGENT_ROLES), k):
        temp = max(0.01, min(2.0, random.gauss(TEMPERATURE_MEAN, TEMPERATURE_STD)))
        top_p = max(0.1, min(1.0, random.gauss(TOP_P_MEAN, TOP_P_STD)))
        configs.append({
            "label": f"stochastic_{i+1}",
            "temperature": round(temp, 3),
            "top_p": round(top_p, 3),
            "planner_addendum": "",
            "executor_addendum": "",
        })

    return configs
