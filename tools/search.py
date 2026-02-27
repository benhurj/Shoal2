# tools/search.py
from tools.base import BaseTool
from models import ToolName


class SearchTool(BaseTool):
    name = ToolName.SEARCH
    description = "Looks up factual information. Input is a search query string."
    usage_example = "search: population of France"

    def execute(self, input_str: str) -> str:
        # MOCK for prototype — replace with SerpAPI, Tavily, or Brave Search later
        knowledge = {
            "population of france": "Approximately 68 million as of 2025.",
            "speed of light": "299,792,458 metres per second.",
            "capital of japan": "Tokyo.",
        }
        key = input_str.strip().lower()
        for k, v in knowledge.items():
            if k in key or key in k:
                return v
        return f"No results found for: {input_str}"