# tools/search.py
from tools.base import BaseTool
from models import ToolName
from ddgs import DDGS


class SearchTool(BaseTool):
    name = ToolName.SEARCH
    description = "Searches the web for real-time information. Input is a search query string."
    usage_example = "search: population of France"

    def execute(self, input_str: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(input_str.strip(), max_results=3))
            if not results:
                return f"No results found for: {input_str}"
            return "\n".join(
                f"- {r['title']}: {r['body']}" for r in results
            )
        except Exception as e:
            return f"Search failed: {e}"