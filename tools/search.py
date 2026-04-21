# tools/search.py
import asyncio
from tools.base import BaseTool
from models import ToolName
from ddgs import DDGS


class SearchTool(BaseTool):
    name = ToolName.SEARCH
    description = "Searches the web for real-time information. Input is a search query string."
    usage_example = "search: population of France"

    def _sync_search(self, query: str) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No results found for: {query}"
        return "\n".join(
            f"- {r['title']}: {r['body']}" for r in results
        )

    async def execute(self, input_str: str) -> str:
        try:
            return await asyncio.to_thread(self._sync_search, input_str.strip())
        except Exception as e:
            return f"Search failed: {e}"