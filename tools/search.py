# tools/search.py
import asyncio
import re
import httpx
from tools.base import BaseTool
from models import ToolName
from ddgs import DDGS
from config import SEARCH_TOP_N, SEARCH_MAX_CHARS

_JINA_ERROR_MARKERS = (
    "SecurityCompromiseError:",
    "RateLimitTriggeredError:",
    "Warning: Target URL returned error",
)


class SearchTool(BaseTool):
    name = ToolName.SEARCH
    description = (
        "Searches the web and returns full page content for the top results. "
        "Input is a search query string. Optionally append [n=N] to fetch N pages (e.g. 'query [n=5]')."
    )
    usage_example = "search: latest Python 3.13 features [n=5]"

    def _parse_input(self, input_str: str) -> tuple[str, int]:
        """Extract query and optional n from input like 'query [n=5]'."""
        match = re.search(r'\[n=(\d+)\]\s*$', input_str)
        if match:
            n = int(match.group(1))
            query = input_str[:match.start()].strip()
        else:
            n = SEARCH_TOP_N
            query = input_str.strip()
        return query, n

    def _sync_search(self, query: str, n: int) -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=n))

    # Short all-caps or title-case navigation words (HOME, ARTICLES, About, etc.)
    _NAV_WORDS = re.compile(
        r'^(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$'
    )

    @staticmethod
    def _clean_jina(text: str) -> str:
        """Strip boilerplate navigation lines from Jina reader output.

        Removes lines that are purely markdown list-links, standalone images,
        checkbox lines, and short nav-word lines (HOME, ARTICLES, etc.).
        Collapses consecutive blank lines.
        """
        nav_line = re.compile(
            r'^\s*'
            r'(?:'
            r'\*\s+\[.*?\]\(.*?\)'       # * [text](url)
            r'|[-•]\s+\[.*?\]\(.*?\)'    # - [text](url)  •  [text](url)
            r'|\[x\]\s'                  # [x] checkbox lines
            r'|\[\s*\]\s'               # [ ] empty checkbox
            r'|!\[.*?\]\(.*?\)'          # ![img](url) standalone
            r')\s*$'
        )
        lines = text.splitlines()
        cleaned = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            if nav_line.match(line):
                continue
            # Drop short all-caps or title-case nav words (HOME, ARTICLES, About Us)
            if stripped and len(stripped) <= 30 and SearchTool._NAV_WORDS.match(stripped):
                continue
            is_blank = not stripped
            if is_blank and prev_blank:
                continue  # collapse multiple blanks
            cleaned.append(line)
            prev_blank = is_blank
        return "\n".join(cleaned).strip()

    async def _fetch_page(self, url: str, snippet: str, client: httpx.AsyncClient) -> str:
        """Fetch page content via Jina Reader, fall back to DDGS snippet."""
        try:
            resp = await client.get(
                f"https://r.jina.ai/{url}",
                timeout=20,
                headers={"Accept": "text/plain"},
                follow_redirects=True,
            )
            text = resp.text.strip()
            if text and not any(text.startswith(m) for m in _JINA_ERROR_MARKERS):
                return self._clean_jina(text)[:SEARCH_MAX_CHARS]
        except Exception:
            pass
        return snippet  # fallback

    async def execute(self, input_str: str) -> str:
        query, n = self._parse_input(input_str)

        try:
            results = await asyncio.to_thread(self._sync_search, query, n)
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"No results found for: {query}"

        async with httpx.AsyncClient() as client:
            pages = await asyncio.gather(*[
                self._fetch_page(r["href"], r["body"], client)
                for r in results
            ])

        output = []
        for r, content in zip(results, pages):
            output.append(f"## {r['title']}\n{r['href']}\n\n{content}")

        return "\n\n---\n\n".join(output)
