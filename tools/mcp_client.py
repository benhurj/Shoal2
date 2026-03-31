# tools/mcp_client.py — Async client for the Shoal MCP tool server (stdio transport)
import asyncio
import contextlib
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Long-lived async client for the Shoal MCP server subprocess.

    Usage:
        client = MCPClient()
        await client.start()
        result = await client.call_tool("search", "Python 3.13 features")
        await client.stop()
    """

    def __init__(self, server_script: str | None = None):
        if server_script is None:
            server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        self._server_script = server_script
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tool_cache: list[dict] | None = None
        self._cache_lock = asyncio.Lock()
        # Serialise all session calls: anyio cancel scopes inside the MCP
        # transport must not be entered/exited from different asyncio tasks.
        self._session_lock = asyncio.Lock()

    async def start(self):
        """Start the MCP server subprocess and initialise the session."""
        params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
        )
        self._exit_stack = contextlib.AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def stop(self):
        """Shut down the session and kill the subprocess."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None
            self._session = None

    async def list_tools(self) -> list[dict]:
        """Return cached list of tool dicts with 'name' and 'description' keys."""
        if self._tool_cache is not None:
            return self._tool_cache
        async with self._cache_lock:
            if self._tool_cache is not None:
                return self._tool_cache
            async with self._session_lock:
                result = await self._session.list_tools()
            self._tool_cache = [
                {"name": t.name, "description": t.description or ""}
                for t in result.tools
            ]
        return self._tool_cache

    async def call_tool(self, name: str, input_str: str) -> str:
        """Call a tool by name with a single string input, return string output."""
        async with self._session_lock:
            result = await self._session.call_tool(name, {"input": input_str})
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts) if texts else "(no output)"

    async def tool_descriptions(self) -> str:
        """Return formatted tool descriptions suitable for inclusion in prompts.

        Each tool is rendered as:
          - name: first-paragraph summary of the full docstring
        """
        tools = await self.list_tools()
        parts = []
        for t in tools:
            # First paragraph only (up to first blank line)
            first_para = t["description"].strip().split("\n\n")[0]
            short = " ".join(first_para.splitlines()).strip()
            parts.append(f"  - {t['name']}: {short}")
        return "\n".join(parts)

    def is_valid_tool(self, name: str) -> bool:
        """Check if a tool name is known (uses cache; returns False before list_tools)."""
        if self._tool_cache is None:
            return False
        return any(t["name"] == name for t in self._tool_cache)
