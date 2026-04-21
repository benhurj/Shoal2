#!/usr/bin/env python
# tools/mcp_server.py — Shoal MCP tool server (stdio transport)
# Invoked as a subprocess by MCPClient. Run directly to test:
#   python tools/mcp_server.py

import os
import sys

# Make project root importable when run as subprocess
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcp.server.fastmcp import FastMCP
from tools.search import SearchTool
from tools.calculator import CalculatorTool
from tools.datetime_tool import DateTimeTool

mcp = FastMCP("shoal-tools")

_search = SearchTool()
_calc = CalculatorTool()
_dt = DateTimeTool()


@mcp.tool()
async def search(input: str) -> str:
    """Search the web and return full page content for the top results.

    Input: a search query string. Optionally append [n=N] to fetch N pages.

    Examples of VALID inputs:
      - "latest Python 3.13 features"
      - "Tesla Model Y range 2024 [n=5]"
      - "SpaceX Starship launch date"
    """
    return await _search.execute(input)


@mcp.tool()
async def calculator(input: str) -> str:
    """Evaluate a Python math expression (digits and operators only, e.g. '17 * 23'). Input must be a valid Python expression — NOT natural language.

    Allowed operators: + - * / // % **
    Allowed functions: abs(), round(), min(), max(), pow()

    VALID: "17 * 23", "1024 * 0.9", "(100 - 15) / 4", "abs(-42) + pow(2, 10)"
    INVALID: "calculate the total", "x + y", "aggregate data"
    """
    return await _calc.execute(input)


@mcp.tool()
async def datetime(input: str) -> str:
    """Return the current date and time in UTC.

    Input: anything (ignored). Use "now" by convention.

    Example: "now"
    """
    return await _dt.execute(input)


if __name__ == "__main__":
    mcp.run(transport="stdio")
