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
    """Evaluate a Python arithmetic expression and return the numeric result.

    Input MUST be a valid Python arithmetic expression using numbers and operators.
    Allowed operators: + - * / // % **
    Allowed functions: abs(), round(), min(), max(), pow()

    Examples of VALID inputs:
      - "2 + 3"
      - "1024 * 0.9"
      - "round(3.14159, 2)"
      - "(100 - 15) / 4"
      - "abs(-42) + pow(2, 10)"

    Examples of INVALID inputs — DO NOT use these:
      - "aggregate data into a table"    ← text descriptions are INVALID
      - "calculate the average score"    ← natural language is INVALID
      - "compare range and cargo volume" ← comparisons in words are INVALID
      - "x + y"                          ← variable names are INVALID

    Only call this tool when you have actual numeric values to compute.
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
