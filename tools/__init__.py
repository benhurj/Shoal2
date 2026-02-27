# tools/__init__.py
from tools.calculator import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.search import SearchTool
from models import ToolName

TOOL_REGISTRY: dict = {
    ToolName.CALCULATOR: CalculatorTool(),
    ToolName.SEARCH: SearchTool(),
    ToolName.DATETIME: DateTimeTool(),
}