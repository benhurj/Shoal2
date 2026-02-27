from tools.base import BaseTool
from models import ToolName


class CalculatorTool(BaseTool):
    name = ToolName.CALCULATOR
    description = "Evaluates a mathematical expression. Input must be a valid Python math expression."
    usage_example = "calculator: 2**10 + 3*4"

    def execute(self, input_str: str) -> str:
        try:
            # Restricted eval — only math operations
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
            result = eval(input_str, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"