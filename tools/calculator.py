import ast
import math
import operator
from tools.base import BaseTool
from models import ToolName

# Whitelisted binary/unary operators
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted callable names
_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
}

# Whitelisted named constants (e.g. pi, e)
_SAFE_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _safe_eval_node(node: ast.AST):
    """Recursively evaluate an AST node, allowing only safe math operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.left), _safe_eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls allowed (no methods/attributes)")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Disallowed function: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCS[func_name](*args)

    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise ValueError(f"Variable references not allowed: {node.id}")

    raise ValueError(f"Disallowed expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = ToolName.CALCULATOR
    description = "Evaluates a mathematical expression. Input must be a valid Python math expression."
    usage_example = "calculator: 2**10 + 3*4"

    async def execute(self, input_str: str) -> str:
        try:
            tree = ast.parse(input_str.strip(), mode='eval')
            result = _safe_eval_node(tree)
            return str(result)
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
            return f"Error: {e}"