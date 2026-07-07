"""Stage 4: a small custom function tool example."""

from __future__ import annotations

import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculate_expression(expression: str) -> float:
    """Evaluate a small arithmetic expression for learning purposes."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Unsupported expression") from exc

    return float(_evaluate_node(tree.body))


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    raise ValueError("Unsupported expression")


def main() -> None:
    from agentscope.tool import FunctionTool

    calculator = FunctionTool(calculate_expression)
    print(calculator)


if __name__ == "__main__":
    main()
