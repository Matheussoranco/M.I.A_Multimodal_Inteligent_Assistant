"""Utilities for safely evaluating basic arithmetic expressions.

This intentionally supports a small subset of Python expressions:
- integers / floats
- unary + and -
- binary operators: +, -, *, /, //, %, **
- parentheses

No names, attributes, function calls, indexing, comprehensions, etc.
"""

from __future__ import annotations

import ast
from typing import Union


Number = Union[int, float]


_ALLOWED_BINOPS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd, ast.USub}


def safe_eval_arithmetic(expression: str) -> Number:
    """Safely evaluate a basic arithmetic expression.

    Raises:
        ValueError: if expression contains unsupported syntax.
    """

    expr = (expression or "").strip()
    if not expr:
        raise ValueError("empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression") from exc

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Python 3.8+: numbers are ast.Constant
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError("only numeric constants are allowed")

        # Back-compat for very old AST nodes
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # pragma: no cover
            return node.n  # type: ignore[attr-defined]

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARYOPS:
                raise ValueError("unsupported unary operator")
            value = _eval(node.operand)
            return +value if isinstance(node.op, ast.UAdd) else -value

        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINOPS:
                raise ValueError("unsupported operator")
            left = _eval(node.left)
            right = _eval(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right

            raise ValueError("unsupported operator")

        # Disallow everything else: Name, Call, Attribute, Subscript, etc.
        raise ValueError("unsupported expression")

    return _eval(tree)
