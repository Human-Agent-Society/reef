"""The FINAL judge, this task's teaching point: held-out grading.

The agent only ever sees the training points; this file runs ONCE, on the
best submission, and scores the
expression on HELD-OUT points the agent has never seen. Memorizing the
training set (a giant interpolating polynomial) collapses on the held-out
range; only the true structure generalizes.

Expressions are evaluated through an AST allowlist, never eval(), so the
artifact cannot execute anything.
"""

import ast
import json
import math
import operator
from pathlib import Path

HELDOUT = json.loads((Path(__file__).parent / "heldout.json").read_text())["points"]
MAX_EXPR_LEN = 2000

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}


def _eval_node(node: ast.AST, x: float) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, x)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "x":
        return x
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _eval_node(node.left, x), _eval_node(node.right, x)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand, x))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _FUNCS[node.func.id](_eval_node(node.args[0], x))
    raise ValueError(f"disallowed syntax: {ast.dump(node)[:80]}")


def evaluate(expr: str, x: float) -> float:
    return _eval_node(ast.parse(expr, mode="eval"), x)


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution.json artifact"}
    try:
        expr = json.loads(Path(artifact).read_text())["expr"]
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}
    if not isinstance(expr, str) or len(expr) > MAX_EXPR_LEN:
        return {
            "reward": 0.0,
            "reason": f"expr must be a string <= {MAX_EXPR_LEN} chars",
        }

    errors = []
    for x, y in HELDOUT:
        try:
            prediction = evaluate(expr, x)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            return {"reward": 0.0, "reason": f"expr failed at x={x}: {e}"}
        if not math.isfinite(prediction):
            return {"reward": 0.0, "reason": f"non-finite prediction at x={x}"}
        errors.append((prediction - y) ** 2)
    rmse = math.sqrt(sum(errors) / len(errors))
    return {"reward": 1.0 / (1.0 + rmse), "rmse_heldout": rmse}
