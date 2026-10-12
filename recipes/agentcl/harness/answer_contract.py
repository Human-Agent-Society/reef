"""Restricted final Python modules for AgentCL; Harbor remains the security boundary.

This syntax contract rejects known process-control, introspection and grader
mutation patterns. It is not a proof against every attack through Python or its
scientific dependencies. Development tool actions are sandboxed separately.
"""

from __future__ import annotations

import ast
import builtins

CONTRACT_VERSION = "agentcl-final-module-v2"
ALLOWED_MODULES = frozenset(
    {
        "array",
        "base64",
        "bisect",
        "collections",
        "copy",
        "csv",
        "datetime",
        "dateutil",
        "decimal",
        "enum",
        "faker",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "holidays",
        "io",
        "itertools",
        "json",
        "math",
        "matplotlib",
        "numpy",
        "operator",
        "pandas",
        "random",
        "re",
        "regex",
        "scipy",
        "seaborn",
        "sklearn",
        "statistics",
        "statsmodels",
        "string",
        "struct",
        "time",
        "typing",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "BaseException",
        "SystemExit",
        "KeyboardInterrupt",
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
    }
)


class AnswerContractError(ValueError):
    """A final module uses syntax outside the declared benchmark answer contract."""


def validate_final_module(source: str) -> None:
    """Require a full module without dynamic imports, private access or module mutation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AnswerContractError("The final Python module has a syntax error.") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES or any(
                    part.startswith("_") for part in alias.name.split(".")
                ):
                    raise AnswerContractError("The final module imports an unsupported module.")
                if alias.asname and (alias.asname.startswith("__") or alias.asname in vars(builtins)):
                    raise AnswerContractError("Private interpreter and builtin import aliases are not permitted.")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (
                node.level
                or module.split(".")[0] not in ALLOWED_MODULES
                or any(part.startswith("_") for part in module.split("."))
            ):
                raise AnswerContractError("The final module imports an unsupported module.")
            if any(alias.name == "*" or alias.name.startswith("_") for alias in node.names):
                raise AnswerContractError("Wildcard and private imports are not permitted.")
            if any(
                alias.asname and (alias.asname.startswith("__") or alias.asname in vars(builtins))
                for alias in node.names
            ):
                raise AnswerContractError("Private interpreter and builtin import aliases are not permitted.")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise AnswerContractError("Process control and interpreter introspection are not permitted.")
            if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in vars(builtins):
                raise AnswerContractError("Builtin names cannot be replaced or deleted.")
        elif isinstance(node, (ast.MatchAs, ast.MatchStar, ast.MatchMapping)):
            capture_name = node.rest if isinstance(node, ast.MatchMapping) else node.name
            if capture_name is not None:
                if capture_name in FORBIDDEN_NAMES or capture_name.startswith("__"):
                    raise AnswerContractError("Process control and interpreter introspection are not permitted.")
                if capture_name in vars(builtins):
                    raise AnswerContractError("Builtin names cannot be replaced or deleted.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or isinstance(node.ctx, (ast.Store, ast.Del)):
                raise AnswerContractError("Private attributes and attribute mutation are not permitted.")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("__") or node.name in vars(builtins):
                raise AnswerContractError("Private interpreter and builtin definitions are not permitted.")
        elif isinstance(node, ast.Global):
            raise AnswerContractError("Global namespace mutation is not permitted.")
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body):
        raise AnswerContractError("Submit a complete module with a top-level function definition.")
