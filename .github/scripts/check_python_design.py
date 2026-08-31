#!/usr/bin/env python3
"""Reject Python design shortcuts that hide dependencies and interfaces."""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / ".github" / "python-design-baseline.txt"
DESIGN_ROOTS = (ROOT / "reef", ROOT / "recipes", ROOT / "tests")
TYPE_CHECKING_ROOTS = DESIGN_ROOTS


def _is_example(path: Path) -> bool:
    """The runnable examples (``recipes/<method>/examples/``, ``recipes/basic``) are type-checked only."""
    parts = path.relative_to(ROOT).parts
    return parts[0] == "recipes" and (parts[1] == "basic" or parts[2:3] == ("examples",))


@dataclass(frozen=True, order=True)
class Finding:
    code: str
    path: str
    owner: str
    subject: str
    line: int
    message: str

    @property
    def fingerprint(self) -> str:
        return "|".join((self.code, self.path, self.owner, self.subject))

    def diagnostic(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _contains_callable(annotation: ast.AST | None, aliases: set[str]) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval")
        except SyntaxError:
            return False
    return any(
        (isinstance(node, ast.Name) and node.id in aliases)
        or (isinstance(node, ast.Attribute) and node.attr == "Callable")
        for node in ast.walk(annotation)
    )


def _callable_aliases(tree: ast.Module) -> set[str]:
    aliases = {"Callable"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "collections.abc"}:
            for imported in node.names:
                if imported.name == "Callable":
                    aliases.add(imported.asname or imported.name)

    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None or not _contains_callable(value, aliases):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return aliases


def _arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    return arguments


class DesignVisitor(ast.NodeVisitor):
    def __init__(self, path: str, aliases: set[str]) -> None:
        self.path = path
        self.aliases = aliases
        self.findings: list[Finding] = []
        self.scopes: list[tuple[str, str]] = [("<module>", "module")]

    def _qualname(self) -> str:
        return ".".join(name for name, kind in self.scopes if kind != "module") or "<module>"

    def _class_owner(self) -> str:
        names: list[str] = []
        for name, kind in self.scopes:
            if kind == "class":
                names.append(name)
        return ".".join(names) or "<module>"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        callable_arguments = [
            argument.arg for argument in _arguments(node) if _contains_callable(argument.annotation, self.aliases)
        ]
        owner = ".".join([*(name for name, kind in self.scopes if kind != "module"), node.name])
        parent_is_class = self.scopes[-1][1] == "class"
        if node.name == "__init__" and parent_is_class and callable_arguments:
            subject = ",".join(callable_arguments)
            self.findings.append(
                Finding(
                    "PYD002",
                    self.path,
                    owner,
                    subject,
                    node.lineno,
                    f"constructor stores behavior as Callable ({subject}); use a named Protocol or object interface",
                )
            )
        elif len(callable_arguments) > 1:
            subject = ",".join(callable_arguments)
            self.findings.append(
                Finding(
                    "PYD004",
                    self.path,
                    owner,
                    subject,
                    node.lineno,
                    f"signature bundles multiple Callable parameters ({subject}); introduce a cohesive interface",
                )
            )

        self.scopes.append((node.name, "function"))
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.append((node.name, "class"))
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not _contains_callable(node.annotation, self.aliases):
            self.generic_visit(node)
            return

        direct_state = self.scopes[-1][1] in {"module", "class"}
        instance_state = (
            isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id in {"self", "cls"}
        )
        if direct_state or instance_state:
            subject = ast.unparse(node.target)
            owner = self._class_owner()
            self.findings.append(
                Finding(
                    "PYD003",
                    self.path,
                    owner,
                    subject,
                    node.lineno,
                    f"long-lived state {subject} is typed as Callable; use a named Protocol or object interface",
                )
            )
        self.generic_visit(node)


def _type_checking_finding(tree: ast.Module, path: str) -> Finding | None:
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if (
            (isinstance(node, ast.ImportFrom) and any(imported.name == "TYPE_CHECKING" for imported in node.names))
            or (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING")
            or (isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING")
        )
    ]
    if not lines:
        return None
    return Finding(
        "PYD001",
        path,
        "<module>",
        "TYPE_CHECKING",
        min(lines),
        "TYPE_CHECKING is banned; fix the module boundary or use a runtime/lazy import",
    )


def inspect_source(source: str, path: str, *, check_callables: bool = True) -> list[Finding]:
    tree = ast.parse(source, filename=path)
    findings: list[Finding] = []
    if check_callables:
        aliases = _callable_aliases(tree)
        visitor = DesignVisitor(path, aliases)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    type_checking = _type_checking_finding(tree, path)
    if type_checking is not None:
        findings.append(type_checking)
    return findings


def inspect_file(path: Path, *, check_callables: bool = True) -> list[Finding]:
    relative_path = path.relative_to(ROOT).as_posix()
    try:
        return inspect_source(path.read_text(encoding="utf-8"), relative_path, check_callables=check_callables)
    except (SyntaxError, UnicodeDecodeError) as error:
        return [Finding("PYD000", relative_path, "<module>", "parse", 1, f"cannot inspect file: {error}")]


def _python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        yield from sorted(root.rglob("*.py"))


def _read_baseline() -> set[str]:
    return {
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> int:
    callable_findings = [
        finding
        for path in _python_files(DESIGN_ROOTS)
        if not _is_example(path)
        for finding in inspect_file(path)
        if finding.code != "PYD001"
    ]
    type_checking_findings = [
        finding for path in _python_files(TYPE_CHECKING_ROOTS) for finding in inspect_file(path, check_callables=False)
    ]
    findings = sorted([*callable_findings, *type_checking_findings])
    actual = {finding.fingerprint for finding in findings if finding.code != "PYD001"}
    expected = _read_baseline()
    unexpected = [finding for finding in findings if finding.code == "PYD001" or finding.fingerprint not in expected]
    stale = sorted(expected - actual)

    if unexpected:
        for finding in unexpected:
            print(finding.diagnostic())
    if stale:
        print("Stale Python design baseline entries (remove them):")
        for fingerprint in stale:
            print(f"  {fingerprint}")
    if unexpected or stale:
        print(
            "Python design policy failed. Prefer named Protocols and cohesive objects over stored Callable behavior."
        )
        return 1

    print(f"Python design policy passed ({len(expected)} documented legacy Callable patterns).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
