#!/usr/bin/env python3
"""Ban optimization-sensitive assert and delete statements in Reef source."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "reef"


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    column: int
    statement: str

    def diagnostic(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: forbidden {self.statement} statement"


def inspect_source(source: str, path: str) -> list[Finding]:
    tree = ast.parse(source, filename=path)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            findings.append(Finding(path, node.lineno, node.col_offset + 1, "assert"))
        elif isinstance(node, ast.Delete):
            findings.append(Finding(path, node.lineno, node.col_offset + 1, "del"))
    return sorted(findings)


def inspect_file(path: Path) -> list[Finding]:
    relative_path = path.relative_to(ROOT).as_posix()
    try:
        return inspect_source(path.read_text(encoding="utf-8"), relative_path)
    except (SyntaxError, UnicodeDecodeError) as error:
        print(f"{relative_path}: cannot inspect file: {error}")
        return [Finding(relative_path, 1, 1, "unparseable")]


def main() -> int:
    findings = [finding for path in sorted(SOURCE_ROOT.rglob("*.py")) for finding in inspect_file(path)]
    if findings:
        for finding in findings:
            print(finding.diagnostic())
        print("Python statement policy failed: use explicit checks and mutation APIs instead of assert or del.")
        return 1

    print("Python statement policy passed (no assert or del statements in reef/).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
