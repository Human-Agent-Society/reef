#!/usr/bin/env python3
"""Keep the root English and Chinese READMEs structurally synchronized."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "README.md"
TRANSLATION = ROOT / "README.zh.md"
RECORD = ROOT / "README.i18n.yaml"
PAIR_PATHS = (SOURCE, TRANSLATION)

FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+")
LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)\s]+)(?:\s+['\"].*?['\"])?\)")
HTML_TARGET_RE = re.compile(r"\b(?:href|src|srcset)=[\"']([^\"']+)[\"']")
HASH_RE = re.compile(r"^(README(?:\.zh)?\.md):\s*([0-9a-f]+)\s*$")


class ReadmeContractError(RuntimeError):
    """A README pair violates the synchronization contract."""


def git_blob_hash(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_record() -> dict[str, str]:
    if not RECORD.is_file():
        raise ReadmeContractError("README.i18n.yaml is missing")
    values: dict[str, str] = {}
    for line in RECORD.read_text(encoding="utf-8").splitlines():
        match = HASH_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    expected = {path.name for path in PAIR_PATHS}
    if set(values) != expected:
        raise ReadmeContractError("README.i18n.yaml must record exactly README.md and README.zh.md")
    return values


def fenced_blocks(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    outside: list[str] = []
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = FENCE_RE.match(lines[index])
        if not match:
            outside.append(lines[index])
            index += 1
            continue
        marker, info = match.groups()
        body: list[str] = []
        index += 1
        while index < len(lines) and not re.match(rf"^{re.escape(marker[0])}{{{len(marker)},}}\s*$", lines[index]):
            body.append(lines[index])
            index += 1
        if index == len(lines):
            raise ReadmeContractError(f"unclosed code fence: {marker}{info}")
        blocks.append((info.strip(), "\n".join(body)))
        index += 1
    return outside, blocks


def table_shapes(lines: list[str]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        table: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            table.append(lines[index].strip())
            index += 1
        if len(table) >= 2 and re.fullmatch(r"\|[|:\- ]+\|", table[1]):
            columns = len(table[0].strip("|").split("|"))
            shapes.append((len(table), columns))
    return shapes


def normalized_target(target: str) -> str:
    if target in {"README.md", "README.zh.md"}:
        return "<readme-counterpart>"
    return target


def structure(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    outside, blocks = fenced_blocks(text)
    outside_text = "\n".join(outside)
    links = [normalized_target(target) for target in MARKDOWN_LINK_RE.findall(outside_text)]
    html_targets = [normalized_target(target) for target in HTML_TARGET_RE.findall(outside_text)]
    return {
        "heading levels": [len(match.group(1)) for line in outside for match in [HEADING_RE.match(line)] if match],
        "list markers": [match.group(0).strip() for line in outside for match in [LIST_RE.match(line)] if match],
        "table shapes": table_shapes(outside),
        "link targets": links,
        "HTML targets": html_targets,
        "code fences": blocks,
    }


def verify_structure() -> None:
    source = structure(SOURCE)
    translation = structure(TRANSLATION)
    differences = [name for name in source if source[name] != translation[name]]
    if differences:
        details = ", ".join(differences)
        raise ReadmeContractError(
            f"README.md and README.zh.md differ in: {details}. "
            "Keep headings, lists, tables, targets, and fenced code synchronized."
        )


def render_record() -> str:
    hashes = {path.name: git_blob_hash(path) for path in PAIR_PATHS}
    return (
        "# Last human-confirmed consistent README pair.\n"
        "# After updating and reviewing both files, run:\n"
        "#   python .github/scripts/check_readme_i18n.py --write\n"
        f"README.md: {hashes['README.md']}\n"
        f"README.zh.md: {hashes['README.zh.md']}\n"
    )


def verify_hashes() -> None:
    recorded = parse_record()
    stale = [path.name for path in PAIR_PATHS if recorded[path.name] != git_blob_hash(path)]
    if stale:
        names = ", ".join(stale)
        raise ReadmeContractError(
            f"README translation record is stale for: {names}. Update and review "
            "both READMEs, then run this command with --write."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="record the current pair after structural checks and human review",
    )
    args = parser.parse_args()
    try:
        verify_structure()
        if args.write:
            RECORD.write_text(render_record(), encoding="utf-8")
            print("Recorded the reviewed README pair in README.i18n.yaml.")
        verify_hashes()
    except (ReadmeContractError, subprocess.CalledProcessError) as error:
        print(f"README i18n check failed: {error}", file=sys.stderr)
        return 1
    print("README.md and README.zh.md are synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
