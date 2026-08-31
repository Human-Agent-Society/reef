#!/usr/bin/env python3
"""Validate Reef's community-maintenance configuration."""

from __future__ import annotations

import csv
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GITHUB_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
ACTION_SHA = re.compile(r"^[0-9a-f]{40}$")
LABEL_COLOR = re.compile(r"^[0-9a-fA-F]{6}$")
USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
ISSUE_LABELS_LINE = re.compile(r"^\s*labels:\s*(.*?)\s*$")
ISSUE_TITLE_LINE = re.compile(r"""^\s*title:\s*["']?(\[[A-Za-z]+\]\s)""")
ISSUE_TITLE_PREFIXES = {
    "[Bug] ",
    "[Feature] ",
    "[Performance] ",
    "[Task] ",
    "[Experiment] ",
    "[Example] ",
    "[RFC] ",
    "[Roadmap] ",
    "[Question] ",
}


def fail(message: str) -> None:
    print(f"community-config: {message}", file=sys.stderr)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def pattern_matches(pattern: str, paths: list[str]) -> bool:
    if pattern == "*":
        return bool(paths)
    normalized = pattern.removeprefix("/")
    if normalized.endswith("/"):
        # A Git submodule is tracked as the directory entry itself, while the
        # trailing-slash CODEOWNERS rule intentionally covers its contents.
        return any(path == normalized.removesuffix("/") or path.startswith(normalized) for path in paths)
    return any(fnmatch.fnmatchcase(path, normalized) for path in paths)


def validate_codeowners(paths: list[str]) -> set[str]:
    codeowners = ROOT / ".github" / "CODEOWNERS"
    owners: set[str] = set()
    errors = 0
    for number, raw_line in enumerate(codeowners.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            fail(f"CODEOWNERS:{number}: expected a pattern and at least one owner")
            errors += 1
            continue
        pattern, *line_owners = fields
        if pattern.startswith("!") or any(char in pattern for char in "[]"):
            fail(f"CODEOWNERS:{number}: unsupported pattern syntax: {pattern}")
            errors += 1
        if not pattern_matches(pattern, paths):
            fail(f"CODEOWNERS:{number}: pattern matches no tracked file: {pattern}")
            errors += 1
        for owner in line_owners:
            if not owner.startswith("@") or not all(GITHUB_LOGIN.fullmatch(part) for part in owner[1:].split("/", 1)):
                fail(f"CODEOWNERS:{number}: invalid owner: {owner}")
                errors += 1
            owners.add(owner.removeprefix("@").split("/", 1)[-1])
    if errors:
        raise ValueError("invalid CODEOWNERS")
    return owners


def validate_oncall(owners: set[str]) -> None:
    path = ROOT / ".github" / "merge-oncall.json"
    config = json.loads(path.read_text())
    if set(config) != {"login"}:
        raise ValueError("merge-oncall.json must contain exactly the 'login' key")
    login = config["login"]
    if not isinstance(login, str) or not GITHUB_LOGIN.fullmatch(login):
        raise ValueError("merge-oncall.json contains an invalid GitHub login")
    if login not in owners:
        raise ValueError("the merge oncall must also appear in CODEOWNERS")


def validate_labels() -> set[str]:
    labels_path = ROOT / ".github" / "labels.json"
    labels = json.loads(labels_path.read_text())
    if not isinstance(labels, list):
        raise ValueError("labels.json must contain a list")

    names: set[str] = set()
    for label in labels:
        if not isinstance(label, dict) or set(label) != {"name", "color", "description"}:
            raise ValueError("each label must contain name, color, and description")
        name = label["name"]
        color = label["color"]
        description = label["description"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each label must have a non-empty name")
        if name in names:
            raise ValueError(f"duplicate label: {name}")
        if not isinstance(color, str) or not LABEL_COLOR.fullmatch(color):
            raise ValueError(f"label {name!r} has an invalid color")
        if not isinstance(description, str) or len(description) > 100:
            raise ValueError(f"label {name!r} has an invalid description")
        names.add(name)

    labeler_path = ROOT / ".github" / "labeler.yml"
    for number, raw_line in enumerate(labeler_path.read_text().splitlines(), 1):
        if not raw_line or raw_line[0].isspace() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.endswith(":"):
            raise ValueError(f"labeler.yml:{number}: expected a top-level label")
        raw_name = raw_line[:-1]
        name = json.loads(raw_name) if raw_name.startswith('"') else raw_name
        if name not in names:
            raise ValueError(f"labeler.yml:{number}: undefined label: {name}")
    return names


def parse_issue_labels(raw_value: str, path: Path, number: int) -> list[str]:
    if not raw_value:
        raise ValueError(f"{path.name}:{number}: labels must use an inline list or scalar")
    if raw_value.startswith("["):
        if not raw_value.endswith("]"):
            raise ValueError(f"{path.name}:{number}: malformed inline labels list")
        return [value.strip().strip("'\"") for value in next(csv.reader([raw_value[1:-1]], skipinitialspace=True))]
    return [raw_value.strip().strip("'\"")]


def validate_issue_templates(defined_labels: set[str]) -> None:
    template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    for path in sorted(template_dir.glob("*")):
        if path.name == "config.yml" or path.suffix not in {".md", ".yaml", ".yml"}:
            continue
        has_labels = False
        title_prefix = None
        for number, line in enumerate(path.read_text().splitlines(), 1):
            title_match = ISSUE_TITLE_LINE.match(line)
            if title_match:
                title_prefix = title_match.group(1)
                if title_prefix not in ISSUE_TITLE_PREFIXES:
                    raise ValueError(f"{path.name}:{number}: unsupported issue title prefix: {title_prefix!r}")
            match = ISSUE_LABELS_LINE.match(line)
            if not match:
                continue
            has_labels = True
            for label in parse_issue_labels(match.group(1), path, number):
                if label not in defined_labels:
                    raise ValueError(f"{path.name}:{number}: undefined label: {label}")
        if not has_labels:
            raise ValueError(f"{path.name}: issue template must declare managed labels")
        if path.suffix in {".yaml", ".yml"} and title_prefix is None:
            raise ValueError(f"{path.name}: issue form must declare a controlled title prefix")


def validate_workflows() -> None:
    errors = 0
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        lines = workflow.read_text().splitlines()
        for number, line in enumerate(lines, 1):
            if line.lstrip().startswith("#"):
                continue
            match = USES_LINE.match(line)
            if not match:
                continue
            action = match.group(1)
            if action.startswith("./"):
                continue
            if "@" not in action:
                fail(f"{workflow.name}:{number}: action has no ref: {action}")
                errors += 1
                continue
            _, ref = action.rsplit("@", 1)
            if not ACTION_SHA.fullmatch(ref):
                fail(f"{workflow.name}:{number}: action is not pinned to a commit SHA")
                errors += 1
            if action.startswith("actions/checkout@"):
                following = "\n".join(lines[number : number + 4])
                if not re.search(r"^\s+persist-credentials:\s*false\s*$", following, re.M):
                    fail(f"{workflow.name}:{number}: checkout must set persist-credentials: false")
                    errors += 1
    if errors:
        raise ValueError("invalid workflow action references")


def main() -> int:
    try:
        paths = tracked_files()
        owners = validate_codeowners(paths)
        validate_oncall(owners)
        labels = validate_labels()
        validate_issue_templates(labels)
        validate_workflows()
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        fail(str(error))
        return 1
    print("Community configuration is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
