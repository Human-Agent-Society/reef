"""Claims the docs make that a test can check.

Two of them: every relative link and anchor resolves, and every stated
task count matches the tasks actually on disk. The counts appear in
three places, which is one page for the site, one for the repo landing
page, and one beside the tasks themselves; checking them here is what
keeps three copies from drifting apart.

Vendored and generated task directories are skipped: their markdown comes
from upstream and is not ours to fix.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SKIP = (
    "tasks/autoresearch/edgebench/",
    "tasks/autoresearch/frontier-cs/frontier-cs-",
    "tasks/continual-learning/cl-bench/",
    "tasks/continual-learning/terminal-bench/",
    "tasks/continual-learning/swebench-verified/",
    ".venv/",
    "runs/",
    "site/",
)


def _ours(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return not any(skip in rel for skip in SKIP)


DOCS = sorted(p for p in ROOT.rglob("*.md") if _ours(p))


def _anchors(text: str) -> set[str]:
    """GitHub-style slugs for every heading in a page."""
    found = set()
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if not heading:
            continue
        slug = heading.group(1).strip().lower()
        slug = re.sub(r"[`*.,:;()\[\]/?!'\"]", "", slug)
        found.add(re.sub(r"\s+", "-", slug))
    return found


def test_docs_were_found():
    assert len(DOCS) > 10, f"expected the doc set, found {DOCS}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_and_anchors_resolve(doc):
    text = doc.read_text()
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        link = match.group(1)
        if link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target, _, fragment = link.partition("#")
        path = (doc.parent / target).resolve()
        assert path.exists(), f"{doc.name} links to a missing path: {link}"
        if fragment and path.suffix == ".md":
            assert fragment.lower() in _anchors(path.read_text()), (
                f"{doc.name} links to a missing anchor: {link}"
            )


# Benchmarks whose tasks are committed, so the count is checkable here.
# SWE-bench Verified is fetched (its upstream has no license), so its 500
# tasks exist only on a machine that ran `reef-eval fetch`.
BENCHMARKS = [
    "autoresearch/first-party",
    "autoresearch/edgebench",
    "autoresearch/frontier-cs",
    "continual-learning/terminal-bench",
    "continual-learning/cl-bench",
]
COUNTED_IN = ["README.md", "docs/tasks.md", "tasks/README.md"]


def _task_count(benchmark: str) -> int:
    from reef_eval.targets import tasks_under

    return len(tasks_under(ROOT / "tasks" / benchmark))


def _count_rows(benchmark: str) -> list[tuple[str, str]]:
    """Table rows across the docs that link *benchmark* and so state a count.

    Only table rows: prose links a benchmark to talk about it, and a
    sentence is not claiming to say how many tasks it has.
    """
    rows = []
    for doc in COUNTED_IN:
        for line in (ROOT / doc).read_text().splitlines():
            if line.startswith("|") and f"{benchmark})" in line:
                rows.append((doc, line))
    return rows


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_stated_task_counts_match_the_tasks_on_disk(benchmark):
    """Every table row linking a benchmark states its real task count.

    The count lives in three tables: one for the docs site, one for the
    repo landing page, one beside the tasks. A benchmark that gains tasks
    leaves the reader unable to tell which table is right, so the number
    has to come from the folder.
    """
    real, rows = _task_count(benchmark), _count_rows(benchmark)
    assert rows, f"no table in {COUNTED_IN} links tasks/{benchmark} any more"
    for doc, line in rows:
        assert str(real) in re.findall(r"\d+", line), (
            f"{doc} lists tasks/{benchmark} without its {real} tasks: "
            f"{line.strip()[:130]}"
        )
