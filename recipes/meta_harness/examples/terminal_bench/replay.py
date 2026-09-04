"""Phase 0: prove Reef's selection matches upstream Meta-Harness, deterministically.

The reproduction claim rests here rather than on a live run. At any affordable
scale two live arms can select different candidates from sampling noise alone,
and nothing distinguishes that from an implementation difference. Replay
removes the noise: both selectors see the same recorded scores, so any
disagreement is a real divergence.

Upstream's rule lives in ``update_frontier``: a candidate's average is compared
against ``frontier["_best"]["avg_pass_rate"]`` and replaces it when strictly
greater, so the bar is a high-water mark and the incumbent is never re-run.
Reef's is :class:`MetaHarnessSelector`, comparing against the score the
incumbent was admitted on. This module drives both over the same score
sequences and reports any step where they disagree.

Run it with the upstream checkout on the path::

    METAHARNESS_DIR=/path/to/meta-harness/reference_examples/terminal_bench_2 \
        python -m recipes.meta_harness.examples.terminal_bench.replay
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from recipes.meta_harness.method import MetaHarnessSelector
from recipes.meta_harness.population import Population, PopulationStore
from reef.harness.render import render_composition  # noqa: F401  (import parity with the method)

SEED_ENTRIES = ({"id": "rules", "name": "rules", "config": {"text": "seed"}},)


@dataclass(frozen=True)
class Step:
    """One recorded iteration: what the candidate scored, and the incumbent."""

    candidate: tuple[float, ...]
    current: tuple[float, ...]

    @property
    def candidate_mean(self) -> float:
        return fmean(self.candidate)


def upstream_decisions(steps: Sequence[Step], baseline: tuple[float, ...], frontier_path: Path) -> list[bool]:
    """Replay upstream's frontier rule, using its own ``update_frontier``.

    The baseline phase runs first, because upstream always seeds the frontier
    from its baselines before any candidate ("Always seed frontier from
    baselines if results exist"). Without it the empty frontier reads -1 and
    the first candidate wins unconditionally.
    """
    import meta_harness

    meta_harness.FRONTIER_VAL = frontier_path
    frontier_path.write_text(json.dumps({}))
    meta_harness.update_frontier({"baseline": ({f"task-{i}": v for i, v in enumerate(baseline)}, fmean(baseline))})
    decisions = []
    for index, step in enumerate(steps):
        before = json.loads(frontier_path.read_text()).get("_best", {}).get("avg_pass_rate", -1)
        # Upstream keys per-task rates by task name; one synthetic task per score.
        per_task = {f"task-{position}": value for position, value in enumerate(step.candidate)}
        meta_harness.update_frontier({f"cand-{index}": (per_task, step.candidate_mean)})
        after = json.loads(frontier_path.read_text()).get("_best", {}).get("avg_pass_rate", -1)
        decisions.append(after != before)
    return decisions


def reef_decisions(steps: Sequence[Step], baseline: tuple[float, ...], mirror_path: Path) -> list[bool]:
    """Replay Reef's selector over the same recorded scores."""
    store = PopulationStore(mirror_path)
    population = Population()
    genesis = population.sync_served(SEED_ENTRIES, step=0)
    genesis.scores = tuple(baseline)  # the same baseline phase upstream runs
    store.restore_committed(population.to_dict())

    decisions = []
    for index, step in enumerate(steps):
        working = store.begin(store.committed.to_dict())
        entries = ({"id": "rules", "name": "rules", "config": {"text": f"candidate-{index}"}},)
        working.stage_candidate(entries, parent_id=working.served_id, step=index + 1)
        decision = MetaHarnessSelector(store).decide(
            _Candidate(),
            _Evaluation({"candidate_scores": list(step.candidate), "current_scores": list(step.current)}),
        )
        decisions.append(bool(decision.selected))
        store.commit_applied(working.to_dict())
    return decisions


class _Candidate:
    candidate_id = "replay"


class _Evaluation:
    def __init__(self, metrics: dict[str, Any]) -> None:
        self.metrics = metrics


def random_steps(count: int, tasks: int, rng: random.Random) -> list[Step]:
    """Score sequences shaped like a real run: noisy, occasionally improving."""
    return [
        Step(
            candidate=tuple(float(rng.random() < rng.uniform(0.1, 0.8)) for _ in range(tasks)),
            current=tuple(float(rng.random() < rng.uniform(0.1, 0.8)) for _ in range(tasks)),
        )
        for _ in range(count)
    ]


def compare(steps: Sequence[Step], baseline: tuple[float, ...], workdir: Path) -> list[int]:
    """Indices where the two selectors disagree."""
    upstream = upstream_decisions(steps, baseline, workdir / "frontier.json")
    reef = reef_decisions(steps, baseline, workdir / "mirror.json")
    return [index for index, (a, b) in enumerate(zip(upstream, reef, strict=True)) if a != b]


def main(argv: Sequence[str] | None = None) -> int:
    checkout = os.environ.get("METAHARNESS_DIR")
    if not checkout:
        print("METAHARNESS_DIR must name the upstream terminal_bench_2 directory", file=sys.stderr)
        return 2
    sys.path.insert(0, checkout)

    rng = random.Random(20260904)
    failures = 0
    with tempfile.TemporaryDirectory() as raw:
        workdir = Path(raw)
        for trial in range(200):
            tasks = rng.randint(1, 10)
            baseline = tuple(float(rng.random() < 0.5) for _ in range(tasks))
            steps = random_steps(count=rng.randint(1, 8), tasks=tasks, rng=rng)
            trial_dir = workdir / f"t{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            disagreements = compare(steps, baseline, trial_dir)
            if disagreements:
                failures += 1
                print(f"trial {trial}: disagreement at steps {disagreements}")
                for index in disagreements:
                    print(f"    candidate={steps[index].candidate} current={steps[index].current}")
    if failures:
        print(f"\nFAIL: {failures} of 200 sequences disagree")
        return 1
    print("OK: 200 random score sequences, upstream and Reef select identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
