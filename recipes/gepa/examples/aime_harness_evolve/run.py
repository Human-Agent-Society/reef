"""The four-cell GEPA reproduction: the pinned upstream quickstart, then the same
search over Reef harness compositions.

Cells: ``reference`` is upstream GEPA's own adapter and LiteLLM transport;
``frozen`` scores the seed composition on the test split with no search;
``rules`` evolves one Reef rules node that Pi receives as its exact system
prompt (the conformance arm); ``multi`` evolves rules and a skill node in
Pi's complete harness (the extension). Before any search cell, the baseline
gate scores the seed prompt through the direct path and through Reef and
stops the run on a material gap, so a broken bridge cannot masquerade as a
search result.

An output directory is resumable: the run identity is written once and a
later invocation with different settings is refused; a cell with a done.json
is skipped; GEPA resumes from its own checkpoint; held-out evaluations resume
from per-example checkpoints; token usage and observed spend persist across
restarts. The spend cap is required for a live run and stops new calls, not
calls in flight - the account-side budget remains the hard ceiling.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.adapter import MULTI_NODE, RULES_ONLY, ReefAdapter, run_quickstart_episode
from harness.baseline import run_baseline_alignment
from harness.config import (
    AIME_DATASET_SHA256,
    AIME_SPLIT_SIZES,
    AIME_TEST_REVISION,
    AIME_TRAIN_REVISION,
    API_KEY_ENV,
    BASELINE_REPETITIONS,
    GEPA_VERSION,
    HELDOUT_WORKERS,
    OPENAI_BASE_URL,
    PI_TASK_MAX_TOKENS,
    PI_VERSION,
    REEF_WORKERS,
    REFERENCE_WORKERS,
    REFLECTION_MODEL,
    SEARCH_BUDGET,
    SEEDS,
    TASK_MODEL,
)
from harness.data import MULTI_NODE_SEED_CANDIDATE, RULES_SEED_CANDIDATE, load_aime_splits
from harness.files import read_json, write_json, write_once
from harness.heldout import CheckpointedEvaluator
from harness.models import REFLECTION_MODEL_PRICE, TASK_MODEL_PRICE, SpendCap, TrackedChatModel, TrackedGEPALM
from harness.publication import publish_candidate
from harness.reference import OFFICIAL_SEED_CANDIDATE, OfficialAIMEAdapter
from harness.reporting import write_aggregate_report, write_search_report
from harness.search import run_sealed_search
from reef.harness import run_episode
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
CELLS = ("reference", "frozen", "rules", "multi")
# The smoke run: two examples per split and an eight-call budget, one worker,
# and reflection even on a perfect minibatch so the plumbing is exercised.
SMOKE_BUDGET = 8
SMOKE_EXAMPLES = 2


@dataclass(frozen=True)
class Session:
    """What every cell of one invocation shares: the credential, the pinned
    binary, the spend cap, and the concurrency plan."""

    api_key: str
    pi_binary: str
    spend_cap: SpendCap | None
    budget: int
    smoke: bool
    workers: Mapping[str, int]

    def task_binding(self) -> ModelBinding:
        return ModelBinding(OPENAI_BASE_URL, TASK_MODEL, api_key=self.api_key)

    def reflection_model(self, output_dir: Path) -> TrackedChatModel:
        return TrackedChatModel(
            ModelBinding(OPENAI_BASE_URL, REFLECTION_MODEL, api_key=self.api_key),
            price=REFLECTION_MODEL_PRICE,
            spend_cap=self.spend_cap,
            usage_path=output_dir / "reflection-usage.json",
        )

    def official_adapter(self, usage_path: Path, *, max_tokens: int | None = None) -> OfficialAIMEAdapter:
        model = TrackedGEPALM(
            TASK_MODEL,
            api_key=self.api_key,
            base_url=OPENAI_BASE_URL,
            max_completion_tokens=max_tokens,
            price=TASK_MODEL_PRICE,
            spend_cap=self.spend_cap,
            usage_path=usage_path,
        )
        return OfficialAIMEAdapter(model, max_workers=self.workers["reference"])

    def reef_adapter(self, cell: str, usage_path: Path, *, max_workers: int | None = None) -> ReefAdapter:
        """The rules arm runs the quickstart envelope; every other arm runs
        Pi's complete harness."""
        rules_only = cell == "rules"
        return ReefAdapter(
            descriptor=get_adapter("pi"),
            task_model=self.task_binding(),
            components=RULES_ONLY if rules_only else MULTI_NODE,
            binary=self.pi_binary,
            max_workers=max_workers or self.workers["reef"],
            episode_runner=run_quickstart_episode if rules_only else run_episode,
            spend_cap=self.spend_cap,
            usage_path=usage_path,
            task_max_tokens=PI_TASK_MAX_TOKENS,
        )

    def heldout(self, adapter: Any, output_dir: Path, **kwargs: Any) -> CheckpointedEvaluator:
        return CheckpointedEvaluator(
            adapter, output_dir / "heldout-checkpoints", batch_size=self.workers["heldout"], **kwargs
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", choices=(*CELLS, "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--budget", type=int, default=SEARCH_BUDGET, help="metric calls per search cell")
    parser.add_argument("--workers", type=int, help="override every evaluation concurrency; sampling is unchanged")
    parser.add_argument("--pi-binary", default=os.environ.get("REEF_PI_BINARY", "pi"))
    parser.add_argument("--output-dir", type=Path, help="reuse a directory to resume its run")
    parser.add_argument("--max-observed-cost-usd", type=float, help="required for a live run; stops new calls")
    parser.add_argument("--smoke", action="store_true", help="two examples per split, eight calls, no authority")
    parser.add_argument("--dry-run", action="store_true", help="validate pins and print the plan; no model calls")
    parser.add_argument("--baseline-only", action="store_true", help="run the baseline gate, then stop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = CELLS if args.cell == "all" else (args.cell,)
    pins = verify_pins(args.pi_binary)
    budget = SMOKE_BUDGET if args.smoke else args.budget
    workers = {
        name: 1 if args.smoke else (args.workers or default)
        for name, default in (("reference", REFERENCE_WORKERS), ("reef", REEF_WORKERS), ("heldout", HELDOUT_WORKERS))
    }
    # Everything a resumed invocation must agree on. Cells, seeds, and the
    # spend cap may be staged across invocations of the same directory.
    identity = {
        "smoke": args.smoke,
        "budget": budget,
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "base_url": OPENAI_BASE_URL,
        "pi_task_max_tokens": PI_TASK_MAX_TOKENS,
        "workers": workers,
        "baseline_repetitions": BASELINE_REPETITIONS,
        "pins": pins,
        "dataset_sha256": AIME_DATASET_SHA256,
        "dataset_sizes": AIME_SPLIT_SIZES,
    }
    if args.dry_run:
        test_size = SMOKE_EXAMPLES if args.smoke else AIME_SPLIT_SIZES["test"]
        validation_size = SMOKE_EXAMPLES if args.smoke else AIME_SPLIT_SIZES["validation"]
        plan = {
            **identity,
            "cells": cells,
            "seeds": args.seeds,
            "planned_task_evaluations": planned_task_evaluations(cells, len(args.seeds), budget, test_size)
            + (2 * BASELINE_REPETITIONS * validation_size if cells != ("frozen",) else 0),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    if args.max_observed_cost_usd is None or args.max_observed_cost_usd <= 0:
        raise SystemExit("--max-observed-cost-usd must be positive for a live run; no model calls were made")
    if pins["reef_dirty"]:
        raise SystemExit("tracked Reef source has uncommitted changes; no model calls were made")
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(f"{API_KEY_ENV} is not set; no model calls were made")
    trainset, valset, testset = load_aime_splits()
    if args.smoke:
        trainset, valset, testset = trainset[:SMOKE_EXAMPLES], valset[:SMOKE_EXAMPLES], testset[:SMOKE_EXAMPLES]

    output_root = args.output_dir or HERE / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    write_once(output_root / "run-identity.json", identity, "run identity")
    session = Session(
        api_key=api_key,
        pi_binary=args.pi_binary,
        spend_cap=SpendCap(output_root / "observed-cost.json", args.max_observed_cost_usd),
        budget=budget,
        smoke=args.smoke,
        workers=workers,
    )
    # The frozen cell alone measures nothing about the bridge, so it is the
    # only selection that skips the gate.
    if cells != ("frozen",):
        baseline_dir = output_root / "baseline-alignment"
        result_path = baseline_dir / "result.json"
        if not (result_path.is_file() and read_json(result_path).get("baseline_aligned") is True):
            # Both arms get the same concurrency and the same output cap, so
            # the comparison differs only in the path the request takes.
            run_baseline_alignment(
                official=session.official_adapter(
                    baseline_dir / "official-task-usage.json", max_tokens=PI_TASK_MAX_TOKENS
                ),
                reef=session.reef_adapter(
                    "rules", baseline_dir / "reef-task-usage.json", max_workers=workers["reference"]
                ),
                valset=valset,
                output_dir=baseline_dir,
                workers=workers["reference"],
            )
    if args.baseline_only:
        return

    for cell in cells:
        for seed in args.seeds:
            cell_dir = output_root / cell / f"seed-{seed}"
            if (cell_dir / "done.json").is_file():
                print(f"skip completed cell {cell} seed {seed}: {cell_dir}")
                continue
            cell_dir.mkdir(parents=True, exist_ok=True)
            if cell == "reference":
                run_reference(session, seed, cell_dir, trainset, valset, testset)
            elif cell == "frozen":
                run_frozen(session, seed, cell_dir, testset)
            else:
                run_reef_search(session, cell, seed, cell_dir, trainset, valset, testset)
            write_json(cell_dir / "done.json", {"cell": cell, "seed": seed})
    write_aggregate_report(output_dir=output_root, cells=cells, seeds=args.seeds)


def run_reference(session: Session, seed: int, output_dir: Path, trainset, valset, testset) -> None:
    """Upstream's own adapter and transport; only the accounting is Reef's."""
    adapter = session.official_adapter(output_dir / "task-usage.json")
    reflection = session.reflection_model(output_dir)
    outcome = run_sealed_search(
        seed_candidate=OFFICIAL_SEED_CANDIDATE,
        trainset=trainset,
        valset=valset,
        testset=testset,
        adapter=adapter,
        reflection_lm=reflection,
        max_metric_calls=session.budget,
        seed=seed,
        run_dir=output_dir / "search",
        # A provider failure in the direct path scores zero rather than
        # aborting a 150-example pass; Reef episodes already score that way.
        heldout_evaluator=session.heldout(adapter, output_dir, failure_score=0.0),
        skip_perfect_score=not session.smoke,
    )
    write_search_report(
        output_dir=output_dir,
        cell="reference",
        seed=seed,
        outcome=outcome,
        config={
            **cell_config(session, "reference", seed),
            "solver": "gepa.DefaultAdapter",
            "pi_task_max_tokens": None,
        },
        task_usage=adapter.usage.snapshot(),
        reflection_usage=reflection.usage.snapshot(),
    )


def run_frozen(session: Session, seed: int, output_dir: Path, testset) -> None:
    """The seed composition on the test split, published like a selected one."""
    adapter = session.reef_adapter("frozen", output_dir / "task-usage.json")
    started = datetime.now(timezone.utc)
    evaluated = session.heldout(adapter, output_dir).evaluate("frozen", testset, MULTI_NODE_SEED_CANDIDATE)
    score = sum(evaluated.scores) / len(evaluated.scores)
    usage = adapter.usage.snapshot()
    write_json(
        output_dir / "summary.json",
        {
            "cell": "frozen",
            "seed": seed,
            "test_score": score,
            "test_scores": evaluated.scores,
            "usage": {"task": usage},
            "estimated_cost_usd": TASK_MODEL_PRICE.estimate(usage),
            "pricing": asdict(TASK_MODEL_PRICE),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_json(output_dir / "config.json", cell_config(session, "frozen", seed))
    publish_candidate(
        adapter=adapter,
        candidate=MULTI_NODE_SEED_CANDIDATE,
        output_dir=output_dir,
        scenario=f"gepa-frozen-{seed}",
        metadata={"cell": "frozen", "seed": seed, "test_score": score},
    )


def run_reef_search(session: Session, cell: str, seed: int, output_dir: Path, trainset, valset, testset) -> None:
    """GEPA search over Reef nodes, then the sealed test split, then publication."""
    adapter = session.reef_adapter(cell, output_dir / "task-usage.json")
    candidate = RULES_SEED_CANDIDATE if cell == "rules" else MULTI_NODE_SEED_CANDIDATE
    reflection = session.reflection_model(output_dir)
    outcome = run_sealed_search(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        testset=testset,
        adapter=adapter,
        reflection_lm=reflection,
        max_metric_calls=session.budget,
        seed=seed,
        run_dir=output_dir / "search",
        heldout_evaluator=session.heldout(adapter, output_dir),
        skip_perfect_score=not session.smoke,
    )
    write_search_report(
        output_dir=output_dir,
        cell=cell,
        seed=seed,
        outcome=outcome,
        config=cell_config(session, cell, seed),
        task_usage=adapter.usage.snapshot(),
        reflection_usage=reflection.usage.snapshot(),
    )
    publish_candidate(
        adapter=adapter,
        candidate=outcome.result.candidates[outcome.promotion.candidate_idx],
        output_dir=output_dir,
        scenario=f"gepa-{cell}-{seed}",
        metadata={
            "cell": cell,
            "seed": seed,
            "promotion": asdict(outcome.promotion),
            "frozen_test_score": outcome.frozen_test_score,
            "selected_test_score": outcome.selected_test_score,
        },
    )


def cell_config(session: Session, cell: str, seed: int) -> dict[str, Any]:
    return {
        "cell": cell,
        "seed": seed,
        "max_metric_calls": 0 if cell == "frozen" else session.budget,
        "task_model": TASK_MODEL,
        "reflection_model": REFLECTION_MODEL,
        "base_url": OPENAI_BASE_URL,
        "api_key_env": API_KEY_ENV,
        "gepa_version": GEPA_VERSION,
        "pi_version": PI_VERSION,
        "revisions": {"AI-MO/aimo-validation-aime": AIME_TRAIN_REVISION, "MathArena/aime_2025": AIME_TEST_REVISION},
        "workers": dict(session.workers),
        "skip_perfect_score": None if cell == "frozen" else not session.smoke,
        "task_sampling": "provider defaults",
        "pi_task_max_tokens": PI_TASK_MAX_TOKENS,
    }


def planned_task_evaluations(cells, seed_count: int, budget: int, test_size: int) -> int:
    """The nominal count: the search budget plus two test passes per search
    cell, one test pass for the frozen cell. Reflection calls are extra."""
    per_seed = sum(test_size if cell == "frozen" else budget + 2 * test_size for cell in cells)
    return per_seed * seed_count


def verify_pins(pi_binary: str) -> dict[str, Any]:
    """Refuse to run on anything but the pinned GEPA and Pi releases, and record
    the exact Reef commit the run is about to use."""
    installed = importlib.metadata.version("gepa")
    if installed != GEPA_VERSION:
        raise SystemExit(f"GEPA version is {installed!r}, expected {GEPA_VERSION}; pip install -e . in {HERE}")
    pi_version = _command([pi_binary, "--version"], cwd=HERE)
    if pi_version != PI_VERSION:
        raise SystemExit(f"Pi version is {pi_version!r}, expected {PI_VERSION}; set REEF_PI_BINARY")
    _command(["git", "lfs", "version"], cwd=REPO_ROOT)
    return {
        "gepa_version": GEPA_VERSION,
        "pi_version": PI_VERSION,
        "reef_commit": _command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT),
        "reef_dirty": bool(_command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT)),
    }


def _command(command: list[str], *, cwd: Path) -> str:
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"{command[0]!r} is required: {exc}") from exc


if __name__ == "__main__":
    main()
