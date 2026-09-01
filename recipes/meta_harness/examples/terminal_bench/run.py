"""Run the pinned frozen, incumbent-only, and full-history experiment cells."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.budget import ObservedCostLedger
from harness.codex import DEFAULT_CODEX_PATH, CodexProposer, verify_permission_profile
from harness.composition import genesis_composition
from harness.config import (
    DEV_TASKS,
    HARBOR_VERSION,
    HARD_TASKS,
    META_HARNESS_COMMIT,
    PI_VERSION,
    REEF_COMMIT,
    SMOKE_DEV_TASKS,
    SMOKE_TEST_TASKS,
    SMOKE_TRAIN_TASKS,
    TARGET_MODEL_PRICING,
    TERMINAL_BENCH_COMMIT,
    TERMINAL_BENCH_DATASET_COMMIT,
    TERMINAL_BENCH_DATASET_URL,
    TERMINAL_BENCH_VERSION,
    TEST_TASKS,
    TRAIN_TASKS,
    ExperimentConfig,
)
from harness.harbor_eval import HarborEvaluator, verify_dataset_registry_pin
from harness.publication import publish_candidate
from harness.reporting import write_aggregate_report, write_cell_report
from harness.search import CandidateRecord, MetaHarnessSearch, Population
from harness.workspace import EvolutionWorkspace
from reef.harness.adapters import get_adapter

HERE = Path(__file__).resolve().parent
CELLS = ("frozen", "incumbent_only", "full_history")
SPLIT_NAMES = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=(*CELLS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pi-binary", default=os.environ.get("REEF_PI_BINARY", "pi"))
    parser.add_argument("--codex-binary", type=Path, default=DEFAULT_CODEX_PATH)
    parser.add_argument("--rounds", type=int, default=ExperimentConfig().rounds)
    parser.add_argument("--trials-per-task", type=int, default=ExperimentConfig().trials_per_task)
    parser.add_argument(
        "--max-observed-cost-usd",
        type=float,
        default=None,
        help="Required for live runs; stops before a new target trial once recorded cost reaches this value",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate exact pins and print the plan without calls")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one pinned task per split while preserving rounds and trials",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(rounds=args.rounds, trials_per_task=args.trials_per_task)
    selected_cells = CELLS if args.cell == "all" else (args.cell,)
    splits = selected_task_splits(args.smoke)
    pi_binary = verify_runtime_pins(args.pi_binary, args.codex_binary)
    output_root = args.output_dir or HERE / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan = build_plan(
        config,
        selected_cells,
        output_root,
        pi_binary,
        args.codex_binary,
        splits=splits,
        smoke=args.smoke,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if plan["pins"]["reef_source_dirty"]:
        raise SystemExit("tracked Reef source has uncommitted changes; commit or stash them before a live run")
    if (
        args.max_observed_cost_usd is None
        or not math.isfinite(args.max_observed_cost_usd)
        or args.max_observed_cost_usd <= 0
    ):
        raise SystemExit("--max-observed-cost-usd must be finite and positive; no model calls were made")
    if not os.environ.get(config.target_api_key_env, "").strip():
        raise SystemExit(f"{config.target_api_key_env} is empty; no model calls were made")

    output_root.mkdir(parents=True, exist_ok=True)
    run_identity_sha256 = ensure_run_identity(output_root / "run-identity.json", plan)
    _write_json(output_root / "plan.json", plan)
    ledger = ObservedCostLedger(output_root / "observed-cost.json", args.max_observed_cost_usd)
    descriptor = get_adapter("pi")
    for cell in selected_cells:
        cell_dir = output_root / cell
        if completed_cell(cell_dir, cell, run_identity_sha256):
            print(f"skip completed cell {cell}: {cell_dir}")
            continue
        run_cell(
            cell,
            cell_dir,
            config=config,
            descriptor=descriptor,
            pi_binary=Path(pi_binary),
            codex_binary=args.codex_binary,
            ledger=ledger,
            splits=splits,
            run_identity_sha256=run_identity_sha256,
        )
    _write_json(
        output_root / "task-splits.json",
        splits,
    )
    aggregate = write_aggregate_report(output_root, selected_cells)
    if not aggregate["equal_target_episode_budget"] or not aggregate["equal_task_checksums"]:
        raise RuntimeError("completed cells do not have equal target budgets and pinned task checksums")


def run_cell(
    cell: str,
    output_dir: Path,
    *,
    config: ExperimentConfig,
    descriptor: Any,
    pi_binary: Path,
    codex_binary: Path,
    ledger: ObservedCostLedger,
    splits: Mapping[str, Sequence[str]] | None = None,
    run_identity_sha256: str = "test-run",
) -> None:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_splits = _normalize_task_splits(splits)
    evaluator = HarborEvaluator(
        splits=selected_splits,
        trials_per_task=config.trials_per_task,
        output_dir=output_dir / "private-evaluations",
        target_model=config.target_model,
        target_base_url=config.target_base_url,
        target_api_key_env=config.target_api_key_env,
        pi_binary=pi_binary,
        before_trial=ledger.before_trial,
        after_trial=ledger.record_trial,
        budget_namespace=cell,
    )
    genesis = genesis_composition()
    if cell == "frozen":
        resumed = evaluator.completed_evaluations("train") > 0
        train = dev = None
        for round_index in range(config.rounds + 1):
            train = evaluator.evaluate(genesis, split="train", round_index=round_index)
            dev = evaluator.evaluate(genesis, split="dev", round_index=round_index)
        if train is None or dev is None:
            raise RuntimeError("frozen cell produced no search evaluations")
        population = Population()
        record, _, _ = population.add(
            CandidateRecord(
                candidate=genesis,
                round_index=0,
                alias="genesis",
                train_score=train.score,
                dev_score=dev.score,
                outcome="selected",
            )
        )
        selected = genesis
        records = tuple(population.records)
        sessions = ()
        selected_dev_score = record.dev_score
    else:
        workspace = EvolutionWorkspace(output_dir / "workspace", descriptor)
        proposer = CodexProposer(
            model=config.proposer_model,
            reasoning_effort=config.proposer_reasoning_effort,
            codex_path=codex_binary,
        )
        outcome = MetaHarnessSearch(
            workspace=workspace,
            evaluator=evaluator,
            proposer=proposer,
            mode=cell,
            rounds=config.rounds,
        ).run(genesis)
        selected = outcome.selected
        records = outcome.population
        sessions = outcome.proposer_sessions
        selected_dev_score = next(
            record.dev_score for record in records if record.candidate.content_hash == selected.content_hash
        )
        resumed = outcome.resumed

    fill_count = _fill_target_budget(evaluator, selected, config.rounds)
    test_result = evaluator.evaluate(selected, split="test", round_index=9000)
    publish_candidate(
        selected,
        descriptor=descriptor,
        output_dir=output_dir,
        scenario=f"meta-harness-{cell}",
        metadata={"cell": cell, "test_score": test_result.score},
    )
    summary = write_cell_report(
        output_dir=output_dir,
        cell=cell,
        selected=selected,
        population=records,
        proposer_sessions=sessions,
        test_result=test_result,
        selected_dev_score=selected_dev_score,
        resumed=resumed,
        budget_fill_evaluations=fill_count,
        wall_time_s=time.monotonic() - started,
        config={**asdict(config), "cell": cell},
    )
    expected = planned_target_episodes(config, selected_splits)
    if summary["target_episode_count"] != expected:
        raise RuntimeError(
            f"cell {cell} retained {summary['target_episode_count']} target episodes; expected {expected}"
        )
    if len(summary["task_checksums"]) != sum(len(tasks) for tasks in selected_splits.values()):
        raise RuntimeError("cell did not retain a checksum for every Terminal-Bench task")
    if cell == "full_history" and summary["later_round_history_read"] is not True:
        raise RuntimeError("later full-history proposer round did not retain evidence of reading prior history")
    mark_done(output_dir, cell, run_identity_sha256)


def _fill_target_budget(evaluator: HarborEvaluator, selected: Any, rounds: int) -> int:
    expected = rounds + 1
    fills = 0
    for split in ("train", "dev"):
        completed = evaluator.completed_evaluations(split)
        while completed < expected:
            evaluator.evaluate(selected, split=split, round_index=8000 + completed)
            completed += 1
            fills += 1
        if completed > expected:
            raise RuntimeError(f"{split} evaluation budget exceeded before held-out testing")
    return fills


def build_plan(
    config: ExperimentConfig,
    cells: tuple[str, ...],
    output_dir: Path,
    pi_binary: str,
    codex_binary: Path,
    *,
    splits: Mapping[str, Sequence[str]] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    selected_splits = _normalize_task_splits(splits)
    per_cell = planned_target_episodes(config, selected_splits)
    return {
        "cells": cells,
        "smoke": smoke,
        "output_dir": str(output_dir),
        "config": asdict(config),
        "target_model_pricing": TARGET_MODEL_PRICING.to_jsonable(),
        "pins": {
            "reef_source_commit": _command_output(["git", "rev-parse", "HEAD"]),
            "reef_source_dirty": bool(_command_output(["git", "status", "--porcelain", "--untracked-files=no"])),
            "reef_base_commit": REEF_COMMIT,
            "meta_harness_commit": META_HARNESS_COMMIT,
            "terminal_bench_historical_commit": TERMINAL_BENCH_COMMIT,
            "terminal_bench_dataset_url": TERMINAL_BENCH_DATASET_URL,
            "terminal_bench_dataset_commit": TERMINAL_BENCH_DATASET_COMMIT,
            "terminal_bench_version": TERMINAL_BENCH_VERSION,
            "harbor_version": HARBOR_VERSION,
            "pi_version": PI_VERSION,
        },
        "binaries": {
            "pi": pi_binary,
            "codex": str(Path(codex_binary).resolve()),
            "codex_version": _command_output([str(codex_binary), "--version"]),
        },
        "split_sizes": {name: len(tasks) for name, tasks in selected_splits.items()},
        "split_identity_sha256": hashlib.sha256(json.dumps(selected_splits, sort_keys=True).encode()).hexdigest(),
        "planned_target_episodes_per_cell": per_cell,
        "planned_target_episodes_total": per_cell * len(cells),
        "planned_proposer_turns": config.rounds * sum(cell != "frozen" for cell in cells),
    }


def planned_target_episodes(
    config: ExperimentConfig,
    splits: Mapping[str, Sequence[str]] | None = None,
) -> int:
    selected_splits = _normalize_task_splits(splits)
    search = (
        (config.rounds + 1) * (len(selected_splits["train"]) + len(selected_splits["dev"])) * config.trials_per_task
    )
    heldout = len(selected_splits["test"]) * config.trials_per_task
    return search + heldout


def selected_task_splits(smoke: bool) -> dict[str, tuple[str, ...]]:
    """Return either the exact reproduction split or its bounded live smoke."""
    if smoke:
        return {
            "train": SMOKE_TRAIN_TASKS,
            "dev": SMOKE_DEV_TASKS,
            "test": SMOKE_TEST_TASKS,
        }
    return {"train": TRAIN_TASKS, "dev": DEV_TASKS, "test": TEST_TASKS}


def _normalize_task_splits(
    splits: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    source = selected_task_splits(False) if splits is None else splits
    if set(source) != set(SPLIT_NAMES):
        raise ValueError(f"task splits must contain exactly {SPLIT_NAMES}")
    normalized = {name: tuple(source[name]) for name in SPLIT_NAMES}
    if any(not tasks for tasks in normalized.values()):
        raise ValueError("task splits must be non-empty")
    flattened = tuple(task for name in SPLIT_NAMES for task in normalized[name])
    if len(flattened) != len(set(flattened)):
        raise ValueError("task splits must be disjoint")
    unknown = set(flattened).difference(HARD_TASKS)
    if unknown:
        raise ValueError(f"task splits contain unpinned tasks: {sorted(unknown)}")
    return normalized


def verify_runtime_pins(pi_binary: str, codex_binary: Path) -> str:
    if importlib.metadata.version("harbor") != HARBOR_VERSION:
        raise SystemExit(f"Harbor must be exactly {HARBOR_VERSION}")
    if importlib.metadata.version("terminal-bench") != TERMINAL_BENCH_VERSION:
        raise SystemExit(f"terminal-bench must be exactly {TERMINAL_BENCH_VERSION}")
    try:
        asyncio.run(verify_dataset_registry_pin())
    except Exception as exc:
        raise SystemExit(f"Terminal-Bench registry pin verification failed: {exc}") from exc
    resolved_pi = shutil.which(pi_binary) if not Path(pi_binary).is_absolute() else str(Path(pi_binary).resolve())
    if not resolved_pi:
        raise SystemExit(f"Pi binary {pi_binary!r} was not found")
    if _command_output([resolved_pi, "--version"]) != PI_VERSION:
        raise SystemExit(f"Pi must be exactly {PI_VERSION}")
    if not Path(codex_binary).is_file():
        raise SystemExit(f"Codex binary was not found: {codex_binary}")
    try:
        verify_permission_profile(codex_binary)
    except Exception as exc:
        raise SystemExit(f"Codex permission-profile verification failed: {exc}") from exc
    _command_output(["docker", "info", "--format", "{{.ServerVersion}}"])
    _command_output(["git", "lfs", "version"])
    base_ok = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REEF_COMMIT, "HEAD"],
        cwd=HERE,
        check=False,
    ).returncode
    if base_ok != 0:
        raise SystemExit(f"current Reef source does not descend from pinned base {REEF_COMMIT}")
    return resolved_pi


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"runtime check failed for {command[0]!r}: {exc}") from exc


def ensure_run_identity(path: Path, plan: Mapping[str, Any]) -> str:
    identity = {
        "schema_version": 1,
        "smoke": bool(plan["smoke"]),
        "config": plan["config"],
        "target_model_pricing": plan["target_model_pricing"],
        "pins": {key: value for key, value in plan["pins"].items() if key != "reef_source_dirty"},
        "binary_versions": {
            "pi": plan["pins"]["pi_version"],
            "codex": plan["binaries"]["codex_version"],
        },
        "split_sizes": plan["split_sizes"],
        "split_identity_sha256": plan["split_identity_sha256"],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    payload = {**identity, "sha256": digest}
    if path.exists():
        if _read_json_object(path) != payload:
            raise RuntimeError(f"existing run identity does not match this invocation: {path}")
    else:
        _write_json(path, payload)
    return digest


def completed_cell(output_dir: Path, cell: str, run_identity_sha256: str) -> bool:
    marker_path = output_dir / "done.json"
    if not marker_path.is_file():
        return False
    marker = _read_json_object(marker_path)
    if (
        marker.get("complete") is not True
        or marker.get("cell") != cell
        or marker.get("run_identity_sha256") != run_identity_sha256
    ):
        raise RuntimeError(f"completed cell marker does not match this run: {marker_path}")
    expected_hashes = marker.get("files")
    if not isinstance(expected_hashes, dict) or expected_hashes != cell_evidence_hashes(output_dir):
        raise RuntimeError(f"completed cell evidence changed after completion: {output_dir}")
    publication = _read_json_object(output_dir / "publication.json")
    repository = Path(str(publication.get("repository") or ""))
    if not repository.is_absolute():
        repository = output_dir / repository
    if not repository.is_dir():
        raise RuntimeError(f"completed cell artifact repository is unavailable: {output_dir}")
    return True


def mark_done(output_dir: Path, cell: str, run_identity_sha256: str) -> None:
    _write_json(
        output_dir / "done.json",
        {
            "complete": True,
            "cell": cell,
            "run_identity_sha256": run_identity_sha256,
            "files": cell_evidence_hashes(output_dir),
        },
    )


def cell_evidence_hashes(output_dir: Path) -> dict[str, str]:
    excluded_roots = {"artifacts.git", "artifact-work", "artifact-cache"}
    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir)
        if not path.is_file() or path.name == "done.json" or excluded_roots.intersection(relative.parts):
            continue
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
