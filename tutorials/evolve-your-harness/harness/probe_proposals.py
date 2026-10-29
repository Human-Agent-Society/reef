"""Sample the native proposer over one failing task and count what validation lets through.

The proposer is ``harness.native_evolution.propose`` with the serve file's
seed tree and one recorded failure; each sample is one model call. A sample
is parsed when it yields a patch and admitted when the patch passes
``validate_patch`` for the native adapter, the same check the evolve step
runs before any rollout. The table this prints is the one the README's
results carry.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the harness package
sys.path.insert(0, str(HERE.parents[3]))  # the reef checkout, for a source run

from harness import native_evolution
from reef.core import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, locate, validate_patch

KINDS = ("skill", "native_tool", "native_hook", "native_graph", "native_agent")


class Models:
    """What ``propose`` reads: ``models.served``, one binding that records its replies."""

    def __init__(self, binding: ModelBinding) -> None:
        self.served = binding


def seed_tree(serve: Path) -> HarnessTree:
    """The serve file's seed tree; its ``evolution.seed`` path is relative to the tutorial directory."""
    config = yaml.safe_load(serve.read_text())
    return HarnessTree.read(HERE.parent / config["recipe"]["config"]["evolution"]["seed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample the native proposer and count admitted changes by kind.")
    parser.add_argument("--serve", default=HERE.parent / "configs" / "serve-native.yaml", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="the OpenAI compatible endpoint")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--samples", default=30, type=int)
    parser.add_argument("--task", default="[fib]", help="the prefix of the task to fail on")
    parser.add_argument("--rows", type=Path, help="write one JSON line per sample here")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.serve.read_text())
    task = next(t for t in config["recipe"]["config"]["evolution"]["tasks"] if t.startswith(args.task))
    tree = seed_tree(args.serve)
    descriptor = get_adapter("native")
    models = Models(ModelBinding(base_url=args.base_url, model=args.model, api_key="none"))
    sample = make_trajectory(
        (
            AgentRecord.create(
                scenario="probe",
                request_type=RequestType.INFERENCE,
                agent_record_id="probe",
                payload={"messages": [{"role": "user", "content": task}]},
            ),
        ),
        reward=0.0,
    )

    rows = []
    for i in range(args.samples):
        started = time.time()
        patch = native_evolution.propose(tree, (sample,), models)
        row = {"i": i, "seconds": round(time.time() - started), "parsed": patch is not None}
        if patch is not None:
            # The first path is the proposal's own file; an agent's graph edit rides beside it.
            location = locate(next(iter(patch.changes)))
            row["kind"] = "?" if location is None else location.kind.value
            validation = validate_patch(tree, patch, descriptor)
            row["admitted"] = validation.passed
            if not validation.passed:
                row["reason"] = validation.message()[:160]
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.rows:
        args.rows.write_text("".join(json.dumps(row) + "\n" for row in rows))

    parsed = [row for row in rows if row["parsed"]]
    by_kind = Counter(row["kind"] for row in parsed)
    admitted = Counter(row["kind"] for row in parsed if row["admitted"])
    print(f"\nsamples {len(rows)}  parsed {len(parsed)}  unparsed {len(rows) - len(parsed)}")
    for kind in KINDS:
        if by_kind[kind]:
            print(f"{kind}: {by_kind[kind]} proposed, {admitted[kind]} admitted")
    for row in parsed:
        if not row["admitted"]:
            print(f"refused: {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
