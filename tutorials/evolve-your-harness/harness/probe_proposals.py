"""Sample the native proposer over one failing task and count what admission lets through.

The proposer is ``harness.native_evolution.propose`` with the serve file's
seed nodes and one recorded failure; each sample is one model call. A sample
is parsed when it yields a mutation and admitted when that mutation passes
its node kind's admission rules, the same check the evolve step runs before
any episode. The table this prints is the one the README's results carry.
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
from reef.harness.model_binding import ModelBinding
from reef.harness.native.seed import SEED_NODES
from reef.harness.nodes import NODE_KINDS
from reef.train.types import TraceSample

KINDS = ("skill", "native_tool", "native_hook", "native_graph", "native_agent")


class _Models:
    """What ``propose`` reads: ``models.served``, one binding that records its replies."""

    def __init__(self, binding: ModelBinding) -> None:
        self.served = binding


def _seed(serve: Path) -> tuple[tuple[str, dict], ...]:
    config = yaml.safe_load(serve.read_text())
    nodes = []
    for entry in config["evolution"]["seed"]:
        if isinstance(entry, str):
            nodes.extend((node["name"], node["config"]) for node in SEED_NODES)
        else:
            nodes.append((entry["name"], entry["config"]))
    return tuple(nodes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample the native proposer and count admitted mutations by kind.")
    parser.add_argument("--serve", default=HERE.parent / "configs" / "serve-native.yaml", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="the OpenAI compatible endpoint")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--samples", default=30, type=int)
    parser.add_argument("--task", default="[fib]", help="the prefix of the task to fail on")
    parser.add_argument("--rows", type=Path, help="write one JSON line per sample here")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.serve.read_text())
    task = next(t for t in config["evolution"]["tasks"] if t.startswith(args.task))
    nodes = _seed(args.serve)
    models = _Models(ModelBinding(base_url=args.base_url, model=args.model, api_key="none"))
    sample = TraceSample("probe", {"messages": [{"role": "user", "content": task}]}, 0.0)

    rows = []
    for i in range(args.samples):
        started = time.time()
        mutation = native_evolution.propose(nodes, (sample,), models)
        row = {"i": i, "seconds": round(time.time() - started), "parsed": mutation is not None}
        if mutation is not None:
            kind = mutation.options["name"]
            row["kind"] = kind
            try:
                NODE_KINDS[kind](None, mutation.options["config"])
                row["admitted"] = True
            except ValueError as exc:
                row["admitted"] = False
                row["reason"] = str(exc)[:160]
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
