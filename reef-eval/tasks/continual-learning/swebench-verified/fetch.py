"""Fetch SWE-bench Verified as stock Harbor tasks, pinned.

    cd tasks/continual-learning/swebench-verified
    python fetch.py --limit 50              # a stream-sized subset
    python fetch.py django__django-15098    # named tasks
    python fetch.py                         # all 500

SWE-bench Verified is one of the six benchmarks the AgentStream paper
(arXiv:2608.00155) builds its task streams from, and of those six it is
the hardest one published in the Harbor task format so far. The pin below
is the exact commit the Harbor registry publishes as v1.0. The upstream
dataset repository carries no license, so these tasks are never committed
here. This script fetches them onto your machine, and the blob filter
keeps a `--limit` fetch small even though the full repository is huge.

    reef-eval stream swebench-verified --agent claude-code --model anthropic/claude-opus-5
"""

import argparse
from pathlib import Path

from reef_eval.fetch import REGISTRY, fetch_pinned_tasks

# The pin lives in reef_eval.fetch.REGISTRY, so this script and
# `fetch.benchmark("swebench-verified")` can never resolve different commits.
# Upstream: harbor-datasets (no license, so never vendored).
PIN = REGISTRY["swebench-verified"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fetch SWE-bench Verified tasks (pinned)"
    )
    parser.add_argument("tasks", nargs="*", help="task names (default: all 500)")
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N", help="only the first N tasks"
    )
    args = parser.parse_args()

    dest = Path(__file__).parent
    copied = fetch_pinned_tasks(
        PIN.repo,
        PIN.ref,
        dest,
        subdir=PIN.subdir,
        only=args.tasks or None,
        limit=args.limit,
    )
    print(f"fetched {len(copied)} SWE-bench Verified task(s) -> {dest}")
    print("stream them: reef-eval stream swebench-verified --agent <a> --model <m>")


if __name__ == "__main__":
    main()
