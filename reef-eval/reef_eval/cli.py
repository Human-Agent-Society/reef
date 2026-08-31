"""The reef-eval CLI: one command from any task to a scored result.

    reef-eval list                                           # everything runnable here
    reef-eval run tasks/autoresearch/first-party/circle-packing --agent oracle
    reef-eval run autoresearch/first-party --agent claude-code --model anthropic/claude-opus-5
    reef-eval run edgebench/ann_vector_search_qps --agent codex --budget 2
    reef-eval stream terminal-bench --agent claude-code   # continual: carried state
    reef-eval report                                         # summarize the results store

Targets resolve in order: an explicit task directory, then every task inside
a category or benchmark folder, then anything else passed to Harbor as a
registry id. The CLI is a thin caller of :class:`Lab`; everything it runs
lands in the same tagged results store (``--lab``, default ``runs/cli``), so
re-running resumes and ``reef-eval report`` is a query.

Protocols the CLI cannot express (custom schedules, control arms) are plain
Python scripts over :class:`Lab`; see ``examples/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from reef_eval.targets import find_tasks_root as _find_tasks_root
from reef_eval.targets import tasks_under as _tasks_under

if TYPE_CHECKING:
    from reef_eval import Budget, Lab
    from reef_eval.types import Row


def resolve_targets(targets: list[str], tasks_root: Path | None) -> list[str]:
    """Expand CLI targets into runnable task references.

    The resolution itself lives in :mod:`reef_eval.targets`, which scripts call
    as ``reef_eval.tasks``. Here a target that names nothing runnable ends the
    process with the message instead of raising.
    """
    from reef_eval.targets import resolve

    try:
        return resolve(targets, tasks_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def _parse_pairs(pairs: list[str] | None) -> dict:
    """``K=V`` flags, with the value read as JSON when it parses as JSON.

    So ``--tag budget=2`` is the number 2 and ``--agent-arg
    extra_allowed_hosts='["a"]'`` is a list, while a plain word stays a
    string.
    """
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _build_agent(args: argparse.Namespace) -> dict:
    agent: dict = {"name": args.agent}
    if args.model:
        agent["model_name"] = args.model
    agent.update(_parse_pairs(args.agent_arg))
    if args.command:
        agent["command"] = args.command
    return agent


def _build_budget(args: argparse.Namespace) -> Budget | None:
    """Assemble a Budget from the budget flags (all optional)."""
    from reef_eval import Budget
    from reef_eval.budget import parse_duration_hours

    budget = Budget(
        time_h=None if args.budget is None else parse_duration_hours(args.budget),
        max_submissions=args.max_evals,
        max_tokens=_parse_count(args.max_tokens),
    )
    return None if budget.is_empty else budget


def _parse_count(value: str | None) -> int | None:
    """Accept plain ints or human suffixes: 500k, 2m, 1.5M."""
    if value is None:
        return None
    text = value.strip().lower()
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if text and text[-1] in mult:
        return int(float(text[:-1]) * mult[text[-1]])
    return int(text)


def _stream_name(name: str | None, targets: list[str]) -> str:
    """The stream's label: ``--name`` when given, else derived from what was
    asked for.

    The derived form is the first target's own name, plus how many more
    followed, so ``reef-eval stream terminal-bench cl-bench`` becomes
    ``terminal-bench+1more``. It is only a label: two streams with the same
    name but different tasks or setups still keep separate state.
    """
    if name is None:
        first = targets[0].rstrip("/").split("/")[-1]
        extra = len(targets) - 1
        return f"{first}+{extra}more" if extra else first
    if not name.strip():
        raise SystemExit("--name cannot be empty")
    bad = [c for c in "/\\" if c in name]
    if bad or any(c.isspace() for c in name):
        raise SystemExit(
            f"invalid --name {name!r}: it becomes a directory name, so it "
            "cannot contain path separators or whitespace"
        )
    return name


def _resolve_agent_flags(args: argparse.Namespace) -> None:
    """Settle --agent / --local / --command, which run and stream share.

    ``--local`` supplies the command itself, so it names its own agent;
    every other run has to say which agent to use.
    """
    if args.local and not args.command:
        raise SystemExit(
            "--local runs your own command: add --command '<shell command>'"
        )
    if args.command and not args.local:
        raise SystemExit("--command only works with --local")
    if args.local:
        args.agent = args.agent or "local-command"
    elif not args.agent:
        raise SystemExit("--agent is required (or use --local with --command)")


def _report_rows(rows: list[Row], lab: str, position: bool = False) -> int:
    """Print one line per episode and return the process exit code.

    An episode counts as a failure when it recorded no reward or recorded
    an error. A non-zero agent exit is noted but does not fail the run,
    since spending the whole time budget ends that way.
    """
    failures = 0
    for row in rows:
        error = row.tags.get("error")
        ok = row.rewards.get("reward") is not None and not error
        failures += not ok
        where = f"[{row.tags.get('position'):>3}] " if position else ""
        name = row.task.rstrip("/").split("/")[-1]
        print(
            f"  {'OK ' if ok else 'ERR'} {where}{name}: "
            f"{row.rewards or error}{_exit_note(row)}"
        )
    print(f"\nresults stored in {lab}: `reef-eval report --lab {lab}`")
    return 1 if failures else 0


def _exit_note(row: Row) -> str:
    """Flag an agent that exited non-zero.

    The score still stands, so this is not an error, but a reward earned by
    an agent that died reads the same as one it worked for unless the exit is
    shown. Spending the whole time budget also ends non-zero.
    """
    code = row.tags.get("agent_exit_code")
    return f"  (agent exited {code})" if code else ""


def _enable_progress() -> None:
    """Show the reef-eval logger's per-episode progress lines on stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("reef_eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)


def _first_build_hint(args: argparse.Namespace) -> None:
    if not getattr(args, "fake", False) and not getattr(args, "local", False):
        print(
            "note: first runs build container images and may download data; "
            "Docker caches both, so later runs start fast"
        )


def _make_lab(args: argparse.Namespace) -> Lab:
    from reef_eval import FakeExecutor, Lab, LocalExecutor
    from reef_eval.executors import Executor

    executor: Executor | None
    if getattr(args, "fake", False):
        executor = FakeExecutor()
    elif getattr(args, "local", False):
        executor = LocalExecutor()
    else:
        executor = None  # the default HarborExecutor (containers)
    return Lab(args.lab, executor=executor, concurrency=args.concurrent)


# ------------------------------------------------------------------ commands


def _downloaded_benchmarks() -> list[tuple[str, Path]]:
    """Benchmarks already downloaded into the cache, as (name, directory).

    A pip install has no ``tasks/`` folder, so this is where its tasks live
    once ``reef-eval fetch`` has run.
    """
    from reef_eval import fetch

    root = fetch.cache_home() / "tasks"
    if not root.is_dir():
        return []
    found = []
    for ref_dir in sorted(root.iterdir()):
        if not ref_dir.is_dir():
            continue
        for bench in sorted(ref_dir.iterdir()):
            if bench.is_dir():
                found.append((bench.name, bench))
    return found


def cmd_list(args: argparse.Namespace) -> int:
    from reef_eval import fetch

    tasks_root = _find_tasks_root(args.tasks_dir)
    listed = False
    from_checkout = False

    if tasks_root is not None:
        tasks = _tasks_under(tasks_root)
        if tasks:
            listed = from_checkout = True
            print(f"tasks under {tasks_root}:\n")
            for task in tasks:
                print(f"  {task.relative_to(tasks_root)}")

    for name, path in _downloaded_benchmarks():
        count = len(_tasks_under(path))
        if not count:
            continue
        listed = True
        print(f"\ndownloaded: {name} ({count} tasks) in {path}")
        print(f"  run with: reef-eval run {name}/<task> --agent <a>")

    if not listed:
        print("Nothing runnable here yet.\n")
        print("Download a benchmark:  reef-eval fetch <name>")
        print(f"  known: {', '.join(fetch.known_benchmarks())}")
        print("\nA source checkout ships every task in tasks/; from anywhere")
        print("else, pass --tasks-dir, or give `reef-eval run` a Harbor id (org/name).")
        return 1

    print("\nRun one:      reef-eval run <name-above> --agent oracle")
    if from_checkout:
        print("Run a folder: reef-eval run autoresearch/first-party --agent oracle")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _resolve_agent_flags(args)
    targets = resolve_targets(args.targets, _find_tasks_root(args.tasks_dir))
    agent = _build_agent(args)
    base_tags = _parse_pairs(args.tag)
    budget = _build_budget(args)

    _enable_progress()
    _first_build_hint(args)
    lab = _make_lab(args)

    async def _run() -> list:
        calls = [
            {
                "task": target,
                "agent": agent,
                "tags": {**base_tags, "attempt": attempt},
                "budget": budget,
            }
            for target in targets
            for attempt in range(args.attempts)
        ]
        return await lab.run_many(calls)

    print(
        f"running {len(targets)} task(s) x {args.attempts} attempt(s) "
        f"as agent '{args.agent}' -> {args.lab}"
    )
    return _report_rows(asyncio.run(_run()), args.lab)


def cmd_stream(args: argparse.Namespace) -> int:
    _resolve_agent_flags(args)
    # Derive the label from what was typed, not from the expansion, so
    # `reef-eval stream terminal-bench` is named for the benchmark.
    name = _stream_name(args.name, args.targets)
    targets = resolve_targets(args.targets, _find_tasks_root(args.tasks_dir))
    agent = _build_agent(args)
    tags = _parse_pairs(args.tag)
    budget = _build_budget(args)
    if args.shuffle is not None:
        # The seed becomes a tag, so each seed is its own stream with its
        # own state, keys, and resume.
        import random

        random.Random(args.shuffle).shuffle(targets)
        tags = {**tags, "shuffle_seed": args.shuffle}

    from reef_eval import Stream

    _enable_progress()
    _first_build_hint(args)
    stream = Stream(name, targets)
    lab = _make_lab(args)
    print(
        f"stream '{name}': {len(targets)} task(s) in order, "
        f"agent '{args.agent}', state carried between episodes -> {args.lab}"
    )
    print(f"state: {stream.state_root(lab, agent, tags=tags, budget=budget)}")
    rows = asyncio.run(stream.run(lab, agent, tags=tags, budget=budget))
    return _report_rows(rows, args.lab, position=True)


def cmd_report(args: argparse.Namespace) -> int:
    from reef_eval import Lab

    lab = Lab(args.lab)
    df = lab.df(args.kind or None)
    if df.empty:
        print(f"no rows in {args.lab}")
        return 1
    if "reward" in df.columns:
        df = df.assign(
            task=df["task"].map(lambda t: "/".join(t.rstrip("/").split("/")[-2:]))
        )
        summary = (
            df.groupby("task")["reward"].agg(["count", "mean", "max"]).sort_index()
        )
        print(summary.to_string())
    else:
        print(df.to_string(max_rows=50))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    # In a checkout, a benchmark's own fetch.py regenerates from upstream.
    tasks_root = _find_tasks_root(args.tasks_dir)
    script = None
    if tasks_root is not None:
        matches = sorted(tasks_root.glob(f"*/{args.benchmark}/fetch.py")) + [
            tasks_root / args.benchmark / "fetch.py"
        ]
        script = next((m for m in matches if m.is_file()), None)
    if script is None:
        from reef_eval import fetch

        if args.benchmark in fetch.BENCHMARKS or args.benchmark in fetch.REGISTRY:
            dest = fetch.benchmark(args.benchmark)
            print(f"downloaded to {dest}")
            print(f"run with: reef-eval run {args.benchmark}/<task> --agent <a>")
            return 0
        known = fetch.known_benchmarks()
        raise SystemExit(f"unknown benchmark '{args.benchmark}'; known: {known}")
    import subprocess

    return subprocess.run(
        [sys.executable, str(script), *args.rest], check=False
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reef-eval",
        description="autoresearch evaluation on the Harbor task standard",
    )
    from reef_eval import __version__

    parser.add_argument(
        "--version", action="version", version=f"reef-eval {__version__}"
    )
    parser.add_argument("--tasks-dir", default=None, help="tasks catalog root")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list runnable tasks")
    p_list.set_defaults(func=cmd_list)

    def add_shared_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--agent",
            default=None,
            help="Harbor agent name (e.g. oracle, claude-code)",
        )
        p.add_argument("--model", default=None, help="model name for the agent")
        p.add_argument(
            "--budget",
            default=None,
            metavar="DURATION",
            help="time budget: 2h / 30m / 90s (a bare number = hours). "
            "HARD: sets the container timeout",
        )
        p.add_argument(
            "--max-evals",
            type=int,
            default=None,
            metavar="N",
            help="submission/eval budget (agent signal; judge caps at the "
            "task's own limit)",
        )
        p.add_argument(
            "--max-tokens",
            default=None,
            metavar="N",
            help="token budget, e.g. 500k or 2m (soft: signalled to the agent, "
            "actuals recorded)",
        )
        p.add_argument("--lab", default="runs/cli", help="results directory")
        p.add_argument(
            "--tag", action="append", metavar="K=V", help="extra tags (repeatable)"
        )
        p.add_argument(
            "--agent-arg",
            action="append",
            metavar="K=V",
            help="extra AgentConfig fields (repeatable)",
        )
        p.add_argument(
            "--fake",
            action="store_true",
            help="use the offline fake executor (no Docker; smoke tests)",
        )
        p.add_argument(
            "--local",
            action="store_true",
            help="run --command on this machine against the real scorer and grader"
            " (no Docker; scores are not isolation-backed)",
        )
        p.add_argument(
            "--command",
            default=None,
            metavar="CMD",
            help="the shell command --local runs; it reads $JUDGE_URL and $BUDGET_SEC",
        )

    p_run = sub.add_parser("run", help="run tasks/folders/registry ids")
    p_run.add_argument("targets", nargs="+")
    add_shared_flags(p_run)
    p_run.add_argument("--attempts", "-n", type=int, default=1)
    p_run.add_argument("--concurrent", type=int, default=4)
    p_run.set_defaults(func=cmd_run)

    p_stream = sub.add_parser(
        "stream",
        help="run tasks as one continual-learning stream (carried agent state)",
    )
    p_stream.add_argument("targets", nargs="+")
    add_shared_flags(p_stream)
    p_stream.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="label for this stream, recorded as the `stream` tag and used as "
        "its state directory (default: derived from the targets). Pass a new "
        "name to start the same tasks over with empty memory",
    )
    p_stream.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="deterministically shuffle the task order with this seed "
        "(interleaved streams; the seed is recorded as a shuffle_seed tag)",
    )
    p_stream.set_defaults(func=cmd_stream, concurrent=1)

    p_report = sub.add_parser("report", help="summarize a results store")
    p_report.add_argument("--lab", default="runs/cli")
    p_report.add_argument(
        "--kind", default="episode", help="episode | trace | '' for all"
    )
    p_report.set_defaults(func=cmd_report)

    p_fetch = sub.add_parser("fetch", help="fetch an external benchmark's tasks")
    p_fetch.add_argument("benchmark")
    # REMAINDER so flags like `--limit 10` pass through to the fetch script.
    p_fetch.add_argument("rest", nargs=argparse.REMAINDER)
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
