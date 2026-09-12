#!/usr/bin/env python3
"""Run CORAL test-time training against a live Reef stack.

This is the piece that imports CORAL. The adapter modules under
``recipes/beta/coral/`` stay import-free of it so they test standalone; this
entry point drives the real CORAL runtime against Reef:

1. loads the real CORAL task in ``task/`` (task.yaml, seed repo, packaged
   grader) and builds CORAL's ``AgentManager`` from it — the same
   orchestration ``coral start`` runs: agent worktrees, the embedded
   LiteLLM gateway (whose only upstream is the Reef service ``run.sh``
   started), the grader daemon, heartbeats, restarts,
2. splices the Reef correlation layer under the manager's gateway
   (``attach_reef_adapter_to_agent_manager``) before any agent spawns,
3. starts an ``AttemptWatcher`` beside CORAL's monitor loop: every attempt
   the grader daemon finalizes is reported to Reef with its exact captured
   inference references — Reef groups siblings, trains, and serves the
   updated weights to the agents' next calls,
4. writes the run's result bundle to ``<work>/bundle.json`` when the run
   auto-stops at its attempt budget.

The agents are CORAL's real runtimes (``opencode`` by default — any
runtime CLI registered with CORAL works via ``--runtime``), prompted by
CORAL's own generated task instructions and submitting through
``coral eval``. Nothing here scripts their behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path

from coral.agent.manager import AgentManager  # CORAL: pinned commit, see README
from coral.config import CoralConfig
from recipes.beta.coral.bundle import build_result_bundle
from recipes.beta.coral.gateway_launcher import attach_reef_adapter_to_agent_manager
from recipes.beta.coral.watcher import AttemptWatcher

DEFAULT_REEF_URL = "http://127.0.0.1:8900"
SCENARIO = "coral-demo"
TASK_DIR = Path(__file__).resolve().parent / "task"


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5):
            return True
    except Exception:
        return False


def wait_healthy(url: str, deadline_s: int = 600) -> None:
    end = time.time() + deadline_s
    while time.time() < end:
        if _probe(url):
            return
        time.sleep(5)
    raise RuntimeError(f"{url} not healthy within {deadline_s}s")


def load_config(args: argparse.Namespace, state: Path) -> CoralConfig:
    """The task config with the run-scoped values resolved for this launch.

    ``repo_path``/``results_dir`` become absolute (CORAL resolves them
    against the CWD otherwise) and ``task_dir`` is set the way
    ``coral start --config`` sets it, so the gateway config reference and
    the grader install resolve against ``task/``.
    """
    config = CoralConfig.from_yaml(TASK_DIR / "task.yaml")
    config.task_dir = TASK_DIR
    config.workspace.repo_path = str(TASK_DIR / "seed")
    config.workspace.results_dir = str(state / "coral-results")
    config.run.session = "local"
    config.run.verbose = True
    config.run.stop.max_real_attempts = args.max_attempts
    config.agents.count = args.agents
    if args.runtime:
        config.agents.runtime = args.runtime
    if args.model:
        config.agents.model = args.model
    config.agents.gateway.enabled = True
    if args.gateway_port:
        config.agents.gateway.port = args.gateway_port
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work") / "coral-demo")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=8, help="run budget: real attempts before auto-stop")
    parser.add_argument("--runtime", default="", help="CORAL runtime (default: task.yaml's, opencode)")
    parser.add_argument("--model", default="", help="model name the runtime asks the gateway for")
    parser.add_argument("--gateway-port", type=int, default=0, help="override the gateway port")
    parser.add_argument("--reef-token", default=os.environ.get("REEF_TOKEN", "reef-local"))
    parser.add_argument("--reef-url", default=os.environ.get("REEF_URL", DEFAULT_REEF_URL))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    state = args.work.resolve()
    state.mkdir(parents=True, exist_ok=True)
    os.environ["REEF_TOKEN"] = args.reef_token
    os.environ["REEF_API_BASE"] = args.reef_url.rstrip("/") + "/v1"

    wait_healthy(f"{args.reef_url}/healthz")

    config = load_config(args, state)
    run_id = f"coral-ttt-{time.strftime('%Y%m%d-%H%M%S')}"
    manager = AgentManager(config, verbose=True, config_dir=TASK_DIR)
    journal = attach_reef_adapter_to_agent_manager(
        manager,
        scenario=SCENARIO,
        journal_path=state / "reef" / "calls.jsonl",
        extra_tags={"coral-run": run_id},
    )

    manager.start_all()
    if manager.paths is None:
        raise RuntimeError("CORAL manager did not initialize run paths")
    print(f"CORAL run dir: {manager.paths.run_dir}  (reef run id: {run_id})")

    watcher = AttemptWatcher(
        coral_dir=manager.paths.coral_dir,
        journal=journal,
        reef_url=args.reef_url,
        scenario=SCENARIO,
        run_id=run_id,
        token=args.reef_token,
        state_path=state / "reef" / "reported.json",
    )
    stop_reporting = threading.Event()
    reporter_thread = threading.Thread(
        target=watcher.run, args=(stop_reporting,), name="reef-attempt-watcher", daemon=True
    )
    reporter_thread.start()

    try:
        # Blocks: feedback/restart supervision until the attempt budget
        # auto-stops the run (or Ctrl+C).
        manager.monitor_loop()
    finally:
        manager.stop_all()
        stop_reporting.set()
        reporter_thread.join(timeout=60)
        watcher.poll_once()  # final drain: anything graded during shutdown

    bundle = build_result_bundle(journal, watcher.reports, run_id=run_id)
    bundle_path = state / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2))
    print(f"bundle: {bundle_path}")
    print(json.dumps(bundle["token_accounting"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
