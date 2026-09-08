#!/usr/bin/env python3
"""Run CORAL test-time training against a live Reef stack.

This is the piece that imports CORAL. The adapter modules under
``recipes/coral/`` stay import-free of it so they test standalone; this entry
point wires the two systems for a real run:

1. builds CORAL's ``GatewayManager`` (its embedded LiteLLM proxy) with a
   model entry that routes to the Reef service ``run.sh`` started,
2. splices the Reef correlation layer under CORAL's middleware
   (``attach_reef_adapter``),
3. runs a small demo loop — one or two agents making attempts from git
   worktrees, graded by a deterministic local grader — reporting every
   finalized attempt to Reef, which trains on sibling groups and serves the
   updated weights to the next attempts,
4. writes the run's result bundle to ``work/<run>/bundle.json``.

The demo agent is scripted (its "intelligence" is a fixed prompt); every
wire interaction — gateway key swap, header stamping, receipt capture,
grading, reporting, training, serving update — is real. Replace
``demo_attempt`` with CORAL's real agent runtimes for a full deployment;
the wiring does not change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from coral.gateway.server import GatewayManager  # CORAL: pinned commit, see README
from recipes.coral.bundle import build_result_bundle
from recipes.coral.gateway_launcher import attach_reef_adapter
from recipes.coral.journal import CallJournal
from recipes.coral.reporter import AttemptReport, report_attempt

REEF_URL = "http://127.0.0.1:8900"
GATEWAY_PORT = 8091
SCENARIO = "coral-demo"


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


def chat(api_key: str, prompt: str, max_tokens: int = 512) -> str:
    body = json.dumps(
        {
            "model": "reef-policy",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def grade(text: str) -> float:
    """Deterministic demo grader: reward proximity to a 12-line answer."""
    lines = [line for line in text.splitlines() if line.strip()]
    return max(0.0, 1.0 - abs(len(lines) - 12) / 12.0)


def git(worktree: Path, *args: str, capture: bool = False) -> str | None:
    result = subprocess.run(
        ["git", "-c", "user.name=coral-demo", "-c", "user.email=demo@localhost", *args],
        cwd=worktree,
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work") / "coral-demo")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--siblings", type=int, default=2, help="attempts per agent per generation")
    parser.add_argument("--reef-token", default="reef-local")
    args = parser.parse_args()
    state = args.work.resolve()
    state.mkdir(parents=True, exist_ok=True)

    wait_healthy(f"{REEF_URL}/healthz")

    config_path = state / "litellm_config.yaml"
    config_path.write_text(
        "model_list:\n"
        "  - model_name: reef-policy\n"
        "    litellm_params:\n"
        "      model: openai/reef-policy\n"
        f"      api_base: {REEF_URL}/v1\n"
        f"      api_key: {args.reef_token}\n"
        "litellm_settings:\n"
        "  drop_params: true\n"
        "  return_response_headers: true\n"
    )
    manager = GatewayManager(port=GATEWAY_PORT, config_path=str(config_path), log_dir=state / "gateway")
    journal: CallJournal = attach_reef_adapter(
        manager,
        scenario=SCENARIO,
        journal_path=state / "reef" / "calls.jsonl",
        extra_tags={"coral-run": "demo-1"},
    )
    manager.start()

    agents = [f"agent-{n + 1}" for n in range(args.agents)]
    worktrees: dict[str, Path] = {}
    keys: dict[str, str] = {}
    bases: dict[str, str] = {}
    for agent in agents:
        worktree = state / "worktrees" / agent
        worktree.mkdir(parents=True, exist_ok=True)
        git(worktree, "init", "-q")
        git(worktree, "commit", "-q", "--allow-empty", "-m", "seed")
        worktrees[agent] = worktree
        keys[agent] = manager.register_agent(agent, worktree)
        bases[agent] = git(worktree, "rev-parse", "HEAD", capture=True)[:12]

    reports: list[AttemptReport] = []
    for generation in range(args.generations):
        for sibling in range(args.siblings):
            for agent in agents:
                worktree = worktrees[agent]
                git(worktree, "checkout", "-q", bases[agent])
                cursor = journal.size()
                answer = chat(keys[agent], "Write a Python function (about 12 lines) that merges two sorted lists.")
                (worktree / "solution.py").write_text(answer)
                git(worktree, "add", "-A")
                git(worktree, "commit", "-q", "--allow-empty", "-m", f"{agent} gen{generation} sib{sibling}")
                commit = git(worktree, "rev-parse", "HEAD", capture=True)[:12]
                score = grade(answer)
                report = AttemptReport(
                    scenario=SCENARIO,
                    agent_id=agent,
                    commit_hash=commit,
                    score=score,
                    status="improved",
                    parent_hash=bases[agent],
                    run_id="demo-1",
                    references=tuple(journal.record_ids_since(cursor, agent, bases[agent])),
                )
                ack = report_attempt(REEF_URL, report, token=args.reef_token)
                reports.append(report)
                print(
                    f"{agent} gen{generation} sib{sibling}: score={score:.2f} refs={len(report.references)} ack={ack.get('agent_record_id')}"
                )
        # next generation branches from each agent's best sibling would go
        # here in a real run; the demo keeps the base fixed per agent and
        # relies on the trained weights (served after each sibling group
        # completes) to move the answers.
        time.sleep(30)

    bundle = build_result_bundle(journal, reports, run_id="demo-1")
    bundle_path = state / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2))
    print(f"bundle: {bundle_path}")
    print(json.dumps(bundle["token_accounting"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
