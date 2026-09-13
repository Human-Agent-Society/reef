#!/usr/bin/env python3
"""Collect CORAL benchmark runs into retained JSON records and a summary table.

Reads each run's run.log (grader daemon lines, watcher reports) and the reef
service status captured by sweep, writes results/<run>.json, and prints a
table. Usage: collect.py <runs_dir> <out_dir>
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

GRADED = re.compile(r"^(\S+ \S+) coral\.grader\.daemon INFO Graded #(\d+) (\w+) -> score=(\S+) status=(\S+)")
REPORTED = re.compile(
    r"^(\S+ \S+) recipes\.beta\.coral\.watcher INFO reported attempt (\w+) \(agent=(\S+) score=(\S+) refs=(\d+)"
)
RUN_ID = re.compile(r"reef run id: (\S+)")


def parse_run(run_dir):
    log = os.path.join(run_dir, "run.log")
    attempts, reports, run_id = [], [], None
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                m = GRADED.match(line)
                if m:
                    ts, k, commit, score, status = m.groups()
                    try:
                        score = float(score)
                    except ValueError:
                        score = None
                    attempts.append(
                        {"time": ts, "eval_index": int(k), "commit": commit, "score": score, "status": status}
                    )
                    continue
                m = REPORTED.match(line)
                if m:
                    ts, commit, agent, score, refs = m.groups()
                    reports.append(
                        {
                            "time": ts,
                            "commit": commit,
                            "agent": agent,
                            "score": None if score == "None" else float(score),
                            "refs": int(refs),
                        }
                    )
                    continue
                m = RUN_ID.search(line)
                if m:
                    run_id = m.group(1)
    except OSError:
        pass
    return attempts, reports, run_id


def main(runs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*-ttt*")) + glob.glob(os.path.join(runs_dir, "*-base*"))):
        name = os.path.basename(run_dir)
        task, arm = name.rsplit("-", 1)
        attempts, reports, run_id = parse_run(run_dir)
        scored = [a for a in attempts if a["score"] is not None]
        status_path = os.path.join(run_dir, "reef-status.json")
        reef_status = None
        if os.path.exists(status_path):
            try:
                with open(status_path) as fh:
                    reef_status = json.load(fh)
            except (OSError, json.JSONDecodeError):
                reef_status = None
        record = {
            "run": name,
            "task": task,
            "arm": arm,
            "agents": 4 if arm.endswith("4") else 1,
            "training": arm.startswith("ttt"),
            "coral_run_id": run_id,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "reports": reports,
            "reef_status": reef_status,
            "summary": {
                "graded": len(attempts),
                "scored": len(scored),
                "best_score": max((a["score"] for a in scored), default=None),
                "first_score": scored[0]["score"] if scored else None,
                "last_score": scored[-1]["score"] if scored else None,
                "improvements": sum(1 for a in attempts if a["status"] == "improved"),
                "reported": len(reports),
                "reports_with_refs": sum(1 for r in reports if r["refs"] > 0),
            },
        }
        with open(os.path.join(out_dir, name + ".json"), "w") as fh:
            json.dump(record, fh, indent=1)
        s = record["summary"]
        step = None
        if reef_status:
            try:
                step = next(iter(reef_status["scenarios"].values()))["scenario_step"]
            except (KeyError, StopIteration, TypeError):
                step = None
        rows.append(
            (
                name,
                s["graded"],
                s["scored"],
                s["best_score"],
                s["improvements"],
                s["reported"],
                s["reports_with_refs"],
                step,
            )
        )
    print(f"{'run':28s} {'graded':>6} {'scored':>6} {'best':>10} {'impr':>5} {'rep':>4} {'refs>0':>6} {'step':>5}")
    for r in rows:
        best = "-" if r[3] is None else f"{r[3]:.5g}"
        print(f"{r[0]:28s} {r[1]:>6} {r[2]:>6} {best:>10} {r[4]:>5} {r[5]:>4} {r[6]:>6} {r[7]!s:>5}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
