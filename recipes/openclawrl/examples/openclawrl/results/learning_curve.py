"""Per-session accept and style metrics of a GSM8K homework stream.

Reef's training run logs loss and rewards; the judge sidecar's verdict, whether
the agent's first reply satisfies the student, lands per session in the REEF Eval
lab as ``verifier/final.json``, which Reef never sees. This script reads those
verdicts and turns them into the stream's learning curve, in two ways.

While a stream runs (``run.sh`` starts it when ``WANDB_API_KEY`` is set), it
tails the lab and logs one point per session into a run of the same W&B group
as Reef's, so the accept and style curves sit beside the training curves::

    uv run --no-project --with wandb results/learning_curve.py \\
        --lab "$RUN_DIR/lab" --reef-url http://host:28900 --reef-token "$(cat token)"

After a stream, it exports the same points to a CSV and draws the figure the
README keeps, with no W&B involved::

    uv run --no-project --with matplotlib results/learning_curve.py \\
        --lab "$RUN_DIR/lab" --csv results/<run>/learning_curve.csv --plot results/<run>/learning_curve.png
    uv run --no-project --with matplotlib results/learning_curve.py \\
        --from-csv results/<run>/learning_curve.csv --plot results/<run>/learning_curve.png --max-session 35

Per session: ``accept`` (the strict criterion), the style markers the judge
flagged (``bold``, ``bullets``, ``numbered``, ``list``), ``style_clean``,
``no_shown_work``, ``no_gold_answer``, ``turns``, ``first_reply_chars``, the
cumulative accept count, and the rolling ten-session accept, bold, list, and
style-clean rates. W&B keys carry the ``eval/`` prefix; the CSV columns are
the same names without it.

In logging mode the script runs until killed (``run.sh`` does that when the
stream ends); ``--once`` logs whatever the lab holds and exits, which also
backfills a finished run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

TASK_RE = re.compile(r"gsm8k-s(\d+)__")
WINDOW = 10
RATE_KEYS = ("accept", "bold", "list", "style_clean")
COLUMNS = (
    "session",
    "accept",
    "bold",
    "bullets",
    "numbered",
    "list",
    "no_shown_work",
    "no_gold_answer",
    "style_clean",
    "turns",
    "first_reply_chars",
    "accept_cum",
    *(f"{key}_rate_{WINDOW}" for key in RATE_KEYS),
)
BLUE = "#2563a6"
ORANGE = "#d97706"
PURPLE = "#6b46c1"
GREEN = "#2f855a"
TITLE = "OpenClaw-RL training over the GSM8K homework stream"


def classify(violations: list[str]) -> dict[str, int]:
    text = " ".join(violations)
    bold = int("\\*\\*" in text)
    bullets = int("[-*" in text)
    numbered = int("\\d+[.)]" in text)
    return {
        "bold": bold,
        "bullets": bullets,
        "numbered": numbered,
        "list": int(bullets or numbered),
        "no_shown_work": int("shown-work" in text),
        "no_gold_answer": int("gold" in text),
        "style_clean": int(not (bold or bullets or numbered)),
    }


def read_sessions(lab: Path) -> dict[int, dict]:
    sessions: dict[int, dict] = {}
    for final in lab.glob("trials/gsm8k-s*/verifier/final.json"):
        match = TASK_RE.search(final.parent.parent.name)
        if match is None:
            continue
        try:
            data = json.loads(final.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sessions[int(match.group(1))] = data
    return sessions


def session_point(position: int, data: dict, earlier: list[dict]) -> dict:
    """The metrics of one session, given the points of the sessions before it."""
    point = {
        "session": position,
        "accept": int(data.get("reward") == 1.0),
        **classify(list(data.get("violations") or [])),
        "turns": data.get("turns"),
        "first_reply_chars": len(data.get("first_reply") or ""),
    }
    window = [*earlier, point][-WINDOW:]
    point["accept_cum"] = sum(item["accept"] for item in earlier) + point["accept"]
    for key in RATE_KEYS:
        point[f"{key}_rate_{WINDOW}"] = sum(item[key] for item in window) / len(window)
    return point


def session_rows(sessions: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for position in sorted(sessions):
        rows.append(session_point(position, sessions[position], rows))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: ("" if row.get(key) is None else row[key]) for key in COLUMNS} for row in rows)


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key.endswith(f"_rate_{WINDOW}"):
                    row[key] = float(value)
                else:
                    row[key] = int(float(value))
            rows.append(row)
    return sorted(rows, key=lambda row: row["session"])


def sessions_to_adaptation(rows: list[dict]) -> int | None:
    """The first session that starts three consecutive accepted sessions."""
    accepts = [row["accept"] for row in rows]
    for index in range(len(accepts) - 2):
        if accepts[index] and accepts[index + 1] and accepts[index + 2]:
            return rows[index]["session"]
    return None


def plot(rows: list[dict], path: Path, max_session: int | None = None) -> None:
    import matplotlib.pyplot as plt

    adaptation = sessions_to_adaptation(rows)
    if max_session is not None:
        rows = [row for row in rows if row["session"] <= max_session]
    sessions = [row["session"] for row in rows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    top.plot(
        sessions,
        [row["accept_cum"] for row in rows],
        color=BLUE,
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        label="accumulated accepts (eval/accept_cum)",
    )
    top.set_ylabel("Accepted sessions")
    bottom.plot(
        sessions,
        [row[f"bold_rate_{WINDOW}"] for row in rows],
        color=ORANGE,
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        label=f"bold rate, {WINDOW}-session window (eval/bold_rate_{WINDOW})",
    )
    bottom.plot(
        sessions,
        [row[f"list_rate_{WINDOW}"] for row in rows],
        color=PURPLE,
        linewidth=1.5,
        marker="o",
        markersize=2.5,
        label=f"list rate, {WINDOW}-session window (eval/list_rate_{WINDOW})",
    )
    bottom.set_ylim(-0.02, 1.02)
    bottom.set_ylabel("Rate in the first reply")
    bottom.set_xlabel("Session index")
    for axis in (top, bottom):
        if adaptation is not None and adaptation <= sessions[-1]:
            axis.axvline(
                adaptation,
                color=GREEN,
                linestyle="--",
                linewidth=1.2,
                label=f"sessions-to-adaptation = {adaptation}" if axis is top else None,
            )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="upper left" if axis is top else "upper right")
    figure.suptitle(TITLE, fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def scenario_of(reef_url: str, token: str) -> str | None:
    request = urllib.request.Request(f"{reef_url}/reef/status", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            names = list(json.load(response).get("scenarios", {}))
    except Exception:
        return None
    return names[0] if names else None


def log_to_wandb(args: argparse.Namespace) -> int:
    scenario = args.scenario
    while scenario is None:
        scenario = scenario_of(args.reef_url, args.reef_token)
        if scenario is None:
            if args.once:
                print("learning_curve: reef reports no scenario yet", file=sys.stderr)
                return 1
            time.sleep(args.poll_seconds)

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=f"{args.group_prefix}/{scenario}",
        name=args.name or f"{args.group_prefix}-eval-{scenario[-12:]}",
        job_type="eval",
        tags=[t for t in args.tags.split(",") if t],
        id=hashlib.sha1(f"eval:{scenario}".encode()).hexdigest()[:32],
        resume="allow",
        dir=args.wandb_dir or str(args.lab),
        config={"scenario": scenario, "lab": str(args.lab)},
    )
    run.define_metric("eval/session")
    run.define_metric("eval/*", step_metric="eval/session")

    logged: list[dict] = []
    while True:
        sessions = read_sessions(args.lab)
        last = logged[-1]["session"] if logged else -1
        pending = sorted(position for position in sessions if position > last)
        for position in pending:
            point = session_point(position, sessions[position], logged)
            logged.append(point)
            run.log({f"eval/{key}": value for key, value in point.items()})
        if pending:
            print(
                f"learning_curve: logged up to s{logged[-1]['session']:03d} "
                f"(accepts {logged[-1]['accept_cum']}/{len(logged)})",
                flush=True,
            )
        if args.once:
            break
        time.sleep(args.poll_seconds)
    run.finish()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lab", type=Path, help="the REEF Eval lab directory ($RUN_DIR/lab)")
    parser.add_argument("--from-csv", type=Path, help="read the per-session rows from a CSV instead of a lab")
    parser.add_argument("--csv", type=Path, help="export the per-session rows to this CSV and exit")
    parser.add_argument("--plot", type=Path, help="draw the learning curve to this PNG and exit")
    parser.add_argument("--max-session", type=int, default=None, help="last session index the figure shows")
    parser.add_argument("--reef-url", help="Reef's URL, to find the scenario (logging mode)")
    parser.add_argument("--reef-token", help="Reef's bearer token (logging mode)")
    parser.add_argument("--project", default="reef")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group-prefix", default="openclawrl")
    parser.add_argument("--name", default=None, help="run name (default: <group-prefix>-eval-<scenario tail>)")
    parser.add_argument("--tags", default="openclawrl,gsm8k-stream,eval")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--once", action="store_true", help="log what the lab holds and exit")
    parser.add_argument("--scenario", default=None, help="skip the status query (backfilling a finished run)")
    parser.add_argument("--wandb-dir", default=None, help="where W&B writes its run files (default: the lab dir)")
    args = parser.parse_args()

    if args.csv or args.plot:
        if args.from_csv:
            rows = read_csv(args.from_csv)
        elif args.lab:
            rows = session_rows(read_sessions(args.lab))
        else:
            parser.error("--csv/--plot need --lab or --from-csv")
        if not rows:
            print("learning_curve: no sessions found", file=sys.stderr)
            return 1
        if args.csv:
            write_csv(rows, args.csv)
            print(f"learning_curve: wrote {args.csv} ({len(rows)} sessions)")
        if args.plot:
            plot(rows, args.plot, args.max_session)
            print(f"learning_curve: wrote {args.plot}")
        return 0

    if args.lab is None or (args.scenario is None and (args.reef_url is None or args.reef_token is None)):
        parser.error("logging mode needs --lab and either --scenario or --reef-url with --reef-token")
    return log_to_wandb(args)


if __name__ == "__main__":
    raise SystemExit(main())
