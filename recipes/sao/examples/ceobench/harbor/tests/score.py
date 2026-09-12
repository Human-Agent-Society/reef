"""Score one CEO-Bench run from its ``world.nmdb``.

Usage: ``score.py <runs-dir> <verifier-dir>``. Finds the run directory the
harness left under ``<runs-dir>``, decrypts its ``world.nmdb`` with the
benchmark's own reader, and writes ``<verifier-dir>/reward.json``:

    reward         final cash over the starting balance (1.0 = break-even)
    final_cash     the benchmark's primary metric, in dollars
    survival_days  the last simulated day the run reached
    bankrupt       1 if the run ended below zero cash

The harness-level ``world.nmdb`` is the copy the runner makes at every
checkpoint; when a run was cut off before its first checkpoint, the live
session database is scored instead. Nothing here trusts the agent's own
accounting (config.json only supplies the starting balance).
"""

import json
import sys
from pathlib import Path

DAY_TABLES = ("config_history", "service_day", "ledger")


def find_run_dir(runs_dir: Path) -> Path:
    runs = sorted(path for path in runs_dir.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run_* directory under {runs_dir}")
    if len(runs) > 1:
        raise RuntimeError(f"expected one run under {runs_dir}, found {len(runs)}")
    return runs[0]


def find_world_db(run_dir: Path) -> Path:
    checkpointed = run_dir / "world.nmdb"
    if checkpointed.exists():
        return checkpointed
    live = sorted((run_dir / "agent_workspace" / "sessions").glob("*/world.nmdb"))
    if live:
        return live[-1]
    raise FileNotFoundError(f"no world.nmdb under {run_dir}")


def score(run_dir: Path) -> dict:
    from saas_bench.db_protection import load_session_db  # the checkout's own SQLCipher reader

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    initial_cash = float(config["initial_cash"])
    conn = load_session_db(find_world_db(run_dir))
    final_cash = float(conn.execute("SELECT COALESCE(SUM(amount), 0) FROM ledger").fetchone()[0])
    survival_days = 0
    for table in DAY_TABLES:
        last_day = conn.execute(f"SELECT COALESCE(MAX(day), 0) FROM {table}").fetchone()[0]
        survival_days = max(survival_days, int(last_day or 0))
    return {
        "reward": final_cash / initial_cash,
        "final_cash": final_cash,
        "survival_days": survival_days,
        "bankrupt": 1 if final_cash < 0 else 0,
    }


def main() -> None:
    runs_dir, verifier_dir = Path(sys.argv[1]), Path(sys.argv[2])
    verifier_dir.mkdir(parents=True, exist_ok=True)
    run_dir = find_run_dir(runs_dir)
    rewards = score(run_dir)
    (verifier_dir / "reward.json").write_text(json.dumps(rewards) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **rewards}))


if __name__ == "__main__":
    main()
