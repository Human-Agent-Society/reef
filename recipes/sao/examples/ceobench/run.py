"""The loop, written out: one CEO-Bench episode per seed, trained between seeds.

For each seed, in order:

    solve  — reef-eval runs the harbor/ task under our agent; the benchmark's
             bash agent plays the configured number of days with every model
             call served by Reef
    verify — Harbor's verifier scores the run from its world.nmdb
    learn  — the agent reports the episode score against every turn's receipt;
             Reef's SAO recipe trains one step per accepted turn, and the next
             seed is served by the updated weights

``CEOBENCH_SEEDS`` (comma separated, default ``42``) and ``CEOBENCH_DAYS``
(default ``500``) pick the episodes. After each episode the loop waits for
the scenario's version chain to stop growing, so the next seed measures what
this one taught; stale turns the recipe declines are not waited for.
"""

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from reef_eval import Lab

MODEL = "reef"  # model name the agent sends; Reef's SGLang serves it

HERE = Path(__file__).resolve().parent
SERVICE_URL = os.environ["REEF_SERVICE_URL"].rstrip("/")
SCENARIO = os.environ.get("REEF_SCENARIO", "ceobench-sao")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")
SEEDS = [int(seed) for seed in os.environ.get("CEOBENCH_SEEDS", "42").split(",") if seed.strip()]
DAYS = int(os.environ.get("CEOBENCH_DAYS", "500"))
#: Training is quiescent once the release count holds for this long.
TRAIN_QUIET_S = 120.0
#: Ceiling on waiting for one episode's training steps before moving on with a warning.
TRAIN_DRAIN_TIMEOUT_S = 7200.0


#: The service is gone or rejecting requests; waiting cannot help.
_SERVICE_GONE = -1


def training_release_count() -> int | None:
    """Training releases committed so far; ``None`` while the service is busy.

    The scenario's registry lock serializes release reads with training, so a
    timeout is "try again". A refused connection or an HTTP rejection is
    terminal: the stack is gone, or this loop is talking to something that is
    not its deployment.
    """
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError:
        return _SERVICE_GONE
    except urllib.error.URLError as error:
        if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
            return _SERVICE_GONE
        return None
    except TimeoutError:
        return None
    return sum(1 for row in payload["releases"] if row.get("operation") == "training")


def wait_for_training() -> int:
    """Block until the training release count stops moving; return that count."""
    deadline = time.time() + TRAIN_DRAIN_TIMEOUT_S
    last, last_change = training_release_count(), time.time()
    while time.time() < deadline:
        if last == _SERVICE_GONE:
            print("    WARNING: the Reef service is not reachable; skipping the training drain")
            return 0
        time.sleep(10)
        current = training_release_count()
        if current is not None and current != last:
            last, last_change = current, time.time()
        elif time.time() - last_change >= TRAIN_QUIET_S:
            return last or 0
    print(f"    WARNING: training still moving after {TRAIN_DRAIN_TIMEOUT_S}s")
    return last or 0


async def main():
    lab = Lab(HERE / "work" / "lab")
    for position, seed in enumerate(SEEDS):
        agent = {"name": "harness:HarborAgent", "model_name": MODEL, "kwargs": {"seed": seed, "days": DAYS}}
        row = await lab.run(str(HERE / "harbor"), agent, tags={"position": position, "seed": seed, "days": DAYS})
        print(f"[{position}] seed {seed}: reward {row.rewards}")
        if row.tags.get("error"):
            raise RuntimeError(f"Harbor trial failed: {row.tags['error']}")
        print(f"    trained: {wait_for_training()} releases committed so far")


asyncio.run(main())
