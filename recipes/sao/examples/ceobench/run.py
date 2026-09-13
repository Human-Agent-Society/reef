"""One CEO-Bench episode, trained while it is played.

    solve  — reef-eval runs the harbor/ task under our agent; the benchmark's
             bash agent plays the configured number of days with every model
             call served by Reef
    learn  — when a week ends, the agent reports that week's turns with the
             week's cash change; Reef's SAO recipe trains one step per
             accepted turn while the agent is already playing the next week,
             and the engine serves the updated adapter
    verify — Harbor's verifier scores the finished run from its world.nmdb

This is test-time training: the policy adapts inside the episode it is
scored on. ``CEOBENCH_SEED`` (default ``42``) and ``CEOBENCH_DAYS`` (default
``500``) pick the episode. Replicates are independent runs from the base
model, one stack each: a Reef process trains one scenario for its lifetime,
so a second seed on the same stack would start from the first seed's adapter.
After the episode the loop waits for the scenario's version chain to stop
growing, so the adapter on disk is the one the episode ended with.
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
SEED = int(os.environ.get("CEOBENCH_SEED", "42"))
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
    agent = {"name": "harness:HarborAgent", "model_name": MODEL, "kwargs": {"seed": SEED, "days": DAYS}}
    row = await lab.run(str(HERE / "harbor"), agent, tags={"seed": SEED, "days": DAYS})
    print(f"seed {SEED}: reward {row.rewards}")
    if row.tags.get("error"):
        raise RuntimeError(f"Harbor trial failed: {row.tags['error']}")
    print(f"    trained: {wait_for_training()} releases committed")


asyncio.run(main())
