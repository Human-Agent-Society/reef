"""The continual-learning loop, written out.

For each task, in order:

    solve  — reef-eval runs the task under our agent (six attempts, each an
             inference through Reef)
    verify — Harbor's isolated verifier scores every attempt
    learn  — the agent reports each score against its receipt; Reef's SAO
             recipe trains on the rollouts, and the *next* task is served
             by the updated weights

The ordering is the experiment: task N+1 measures what task N taught. That
only holds if task N's rollouts have actually trained before task N+1
starts, so after each rollout the loop blocks until the scenario's version
log carries one ``training`` step per scored rollout so far. Rollouts
are quick at this model size while a train step also saves a checkpoint and
publishes weights; without the barrier the rollout loop finishes ahead of
training and the process exit tears the stack down over a queue of
never-trained rollouts.
"""

import asyncio
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from reef_eval import Lab

MODEL = "reef"  # model name the agent sends; Reef's SGLang serves it

#: The drain below reads the same deployment the agent talks to. Copied from
#: harness/agent.py rather than imported, because the harness is installed on
#: its own into reef-eval's environment while run.py stays host-side.
SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
SCENARIO = "sao-smoke"
ROLLOUTS = 6

HERE = Path(__file__).resolve().parent
TASKS = ["imo-4", "imo-8", "imo-12"]
AGENT = {"name": "harness:HarborAgent", "model_name": MODEL}
#: Ceiling on waiting for one task's six training steps (checkpoint saves
#: and weight publishes included) before failing the run.
TRAIN_DRAIN_TIMEOUT_S = 1800

#: The service is gone or rejecting requests; waiting cannot help.
_SERVICE_GONE = -1


def training_step_count() -> int | None:
    """Training steps committed so far; ``None`` while the service is busy.

    The scenario's registry lock serializes status reads with training, so
    this request legitimately stalls for the length of an in-flight train
    step (which includes a checkpoint save and a weight publish). A timeout
    is "try again", not an error. A refused connection or an HTTP rejection
    is terminal: the stack is gone or this loop is talking to something that
    is not its deployment, and waiting cannot change either.
    """
    request = urllib.request.Request(f"{SERVICE_URL}/reef/status", headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError:
        return _SERVICE_GONE  # answered and rejected: not our deployment
    except urllib.error.URLError as error:
        if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
            return _SERVICE_GONE
        return None  # stalled behind a train step — try again
    except TimeoutError:
        return None
    # This run activates nothing and sets no training start, so each commit of the scenario is a training step.
    scenario = payload["scenarios"].get(SCENARIO)
    return 0 if scenario is None else int(scenario["scenario_step"])


def wait_for_training(expected: int) -> None:
    """Block until ``expected`` rollouts have each committed a training step."""
    deadline = time.time() + TRAIN_DRAIN_TIMEOUT_S
    trained = None
    while time.time() < deadline:
        trained = training_step_count()
        if trained == _SERVICE_GONE:
            raise RuntimeError("The Reef service is not reachable during the training drain")
        if trained is not None and trained >= expected:
            print(f"    trained: {trained}/{expected} rollouts committed")
            return
        time.sleep(10)
    raise TimeoutError(f"Only {trained}/{expected} rollouts trained within {TRAIN_DRAIN_TIMEOUT_S}s")


async def main():
    lab = Lab(HERE / "work" / "lab")
    for position, name in enumerate(TASKS):
        row = await lab.run(str(HERE / "harbor" / name), AGENT, tags={"position": position})
        print(f"[{position}] {name}: reward {row.rewards}")
        wait_for_training(expected=(position + 1) * ROLLOUTS)


if __name__ == "__main__":
    asyncio.run(main())
