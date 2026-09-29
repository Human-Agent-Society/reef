"""SDPO on SciKnowEval Chemistry, through reef-eval.

One episode is one training run:

    train  — reef-eval runs the ``harbor/chemistry`` task under our agent; the
             agent runs the stage in the task container, which samples each
             step's 32x8 grid through Reef, reports every rollout with its
             coordinates, waits for the step's training release, and evaluates
             avg@n on the test split every few steps
    verify — Harbor's isolated verifier reads the run's curve and reports its
             last avg@n

``Lab.run`` is reef-eval's one primitive: task in, trusted scored row out. The
row is keyed by the Reef scenario and the settings that shape the run, so a
run with other settings is a new episode and a rerun of a recorded one is
skipped. A scenario keeps the weights it trained, so a new run gets a new
``REEF_SCENARIO``. The Reef stack is already serving; ``run.sh`` starts it
from ``serve.yaml`` first.
"""

import asyncio
import os
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
#: Reef serves the trained weights under this name; the stage asks for it.
MODEL = "reef"
SCENARIO = os.environ.get("REEF_SCENARIO", "sdpo-chemistry")
#: The host settings the stage reads in the container; with the scenario they name the run's row.
RUN_SETTINGS = (
    "SDPO_STEPS",
    "SDPO_TRAINING_HOURS",
    "SDPO_SEED",
    "SDPO_EPOCHS",
    "SDPO_GROUPS_PER_STEP",
    "SDPO_ROLLOUTS_PER_GROUP",
    "SDPO_EVAL_EVERY",
    "SDPO_EVAL_SAMPLES",
)

settings = {name: os.environ[name] for name in RUN_SETTINGS if name in os.environ}
key = ":".join(["chemistry", SCENARIO, *(f"{name}={value}" for name, value in settings.items())])
lab = Lab(Path(os.environ.get("SDPO_RUN_DIR", HERE / "work")) / "lab")
agent = {"name": "harness:HarborAgent", "model_name": MODEL}
row = asyncio.run(lab.run(str(HERE / "harbor" / "chemistry"), agent, tags={"scenario": SCENARIO, **settings}, key=key))
print(f"episode {key}: {row.rewards}")
