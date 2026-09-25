"""SDPO on SciKnowEval Chemistry, through reef-eval.

One episode is one training run:

    train  — reef-eval runs the ``harbor/chemistry`` task under our agent; the
             agent runs the stage in the task container, which samples each
             step's 32x8 grid through Reef, reports every rollout with its
             coordinates, waits for the step's training release, and evaluates
             avg@16 on the test split every few steps
    verify — Harbor's isolated verifier reads the run's curve and reports its
             last avg@16

``Lab.run`` is reef-eval's one primitive: task in, trusted scored row out. The
Reef stack is already serving; ``run.sh`` starts it from ``serve.yaml`` first.
"""

import asyncio
import os
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
#: Reef serves the trained weights under this name; the stage asks for it.
MODEL = "reef"

lab = Lab(Path(os.environ.get("SDPO_RUN_DIR", HERE / "work")) / "lab")
agent = {"name": "harness:HarborAgent", "model_name": MODEL}
row = asyncio.run(lab.run(str(HERE / "harbor" / "chemistry"), agent))
print(f"episode reward: {row.rewards}")
