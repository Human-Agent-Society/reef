"""SDPO on an arithmetic grid, through reef-eval.

One episode:

    train  — reef-eval runs the ``harbor/`` task under our agent; the agent
             runs the grid runner in the task container, which samples each
             step's grid through Reef, reports every rollout with its
             coordinates, and waits for the step's training release
    verify — Harbor's isolated verifier reads the run's record and scores the
             last grid

``Lab.run`` is reef-eval's one primitive: task in, trusted scored row out. The
Reef stack is already serving; ``run.sh`` starts it from ``serve.yaml`` first.
"""

import asyncio
import os
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
#: Reef serves the trained weights under this name; the runner asks for it.
MODEL = "reef"

lab = Lab(Path(os.environ.get("RUN_DIR", HERE / "work")) / "lab")
agent = {"name": "harness:HarborAgent", "model_name": MODEL}
row = asyncio.run(lab.run(str(HERE / "harbor"), agent))
print(f"episode reward: {row.rewards}")
