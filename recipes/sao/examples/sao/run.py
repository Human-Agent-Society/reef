"""The continual-learning loop, written out.

For each task, in order:

    solve  — tide runs the task under our agent (six attempts, each an
             inference through Reef)
    verify — Harbor's isolated verifier scores every attempt
    learn  — the agent reports each score against its receipt; Reef's SAO
             recipe trains on the rollouts, and the *next* task is served
             by the updated weights

The ordering is the experiment: task N+1 measures what task N taught.
"""

import asyncio
from pathlib import Path

from tide import Lab

MODEL = "reef"  # model name the agent sends; Reef's SGLang serves it

HERE = Path(__file__).resolve().parent
TASKS = ["imo-4", "imo-8", "imo-12"]
AGENT = {"name": "harness:HarborAgent", "model_name": MODEL}


async def main():
    lab = Lab(HERE / "work" / "lab")
    for position, name in enumerate(TASKS):
        row = await lab.run(str(HERE / "harbor" / name), AGENT, tags={"position": position})
        print(f"[{position}] {name}: reward {row.rewards}")


asyncio.run(main())
