"""The whole loop, in the open.

One episode:

    solve  — REEF Eval runs the harbor/ task under our agent; the agent asks
             Reef for the answer (Reef records the exchange, returns a
             receipt)
    verify — Harbor's isolated verifier scores the attempt
    learn  — the agent reports the verifier score against the receipt;
             Reef trains on it and serves the next version

``Lab.run`` is REEF Eval's one primitive: task in, trusted scored row out.
"""

import asyncio
from pathlib import Path

from reef_eval import Lab

MODEL = "gpt-4o"  # provider model the recipe proxies to

HERE = Path(__file__).resolve().parent

lab = Lab(HERE / "work" / "lab")
agent = {"name": "harness:HarborAgent", "model_name": MODEL}
row = asyncio.run(lab.run(str(HERE / "harbor"), agent))
print(f"episode reward: {row.rewards}")
