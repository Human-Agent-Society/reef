"""The summary-only Guidance-TTT loop, in the open.

One episode:

    guide   — reef-eval runs the harbor/ task under our agent; every rollout asks
              the trainable Qwen policy for guidance through Reef, which
              records the exact tokens and returns a receipt
    execute — a frozen external model turns that guidance into a complete
              C++17 candidate; its weights are never trained
    verify  — the external FrontierCS/go-judge scores the candidate
    learn   — the agent reports each score against its exact guidance
              receipt; Reef's TTTD recipe waits for one complete step grid
              (``groups_per_step`` x ``rollouts_per_group``), updates the
              rank-32 LoRA adapter, and serves it

``Lab.run`` is reef-eval's one primitive: task in, trusted scored row out.
"""

import asyncio
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE / "harbor" / "polyomino_packing"
STATE_DIR = HERE / "work" / "polyomino_packing"
AGENT = {"name": "harness:HarborAgent", "model_name": "Qwen/Qwen3-8B"}


async def main() -> None:
    lab = Lab(STATE_DIR / "lab")
    row = await lab.run(str(TASK_DIR), AGENT)
    if error := row.tags.get("error"):
        raise RuntimeError(f"Harbor trial failed: {error}")
    if not row.rewards:
        raise RuntimeError(f"Harbor trial returned no rewards: {row.uri}")
    print(f"episode reward: {row.rewards}")


asyncio.run(main())
