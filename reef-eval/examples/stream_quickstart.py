"""A continual-learning stream in 30 seconds, no setup.

The agent is simulated: each task it writes one line into its carried
state directory and scores by how many lines it has accumulated, so the
reward rises only because the state travels from task to task. Swap in
real tasks and a real agent when you have Docker; see the README.

    python examples/stream_quickstart.py
"""

import asyncio
from pathlib import Path

from reef_eval import FakeExecutor, Lab, Stream, metrics


def remembering_agent(spec):
    """Score by how much the carried state has accumulated."""
    memory = Path(spec.overrides["state_dir"]) / "memory.txt"
    seen = memory.read_text().splitlines() if memory.exists() else []
    seen.append(spec.task)
    memory.write_text("\n".join(seen) + "\n")
    return {"reward": len(seen) / 10}


async def main():
    lab = Lab("runs/stream-quickstart", executor=FakeExecutor(score=remembering_agent))
    stream = Stream("demo", [f"tasks/demo-{i}" for i in range(1, 6)])
    await stream.run(lab, agent={"name": "fake"})

    curve = metrics.learning_curve(lab.df("episode"), by=["stream"])
    print(curve[["position", "reward", "cum_mean"]].to_string(index=False))
    print()
    print("The reward rises only because the state directory is carried.")
    print("Run this script again and every position is skipped: that is resume.")


if __name__ == "__main__":
    asyncio.run(main())
