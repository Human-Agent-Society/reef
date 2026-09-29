"""Run one Harbor trial containing the selected Guidance-TTT trajectory."""

import asyncio

from reef_eval import Lab

from harness.config import RunConfig


async def main() -> None:
    config = RunConfig.load()
    lab = Lab(config.state_dir / "lab")
    row = await lab.run(str(config.task_dir), {"name": "harness:HarborAgent", "model_name": config.model})
    if error := row.tags.get("error"):
        raise RuntimeError(f"Harbor trial failed: {error}")
    if not row.rewards:
        raise RuntimeError(f"Harbor trial returned no rewards: {row.uri}")
    print(f"episode reward: {row.rewards}")


if __name__ == "__main__":
    asyncio.run(main())
