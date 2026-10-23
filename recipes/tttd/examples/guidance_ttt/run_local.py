"""Run the selected Guidance-TTT trajectory without Harbor.

``run.py`` runs the trajectory inside one Harbor trial and scores the best
candidate from the task's container. A cluster that has Apptainer but no Docker
cannot enforce the trial's network allowlist. This driver runs the same
trajectory on the host, writes the best candidate to the state directory, and
asks the task judge for the final score itself. A FrontierCS 2 task is scored
on its held-out suite (``<task>:eval``); another task is scored again on its
search suite.
"""

import asyncio
import json
import logging

from reef_client import ReefClient

from harness.config import FRONTIERCS2_TASKS, RunConfig
from harness.harbor_agent import CLIENT_TIMEOUT_S, run_trajectory
from harness.scorer import SOURCE_FILES, JudgeScorer


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = RunConfig.load()
    client = ReefClient(config.service_url, token=config.token, timeout_s=CLIENT_TIMEOUT_S)
    instruction = (config.task_dir / "instruction.md").read_text()
    contract, solution, record = await run_trajectory(instruction, config.model, client, emit=logging.info)
    path = config.state_dir / SOURCE_FILES[contract.solution_language.lower()][0]
    path.write_text(solution + "\n")
    final_id = contract.judge_problem_id + (":eval" if config.task in FRONTIERCS2_TASKS else "")
    scorer = JudgeScorer(
        config.judge_url,
        problem_id=final_id,
        language=contract.solution_language,
        timeout_s=2 * config.verifier_timeout_s,
    )
    result = scorer(solution)
    final = {
        "reward": result.reward if result.valid else 0.0,
        "valid": result.valid,
        "raw_score": result.raw_score,
        "message": result.message,
        "beats_reference": result.artifacts.get("beats_reference"),
        "solution": str(path),
        "reef": record,
    }
    (config.state_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"final {final_id}: reward {final['reward']} beats_reference {final['beats_reference']}")


if __name__ == "__main__":
    asyncio.run(main())
