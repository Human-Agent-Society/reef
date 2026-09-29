"""Check one bootstrap through the same judge adapter used by the search."""

import json

from harness.bootstrap import prepare_seed
from harness.config import RunConfig
from harness.scorer import JudgeScorer


def main() -> None:
    config = RunConfig.load()
    contract = config.contract()
    seed = prepare_seed(
        config,
        JudgeScorer(
            config.judge_url,
            problem_id=contract.judge_problem_id,
            language=contract.solution_language,
            timeout_s=config.verifier_timeout_s,
        ),
    )
    print(json.dumps({"task": config.task, "verified_seed": str(seed)}))


if __name__ == "__main__":
    main()
