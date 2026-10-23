"""Write a verified starting file for a FrontierCS 2 task whose shipped file is a stub.

Some tasks ship their editable file with the removed method replaced by
``raise NotImplementedError``. That file cannot enter the archive, because it
scores zero. This asks the frozen executor once for a simple working version
of the stub, through the same execution prompt the search uses, scores it with
the task judge, and keeps the first candidate the judge accepts. Run it after
``prepare.py`` with the judge and the executor configured as for ``run.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harness.config import FRONTIERCS2_TASKS, RunConfig
from harness.execution import OpenAICompatibleExecutionClient
from harness.prompts import build_execution_prompt, extract_terminal_tag_or_none
from harness.scorer import JudgeScorer, extract_solution_code
from harness.state import LibraryEntry, LLMRequest

GUIDANCE = (
    "Implement every removed interface of the file with a simple, standard and correct method, so that the "
    "file runs on every workload of the statement and scores above zero. Keep the required signatures and "
    "the validity rules. Do not reproduce the method of the paper the task comes from."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    config = RunConfig.load()
    if config.task not in FRONTIERCS2_TASKS:
        raise SystemExit(f"{config.task} is not a FrontierCS 2 task")
    spec = json.loads((Path(__file__).parent / "judges/frontiercs2_tasks.json").read_text())[config.task]
    suffix = Path(spec["editable_file"]).suffix
    target = config.state_dir / f"bootstrap{suffix}"
    if target.exists():
        raise SystemExit(f"{target} exists; remove it to write a new bootstrap")
    # prepare.py checks out the pinned repository here.
    problems = Path(__file__).parent / "work/FrontierCS-2/tasks/frontier-cs-2.0-demo/problems"
    repo = next(problems.glob(f"*/{config.task}/agent-base/frontier/task/repo"))
    contract = config.contract()
    parent = LibraryEntry(
        id="stub", parent_id=None, problem_id=config.task, timestep=0, guidance="", execution_thinking="",
        solution=(repo / spec["editable_file"]).read_text(), verifier_reward=0.0, verifier_raw_score=None,
        verifier_status="invalid", verifier_message="the shipped stub raises NotImplementedError",
        summary="The shipped stub.", reusable_idea="", failure_mode=None,
    )  # fmt: skip
    prompt = build_execution_prompt(
        problem_prompt=contract.problem_prompt,
        selected_entry=parent,
        guidance=GUIDANCE,
        solution_language=contract.solution_language,
        solution_contract=contract.solution_contract,
        score_direction=contract.score_direction,
        raw_score_label=contract.raw_score_label,
        execution_context=contract.execution_context,
    )
    client = OpenAICompatibleExecutionClient(config.backend)
    scorer = JudgeScorer(
        config.judge_url,
        problem_id=contract.judge_problem_id,
        language=contract.solution_language,
        timeout_s=config.verifier_timeout_s,
    )
    for attempt in range(1, args.attempts + 1):
        response = client.complete(
            LLMRequest(
                system=prompt.system,
                user=prompt.user,
                model=config.backend.model,
                temperature=config.backend.temperature,
                max_tokens=config.backend.max_tokens,
                metadata={"purpose": "frontiercs2_bootstrap", "attempt": attempt},
            )
        )
        code = extract_solution_code(response.text, contract.solution_language)
        summary = extract_terminal_tag_or_none(response.text, "summary")
        if not code or not summary:
            print(f"attempt {attempt}: the response has no complete solution and summary")
            continue
        result = scorer(code)
        print(f"attempt {attempt}: {result.status} score={result.raw_score} {result.message.splitlines()[0]}")
        if result.valid:
            target.write_text(code if code.endswith("\n") else code + "\n")
            target.with_suffix(".md").write_text(summary.strip() + "\n")
            provenance = {"executor": config.backend.safe_dict(), "attempt": attempt, "train_score": result.raw_score}
            target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
            print(f"wrote {target}")
            return
    raise SystemExit(f"no accepted bootstrap after {args.attempts} attempts")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
