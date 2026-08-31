"""Every first-party task, end to end through its real judge, no Docker.

For each task: start its judge locally, submit the reference solution from
grader_tests.json, and require the final grade to match. The session
score, or the final judge's number where the task has one (this is what
proves symbolic-regression's held-out final judge actually runs).
"""

import json
import sys
from pathlib import Path

import pytest

from reef_eval import Lab, LocalExecutor

TASKS = sorted(
    (Path(__file__).parent.parent / "tasks" / "autoresearch" / "first-party").iterdir()
)


@pytest.mark.parametrize("task_dir", TASKS, ids=lambda p: p.name)
async def test_oracle_through_the_local_judge(task_dir, tmp_path):
    grader_tests = json.loads((task_dir / "tests" / "grader_tests.json").read_text())
    oracle = grader_tests["cases"][0]
    payload = tmp_path / "solution.json"
    payload.write_text(json.dumps(oracle["solution"]))

    submit = (
        f'{sys.executable} -c "import os,urllib.request;'
        f"urllib.request.urlopen(urllib.request.Request("
        f"os.environ['JUDGE_URL']+'/submit',"
        f"data=open('{payload}','rb').read()),timeout=60)\""
    )
    lab = Lab(tmp_path / "lab", executor=LocalExecutor(root=tmp_path))
    row = await lab.run(str(task_dir), {"command": submit, "override_timeout_sec": 60})

    final_cases = grader_tests.get("final_cases")
    expected = final_cases[0]["reward"] if final_cases else oracle["reward"]
    assert row.rewards["reward"] == pytest.approx(expected, abs=1e-6)
    assert len(lab.df("trace")) == 1  # the one submission is in the log
