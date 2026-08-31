"""The minimal harness's search loop, run end to end against the real
circle-packing judge on this machine, guarding the example the README
points beginners at."""

import sys
from pathlib import Path

import pytest

from reef_eval import Lab, LocalExecutor

ROOT = Path(__file__).parent.parent
TASK = str(ROOT / "tasks" / "autoresearch" / "first-party" / "circle-packing")
SEARCH = ROOT / "examples" / "random_search.py"


async def test_random_search_speaks_the_judge_protocol(tmp_path):
    lab = Lab(tmp_path / "lab", executor=LocalExecutor(root=tmp_path))
    row = await lab.run(
        TASK,
        {
            "command": f"{sys.executable} {SEARCH}",
            "override_timeout_sec": 3,
            # Blind random search draws a valid packing about 3% of the time,
            # so pin the seed: this one is valid on its first submission.
            "env": {"SEED": "10"},
        },
    )
    assert row.rewards["reward"] > 0  # a valid packing scored
    trace = lab.df("trace")
    assert len(trace) > 0  # every submission is in the log
    assert trace["score"].max() == pytest.approx(row.rewards["reward"])
