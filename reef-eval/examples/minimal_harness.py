"""The smallest complete agent harness: ~25 lines of adapter, no LLM.

The adapter is this file; the method it runs is ``random_search.py``.
Together they show the whole integration path: put your method in the
container, point it at the judge, and get back a trusted score plus its
submission log as trace rows. Requires Docker and
``pip install reef-eval[harbor]``.

    python examples/minimal_harness.py
"""

import asyncio
import tempfile
from pathlib import Path

from harbor.agents.base import BaseAgent

from reef_eval import Lab, metrics

TASK = str(
    Path(__file__).parent.parent
    / "tasks"
    / "autoresearch"
    / "first-party"
    / "circle-packing"
)
SEARCH_SCRIPT = Path(__file__).with_name("random_search.py")


class RandomSearchHarness(BaseAgent):
    """Uploads the search script and runs it. That is the entire adapter."""

    @staticmethod
    def name() -> str:
        return "random-search"

    def version(self) -> str | None:
        return "0.1"

    async def setup(self, environment) -> None:
        pass  # pure stdlib, nothing to install

    async def run(self, instruction, environment, context) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "search.py").write_text(SEARCH_SCRIPT.read_text())
            await environment.upload_dir(source_dir=tmp, target_dir="/opt/harness")
        # JUDGE_URL is already in the container's environment (set by the
        # task's docker-compose); only the time budget needs passing.
        await environment.exec(
            command="python /opt/harness/search.py",
            env={"BUDGET_SEC": "30"},
        )


async def main():
    lab = Lab("runs/minimal-harness")
    row = await lab.run(
        TASK,
        agent={"import_path": "minimal_harness:RandomSearchHarness"},
        tags={"harness": "random-search"},
    )
    if row.tags.get("error"):
        raise SystemExit(f"the episode failed: {row.tags['error']}")
    print("trusted reward:", row.rewards)  # the judge's final score

    trace = lab.df("trace")
    if not trace.empty:
        curve = metrics.anytime(trace)
        print("\nthe submission log, as ingested:")
        print(curve[["t", "score", "best_so_far"]].to_string(index=False))
        print("anytime AUC:", round(metrics.auc(curve), 4))


if __name__ == "__main__":
    asyncio.run(main())
