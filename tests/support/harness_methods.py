"""Shared pieces of the Meta-Harness tests: a canned proposer model and one scored setup.

``QueueChat`` is the proposer model, ``proposal_reply`` one reply it writes,
and ``meta_harness_sections`` the recipe section over the fake pi binary of
``tests.support.harness_steps``, scored by ``score_marker``. ``report_once``
records one scored trace, enough to wake one step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.support._strategies import resolve_episode_scorer
from tests.support._trajectories import recorded_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import recorded_payload
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.scenario.scenario import Scenario

#: What the scorer looks for in the rules an episode ran with.
MARKER = "META-HARNESS-WINNER"
#: The seed tree, and two improvements of it that carry the marker.
SEED = HarnessTree({"rules.md": "Answer carefully.\n"})
IMPROVED = HarnessTree({"rules.md": f"Answer carefully. {MARKER}\n"})
ALTERNATE = HarnessTree({"rules.md": f"Answer carefully. {MARKER} Check twice.\n"})
TASK = "Solve the supplied task."
#: One failing trace of the task, with its report's feedback.
SAMPLE = recorded_trajectory(
    "trace-1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "attempt"}}]},
    },
    0.0,
    feedback="the attempt failed",
)


class QueueChat(ModelBinding):
    """The proposer model: it answers the canned replies in order and keeps every prompt it was asked."""

    def __init__(self, *replies: str) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="proposer", api_key="dummy")
        object.__setattr__(self, "replies", list(replies))
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def proposal_reply(parent: HarnessTree, tree: HarnessTree, *, hypothesis: str = "improve rules") -> str:
    """The proposer's reply: ``tree`` whole, branched from ``parent``."""
    return json.dumps(
        {
            "parent_id": parent.checksum(),
            "hypothesis": hypothesis,
            "changes": "revise the general instruction",
            "files": dict(tree.files),
        }
    )


def score_marker(task: str, result: EpisodeResult) -> float:
    """1.0 when the episode's rendered rules carry ``MARKER``."""
    last = result.trajectory[-1] if result.trajectory else {}
    return 1.0 if MARKER in str(last.get("rules", "")) else 0.0


def served_runtime(api: str = "openai") -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path="served-model", base_url="http://127.0.0.1:9", api_key="dummy", api=api)


def meta_harness_sections(
    tmp_path: Path, *, scorer: Any = score_marker, adapter: str = "pi", **method: Any
) -> dict[str, Any]:
    """A Meta-Harness recipe section over the fake pi binary in ``tmp_path``, the seed written beside it."""
    seed = tmp_path / "seed"
    if not seed.exists():
        SEED.write(seed)
    return {
        "implementation": "recipes.meta_harness.recipe:MetaHarnessRecipe",
        "model": {"path": "served-model"},
        "evolution": {
            "adapter": adapter,
            "binary": str(tmp_path / "fake-pi"),
            "evaluate": resolve_episode_scorer(scorer),
            "tasks": [TASK],
            "seed": str(seed),
            "meta_harness": dict(method),
            "batch_size": 1,
        },
    }


def report_once(scenario: Scenario, name: str, suffix: str) -> None:
    """One recorded trace and its report: enough to wake one step."""
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.INFERENCE,
            payload=recorded_payload(SAMPLE),
            agent_record_id=f"i{suffix}",
        )
    )
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.REPORT,
            payload={"score": 0.0, "feedback": SAMPLE.metadata.get("feedback"), "references": [f"i{suffix}"]},
            agent_record_id=f"r{suffix}",
        )
    )


__all__ = [
    "ALTERNATE",
    "IMPROVED",
    "MARKER",
    "SAMPLE",
    "SEED",
    "TASK",
    "QueueChat",
    "meta_harness_sections",
    "proposal_reply",
    "report_once",
    "score_marker",
    "served_runtime",
]
