"""Shared pieces of the harness method tests: a canned model, a scorer, and the Meta-Harness deployment.

``QueueChat`` is a model that answers canned replies: the Meta-Harness, ACE
and ReasoningBank tests give it to their method. ``proposal_reply`` is one
reply of the Meta-Harness proposer. ``MarkerScorer`` scores a run of the
fake pi binary of ``tests.support.harness_steps`` by the marker in the rules
it rendered. ``meta_harness_sections`` is a deployment's recipe section over
that binary, and ``meta_harness_recipe`` the recipe built from it with a
``QueueChat``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from tests.support.harness_steps import make_pi_binary

from reef.evaluation.harness import TaskScorer
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.run import TaskResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.recipe.registry import build_recipe

#: What the scorer looks for in the rules a run completed with.
MARKER = "META-HARNESS-WINNER"
#: The seed tree, and two improvements of it that carry the marker.
SEED = HarnessTree({"rules.md": "Answer carefully.\n"})
IMPROVED = HarnessTree({"rules.md": f"Answer carefully. {MARKER}\n"})
ALTERNATE = HarnessTree({"rules.md": f"Answer carefully. {MARKER} Check twice.\n"})
TASK = "Solve the supplied task."


class QueueChat(ModelBinding):
    """A method's model: it answers the canned replies in order and keeps every prompt it was asked."""

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


class MarkerScorer(TaskScorer):
    """1.0 when the run's rendered rules carry ``MARKER``, else 0.0."""

    def score(self, task: str, result: TaskResult) -> float:
        last = result.trajectory[-1] if result.trajectory else {}
        return 1.0 if MARKER in str(last.get("rules", "")) else 0.0


def served_runtime(api: str = "openai") -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path="served-model", base_url="http://127.0.0.1:9", api_key="dummy", api=api)


def meta_harness_sections(tmp_path: Path, *, adapter: str = "pi", **method: Any) -> dict[str, Any]:
    """A Meta-Harness deployment's recipe section over the fake pi binary in ``tmp_path``, the seed written beside it.

    ``method`` is the method's config. The evaluation compares means, as Meta-Harness selects.
    """
    seed = tmp_path / "seed"
    if not seed.exists():
        SEED.write(seed)
    return {
        "implementation": "reef.recipe.harness:HarnessEvolutionRecipe",
        "model": {"path": "served-model"},
        "evolution": {
            "adapter": adapter,
            "binary": str(tmp_path / "fake-pi"),
            "seed": str(seed),
            "state_dir": str(tmp_path / "harness"),
            "method": {"module": "recipes.meta_harness:MetaHarness", "config": dict(method)},
            "batch_size": 1,
            "evaluation": {
                "tasks": [{"id": TASK, "instruction": TASK}],
                "scorer": "tests.support.harness_methods:MarkerScorer",
                "policy": "compare",
                "statistic": "mean",
            },
        },
    }


def meta_harness_recipe(tmp_path: Path, chat: QueueChat, **method: Any) -> HarnessEvolutionRecipe:
    """The recipe ``meta_harness_sections`` names, built over the fake pi binary, with ``chat`` as its proposer."""
    make_pi_binary(tmp_path)
    config = meta_harness_sections(tmp_path, **method)
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=served_runtime())
    if not isinstance(recipe, HarnessEvolutionRecipe):
        raise TypeError(f"the section built a {type(recipe).__name__}")
    return dataclasses.replace(recipe, models={"proposer": chat})


__all__ = [
    "ALTERNATE",
    "IMPROVED",
    "MARKER",
    "SEED",
    "TASK",
    "MarkerScorer",
    "QueueChat",
    "meta_harness_recipe",
    "meta_harness_sections",
    "proposal_reply",
    "served_runtime",
]
