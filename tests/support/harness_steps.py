"""Shared pieces of the harness evolution tests.

A fake pi binary that writes its rendered rules into its session, the
deployment's model binding, a pi seed tree, an in-memory releases reader of
one component, and a recipe and a backend built over them. Rollouts run the
fake binary through the real adapter path, so render, launch and the
trajectory read are the production code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tests.support._strategies import resolve_proposer, resolve_rollout_scorer

from reef.artifact.artifact import ArtifactNotFound
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.rollouts.run import RolloutResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.train.backend import CommittedReleases, ComponentReleases, ReleaseFiles
from reef.train.harness_backend.backend import HarnessBackend

#: The fake pi scores itself: its session event carries the rendered rules text, so a scorer can prefer a tree whose
#: rules hold the marker.
PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = agent_dir / "AGENTS.md"
event = {"type": "agent_end", "prompt": prompt, "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

#: The deployment's model binding: rendered into rollouts at run time, never into a tree.
MODEL = ModelBinding(base_url="http://localhost:8000", model="qwen3-8b", api_key="dummy")

#: A pi seed: one setting, and rules without the marker.
PI_SEED = HarnessTree(
    {
        "config/primary.json": json.dumps({"defaultThinkingLevel": "off"}) + "\n",
        "rules.md": "Answer briefly.\n",
    }
)

#: The component a flat harness scenario evolves.
COMPONENT = "harness"


def make_pi_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def runtime() -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path=MODEL.model, base_url=MODEL.base_url, api_key=MODEL.api_key)


def marker_score(task: str, result: RolloutResult) -> float:
    """1.0 when the rollout's rendered rules carry ``marker``."""
    return 1.0 if "marker" in result.trajectory[-1]["rules"] else 0.0


class TreeReleases(CommittedReleases):
    """An in-memory commit history of one component: its releases by id, the head the latest one.

    ``base`` is the base release; ``publish`` appends one, as a commit that publishes for the component does.
    """

    def __init__(self, base: HarnessTree) -> None:
        self.trees: dict[str, HarnessTree] = {"base": base}
        self.order = ["base"]

    def publish(self, release_id: str, tree: HarnessTree) -> None:
        self.trees[release_id] = tree
        self.order.append(release_id)

    def head(self, component: str) -> ReleaseFiles:
        return self.release(self.order[-1], component)

    def release(self, release_id: str, component: str) -> ReleaseFiles:
        if release_id not in self.trees:
            raise ArtifactNotFound(f"no release {release_id!r}")
        return ReleaseFiles(release_id, dict(self.trees[release_id].files))


def harness_recipe(
    tmp_path: Path,
    propose: Any,
    *,
    seed: HarnessTree = PI_SEED,
    score: Any = marker_score,
    tasks: tuple[str, ...] = ("task one",),
    reads_requests: bool = False,
    **fields: Any,
) -> HarnessEvolutionRecipe:
    """A pi recipe over the fake binary; ``fields`` set any other recipe field."""
    return HarnessEvolutionRecipe(
        resolve_proposer(propose, reads_requests=reads_requests),
        resolve_rollout_scorer(score),
        tasks,
        binary=str(make_pi_binary(tmp_path)),
        seed=seed,
        runtime=runtime(),
        proposals_dir=str(tmp_path / "proposals"),
        **fields,
    )


def harness_backend(
    tmp_path: Path, propose: Any, *, scenario: str = "demo", **recipe_fields: Any
) -> tuple[HarnessBackend, TreeReleases]:
    """The backend ``harness_recipe`` builds for ``scenario``, over an in-memory history whose base is the seed."""
    recipe = harness_recipe(tmp_path, propose, **recipe_fields)
    releases = TreeReleases(recipe.seed)
    return recipe.build_backend(scenario, ComponentReleases(releases, COMPONENT), None), releases


def changed_rules(text: str) -> Mapping[str, str]:
    """A patch body that rewrites the rules."""
    return {"rules.md": text}


__all__ = [
    "COMPONENT",
    "MODEL",
    "PI_FAKE",
    "PI_SEED",
    "TreeReleases",
    "changed_rules",
    "harness_backend",
    "harness_recipe",
    "make_pi_binary",
    "marker_score",
    "runtime",
]
