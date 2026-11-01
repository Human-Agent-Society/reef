"""Shared pieces of the harness evolution tests.

A fake pi binary that writes its rendered rules into its session, the
deployment's model binding, a pi seed tree, an in-memory releases reader of
one component, and a recipe and a backend built over them. Tasks run the
fake binary through the real adapter path, so render, launch and the
trajectory read are the production code. ``evolution_config`` is a
deployment config for the tests that build a recipe from its config, over
the plugins below; ``traced`` makes trace samples of prompts; ``step_services``
gives a method that runs on a workspace of its own the backend's services.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self
from weakref import WeakKeyDictionary

from tests.support._strategies import resolve_method, resolve_task_scorer
from tests.support._trajectories import recorded_trajectory

from reef.artifact.artifact import ArtifactNotFound
from reef.core.job_data import TrajectorySample
from reef.core.plugins import PluginReference
from reef.evaluation.harness import EvaluationSettings, History, TaskCollector, TaskScorer
from reef.harness.adapters import get_adapter
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.harness.tasks.run import TaskResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.harness import EvolutionSettings, HarnessEvolutionRecipe
from reef.runtime.executor.config import ExecutorSettings
from reef.train.backend import CommittedReleases, ReleaseFiles
from reef.train.harness_backend import Method, StepContext, Workspace
from reef.train.harness_backend.agent.settings import AgentSettings
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.calls import StepCalls
from reef.train.harness_backend.services import StepServices
from reef.train.pipeline import PipelineConfig

#: The fake pi scores itself: its session event carries the rendered rules text, so a scorer can prefer a tree whose
#: rules hold the marker.
PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
package = Path(sys.argv[sys.argv.index("-e") + 1])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = package / "rules.md"
event = {"type": "agent_end", "prompt": prompt, "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

#: The deployment's model binding: rendered into runs at run time, never into a tree.
MODEL = ModelBinding(base_url="http://localhost:8000", model="qwen3-8b", api_key="dummy")

#: A pi seed: one setting, and rules without the marker.
PI_SEED = HarnessTree(
    {
        "config/primary.json": json.dumps({"defaultTools": []}) + "\n",
        "rules.md": "Answer briefly.\n",
    }
)

#: The component a flat harness scenario evolves.
COMPONENT = "harness"

#: Why a step skips when its method leaves the head tree as it was.
UNCHANGED = "the method left the head tree unchanged"


def make_pi_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def runtime() -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path=MODEL.model, base_url=MODEL.base_url, api_key=MODEL.api_key)


def marker_score(task: str, result: TaskResult) -> float:
    """1.0 when the run's rendered rules carry ``marker``."""
    return 1.0 if "marker" in result.trajectory[-1]["rules"] else 0.0


class TreeReleases(CommittedReleases):
    """An in-memory commit history of one component: its releases by id, the head the latest one.

    ``base`` is the root release; ``publish`` appends one, as a commit that publishes for the component does.
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
    method: Any,
    *,
    seed: HarnessTree = PI_SEED,
    score: Any = marker_score,
    tasks: tuple[str, ...] = ("task one",),
    evaluation: Mapping[str, Any] | None = None,
    worker_executor: ExecutorSettings | None = None,
    agent: AgentSettings | None = None,
    models: Mapping[str, ModelBinding] | None = None,
    activation: PluginReference | None = None,
    job_samples: int | None = None,
    **evolution: Any,
) -> HarnessEvolutionRecipe:
    """A pi recipe over the fake binary, its state under ``tmp_path``.

    ``evaluation`` sets ``EvaluationSettings`` fields beside ``tasks``, and ``evolution`` sets
    ``EvolutionSettings`` fields. ``worker_executor`` places the run workers (``execution.evolution``),
    ``agent`` is the method's coding agent, ``models`` the method's named models, and ``activation`` the
    scenario's activation rule (the ``activation`` section; none is ``always``). ``job_samples`` is the
    traces of one step (``pipeline.sample2job.config.job_samples``); None keeps the stage's default of 1.
    """
    return HarnessEvolutionRecipe(
        method=resolve_method(method),
        evolution=EvolutionSettings(
            **{"binary": str(make_pi_binary(tmp_path)), "state_dir": str(tmp_path / "harness"), **evolution}
        ),
        evaluation=EvaluationSettings(tasks=tasks, **dict(evaluation or {})),
        scorer=resolve_task_scorer(score),
        seed=seed,
        runtime=runtime(),
        worker_executor=ExecutorSettings() if worker_executor is None else worker_executor,
        agent=agent,
        models=dict(models or {}),
        activation=activation,
        pipeline_stages=(
            None
            if job_samples is None
            else PipelineConfig.from_setting(
                {"sample2job": {"config": {"job_samples": job_samples}}},
                spec=HarnessEvolutionRecipe.pipeline_spec(),
                environ={},
            )
        ),
    )


#: The releases whose head each test backend steps on, as a job names its start.
BACKEND_RELEASES: WeakKeyDictionary[HarnessBackend, TreeReleases] = WeakKeyDictionary()


def build_harness_backend(
    recipe: HarnessEvolutionRecipe, scenario: str = "demo", releases: TreeReleases | None = None
) -> HarnessBackend:
    """``recipe``'s backend of ``COMPONENT`` in ``scenario``, whose steps run on the head of ``releases`` (an
    in-memory history whose base is the seed by default)."""
    backend = recipe.build_backend(scenario, COMPONENT)
    BACKEND_RELEASES[backend] = TreeReleases(recipe.seed) if releases is None else releases
    return backend


def head_of(backend: HarnessBackend) -> ReleaseFiles:
    """The head that ``backend``'s next step runs on: the latest release of its ``build_harness_backend`` history."""
    return BACKEND_RELEASES[backend].head(COMPONENT)


def harness_backend(
    tmp_path: Path, method: Any, *, scenario: str = "demo", **recipe_fields: Any
) -> tuple[HarnessBackend, TreeReleases]:
    """The backend ``harness_recipe`` builds for ``scenario``, over an in-memory history whose base is the seed."""
    recipe = harness_recipe(tmp_path, method, **recipe_fields)
    releases = TreeReleases(recipe.seed)
    return build_harness_backend(recipe, scenario, releases), releases


def changed_rules(text: str) -> Mapping[str, str]:
    """A patch body that rewrites the rules."""
    return {"rules.md": text}


def traced(*prompts: str, clients: tuple[str | None, ...] = ()) -> tuple[TrajectorySample, ...]:
    """One trace sample per prompt, tagged with the client at its position when given."""
    return tuple(
        recorded_trajectory(
            f"a{index}",
            {
                "messages": [{"role": "user", "content": prompt}],
                "metadata": {"tags": {"client": clients[index]}} if index < len(clients) and clients[index] else {},
            },
            0.0,
        )
        for index, prompt in enumerate(prompts)
    )


class IdleMethod(Method):
    """A method that leaves the tree unchanged, so each of its steps skips.

    Purpose: ``evolution_config`` names it, for a test that builds a recipe from its config and never steps.
    """

    def step(self, context: StepContext) -> None:
        return None


class FixedScorer(TaskScorer):
    """A scorer that gives every run one score.

    Values: ``fixed`` is that score, 0.0 unless the config's ``score`` sets it.
    Purpose: ``evolution_config`` names it, for a test that builds a recipe from its config.
    """

    def __init__(self, fixed: float = 0.0) -> None:
        self.fixed = fixed

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(float(config.get("score", 0.0)))

    def score(self, task: str, result: TaskResult) -> float:
        return self.fixed


class NoCollection(TaskCollector):
    """A collector that collects no prompt.

    Purpose: a test names it in ``evolution.evaluation.collector``.
    """

    def collect(self, samples: tuple[TrajectorySample, ...]) -> list[str]:
        return []


def evolution_config(*, evaluation: Mapping[str, Any] | None = None, **evolution: Any) -> dict[str, Any]:
    """A deployment config whose ``evolution`` section runs ``IdleMethod`` on one task that ``FixedScorer`` scores.

    ``evaluation`` sets further keys of ``evolution.evaluation``, and ``evolution`` further keys of ``evolution``.
    """
    return {
        "evolution": {
            "method": "tests.support.harness_steps:IdleMethod",
            "evaluation": {
                "tasks": ["t"],
                "scorer": "tests.support.harness_steps:FixedScorer",
                **dict(evaluation or {}),
            },
            **evolution,
        }
    }


def step_services(
    workspace: Workspace, models: ModelBindings, record: list[dict[str, Any]] | None = None
) -> StepServices:
    """The backend's services of ``workspace``'s step, made before the method runs: ``models`` with no budget, every
    call in ``record``, and no scorer. The head is the tree in ``harness/``."""
    return StepServices(
        workspace=workspace,
        descriptor=get_adapter(str(workspace.host()["adapter"])),
        evaluator=None,
        history=History.open(workspace.history_dir),
        head=workspace.tree().checksum(),
        models=models,
        step_calls=StepCalls(0, [] if record is None else record),
    )


__all__ = [
    "COMPONENT",
    "MODEL",
    "PI_FAKE",
    "PI_SEED",
    "UNCHANGED",
    "FixedScorer",
    "IdleMethod",
    "NoCollection",
    "TreeReleases",
    "changed_rules",
    "evolution_config",
    "harness_backend",
    "harness_recipe",
    "make_pi_binary",
    "marker_score",
    "runtime",
    "step_services",
    "traced",
]
