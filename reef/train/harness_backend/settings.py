"""The configuration of the harness training component: the ``evolution`` section and what a deployment adds to it.

``EvolutionSettings`` is the parsed ``evolution`` section without its ``evaluation`` section
(``reef.evaluation.harness.EvaluationSettings``). ``HarnessSettings`` holds both, with the seed tree, the placement
of the run workers and the kind of the jobs: every value that ``HarnessTrainingComponent`` reads to build itself
and the objects of its jobs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.config import Settings, config_option
from reef.core.errors import ConfigError
from reef.core.tasks import TaskSplitError, manifest_task_paths
from reef.evaluation.harness import EvaluationSettings
from reef.execution.executor.config import ExecutorSettings
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree

#: The directory of the harness training component's jobs and starts under ``evolution.state_dir``.
TRAINING_DIR = ".training"


@dataclass(frozen=True)
class EvolutionSettings(Settings):
    """The parsed ``evolution`` section of a harness evolution deployment, without its ``evaluation`` section.

    Values: ``adapter`` names an adapter ``reef.harness.adapters.get_adapter``
    resolves. ``seed`` is the directory of the first tree, ``None`` for the
    empty tree. ``method`` is a plugin setting that names a ``Method``.
    ``models`` maps a name other than ``served`` to a model section (``url``,
    ``model``, ``api_key_env``); ``model_metadata`` maps a model name to its
    metadata. ``state_dir`` is the directory of every scenario's history
    and job records: ``scenario_state_dir`` and ``component_state_dir``
    name a scenario's and a component's directory under it, and ``training_dir`` the directory of the harness
    training component's jobs and starts. Every count is ≥ 0, where 0 turns its limit off.
    The retired ``publish`` and ``review_kinds`` are refused: the top-level
    ``activation`` section decides whether a kept tree serves.
    ``executor`` and ``sandbox`` choose where tasks run
    (``reef.execution.sandbox.config.build_executor``), and ``agent`` which coding
    agent a method runs and where (``reef.train.harness_backend.agent.settings``),
    ``None`` for none.
    Purpose: ``HarnessEvolutionRecipe`` parses it from the deployment, and
    ``HarnessTrainingComponent`` builds itself and each backend from it.
    """

    method: str | Mapping[str, Any] | None = config_option(None, help="The Method that evolves the tree.")
    adapter: str = config_option("pi", help="The harness adapter the tree renders for.")
    seed: str | None = config_option(None, help="The directory of the first tree; none starts empty.")
    binary: str | None = config_option(None, help="The harness binary; none installs the adapter's pinned one.")
    models: Mapping[str, Any] = config_option(
        default_factory=dict, help="The method's own models by name: url, model and api_key_env."
    )
    model_metadata: Mapping[str, Any] = config_option(default_factory=dict, help="Model metadata by model name.")
    state_dir: str = config_option(".reef/harness", help="Each scenario's history and job records.")
    max_steps: int = config_option(0, help="The most automatic jobs. 0 sets no limit.")
    max_failure_streak: int = config_option(
        0, help="Skip automatic jobs after this many rejections in a row. 0 sets no limit."
    )
    max_model_calls_per_step: int = config_option(0, help="The most model calls of one job. 0 sets no limit.")
    max_runs_per_step: int = config_option(0, help="The most runs a method runs in one job. 0 sets no limit.")
    executor: str = config_option("local", help="Where tasks run: local, sandbox or e2b.")
    sandbox: Mapping[str, Any] | None = config_option(None, help="The sandbox settings of the run executor.")
    agent: Mapping[str, Any] | None = config_option(
        None, help="The coding agent a method runs: adapter, binary, sandbox and timeout_s."
    )

    def __post_init__(self) -> None:
        if not self.state_dir.strip():
            raise ValueError("evolution.state_dir must be a non-empty path")
        if self.seed is not None and not self.seed.strip():
            raise ValueError("evolution.seed must be the path of a tree directory when it is set")
        for label, value in (
            ("max_steps", self.max_steps),
            ("max_failure_streak", self.max_failure_streak),
            ("max_model_calls_per_step", self.max_model_calls_per_step),
            ("max_runs_per_step", self.max_runs_per_step),
        ):
            if value < 0:
                raise ValueError(f"evolution.{label} must be at least 0 (0 turns the limit off)")
        if "served" in self.models:
            raise ValueError("evolution.models may not name a model 'served'; that is the model under test")

    def scenario_state_dir(self, scenario: str) -> Path:
        """The directory of ``scenario`` under ``state_dir``, made absolute."""
        return Path(self.state_dir).expanduser().resolve() / scenario

    def component_state_dir(self, scenario: str, component: str) -> Path:
        """The directory of ``component``'s history, step records and step progress in ``scenario``:
        ``scenario_state_dir(scenario) / component``."""
        return self.scenario_state_dir(scenario) / component

    def training_dir(self) -> Path:
        """The directory of the harness training component's jobs and starts: ``TRAINING_DIR`` under ``state_dir``,
        made absolute."""
        return Path(self.state_dir).expanduser().resolve() / TRAINING_DIR


def plugin_name(setting: str | Mapping[str, Any]) -> str:
    """A stable text form of a plugin setting: the reference, or its JSON with sorted keys."""
    return setting if isinstance(setting, str) else json.dumps(dict(setting), sort_keys=True, default=str)


def configured_tasks(evaluation: EvaluationSettings) -> tuple[str, ...]:
    """The configured evaluation tasks: ``evaluation.tasks``, or the eval tasks that the task manifest names.

    Raise ``ConfigError`` for a manifest that cannot be read and for one that names no eval task.
    """
    if evaluation.task_manifest is None or evaluation.tasks_root is None:
        return evaluation.tasks
    try:
        paths = manifest_task_paths(
            Path(evaluation.task_manifest).expanduser(), Path(evaluation.tasks_root).expanduser(), "eval"
        )
    except TaskSplitError as exc:
        raise ConfigError(str(exc)) from exc
    if not paths:
        raise ConfigError(f"evolution.evaluation.task_manifest {evaluation.task_manifest} names no eval tasks")
    return tuple(str(path) for path in paths)


@dataclass(frozen=True)
class HarnessSettings:
    """The configuration of one harness evolution deployment, as its training component reads it.

    Values: ``evolution`` and ``evaluation`` are the parsed sections. ``seed`` is the first tree of a new scenario,
    the files of ``evolution.seed``. ``worker_executor`` places the run workers (``execution.evolution``).
    ``runs_offline`` is True when the jobs roll out the imported train tasks. Every field is a value that the
    deployment config states; the settings hold no object that runs.
    Purpose: ``HarnessEvolutionRecipe.harness_settings`` makes one from the recipe's fields, and
    ``HarnessTrainingComponent`` builds itself and the backend, the evaluator and the selection policy of each
    scenario component from it.
    """

    evolution: EvolutionSettings
    evaluation: EvaluationSettings
    seed: HarnessTree = field(default_factory=lambda: HarnessTree({}))
    worker_executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    runs_offline: bool = False

    def task_directories(self, descriptor: AdapterDescriptor) -> bool:
        """Whether the runs play Harbor task directories rather than prompts: always for an adapter whose
        command takes a task directory, else when the evaluation reads a task manifest or the imported tasks, or the
        jobs roll out the imported train tasks."""
        return (
            descriptor.takes_task_directory
            or self.evaluation.task_manifest is not None
            or self.evaluation.imported_tasks
            or self.runs_offline
        )


__all__ = ["TRAINING_DIR", "EvolutionSettings", "HarnessSettings", "configured_tasks", "plugin_name"]
