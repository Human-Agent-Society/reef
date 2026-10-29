"""``AgentMethod``: a method that is a folder of prompts, which the deployment's coding agent runs each step.

The folder holds ``START.md``, the prompt of a step; ``RESUME.md``, the
prompt of a step whose ``state/`` holds files from an earlier step
(``START.md`` when the folder has none); and the agent's own harness tree,
such as its ``rules.md`` and ``skills/<name>/SKILL.md``, which Reef renders
for the agent's adapter after its protocol rules. A ``README.md`` is for
people and is not given to the agent. In a prompt, ``{step}`` becomes the
step's number and ``{history}`` the history's path from the workspace.
Every other file of the folder is a file of the agent's tree; a run refuses
one that no part of the agent's adapter takes.

Each step runs the agent once (``Services.run_agent``). The tree it leaves in
``harness/`` is the candidate; a run that does not exit 0 leaves the head
tree, so the step skips. The agent says what the step did in ``output/``
(``summary.md``, ``notes.json``, ``requires.json``), as every method does;
the method adds the run's outcome to the notes under ``agent``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.harness.tree import HarnessTree
from reef.train.harness_backend.agent.run import fill
from reef.train.harness_backend.contract import Method, StepContext
from reef.train.harness_backend.workspace import NOTES_FILE, SUMMARY_FILE, json_object, output_text

START_FILE = "START.md"
RESUME_FILE = "RESUME.md"
#: The folder's file for people, which the agent is not given.
README_FILE = "README.md"


@dataclass(frozen=True)
class AgentMethodSettings(PluginSettings):
    """The config of ``AgentMethod``, from ``evolution.method.config``.

    Values: ``prompts`` is the path of the method folder, which holds
    ``START.md``. ``model`` names the binding the agent runs on: ``served``
    or an ``evolution.models`` entry.
    Purpose: ``AgentMethod.from_config`` parses it.
    """

    prompts: str = config_option("", help="The method folder: START.md, an optional RESUME.md, the agent's tree.")
    model: str = config_option("served", help="The model the agent runs on: served, or an evolution.models name.")

    def __post_init__(self) -> None:
        if not self.prompts.strip():
            raise ValueError("prompts must be the path of the method folder")
        if not self.model.strip():
            raise ValueError("model must name a model binding")


class AgentMethod(Method):
    """A method that is a folder of prompts, which the deployment's coding agent runs each step.

    Values: ``start`` is the prompt of a step and ``resume`` the prompt of a
    step that continues from ``state/``, ``None`` to use ``start``. ``tree``
    is the agent's own harness tree, and ``model`` the binding it runs on.
    Purpose: a deployment names ``reef.train.harness_backend:AgentMethod``
    in ``evolution.method`` with the folder in its config.
    """

    runs_agent = True

    def __init__(self, start: str, resume: str | None, tree: HarnessTree, model: str) -> None:
        self.start = start
        self.resume = resume
        self.tree = tree
        self.model = model

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        settings = AgentMethodSettings.parse(config, plugin_class=cls)
        folder = Path(settings.prompts).expanduser()
        try:
            files = dict(HarnessTree.read(folder).files)
        except ValueError as exc:
            raise PluginConfigError(f"AgentMethod prompts: {exc}") from exc
        start = files.pop(START_FILE, None)
        if start is None or not start.strip():
            raise PluginConfigError(f"AgentMethod prompts: {folder} holds no {START_FILE}")
        resume = files.pop(RESUME_FILE, None)
        files.pop(README_FILE, None)
        # The rest is the agent's own tree; rendering it for the agent's adapter refuses a file no part takes.
        return cls(start, resume, HarnessTree(files), settings.model)

    def step(self, context: StepContext) -> None:
        workspace, services = context.workspace, context.services
        head = workspace.tree()
        prompt = self.start
        if self.resume is not None and any(workspace.state.iterdir()):
            prompt = self.resume
        outcome = services.run_agent(fill(prompt, workspace), tree=self.tree, model=self.model)
        if not outcome.succeeded:
            workspace.write_tree(head)
            workspace.write_output(
                f"the agent {outcome.describe()}; the step keeps the head tree", {"agent": outcome.to_json()}
            )
            return
        text = output_text(workspace.output / NOTES_FILE)
        try:
            notes = json_object(json.loads(text)) if text else {}
        except ValueError:
            # The backend refuses the agent's notes.json as it stands, and records why.
            return
        workspace.write_output(output_text(workspace.output / SUMMARY_FILE), {**notes, "agent": outcome.to_json()})


__all__ = ["AgentMethod", "AgentMethodSettings"]
