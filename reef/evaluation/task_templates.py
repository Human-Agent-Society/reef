"""The builder of the sandbox templates of a task, which an evaluation on e2b calls before it adds the task's rollouts.

The evaluation service gives each task of an evaluation on e2b one template item (``TaskTemplateWorker`` of
``reef.evaluation.rollout_worker``). The item calls the deployment's ``TaskTemplateBuilder``, which the setting
``evaluation.task_template_builder`` names. Without the setting, ``DockerfileTemplateBuilder`` builds the templates from
the task's Dockerfiles, as a trial would. A deployment that keeps task images elsewhere, such as in a registry, names
its own builder, so that the templates come from those images. Such a builder finds a task's image by the owner of the
scenario's items and the task's name (``build_for_owner``), never by the scenario's name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from reef.core.plugins import ConfiguredPlugin


class TaskTemplateBuilder(ConfiguredPlugin, ABC):
    """Makes the sandbox templates of one task ready, so that the task's trials only start sandboxes.

    ``build`` gets the scenario of the evaluation, the name of the task, which is the ID of the task's item in the
    data of the scenario's owner, and the task's Harbor directory on this process. It returns when the templates that a
    trial of the task starts from are ready. It raises ``TimeoutError`` when it found no free builder in the task's
    build time limit: the worker then calls it again, within its wait limit. Any other error fails the attempt of the
    template item, and the item is tried again until it uses up its attempts. Unless the evaluation requires prepared
    templates (``evaluation.require_prepared_templates``), a trial builds a template that is still missing itself.

    ``build_for_owner`` gets the same arguments after ``owner``: the owner that the dispatcher's record of the scenario
    stores (``ScenarioSettings.owner``), whose data holds the task's item. A scenario created without an owner has its
    own name as its owner. ``owner`` is None when the worker knows no owner, such as a worker whose endpoints read no
    scenario record. The worker calls ``build_for_owner``. Its default calls ``build`` without the owner, so a builder
    that needs no owner implements ``build`` alone. A builder that finds templates by owner overrides
    ``build_for_owner``; it must raise for an owner that it cannot use, None too, and never choose another owner.

    Purpose: the evaluation service's ``TaskTemplateWorker`` calls ``build_for_owner`` for each template item.
    """

    @abstractmethod
    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None: ...

    async def build_for_owner(self, owner: str | None, scenario: str, task_name: str, task_directory: Path) -> None:
        await self.build(scenario, task_name, task_directory)


class DockerfileTemplateBuilder(TaskTemplateBuilder):
    """Builds a task's templates from its Dockerfiles in this process, with the aliases, resources and build time
    limits of its trials (``build_task_template`` of Reef's e2b sandbox). It waits for a build of the same template
    that another process started."""

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        # The e2b extra is optional, and only an evaluation on e2b needs it.
        from reef.execution.sandbox.e2b import build_task_template

        await build_task_template(task_directory)


__all__ = ["DockerfileTemplateBuilder", "TaskTemplateBuilder"]
