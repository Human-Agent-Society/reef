"""The builder of the sandbox templates of a task, which an evaluation on e2b calls before it adds the task's rollouts.

The evaluation service gives each task of an evaluation on e2b one template item (``TaskTemplateWorker`` of
``reef.evaluation.rollout_worker``). The item calls the deployment's ``TaskTemplateBuilder``, which the setting
``evaluation.task_template_builder`` names. Without the setting, ``DockerfileTemplateBuilder`` builds the templates from
the task's Dockerfiles, as a trial would. A deployment that keeps task images elsewhere, such as in a registry, names
its own builder, so that the templates come from those images.
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
    template item, and the item is tried again until it uses up its attempts. A trial builds a template that is still
    missing itself.

    Purpose: the evaluation service's ``TaskTemplateWorker`` calls it for each template item.
    """

    @abstractmethod
    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None: ...


class DockerfileTemplateBuilder(TaskTemplateBuilder):
    """Builds a task's templates from its Dockerfiles in this process, with the aliases, resources and build time
    limits of its trials (``build_task_template`` of Reef's e2b sandbox). It waits for a build of the same template
    that another process started."""

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        # The e2b extra is optional, and only an evaluation on e2b needs it.
        from reef.execution.sandbox.e2b import build_task_template

        await build_task_template(task_directory)


__all__ = ["DockerfileTemplateBuilder", "TaskTemplateBuilder"]
