"""Terminus adapter quirks: constructor knobs, skill frontmatter, one context module.

Skills carry the ``name`` and ``description`` frontmatter the instruction
builder reads. ``process_skill`` and ``process_command`` add it
when an evolved node ships bare text, under both skill roots. Terminus 2 has no slash-command surface, so ``agent_command``
renders under the second root, and Harbor lists those skills beside the
others, with nothing that marks them as commands.

One ``code_extension`` can define an Agent subclass of Harbor's Terminus2.
Rendering only checks its syntax. Execution requires Reef's sandbox around
the runner and Harbor's remote E2B environment for the terminal task: the
task container alone does not isolate evolved Python in the outer runner.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer, ExecutionValidator
from reef.harness.runners.terminus.tree import ENVIRONMENT_ENV, TerminusTreeError, extension_source
from reef.harness.tree.render import RenderError
from reef.sandbox.process import EpisodeExecutor, EpisodeLaunchError, SandboxExecutor


class TerminusAdapterRenderer(AdapterRenderer):
    @staticmethod
    def process_skill(path: str, text: str) -> str:
        return with_skill_frontmatter(path, text)

    @staticmethod
    def process_command(path: str, text: str) -> str:
        # A command is a skill under the second root, so it needs the same frontmatter.
        return with_skill_frontmatter(path, text)

    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        try:
            extension_source(files)
        except TerminusTreeError as exc:
            raise RenderError(str(exc)) from exc
        return files


class TerminusExecutionValidator(ExecutionValidator):
    def __call__(self, files: Mapping[str, str], executor: EpisodeExecutor) -> None:
        """Refuse unisolated Python and local Docker nested in Reef's jail."""
        try:
            extension = extension_source(files)
        except TerminusTreeError as exc:
            raise EpisodeLaunchError(str(exc)) from exc
        if isinstance(executor, SandboxExecutor):
            if executor.env.get(ENVIRONMENT_ENV) != "e2b":
                raise EpisodeLaunchError(
                    "terminus Docker cannot run under evolution.executor: sandbox; "
                    "set REEF_TERMINUS_ENVIRONMENT=e2b and include it in evolution.sandbox.env_from"
                )
            if not executor.egress_hosts or not executor.env.get("E2B_API_KEY"):
                raise EpisodeLaunchError(
                    "sandboxed terminus requires egress_hosts and E2B_API_KEY in sandbox.env_from"
                )
        elif extension is not None:
            raise EpisodeLaunchError(
                "terminus code_extension requires evolution.executor: sandbox with remote E2B tasks"
            )


validate_execution = TerminusExecutionValidator()
