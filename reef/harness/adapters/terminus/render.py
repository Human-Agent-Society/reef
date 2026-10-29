"""Terminus adapter render steps: constructor knobs, skill frontmatter, one context module.

Skills carry the ``name`` and ``description`` frontmatter the instruction
builder reads. ``process_skill`` and ``process_command`` add it
when an evolved file ships bare text, under both skill roots. Terminus 2 has no slash-command surface, so a command
renders under the second root, and Harbor lists those skills beside the
others, with nothing that marks them as commands.

One extension can define an Agent subclass of Harbor's Terminus2. Rendering
only checks its syntax; ``reef.harness.adapters.terminus.execution`` checks
where it may run.
"""

from __future__ import annotations

from reef.harness.adapters.common.frontmatter import with_skill_frontmatter
from reef.harness.adapters.descriptor import AdapterRenderer
from reef.harness.runners.terminus.tree import TerminusTreeError, extension_source
from reef.harness.tree.render import RenderError


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
