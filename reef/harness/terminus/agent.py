"""``ReefTerminus``: Terminus 2 with its behavior bound to a rendered tree.

This module imports harbor, so it is imported lazily by the runner and never
by the adapter registry. Every seam below is a no-op on an empty tree, which
is what keeps stock Terminus 2 the measured baseline: a gated tree change then
means a real capability delta rather than a harness artifact.

Seams, verified against harbor 0.22.0:

- ``__init__`` takes the tree's constructor arguments, which win over the
  runner's own, because the tree is the thing under evaluation.
- ``_build_skills_section`` appends the tree's rules, skills, and commands.
- ``_query_llm`` hands the pending call to the tree's context policy.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2

from reef.harness.terminus.tree import TerminusTreeError, context_policy, instruction_text, load_tree, terminus_kwargs

#: Where the runner rendered the tree. The adapter descriptor sets it.
TREE_DIR_ENV = "REEF_TERMINUS_DIR"

logger = logging.getLogger(__name__)


class ReefTerminus(Terminus2):
    """Terminus 2 whose instruction, knobs, and context come from a reef tree."""

    def __init__(self, logs_dir: Path, *args: Any, reef_dir: str | None = None, **kwargs: Any) -> None:
        root = reef_dir or os.environ.get(TREE_DIR_ENV)
        if not root:
            raise TerminusTreeError(f"pass reef_dir or set {TREE_DIR_ENV} to the rendered tree")
        self._reef_files = load_tree(root)
        self._reef_instruction = instruction_text(self._reef_files)
        self._reef_context = context_policy(self._reef_files)
        # The tree is the genome: for the knobs it sets, it outranks the runner.
        super().__init__(logs_dir, *args, **{**kwargs, **terminus_kwargs(self._reef_files)})

    @staticmethod
    def name() -> str:
        return "reef-terminus-2"

    async def _build_skills_section(self, environment: Any) -> str | None:
        """Stock instruction, then the tree's own."""
        section = await super()._build_skills_section(environment)
        if not self._reef_instruction:
            return section
        return f"{section}\n\n{self._reef_instruction}" if section else self._reef_instruction

    async def _query_llm(
        self,
        chat: Any,
        prompt: str,
        original_instruction: str = "",
        session: Any = None,
    ) -> Any:
        """Let the context policy rebuild the message list, then delegate.

        Delegating to the stock method keeps its retry and context-recovery
        behavior, and the rebuild happens once per turn outside that retry
        loop. A policy that raises is logged and skipped, so a defective
        candidate degrades to stock assembly and still earns a score instead
        of killing the trial.
        """
        messages = None
        if self._reef_context is not None:
            request = {"messages": [*chat.messages, {"role": "user", "content": prompt}]}
            try:
                messages = self._reef_context.assemble(request)
            except Exception:  # a tree defect must not end the trial
                logger.warning("%s raised; using stock assembly", self._reef_context.path, exc_info=True)
                messages = None
        if messages and messages[-1].get("role") == "user":
            # The same in-place rewrite Terminus 2 performs after summarizing:
            # replace the history, reset the chain, and send the final user
            # message as this turn's prompt.
            chat._messages = messages[:-1]
            chat.reset_response_chain()
            prompt = str(messages[-1].get("content", ""))
        return await super()._query_llm(chat, prompt, original_instruction=original_instruction, session=session)
