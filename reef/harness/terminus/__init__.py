"""The ``reef-terminus`` runner: the binary the terminus adapter's episodes launch.

Terminus 2 is a harbor agent class rather than a CLI, so this package is the
adapter's own runner. It reads the tree Reef rendered into ``REEF_TERMINUS_DIR``,
binds it to :class:`~reef.harness.terminus.agent.ReefTerminus`, runs the task
harbor names, and writes the ATIF trajectory and the verifier's reward under
``REEF_TERMINUS_SESSION_DIR`` for the episode's trajectory reader.

harbor is an optional dependency (``pip install reef-infra[terminus]``) and is
imported only when the runner actually runs, so importing the adapter registry
stays cheap and CI without Docker still loads the descriptor.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from reef.harness.terminus.tree import (
    ContextPolicy,
    TerminusTreeError,
    context_policy,
    instruction_text,
    load_tree,
    terminus_kwargs,
)


def main(argv: Sequence[str] | None = None) -> int:
    """The ``reef-terminus`` console script: run one task through harbor."""
    parser = argparse.ArgumentParser(
        prog="reef-terminus", description="Run one Terminal-Bench task for a reef episode."
    )
    parser.add_argument("--task", required=True, help="the task name harbor should run")
    arguments = parser.parse_args(argv)
    # Imported here so the adapter, its tests, and the registry never need harbor.
    from reef.harness.terminus.runner import run

    return run(arguments.task)


__all__ = [
    "ContextPolicy",
    "TerminusTreeError",
    "context_policy",
    "instruction_text",
    "load_tree",
    "main",
    "terminus_kwargs",
]
