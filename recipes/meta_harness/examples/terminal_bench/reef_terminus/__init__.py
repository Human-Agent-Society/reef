"""The terminus adapter of the Terminal-Bench campaign: Harbor's Terminus 2 as a Reef harness.

Terminus 2 is a Harbor agent class rather than a CLI. This package is its
adapter (``adapter.yaml``, ``render.py``, ``execution.py``), the runner that
the adapter's episodes launch (``reef-terminus``: ``runner.py``, ``tree.py``),
and the reader of the runner's trials (``trajectory.py``). Installing the
campaign registers the adapter as ``terminus`` on the ``reef.harness_adapters``
entry point group, and the ``reef-terminus`` console script.

The runner reads the tree Reef rendered into ``REEF_TERMINUS_DIR``, hands it to
Harbor's own ``terminus-2`` agent as native configuration, runs the task, and
writes the ATIF trajectory and the verifier's reward under
``REEF_TERMINUS_SESSION_DIR`` for the episode's trajectory reader. Harbor is
imported only when the runner runs, so loading the adapter stays cheap.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from reef_terminus.tree import TerminusTreeError, instruction_paths, load_tree, skill_roots, terminus_kwargs

from reef.harness.adapters.descriptor import ADAPTER_FILE, AdapterDescriptor, load_descriptor


def descriptor() -> AdapterDescriptor:
    """The terminus adapter: the ``reef.harness_adapters`` entry point that installing the campaign registers."""
    return load_descriptor(Path(__file__).parent / ADAPTER_FILE)


def main(argv: Sequence[str] | None = None) -> int:
    """The ``reef-terminus`` console script: run one task through harbor."""
    parser = argparse.ArgumentParser(
        prog="reef-terminus", description="Run one Terminal-Bench task for a reef episode."
    )
    parser.add_argument("--task", required=True, help="the task name harbor should run")
    arguments = parser.parse_args(argv)
    # Imported here so the adapter, its tests, and the registry never need harbor.
    from reef_terminus.runner import run

    return run(arguments.task)


__all__ = [
    "TerminusTreeError",
    "descriptor",
    "instruction_paths",
    "load_tree",
    "main",
    "skill_roots",
    "terminus_kwargs",
]
