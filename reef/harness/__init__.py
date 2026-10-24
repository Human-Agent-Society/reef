"""The harness layer: a harness as a file tree, rendered for one agent, run, and read back.

The map, one subpackage per job:

``tree/``
    what a harness is: ``HarnessTree`` and ``TreePatch``, the layout of its
    files (``layout``), their checks (``checks``, ``validate``), how a tree
    renders for one adapter (``render``), and an editable copy for an agent
    session (``workspace``).
``adapters/``
    one directory per agent, the mapping only: a ``descriptor.yaml`` (schema
    in ``descriptor``) that says which tree kinds the agent takes and where
    each lands, how its binary runs headless and binds a model, plus its
    quirks and the ``harness_facts.yaml`` a text proposer reads about the
    agent. pi, opencode, claude, codex, dsh, hermes, and the two programs
    Reef ships, native and terminus.
``episodes/``
    one headless run and its reading: launch on a rendered root through any
    ``reef.sandbox`` executor (``run``), the adapter as the program that
    executor runs and its install (``vendor_install``), the model binding,
    the trajectory readers, and Harbor task episodes (``tasks``).
``runners/``
    the programs an adapter's ``binary`` points at when Reef ships them:
    ``native`` (the loop, its graph, its seed and the resident ``serve`` form)
    and ``terminus`` (the Harbor runner).
``capture``
    the capture proxy between a harness and Reef: tags and receipts.
``step_result``
    what a settled step means for the person who asked, in the words the
    harness pages render.

This package depends on core values, runtime contracts and sandbox
execution, never on training, recipes, scenario coordination, or the HTTP
service. The evolution loop is ``reef.train.harness_backend``; versioning
(staging, publishing, the commit history, recovery and rollback) is Reef's
artifact stack (``reef.artifact``, ``reef.scenario``).
"""

from reef.harness.adapters.descriptor import AdapterDescriptor, ConfigTarget, DescriptorError, load_descriptor
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, read_opencode_session, read_pi_session
from reef.harness.tree import HarnessTree, RenderError, TreeKind, TreePatch, render_tree, validate_tree

__all__ = [
    "AdapterDescriptor",
    "ConfigTarget",
    "DescriptorError",
    "EpisodeError",
    "EpisodeResult",
    "HarnessTree",
    "RenderError",
    "TrajectoryError",
    "TrajectoryKeepError",
    "TreeKind",
    "TreePatch",
    "load_descriptor",
    "read_opencode_session",
    "read_pi_session",
    "render_tree",
    "run_episode",
    "validate_tree",
]
