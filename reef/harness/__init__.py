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
    quirks. pi, opencode, claude, codex, dsh, hermes, and the two programs
    Reef ships, native and terminus.
``rollouts/``
    one headless run and its reading: launch on a rendered root through any
    ``reef.execution.sandbox`` executor (``run``), the adapter as the program that
    executor runs and its install (``vendor_install``), the model binding,
    the trajectory readers, and Harbor task rollouts (``tasks``).
``runners/``
    the programs an adapter's ``binary`` points at when Reef ships them:
    ``native`` (the loop, its graph and the resident ``serve`` form) and
    ``terminus`` (the Harbor runner).
``capture``
    the capture proxy between a harness and Reef: tags and receipts.
``delivery``
    the surface that serves a release to a client: its source tree rendered
    for the scenario's adapter.

This package depends on core values, runtime and surface contracts, and
sandbox execution, never on training, recipes, scenario coordination, or the
HTTP service. The evolution loop is ``reef.train.harness_backend``; versioning
(staging, publishing, the commit history, recovery and activation) is Reef's
artifact stack (``reef.artifact``, ``reef.scenario``).
"""

from reef.harness.adapters.descriptor import AdapterDescriptor, ConfigTarget, DescriptorError, load_descriptor
from reef.harness.rollouts.run import RolloutError, RolloutResult, TrajectoryKeepError, run_rollout
from reef.harness.rollouts.trajectory import TrajectoryError, read_opencode_session, read_pi_session
from reef.harness.tree import HarnessTree, RenderError, TreeKind, TreePatch, render_tree, validate_tree

__all__ = [
    "AdapterDescriptor",
    "ConfigTarget",
    "DescriptorError",
    "HarnessTree",
    "RenderError",
    "RolloutError",
    "RolloutResult",
    "TrajectoryError",
    "TrajectoryKeepError",
    "TreeKind",
    "TreePatch",
    "load_descriptor",
    "read_opencode_session",
    "read_pi_session",
    "render_tree",
    "run_rollout",
    "validate_tree",
]
