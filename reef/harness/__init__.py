"""The harness layer: a harness as a file tree, rendered for one agent, run, and read back.

The map, one subpackage per job:

``tree/``
    what a harness is: ``HarnessTree`` and ``TreePatch``, the checks of its
    files (``checks``, ``validate``), and how a tree renders for one adapter
    (``render``).
``adapters/``
    one directory per agent, the mapping only: a ``tree.yaml`` (schema in
    ``parts``) that says which parts a tree of the agent holds and where each
    lands, a ``descriptor.yaml`` (schema in ``descriptor``) that says how its
    binary runs headless and binds a model, and a ``render.py`` with its own
    render steps. pi, opencode, claude, codex, dsh, hermes, and the two
    programs Reef ships, native and terminus.
``episodes/``
    one headless run and its reading: launch on a rendered root through any
    ``reef.sandbox`` executor (``run``), the adapter as the program that
    executor runs and its install (``vendor_install``), the model binding,
    the trajectory readers, and Harbor task episodes (``tasks``).
``runners/``
    the programs an adapter's ``binary`` points at when Reef ships them:
    ``native`` (the loop, its graph and the resident ``serve`` form) and
    ``terminus`` (the Harbor runner).
``capture``
    the capture proxy between a harness and Reef: tags and receipts.
``gateway``
    the loopback gateway an agent process reaches its model and its tools
    through, which meters each call.
``delivery``
    the surface that serves a release to a client: its source tree rendered
    for the scenario's adapter.

This package depends on core values, runtime and surface contracts, and
sandbox execution, never on training, recipes, scenario coordination, or the
HTTP service. The evolution loop is ``reef.train.harness_backend``; versioning
(staging, publishing, the commit history, recovery and rollback) is Reef's
artifact stack (``reef.artifact``, ``reef.scenario``).
"""

from reef.harness.adapters.descriptor import AdapterDescriptor, ConfigTarget, DescriptorError, load_descriptor
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, read_opencode_session, read_pi_session
from reef.harness.tree import HarnessTree, RenderError, TreePatch, render_tree, validate_tree

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
    "TreePatch",
    "load_descriptor",
    "read_opencode_session",
    "read_pi_session",
    "render_tree",
    "run_episode",
    "validate_tree",
]
