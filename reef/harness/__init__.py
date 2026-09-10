"""The harness layer: a coding agent's composition as a tree, rendered, run and served.

The map, one subpackage per job:

``tree/``
    what a composition is: the node kinds and their admission (``nodes``), and
    how a tree renders to one harness's files (``render``).
``adapters/``
    one directory per agent, the mapping only: a ``descriptor.yaml`` (schema in
    ``descriptor``) plus quirks. pi, opencode, claude, codex, dsh, hermes, and
    the two programs Reef ships, native and terminus.
``episodes/``
    one headless run and its reading: launch on a rendered root, locally or
    in a jail (``run``, ``executor``), the model binding, the version pin and
    the trajectory readers.
``runners/``
    the programs an adapter's ``binary`` points at when Reef ships them:
    ``native`` (the loop, its graph, its seed and the resident ``serve`` form)
    and ``terminus`` (the Harbor runner).
``client/``
    what runs on a user's machine: the wrapper the install script bakes
    around a pulled harness.

The evolution loop itself, propose, gate and publish, is
``reef.train.cordis_backend``; versioning (staging, publishing, the commit
log, recovery and rollback) is reef's artifact stack (``reef.artifact``,
``reef.scenario.commit_protocol``), not here.

Vocabulary note: ``reef.surface.harnesses`` delivers a *harness artifact* to
a client program; this package evolves the *harness composition itself* by
driving a real coding agent binary per episode.
"""

from reef.harness.adapters.descriptor import AdapterDescriptor, ConfigTarget, DescriptorError, load_descriptor
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, read_opencode_storage, read_pi_session
from reef.harness.tree.nodes import NODE_KINDS
from reef.harness.tree.render import RenderError, render_composition

__all__ = [
    "NODE_KINDS",
    "AdapterDescriptor",
    "ConfigTarget",
    "DescriptorError",
    "EpisodeError",
    "EpisodeResult",
    "RenderError",
    "TrajectoryError",
    "TrajectoryKeepError",
    "load_descriptor",
    "read_opencode_storage",
    "read_pi_session",
    "render_composition",
    "run_episode",
]
