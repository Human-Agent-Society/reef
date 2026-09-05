"""Harness evolution layer: mutation of a coding-agent composition.

A harness composition is a compose Entry tree whose entries are the five
node kinds of ``reef.harness.nodes``. ``reef.harness.render`` turns the
tree into one harness's native files as described by an
``AdapterDescriptor``; ``reef.harness.episode`` runs one headless invocation
inside an isolated, fully relocated root and collects its trajectory.
Versioning — staging, publishing, commit log, release chain, recovery,
and rollback — is handled by reef's native artifact stack
(``reef.artifact``, ``reef.scenario.commit_protocol``), not here.

Vocabulary note: ``reef.surface.harnesses`` delivers a *harness artifact* to
a client program; this package evolves the *harness composition itself* by
driving a real coding-agent binary (pi, opencode) per episode.
"""

from reef.harness.descriptor import AdapterDescriptor, ConfigTarget, DescriptorError, load_descriptor
from reef.harness.episode import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.nodes import NODE_KINDS
from reef.harness.render import RenderError, render_composition
from reef.harness.trajectory import TrajectoryError, read_opencode_storage, read_pi_session

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
