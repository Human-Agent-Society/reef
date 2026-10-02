"""Shared value types, contracts, and root errors: the bottom of the dependency graph.

``config`` holds field declarations and value parsing shared by service,
recipe and runtime components; it does not discover or import components.

Nothing here implements storage behavior or I/O. Record storage interfaces
belong to ``reef.storage.records``; concrete record and artifact adapters build on the
shared identities and wire types defined here.

The admission bar is concrete: a type belongs here only when at least two
packages that do not depend on each other need it, and it carries no I/O.
``batches`` and ``evaluation`` hold the values and candidate contracts shared
by runtimes and training; ``requirements`` validates training-request requirements
and reads their release-chain records.
Anything with one consumer stays in that consumer — the ``x-reef-*`` header
parsing and the HTTP report envelope live in ``service/wire.py`` — while the
typed report *body* is ``reports/`` here because four packages parse it.

The boundary is directional: core imports nothing above it. In particular
``reef.core`` never imports ``reef.artifact`` — ``artifact_ref`` lives here
so identity can be shared without pulling in storage, and
``artifacts/artifact.py`` re-exports it. Pinned by
``tests/reef_service/test_dependency_boundaries.py``.
"""

from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef, RuntimeLoadSpan, parse_runtime_load_spans
from reef.core.errors import ReefError, UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError

__all__ = [
    "AgentRecord",
    "ArtifactRef",
    "LiveWeightArtifactRef",
    "ReefError",
    "ReportBase",
    "ReportValidationError",
    "RequestType",
    "RuntimeLoadSpan",
    "UnknownScenario",
    "parse_runtime_load_spans",
]
