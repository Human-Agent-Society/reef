"""Shared value types and the root error hierarchy: the bottom of the dependency graph.

Nothing here has storage behavior or I/O; the packages above supply that
(``records.py`` stores records, ``artifact/`` stores artifacts) while
sharing these definitions.

The admission bar is concrete: a type belongs here only when at least two
packages that do not depend on each other need it, and it carries no I/O.
Anything with one consumer stays in that consumer — the ``x-reef-*`` header
parsing and the HTTP report envelope live in ``service/wire.py`` — while the
typed report *body* is ``reports/`` here because four packages parse it.

The boundary is directional: core imports nothing above it. In particular
``reef.core`` never imports ``reef.artifact`` — ``artifact_ref`` lives here
so identity can be shared without pulling in storage, and
``artifacts/artifact.py`` re-exports it. Pinned by
``tests/reef_service/test_dependency_boundaries.py``.
"""

from reef.core.artifact_ref import ArtifactRef, LiveWeightArtifactRef, WeightVersionSpan, parse_weight_version_spans
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
    "UnknownScenario",
    "WeightVersionSpan",
    "parse_weight_version_spans",
]
