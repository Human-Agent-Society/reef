"""The serving capabilities attached to one scenario.

A surface describes how one frozen release reaches inference or a
client pulling files. The capabilities are explicit: record-only surfaces
have none, while model, adapter, and harness surfaces compose only the pieces
they use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef, ArtifactValidator


@dataclass(frozen=True)
class AcceptAnyArtifact(ArtifactValidator):
    """Default admission policy for shape-agnostic scenarios."""

    def validate(self, artifact: Artifact) -> None:
        return None


class ServingRuntime(ABC):
    """The runtime shape visible to surface loaders."""

    @property
    @abstractmethod
    def base_url(self) -> str: ...


class WeightRuntime(ServingRuntime):
    """A runtime that can inspect and restore served model weights."""

    @abstractmethod
    def serving_runtime_load_id(self) -> str | None: ...

    @abstractmethod
    def restore_checkpoint(self, artifact: Artifact) -> str: ...

    def activate_checkpoint(self, artifact: Artifact) -> str:
        """Bind a recovered or republished artifact before the scenario serves it.

        Called once a release is final and before traffic routes to it. The
        default binds nothing and returns the release ID: for runtimes whose
        serving update and Reef publication are one operation, the artifact
        already names what the engine serves. A runtime that serves immutable
        remote snapshots overrides this to select the sampler and training
        state the artifact references, and returns the runtime load ID it
        now serves under.
        """
        return artifact.ref.release_id


class AdapterWeightRuntime(WeightRuntime):
    """A weight runtime that can inspect each scenario's resident adapter."""

    @abstractmethod
    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None: ...


class ArtifactLoader(ABC):
    """Runtime-backed artifact loading and startup recovery."""

    @abstractmethod
    def recover(
        self,
        current: ArtifactRef | None,
        checkpoint: ArtifactRef,
        runtime: ServingRuntime | None,
    ) -> ArtifactRef: ...

    @abstractmethod
    def load(self, artifact: Artifact, runtime: ServingRuntime | None) -> str: ...


class ArtifactActivator(ArtifactLoader):
    """Optional loader capability: make a published release servable.

    Called once a release is final — after startup recovery has a
    materializable head and after a publication or rollback commit has
    minted its release — and before the scenario routes traffic to it.
    ``source`` names the artifact whose bytes a rollback republished.
    """

    @abstractmethod
    def activate(
        self, artifact: Artifact, runtime: ServingRuntime | None, *, source: Artifact | None = None
    ) -> str: ...


class InferenceHooks(ABC):
    """Request and response hooks around one provider inference."""

    @abstractmethod
    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None: ...


class InferenceLease(ABC):
    """Serving state held for one inference attempt; released exactly once."""

    @abstractmethod
    def release(self) -> None: ...


class LeasingInferenceHooks(InferenceHooks):
    """Optional inference capability: hold serving state for one attempt.

    ``begin_request`` runs after ``prepare_request`` froze the artifact and
    returns the lease the service releases when the attempt ends, whether it
    completed, aborted, or failed.
    """

    @abstractmethod
    def begin_request(self, artifact: Artifact, path: str) -> InferenceLease: ...


class FileTree(ABC):
    """A client-readable file tree derived from an artifact."""

    @abstractmethod
    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None: ...


@dataclass(frozen=True)
class HarnessInfo:
    """What the harness routes need beyond the file tree: the seed behind the base release and the served model."""

    seed_entries: tuple[Mapping[str, Any], ...] = ()
    served_model: str | None = None
    #: The served model's API dialect: Reef forwards calls unchanged, so an installed client must speak it.
    served_api: str = "openai"
    #: Further models the installed client may pick from; the served one stays the default.
    client_models: tuple[str, ...] = ()
    #: The adapter the tree is rendered for, such as ``pi`` or ``claude``; a client installs ``reef-<adapter>``.
    adapter: str | None = None


@dataclass(frozen=True)
class Surface:
    """The explicit serving capabilities bound to one scenario.

    ``None`` means the capability is absent. Callers inspect the corresponding
    field; every recipe binds an instance of this same type.
    """

    loader: ArtifactLoader | None = None
    inference: InferenceHooks | None = None
    files: FileTree | None = None
    harness: HarnessInfo | None = None


__all__ = [
    "AdapterWeightRuntime",
    "ArtifactActivator",
    "ArtifactLoader",
    "FileTree",
    "HarnessInfo",
    "InferenceHooks",
    "InferenceLease",
    "LeasingInferenceHooks",
    "ServingRuntime",
    "Surface",
    "WeightRuntime",
]
