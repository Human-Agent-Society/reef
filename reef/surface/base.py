"""The serving capabilities attached to one scenario.

A surface describes how one frozen release reaches inference or a
client pulling files. A release binds named components (``weights``,
``harness``, ``skills``, ...), and the surface binds each component's
capabilities explicitly: record-only surfaces have no components, while
model, adapter, and harness surfaces compose only the pieces they use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef, ArtifactValidator
from reef.core.components import validate_component_name


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


@dataclass(frozen=True)
class ComponentSurface:
    """How one named component of a release is admitted, loaded, injected, and read.

    ``None`` means the capability is absent. ``validator`` runs before the
    component is published or restored; it defaults to accepting anything.
    """

    validator: ArtifactValidator = field(default_factory=AcceptAnyArtifact)
    loader: ArtifactLoader | None = None
    inference: InferenceHooks | None = None
    files: FileTree | None = None


class ChainedLease(InferenceLease):
    """Release every component lease, last acquired first, even when one fails."""

    def __init__(self, leases: tuple[InferenceLease, ...]) -> None:
        self.leases = leases

    def release(self) -> None:
        failure: BaseException | None = None
        for lease in reversed(self.leases):
            try:
                lease.release()
            except Exception as exc:
                failure = exc if failure is None else failure
        if failure is not None:
            raise failure


class ChainedInferenceHooks(InferenceHooks):
    """Every component's hooks, applied in component declaration order."""

    def __init__(self, hooks: tuple[tuple[str, InferenceHooks], ...]) -> None:
        self.component_hooks = hooks

    def prepare_request(self, artifact: Artifact, path: str, request: dict[str, Any]) -> dict[str, Any]:
        for name, hooks in self.component_hooks:
            request = hooks.prepare_request(artifact.component(name), path, request)
        return request

    def verify_response(self, artifact: Artifact, path: str, response: Mapping[str, Any]) -> None:
        for name, hooks in self.component_hooks:
            hooks.verify_response(artifact.component(name), path, response)


class LeasingChainedInferenceHooks(ChainedInferenceHooks, LeasingInferenceHooks):
    def begin_request(self, artifact: Artifact, path: str) -> InferenceLease:
        leases: list[InferenceLease] = []
        try:
            for name, hooks in self.component_hooks:
                if isinstance(hooks, LeasingInferenceHooks):
                    leases.append(hooks.begin_request(artifact.component(name), path))
        except Exception:
            ChainedLease(tuple(leases)).release()
            raise
        return ChainedLease(tuple(leases))


class ComponentFileTree(FileTree):
    """One component's file tree read from that component's directory."""

    def __init__(self, name: str, tree: FileTree) -> None:
        self.component = name
        self.tree = tree

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None:
        return self.tree.read_files(artifact.component(self.component))


#: The component a surface built from the flat ``loader``, ``inference`` and ``files`` keywords serves.
FLAT_COMPONENT = "release"


@dataclass(frozen=True, init=False)
class Surface:
    """The explicit serving capabilities bound to one scenario, per release component.

    A scenario with no components records traffic only. A scenario with one
    component serves a flat release, exactly as before components existed. A
    scenario with several components serves a release whose files sit in one
    directory per component; the ``loader``, ``inference``, and ``files``
    views then route each capability to its component's directory. At most
    one component may load into a runtime, and at most one may expose a
    client-pulled file tree. Every recipe binds an instance of this same type.

    The ``loader``, ``inference`` and ``files`` keywords of the surface before
    components existed still build one: a single component,
    ``FLAT_COMPONENT``, or, beside one declared component (as
    ``dataclasses.replace`` passes it), that component with those
    capabilities replaced. A surface of several components takes them per
    component only.
    """

    components: Mapping[str, ComponentSurface] = field(default_factory=dict)
    harness: HarnessInfo | None = None
    #: The whole release's own admission check, run before each component's: on the release a step publishes and
    #: on a release a rollback or promote restores. A recipe binds its checks on ``ComponentSurface.validator``; this
    #: one carries the ``build_artifact_validator`` of a recipe that serves no component, which admits the release as
    #: a whole, as every recipe's did before components existed (a composite moves it onto that recipe's component).
    validator: ArtifactValidator = field(default_factory=AcceptAnyArtifact)

    def __init__(
        self,
        components: Mapping[str, ComponentSurface] | None = None,
        harness: HarnessInfo | None = None,
        validator: ArtifactValidator | None = None,
        *,
        loader: ArtifactLoader | None = None,
        inference: InferenceHooks | None = None,
        files: FileTree | None = None,
    ) -> None:
        declared = {} if components is None else components
        if loader is not None or inference is not None or files is not None:
            if not isinstance(declared, Mapping):
                raise ValueError("surface components must be a mapping of component name to ComponentSurface")
            if not declared:
                declared = {FLAT_COMPONENT: ComponentSurface(loader=loader, inference=inference, files=files)}
            elif len(declared) == 1:
                ((name, component),) = declared.items()
                if not isinstance(component, ComponentSurface):
                    raise ValueError(f"component {name!r} must be a ComponentSurface")
                declared = {
                    name: replace(
                        component,
                        loader=component.loader if loader is None else loader,
                        inference=component.inference if inference is None else inference,
                        files=component.files if files is None else files,
                    )
                }
            else:
                raise ValueError(
                    f"surface serves components {list(declared)}: set loader, inference and files per component"
                )
        object.__setattr__(self, "components", declared)
        object.__setattr__(self, "harness", harness)
        object.__setattr__(self, "validator", AcceptAnyArtifact() if validator is None else validator)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping):
            raise ValueError("surface components must be a mapping of component name to ComponentSurface")
        validated: dict[str, ComponentSurface] = {}
        for name, component in self.components.items():
            if not isinstance(component, ComponentSurface):
                raise ValueError(f"component {name!r} must be a ComponentSurface")
            validated[validate_component_name(name)] = component
        object.__setattr__(self, "components", MappingProxyType(validated))
        loaders = [name for name, component in validated.items() if component.loader is not None]
        if len(loaders) > 1:
            raise ValueError(f"at most one component loads into a runtime, not {loaders}")
        trees = [name for name, component in validated.items() if component.files is not None]
        if len(trees) > 1:
            raise ValueError(f"at most one component exposes a client-pulled file tree, not {trees}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.components)

    @property
    def single(self) -> bool:
        """True when the release is flat: the whole artifact is the one component, or there is none."""
        return len(self.components) <= 1

    @property
    def loader_component(self) -> str | None:
        """The component a runtime loads, if any."""
        return next((name for name, component in self.components.items() if component.loader is not None), None)

    @property
    def files_component(self) -> str | None:
        """The component a client pulls as a file tree, if any."""
        return next((name for name, component in self.components.items() if component.files is not None), None)

    def component_artifact(self, artifact: Artifact, name: str) -> Artifact:
        """``name``'s view of a release: the artifact itself when the release is flat."""
        return artifact if self.single else artifact.component(name)

    @property
    def loader(self) -> ArtifactLoader | None:
        """The runtime-loaded component's loader, for inspection; lifecycle calls go through this surface."""
        name = self.loader_component
        return None if name is None else self.components[name].loader

    @property
    def inference(self) -> InferenceHooks | None:
        """The request hooks for a release: each component's, in declaration order."""
        bound = tuple(
            (name, component.inference)
            for name, component in self.components.items()
            if component.inference is not None
        )
        if not bound:
            return None
        if self.single:
            return bound[0][1]
        if any(isinstance(hooks, LeasingInferenceHooks) for _, hooks in bound):
            return LeasingChainedInferenceHooks(bound)
        return ChainedInferenceHooks(bound)

    def inference_for_evaluation(self, component: str | None) -> InferenceHooks | None:
        """The request hooks of an evaluation call that runs a candidate of ``component``: every other component's.

        The episode renders its own candidate of that component, so the served
        release's hooks for it would mix the served content into the
        candidate's calls (a skill catalog, for one); the rest of the release
        (weights, request defaults) is served as it is. ``None`` names the only
        component of a flat release and no component of a composed one.
        """
        if component is not None and component not in self.components:
            raise ValueError(f"the release has no component {component!r}")
        if self.single:
            return None
        bound = tuple(
            (name, hooks.inference)
            for name, hooks in self.components.items()
            if name != component and hooks.inference is not None
        )
        if not bound:
            return None
        if any(isinstance(hooks, LeasingInferenceHooks) for _, hooks in bound):
            return LeasingChainedInferenceHooks(bound)
        return ChainedInferenceHooks(bound)

    @property
    def files(self) -> FileTree | None:
        name = self.files_component
        if name is None:
            return None
        tree = self.components[name].files
        if tree is None or self.single:
            return tree
        return ComponentFileTree(name, tree)

    def validate(self, artifact: Artifact) -> None:
        """Run the release's own admission check, then every component's against its view of ``artifact``."""
        self.validator.validate(artifact)
        for name, component in self.components.items():
            component.validator.validate(self.component_artifact(artifact, name))

    def component_changed(self, artifact: Artifact, previous: Artifact, name: str) -> bool:
        """Whether ``name``'s content differs between two releases.

        A flat release's content id is its one component's, so the refs
        answer without materializing anything. A composed release answers from
        its manifest; a release without one counts as changed.
        """
        if self.single:
            return artifact.ref.content_id != previous.ref.content_id
        current = artifact.materialize().components if not artifact.is_live else None
        before = previous.materialize().components if not previous.is_live else None
        if current is None or before is None or name not in current.entries or name not in before.entries:
            return True
        return current.entries[name].content_id != before.entries[name].content_id

    def recover(
        self,
        current: ArtifactRef | None,
        checkpoint: ArtifactRef,
        runtime: ServingRuntime | None,
    ) -> ArtifactRef:
        """The head the runtime can still serve after a restart; the checkpoint without a loader."""
        loader = self.loader
        return checkpoint if loader is None else loader.recover(current, checkpoint, runtime)

    def load(self, artifact: Artifact, runtime: ServingRuntime | None) -> None:
        """Load the runtime-loaded component of a rollback target; nothing without a loader."""
        name = self.loader_component
        if name is None:
            return
        loader = self.components[name].loader
        if loader is not None:
            loader.load(self.component_artifact(artifact, name), runtime)

    def activate(
        self,
        artifact: Artifact,
        runtime: ServingRuntime | None,
        *,
        source: Artifact | None = None,
        previous: Artifact | None = None,
    ) -> None:
        """Make the runtime-loaded component servable when its loader activates.

        ``previous`` is the release served before this one: a component whose
        content it already served is not activated again.
        """
        name = self.loader_component
        if name is None:
            return
        loader = self.components[name].loader
        if not isinstance(loader, ArtifactActivator):
            return
        if previous is not None and not self.component_changed(artifact, previous, name):
            return
        loader.activate(
            self.component_artifact(artifact, name),
            runtime,
            source=None if source is None else self.component_artifact(source, name),
        )


__all__ = [
    "AcceptAnyArtifact",
    "AdapterWeightRuntime",
    "ArtifactActivator",
    "ArtifactLoader",
    "ComponentSurface",
    "FileTree",
    "HarnessInfo",
    "InferenceHooks",
    "InferenceLease",
    "LeasingInferenceHooks",
    "ServingRuntime",
    "Surface",
    "WeightRuntime",
]
