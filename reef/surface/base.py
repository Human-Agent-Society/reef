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

from reef.artifact.artifact import Artifact, ArtifactValidator
from reef.artifact.files import ArtifactFiles
from reef.core.components import validate_component_name
from reef.core.evaluation import ARTIFACT_KINDS, ArtifactKind
from reef.core.model_metadata import ModelMetadata


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
    """A runtime that serves published model weights.

    A runtime load ID names the weights that serve a request. A runtime that
    loads weights makes a new ``RuntimeLoadId`` string for each load. A
    runtime that loads nothing, such as a proxy, uses the release ID, since the
    release alone names what serves.
    """

    @abstractmethod
    def current_runtime_load_id(self) -> str | None:
        """The runtime load ID of the last release the runtime activated, of any scenario.

        ``None`` when the runtime activated none, or loads nothing and names no load.
        """

    @abstractmethod
    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the published release ``artifact``; return the runtime load ID that serves it.

        Reef calls it at a commit that activates a release, at an activation
        and at a start, before traffic routes to the release. ``scenario`` names
        the release's scenario exactly when the runtime serves a separate
        adapter per scenario. A raise leaves the runtime serving what it served
        before.
        """


class AdapterWeightRuntime(WeightRuntime):
    """A weight runtime that can inspect each scenario's resident adapter."""

    @abstractmethod
    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None: ...


class ArtifactLoader(ABC):
    """Make a component's published releases servable on the runtime."""

    @abstractmethod
    def activate(self, artifact: Artifact, runtime: ServingRuntime | None, *, source: Artifact | None = None) -> str:
        """Make the published release ``artifact`` servable; return the runtime load ID that serves it.

        Called once a release is final: after startup recovery has read the
        active artifact from the commit history, after a commit has published
        its release, and at an activation. It runs before the scenario routes
        traffic to the release. ``source`` names the staged artifact whose
        bytes a commit published.
        """

    def served_runtime_load_id(self, artifact: Artifact, runtime: ServingRuntime | None) -> str | None:
        """Return the runtime load ID that serves the committed ``artifact``, or None when it is unknown."""
        return None


class FileTree(ABC):
    """A client-readable file tree derived from an artifact."""

    @abstractmethod
    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None: ...


@dataclass(frozen=True)
class HarnessInfo:
    """What the harness routes need beyond the file tree: the served model and the models a client may pick."""

    served_model: str | None = None
    #: The served model's API dialect: Reef forwards calls unchanged, so an installed client must speak it.
    served_api: str = "openai"
    #: Further models the installed client may pick from; the served one stays the default.
    client_models: tuple[str, ...] = ()
    #: Explicit overrides by exact name; never reuse discovered metadata for a different endpoint.
    model_metadata: Mapping[str, ModelMetadata] = field(default_factory=dict)
    #: Discovered metadata for served_model on the default deployment endpoint.
    served_metadata: ModelMetadata | None = None


class ArtifactLayout(ABC):
    """The directory layout of one component's artifacts, which an artifact import must meet (RFC #980).

    ``check(files)`` returns when ``files`` are an artifact of the component, in the layout of the component's
    releases, and every remote path they name resolves on the component's service. Otherwise it raises
    ``ValueError`` naming what does not hold. ``files`` is the source's files, wherever the source keeps them, so a
    check reads files through it and reads no more of them than it needs.

    ``resolves_references`` is true when every reader of the component's artifacts resolves a reference
    (``REFERENCE_FILE``, RFC #995) to the revision's files; then an import from a source that keeps its files fixed
    stores the reference instead of a copy. The default is false: an import always stores a copy.

    Purpose: a recipe binds it on ``ComponentSurface.layout``; the committer checks an import with it and decides
    what the import stores.
    """

    @abstractmethod
    def check(self, files: ArtifactFiles) -> None: ...

    @property
    def resolves_references(self) -> bool:
        return False


@dataclass(frozen=True)
class ComponentSurface:
    """How one named component of a release is admitted, loaded and read, and what kind it is.

    ``None`` means the capability is absent. ``validator`` runs before the
    component is published or restored; it defaults to accepting anything.
    ``kind`` is the ``ArtifactKind`` of every artifact of the component, or
    ``None`` when the component is no side of a policy, such as the records
    of a trainer that serves nothing, or when the surface declares no kind.
    ``layout`` checks an imported directory; ``None`` means the component
    takes no import.
    """

    validator: ArtifactValidator = field(default_factory=AcceptAnyArtifact)
    loader: ArtifactLoader | None = None
    files: FileTree | None = None
    kind: ArtifactKind | None = None
    layout: ArtifactLayout | None = None

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"component kind must be one of {ARTIFACT_KINDS} or None, got {self.kind!r}")


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
    directory per component; the ``loader`` and ``files`` views then route
    each capability to its component's directory. No surface changes a
    request: the inference service only forwards it (RFC #1048). At most
    one component may load into a runtime, and at most one may expose a
    client-pulled file tree. Every recipe binds an instance of this same type.

    The ``loader`` and ``files`` keywords of the surface before
    components existed still build one: a single component,
    ``FLAT_COMPONENT``, or, beside one declared component (as
    ``dataclasses.replace`` passes it), that component with those
    capabilities replaced. A surface of several components takes them per
    component only.
    """

    components: Mapping[str, ComponentSurface] = field(default_factory=dict)
    harness: HarnessInfo | None = None
    #: The whole release's own admission check, run before each component's: on the release a step publishes and
    #: on a release an activation serves. A recipe binds its checks on ``ComponentSurface.validator``; this
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
        files: FileTree | None = None,
    ) -> None:
        declared = {} if components is None else components
        if loader is not None or files is not None:
            if not isinstance(declared, Mapping):
                raise ValueError("surface components must be a mapping of component name to ComponentSurface")
            if not declared:
                declared = {FLAT_COMPONENT: ComponentSurface(loader=loader, files=files)}
            elif len(declared) == 1:
                ((name, component),) = declared.items()
                if not isinstance(component, ComponentSurface):
                    raise ValueError(f"component {name!r} must be a ComponentSurface")
                declared = {
                    name: replace(
                        component,
                        loader=component.loader if loader is None else loader,
                        files=component.files if files is None else files,
                    )
                }
            else:
                raise ValueError(f"surface serves components {list(declared)}: set loader and files per component")
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
        current = artifact.materialize().components
        before = previous.materialize().components
        if current is None or before is None or name not in current.entries or name not in before.entries:
            return True
        return current.entries[name].content_id != before.entries[name].content_id

    def served_runtime_load_id(self, artifact: Artifact, runtime: ServingRuntime | None) -> str | None:
        """Return the runtime load ID that serves the committed release, or None when no component has a loader."""
        name = self.loader_component
        if name is None:
            return None
        loader = self.components[name].loader
        if loader is None:
            return None
        return loader.served_runtime_load_id(self.component_artifact(artifact, name), runtime)

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
        if loader is None:
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
    "ArtifactLoader",
    "ComponentSurface",
    "FileTree",
    "HarnessInfo",
    "ServingRuntime",
    "Surface",
    "WeightRuntime",
]
