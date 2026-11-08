"""The components a scenario's snapshots bind, and how each one is checked, imported and read.

A recipe declares them (``Recipe.build_components``). A snapshot binds named
components (``weights``, ``harness``, ``config``, ...). A scenario with no
component records traffic only. The inference runtime loads the component of
a weight kind; a client pulls the component that has a file tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from reef.artifact.artifact import REPOSITORY_FILES, Artifact, ArtifactValidator
from reef.artifact.files import ArtifactFiles
from reef.artifact.sources import SourceFiles
from reef.core.components import validate_component_name
from reef.core.evaluation import ARTIFACT_KINDS, ArtifactKind

#: The kinds of a component that the inference runtime loads.
WEIGHT_KINDS: tuple[ArtifactKind, ...] = ("model", "model_adapter")
#: The file of every harness artifact that holds its composition entries, at the tree's root. A client's pull
#: leaves it out: it describes the tree, and is no part of the harness.
ENTRIES_FILE = "reef-entries.json"


@dataclass(frozen=True)
class AcceptAnyArtifact(ArtifactValidator):
    """The admission check that accepts every artifact."""

    def validate(self, artifact: Artifact) -> None:
        return None


class FileTree(ABC):
    """A client-readable file tree derived from an artifact."""

    @abstractmethod
    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None: ...


class TextFileTree(FileTree):
    """Read every UTF-8 text file in an artifact, byte-faithfully, without the repository's own files."""

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None:
        local_path = artifact.materialize().local_path
        if local_path is None:
            return None
        files: dict[str, str] = {}
        for path in sorted(Path(local_path).rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(local_path))
            if relative in REPOSITORY_FILES or relative == ENTRIES_FILE:
                continue
            try:
                # Bytes-then-decode avoids universal-newline rewriting. The
                # pulled tree must equal the tree the validator checked.
                files[relative] = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue
        return files or None


class ArtifactLayout(ABC):
    """The directory layout of one component's artifacts, which an artifact import must meet (RFC #980).

    ``check(files)`` returns when ``files`` are an artifact of the component, in the layout of the component's
    snapshots, and every remote path they name resolves on the component's service. Otherwise it raises
    ``ValueError`` naming what does not hold. ``files`` is the source's files, wherever the source keeps them, so a
    check reads files through it and reads no more of them than it needs.

    ``resolves_references`` is true when every reader of the component's artifacts resolves a reference
    (``REFERENCE_FILE``, RFC #995) to the revision's files; then an import from a source that keeps its files fixed
    stores the reference instead of a copy. The default is false: an import always stores a copy.

    ``stored(files, directory)`` checks ``files``, then stores them in the empty directory ``directory``:
    ``files.reference`` when the source keeps the files fixed and ``resolves_references`` is true (RFC #995), else
    a copy of the files (RFC #980). It returns the unpublished artifact in ``directory``, with the ID
    ``artifact_id`` or a new one (``Artifact.local``).

    Purpose: a recipe binds it on ``ComponentSpec.layout``. The committer stores an import with it, and the
    scenario factory stores the artifact item that a new scenario starts from.
    """

    @abstractmethod
    def check(self, files: ArtifactFiles) -> None: ...

    @property
    def resolves_references(self) -> bool:
        return False

    def stored(self, files: SourceFiles, directory: Path, *, artifact_id: str | None = None) -> Artifact:
        self.check(files)
        if files.reference is None or not self.resolves_references:
            files.copy_into(directory)
        else:
            for path, data in files.reference.items():
                (directory / path).write_bytes(data)
        return Artifact.local(directory, artifact_id=artifact_id)


@dataclass(frozen=True)
class ComponentSpec:
    """How one named component of a snapshot is admitted, imported and read, and what kind it is.

    Values: ``validator`` admits an artifact of the component before it is
    published or activated; the default accepts any. ``kind`` is the
    ``ArtifactKind`` of every artifact of the component, or ``None`` when the
    component is no side of a policy. A component of a ``WEIGHT_KINDS`` kind
    is the one the inference runtime loads. ``files`` is the tree a client
    pulls, or ``None`` when no client pulls the component. ``layout`` checks
    an imported directory; ``None`` means the component takes no import.

    Purpose: a recipe makes it in ``build_components``; the scenario reads it.
    """

    validator: ArtifactValidator = field(default_factory=AcceptAnyArtifact)
    files: FileTree | None = None
    kind: ArtifactKind | None = None
    layout: ArtifactLayout | None = None

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in ARTIFACT_KINDS:
            raise ValueError(f"component kind must be one of {ARTIFACT_KINDS} or None, got {self.kind!r}")

    @property
    def loaded(self) -> bool:
        """True when the inference runtime loads the component's artifacts."""
        return self.kind in WEIGHT_KINDS


@dataclass(frozen=True)
class ScenarioComponents:
    """The components that every snapshot of one scenario binds, by name.

    Values: ``components`` maps valid component names to ``ComponentSpec``s, in
    declaration order; it may be empty. At most one component is ``loaded`` and
    at most one has ``files``. ``validator`` is the whole snapshot's admission
    check, run before each component's; it carries the
    ``build_artifact_validator`` of a recipe that serves no component.

    A snapshot with at most one component is flat: the whole artifact is the
    component. A snapshot with several keeps each one in a directory named after
    it, and ``component_artifact`` reads that directory.

    Purpose: a recipe makes it in ``build_components``; the scenario factory,
    the committer, the dispatcher and the service read it.
    """

    components: Mapping[str, ComponentSpec] = field(default_factory=dict)
    validator: ArtifactValidator = field(default_factory=AcceptAnyArtifact)

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping):
            raise ValueError("components must be a mapping of component name to ComponentSpec")
        validated: dict[str, ComponentSpec] = {}
        for name, component in self.components.items():
            if not isinstance(component, ComponentSpec):
                raise ValueError(f"component {name!r} must be a ComponentSpec")
            validated[validate_component_name(name)] = component
        object.__setattr__(self, "components", MappingProxyType(validated))
        loaded = [name for name, component in validated.items() if component.loaded]
        if len(loaded) > 1:
            raise ValueError(f"at most one component loads into the inference runtime, not {loaded}")
        trees = [name for name, component in validated.items() if component.files is not None]
        if len(trees) > 1:
            raise ValueError(f"at most one component exposes a client-pulled file tree, not {trees}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.components)

    @property
    def single(self) -> bool:
        """True when the snapshot is flat: the whole artifact is the one component, or there is none."""
        return len(self.components) <= 1

    @property
    def loaded_component(self) -> str | None:
        """The component the inference runtime loads, if any."""
        return next((name for name, component in self.components.items() if component.loaded), None)

    @property
    def files_component(self) -> str | None:
        """The component a client pulls as a file tree, if any."""
        return next((name for name, component in self.components.items() if component.files is not None), None)

    def import_target(self, component: str | None) -> tuple[str, ArtifactLayout]:
        """The component that an import into ``component`` lands in, and the layout that checks it.

        ``component`` may be ``None`` when the scenario serves one component. Raise ``ValueError`` when it is
        ``None`` and the scenario serves several, for a component that the scenario does not serve, and for a
        component without a ``layout``, which takes no import.
        """
        if component is None:
            if len(self.names) > 1:
                raise ValueError(f"the scenario serves components {list(self.names)}: name the component")
            if not self.names:
                raise ValueError("the scenario serves no component, so it takes no imported artifact")
            component = self.names[0]
        if component not in self.components:
            raise ValueError(f"the scenario has no component {component!r}")
        layout = self.components[component].layout
        if layout is None:
            raise ValueError(f"component {component!r} takes no imported artifact")
        return component, layout

    def component_artifact(self, artifact: Artifact, name: str) -> Artifact:
        """``name``'s view of a snapshot: the artifact itself when the snapshot is flat."""
        return artifact if self.single else artifact.component(name)

    def read_files(self, artifact: Artifact) -> Mapping[str, str] | None:
        """The client-pulled file tree of the snapshot ``artifact``; ``None`` when no component has one."""
        name = self.files_component
        if name is None:
            return None
        tree = self.components[name].files
        if tree is None:
            return None
        return tree.read_files(self.component_artifact(artifact, name))

    def validate(self, artifact: Artifact) -> None:
        """Run the snapshot's own admission check, then every component's against its view of ``artifact``."""
        self.validator.validate(artifact)
        for name, component in self.components.items():
            component.validator.validate(self.component_artifact(artifact, name))


__all__ = [
    "ENTRIES_FILE",
    "WEIGHT_KINDS",
    "AcceptAnyArtifact",
    "ArtifactLayout",
    "ComponentSpec",
    "FileTree",
    "ScenarioComponents",
    "TextFileTree",
]
