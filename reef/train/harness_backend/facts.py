"""What the harness backend states from its settings alone, without a method, an executor or a component.

``HarnessFacts`` reads the seed tree and the adapter of ``HarnessSettings``. ``HarnessTrainingComponent`` holds the
value and answers from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.artifact.artifact import Artifact
from reef.artifact.components import TextFileTree
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.errors import ConfigError
from reef.core.values import JsonValue
from reef.core.weight_layout import ArtifactFormat
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, DescriptorError
from reef.harness.tree import HarnessTree, validate_tree
from reef.storage.job_data import JobDataKind
from reef.train.component import ScenarioRoot, TrainingBackendFacts
from reef.train.harness_backend.settings import HarnessSettings

#: The name of the harness training component, which its jobs carry.
HARNESS_COMPONENT = "harness"


@dataclass(frozen=True)
class HarnessFacts(TrainingBackendFacts):
    """The facts of one harness evolution deployment: its seed tree and its adapter.

    Values: ``seed`` is the first tree of a new scenario. ``descriptor`` is the adapter that the trees render for.

    The members are those of ``TrainingBackendFacts``. A snapshot is a tree, so ``artifact_format`` is None. The
    component keeps the state of each scenario component apart, so scenarios share it; the jobs of one scenario
    run one at a time. The backend declares no parameters schema. ``check_tree(tree)`` raises ``ValueError`` for a
    tree that fails the adapter's validation; ``tree_in(directory)`` is the checked tree of the snapshot in
    ``directory``, its text files without the repository's own files (``TextFileTree``). ``default_start`` is
    the seed tree's files, as ``HarnessEvolutionRecipe.default_start`` gives them. ``check_root_start`` and
    ``check_import_start`` accept files that are a tree that passes ``check_tree``. ``check_artifact`` accepts
    files that decode as such a tree, with nothing left to check on a service.

    Purpose: ``from_settings`` makes one from the deployment's ``HarnessSettings``; ``HarnessTrainingComponent``
    makes one from the same settings and answers from it.
    """

    seed: HarnessTree
    descriptor: AdapterDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.seed, HarnessTree) or not isinstance(self.descriptor, AdapterDescriptor):
            raise TypeError("HarnessFacts holds a HarnessTree and an AdapterDescriptor")

    @classmethod
    def from_settings(cls, settings: HarnessSettings) -> Self:
        """The facts of the deployment whose configuration is ``settings``; ``ConfigError`` when
        ``evolution.adapter`` names no adapter."""
        try:
            descriptor = get_adapter(settings.evolution.adapter)
        except DescriptorError as exc:
            raise ConfigError(str(exc)) from exc
        return cls(settings.seed, descriptor)

    @property
    def name(self) -> str:
        return HARNESS_COMPONENT

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return None

    @property
    def shares_scenarios(self) -> bool:
        return True

    @property
    def concurrent_jobs(self) -> bool:
        return False

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        return {"seed": self.seed.checksum()}

    def default_start(self) -> DefaultStart:
        """The seed tree's files; the empty seed makes a root with no files."""
        return DefaultStart(files=dict(self.seed.files), inputs=self.default_start_inputs)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any] | None:
        return None

    def check_tree(self, tree: HarnessTree) -> None:
        """Raise ``ValueError`` for a tree that fails validation for the deployment's adapter."""
        validation = validate_tree(tree, self.descriptor)
        if not validation.passed:
            raise ValueError(f"the harness tree fails validation: {validation.message()}")

    def tree_in(self, directory: Path) -> HarnessTree:
        """The tree of the snapshot in ``directory``, checked (``check_tree``)."""
        tree = HarnessTree(dict(TextFileTree().read_files(Artifact.local(directory)) or {}))
        self.check_tree(tree)
        return tree

    def check_root_start(self, root: ScenarioRoot) -> None:
        self.tree_in(root.local_directory())

    def check_import_start(self, artifact_dir: Path) -> None:
        self.tree_in(artifact_dir)

    def check_artifact(self, files: ArtifactFiles) -> Mapping[str, JsonValue] | None:
        self.check_tree(HarnessTree({path: files.read_bytes(path).decode("utf-8") for path in files.paths()}))
        return None


__all__ = ["HARNESS_COMPONENT", "HarnessFacts"]
