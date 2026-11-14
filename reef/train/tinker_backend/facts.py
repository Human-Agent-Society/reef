"""What the Tinker and the Spindle full-parameter backends state from their settings alone, without a client.

``TinkerFacts`` and ``SpindleFullFacts`` are the two backends' own values; they share nothing but the contract
(``reef.train.component.TrainingBackendFacts``). Each component holds its value and answers from it. The module
imports no Tinker SDK, so control and inference read the facts without one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.artifact.base_root import BASE_FILE, BaseRoot, read_base_root
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.artifact.sources import REFERENCE_FILE, HuggingFaceRevision, read_reference
from reef.core.values import JsonValue
from reef.core.weight_layout import ArtifactFormat
from reef.storage.job_data import JobDataKind
from reef.tinker.checkpoint import MANIFEST
from reef.train.component import ScenarioRoot, TrainingBackendFacts
from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.config import SpindleFullSettings, TinkerSettings
from reef.train.tinker_backend.parameters import tinker_parameters_schema


@dataclass(frozen=True)
class TinkerFacts(TrainingBackendFacts):
    """The facts of one Tinker deployment, which trains a LoRA of ``model`` on ``service``.

    Values: ``model`` is the non-empty base model. ``service`` is the base URL of the Tinker-compatible service,
    None for hosted Tinker. ``lora_rank`` is a positive int.

    The members are those of ``TrainingBackendFacts``. A snapshot holds only its checkpoint manifest, so
    ``artifact_format`` is None. Each job trains from its own start, so scenarios and the jobs of one scenario
    share the component. ``validated(checkpoint)`` returns ``checkpoint`` when it is a LoRA of ``model`` and
    ``lora_rank`` that ``service`` holds, else raises ``ValueError``; ``checkpoint_in(directory)`` is the validated
    checkpoint of the manifest in ``directory``. ``default_start`` is a base root of ``default_start_inputs``
    (``reef.artifact.base_root``): a job from it trains a new LoRA of ``model``. ``base_root_in(directory)`` is
    the base root that ``directory`` holds, or None; it raises ``ValueError`` for a base root of other inputs.
    ``check_root_start`` and ``check_import_start`` accept files that are such a base root or hold a manifest that
    ``validated`` takes. ``check_artifact`` accepts such a base root with nothing left to check; for a manifest that
    ``validated`` takes it returns ``{"manifest": <the manifest>}``, whose state path the component looks up on
    the service (``TinkerJobComponent.check_remote_artifact``).

    Purpose: ``TinkerComponentFactory.facts`` makes one from the deployment's ``TinkerSettings`` (``of``);
    ``TinkerTrainingComponent`` makes one from the same settings and answers from it.
    """

    model: str
    service: str | None
    lora_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("TinkerFacts.model must be a non-empty string")
        if self.service is not None and (not isinstance(self.service, str) or not self.service):
            raise ValueError("TinkerFacts.service must be a non-empty string or None")
        if isinstance(self.lora_rank, bool) or not isinstance(self.lora_rank, int) or self.lora_rank <= 0:
            raise ValueError("TinkerFacts.lora_rank must be a positive int")

    @classmethod
    def of(cls, model: str, settings: TinkerSettings) -> Self:
        """The facts of the component that trains ``model`` with ``settings``."""
        return cls(model, settings.base_url, settings.lora_rank)

    @property
    def name(self) -> str:
        return "tinker"

    @property
    def artifact_format(self) -> ArtifactFormat | None:
        return None

    @property
    def shares_scenarios(self) -> bool:
        return True

    @property
    def concurrent_jobs(self) -> bool:
        return True

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        inputs = {"model": self.model, "lora_rank": str(self.lora_rank)}
        if self.service is not None:
            inputs["service"] = self.service
        return inputs

    def default_start(self) -> DefaultStart:
        """A base root of ``default_start_inputs``: the untrained LoRA leaves the base model's outputs unchanged."""
        inputs = self.default_start_inputs
        return DefaultStart(files={BASE_FILE: BaseRoot(inputs).to_text()}, inputs=inputs)

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return tinker_parameters_schema(kind)

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self.model, service=self.service)
        if checkpoint.lora_rank != self.lora_rank:
            raise ValueError(
                f"the Tinker checkpoint is no rank {self.lora_rank} LoRA of {self.model}: "
                f"its rank is {checkpoint.lora_rank}"
            )
        return checkpoint

    def checkpoint_in(self, directory: Path) -> TinkerCheckpoint:
        """The checkpoint of the manifest in ``directory``, validated; ``ValueError`` without a manifest."""
        if not (directory / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        return self.validated(TinkerCheckpoint.read(directory))

    def base_root_in(self, directory: Path) -> BaseRoot | None:
        """The base root that the artifact in ``directory`` is, or None for an artifact without ``BASE_FILE``.

        Raise ``ValueError`` when the file is not readable, or when its inputs differ from
        ``default_start_inputs``: it stands for another model, rank or service than this deployment's.
        """
        base = read_base_root(directory)
        if base is not None:
            base.check_inputs(self.default_start_inputs)
        return base

    @property
    def start_files(self) -> tuple[str, ...]:
        """The base root's file and the checkpoint's manifest: a Tinker checkpoint holds no weights."""
        return (BASE_FILE, MANIFEST)

    def check_root_start(self, root: ScenarioRoot) -> None:
        self.check_import_start(root.local_directory())

    def check_import_start(self, artifact_dir: Path) -> None:
        if self.base_root_in(artifact_dir) is None:
            self.checkpoint_in(artifact_dir)

    def check_artifact(self, files: ArtifactFiles) -> Mapping[str, JsonValue] | None:
        if self.is_own_base_root(files):
            return None
        return {"manifest": self.validated(manifest_of(files)).to_dict()}


def manifest_of(files: ArtifactFiles) -> TinkerCheckpoint:
    """The checkpoint of the manifest that ``files`` hold; ``ValueError`` without a readable manifest."""
    if MANIFEST not in files.paths():
        raise ValueError(f"a Tinker artifact holds {MANIFEST}")
    try:
        return TinkerCheckpoint.from_dict(json.loads(files.read_bytes(MANIFEST)))
    except (TypeError, ValueError) as exc:
        # A manifest from outside Reef may have any shape; ``from_dict`` refuses unknown fields with TypeError.
        raise ValueError(f"{MANIFEST} is no Tinker checkpoint manifest: {exc}") from exc


@dataclass(frozen=True)
class SpindleFullFacts(TrainingBackendFacts):
    """The facts of one Spindle full-parameter deployment, which trains all the parameters of ``base`` on ``service``.

    Values: ``model`` is the base model, which is ``base.repo``. ``service`` is the base URL of the Spindle
    deployment, or None. ``base`` is the Hugging Face revision that the deployment's engine trains.

    The members are those of ``TrainingBackendFacts``. Every snapshot refers to full weights on the Hub, so
    ``artifact_format`` is ``ArtifactFormat("full")``. Each job trains from its own start, so scenarios and the jobs
    of one scenario share the component. ``validated(checkpoint)`` returns ``checkpoint`` when it holds all the
    parameters of ``model`` on ``service``, else raises ``ValueError``; ``checkpoint_in(directory)`` is the
    validated checkpoint of the manifest in ``directory``. ``check_base_files(directory)`` raises ``ValueError``
    when ``directory`` refers to another revision than ``base``. ``default_start`` is the reference to ``base``
    (``reef.artifact.sources.REFERENCE_FILE``), not a base root: the base's weights are a commit on the Hub.
    ``check_root_start`` and ``check_import_start`` accept files that hold a manifest that ``validated`` takes,
    and files without a manifest that refer to ``base`` or to nothing. ``check_artifact`` checks no weights
    itself, since they are on the Hub: for a manifest that ``validated`` takes beside a reference it returns
    ``{"manifest", "weights"}``, and for files without a manifest ``{"files": <their checksum>}``, which the
    component compares with the service and the Hub (``SpindleFullTrainingComponent.check_remote_artifact``).

    Purpose: ``SpindleFullComponentFactory.facts`` makes one from the deployment's ``SpindleFullSettings``
    (``of``); ``SpindleFullTrainingComponent`` makes one from the same settings and answers from it.
    """

    model: str
    service: str | None
    base: HuggingFaceRevision

    def __post_init__(self) -> None:
        if not isinstance(self.base, HuggingFaceRevision):
            raise TypeError("SpindleFullFacts.base must be a HuggingFaceRevision")
        if self.model != self.base.repo:
            raise ValueError(f"the base model {self.model} is not the repo of the base {self.base.uri}")
        if self.service is not None and (not isinstance(self.service, str) or not self.service):
            raise ValueError("SpindleFullFacts.service must be a non-empty string or None")

    @classmethod
    def of(cls, model: str, settings: SpindleFullSettings) -> Self:
        """The facts of the component that trains ``model`` with ``settings``."""
        return cls(model, settings.base_url, settings.base_revision)

    @property
    def name(self) -> str:
        return "spindle_full"

    @property
    def artifact_format(self) -> ArtifactFormat:
        """Full weights: each snapshot refers to a commit on the Hub that holds them."""
        return ArtifactFormat("full")

    @property
    def shares_scenarios(self) -> bool:
        return True

    @property
    def concurrent_jobs(self) -> bool:
        return True

    @property
    def default_start_inputs(self) -> Mapping[str, str]:
        return {"model": self.base.uri, "service": str(self.service)}

    def default_start(self) -> DefaultStart:
        """The reference to the base revision, as every artifact of the backend refers to its weights."""
        return DefaultStart(
            files={REFERENCE_FILE: json.dumps(self.base.to_dict()) + "\n"}, inputs=self.default_start_inputs
        )

    def parameters_schema(self, kind: JobDataKind) -> dict[str, Any]:
        return tinker_parameters_schema(kind)

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self.model, service=self.service)
        if checkpoint.lora_rank is not None:
            raise ValueError(f"the Tinker checkpoint is a rank {checkpoint.lora_rank} LoRA, not all the parameters")
        return checkpoint

    def checkpoint_in(self, directory: Path) -> TinkerCheckpoint:
        """The checkpoint of the manifest in ``directory``, validated; ``ValueError`` without a manifest."""
        if not (directory / MANIFEST).exists():
            raise ValueError(f"artifact is missing {MANIFEST}")
        return self.validated(TinkerCheckpoint.read(directory))

    def check_base_files(self, directory: Path) -> None:
        """Raise ``ValueError`` when the artifact in ``directory`` refers to another revision than ``base``."""
        revision = read_reference(directory)
        if revision is not None and revision != self.base:
            raise ValueError(f"the {self.name} training component starts only from its base {self.base.uri}")

    @property
    def start_files(self) -> tuple[str, ...]:
        """The checkpoint's manifest and the reference to the weights' revision: never the weights."""
        return (MANIFEST, REFERENCE_FILE)

    def check_root_start(self, root: ScenarioRoot) -> None:
        self.check_import_start(root.local_directory())

    def check_import_start(self, artifact_dir: Path) -> None:
        if (artifact_dir / MANIFEST).exists():
            self.checkpoint_in(artifact_dir)
        else:
            self.check_base_files(artifact_dir)

    def check_artifact(self, files: ArtifactFiles) -> Mapping[str, JsonValue] | None:
        if MANIFEST not in files.paths():
            return {"files": files.checksum()}
        manifest = self.validated(manifest_of(files)).to_dict()
        if REFERENCE_FILE not in files.paths():
            raise ValueError(f"a {self.name} snapshot holds {REFERENCE_FILE} beside its manifest")
        revision = HuggingFaceRevision.from_dict(json.loads(files.read_bytes(REFERENCE_FILE)))
        return {"manifest": manifest, "weights": revision.to_dict()}


__all__ = ["SpindleFullFacts", "TinkerFacts"]
