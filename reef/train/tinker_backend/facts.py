"""What the Tinker and the Spindle full-parameter backends state from their settings alone, without a client.

``TinkerFacts`` and ``SpindleFullFacts`` are the two backends' own values; they share nothing but the contract
(``reef.train.component.TrainingBackendFacts``). Each component holds its value and answers from it. The module
imports no Tinker SDK, so control and inference read the facts without one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.artifact.sources import HuggingFaceRevision, read_reference
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
    checkpoint of the manifest in ``directory``. ``check_root_start`` accepts a root whose files hold such a
    manifest.

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

    def check_root_start(self, root: ScenarioRoot) -> None:
        self.checkpoint_in(root.local_directory())


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
    when ``directory`` refers to another revision than ``base``. ``check_root_start`` accepts a root whose files
    hold a manifest that ``validated`` takes, and one without a manifest that refers to ``base`` or to nothing.

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

    def check_root_start(self, root: ScenarioRoot) -> None:
        directory = root.local_directory()
        if (directory / MANIFEST).exists():
            self.checkpoint_in(directory)
        else:
            self.check_base_files(directory)


__all__ = ["SpindleFullFacts", "TinkerFacts"]
