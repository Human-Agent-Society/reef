"""Spindle's full-parameter training as a training component (RFC #996).

Spindle, the Tinker-compatible server, trains all the parameters of a model on a deployment's full-parameter engine.
The engine's base is a Hugging Face revision; a job starts from those weights or from a saved checkpoint of them.
The component shares Tinker's job loop (``TinkerJobComponent``); only its parameterization and its base differ.
"""

from __future__ import annotations

import json
from pathlib import Path

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.artifact.sources import REFERENCE_FILE, HuggingFaceRevisionFiles, read_reference
from reef.tinker.checkpoint import MANIFEST, FreshStart, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.component import TinkerJobComponent
from reef.train.tinker_backend.config import SpindleFullConfig


class SpindleFullTrainingComponent(TinkerJobComponent):
    """Train all the parameters of ``config.base`` on a Spindle deployment's full-parameter engine.

    The engine's ``model`` is the base's repo, which is ``base_model``, and its weights are the base's revision.
    Its artifacts are full weights (kind ``model``):

    - the base: a reference to the base revision (``REFERENCE_FILE``), which ``base/`` holds; a job from it starts a
      new model with the base's own weights (``FreshStart``);
    - a saved checkpoint: a manifest of all the parameters (``lora_rank`` null) of the base model on the
      configured service, whose ``state_path`` a job restores.

    An import takes exactly these: a manifest that ``validated`` takes, or files that are the base's files (equal
    content ID), whether Reef keeps them by reference or as a copy. Any other full weights are refused, since the
    engine trains its base only. So every artifact without a manifest has the base's weights: ``train-from`` starts
    it fresh, and Spindle's sampler serves it as the base model.
    """

    def __init__(
        self,
        base_model: str,
        config: SpindleFullConfig,
        client: TinkerClient,
        *,
        sampler: TinkerSampler | None = None,
    ) -> None:
        self._revision = config.base_revision
        if base_model != self._revision.repo:
            raise ValueError(f"the base model {base_model} is not the repo of the base {self._revision.uri}")
        #: The content ID of the base's files, read from the Hub at the first import.
        self.base_content_id: str | None = None
        super().__init__(base_model, config, client, sampler=sampler)

    @property
    def name(self) -> str:
        return "spindle_full"

    def prepare_base(self) -> None:
        reference = self._base / REFERENCE_FILE
        if not reference.exists():
            self._base.mkdir(parents=True, exist_ok=True)
            reference.write_text(json.dumps(self._revision.to_dict()) + "\n", encoding="utf-8")

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self._model, service=self._config.base_url)
        if checkpoint.lora_rank is not None:
            raise ValueError(f"the Tinker checkpoint is a rank {checkpoint.lora_rank} LoRA, not all the parameters")
        return checkpoint

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """A manifest that ``validated`` takes, or a reference to the base, which starts fresh; else ``ValueError``."""
        revision = read_reference(directory)
        if revision is None:
            return super().start_at(directory)
        if revision != self._revision:
            raise ValueError(f"the start {directory} refers to {revision.uri}, not to the base {self._revision.uri}")
        return FreshStart(self._model)

    def base_artifact_source(self) -> BaseArtifactSource:
        """The base's release: the reference to the base revision."""
        inputs = {"model": self._revision.uri, "service": str(self._config.base_url)}
        return BaseArtifactSource(model_directory=self._base, inputs=inputs)

    def check_release(self, files: ArtifactFiles) -> None:
        """A full-parameter manifest of this component's model and service whose state resolves, or the base's files."""
        if MANIFEST in files.paths():
            self.check_manifest(files)
        elif files.content_id() != self.base_files_id():
            raise ValueError(
                f"the {self.name} training component trains only its base {self._revision.uri}: an import of full "
                "weights holds the base's files, or a manifest of a full-parameter checkpoint of it"
            )

    def import_start(self, release: Path) -> str:
        """A manifest's start, else the base start: every artifact of this component without a manifest has the base's
        files, since its base holds them and ``check_release`` admits no other."""
        if (release / MANIFEST).exists():
            return self.import_manifest(release)
        revision = read_reference(release)
        if revision is not None and revision != self._revision:
            raise ValueError(f"the {self.name} training component starts only from its base {self._revision.uri}")
        return self.base_start

    def base_files_id(self) -> str:
        """The content ID of the base's files on the Hub, listed once."""
        if self.base_content_id is None:
            self.base_content_id = HuggingFaceRevisionFiles(self._revision.repo, self._revision.commit).content_id()
        return self.base_content_id
