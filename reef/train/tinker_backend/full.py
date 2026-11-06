"""Spindle's full-parameter training as a training component (RFC #996).

Spindle, the Tinker-compatible server, trains all the parameters of a model on a deployment's full-parameter engine.
The engine's base is a Hugging Face revision; a job starts from those weights or from a saved checkpoint of them.
Each saved checkpoint's weights are exported to a Hugging Face repo, so every snapshot refers to weights on the Hub,
which any serving runtime loads. The component shares Tinker's job loop (``TinkerJobComponent``); its
parameterization, its base and its snapshots differ.
"""

from __future__ import annotations

import json
from pathlib import Path

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.artifact.sources import REFERENCE_FILE, HuggingFaceRevision, HuggingFaceRevisionFiles, read_reference
from reef.core.weight_layout import ArtifactFormat
from reef.tinker.checkpoint import MANIFEST
from reef.tinker.sampling import TinkerSampler
from reef.train.tinker_backend.checkpoint import FreshStart, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.component import TinkerBaseFiles, TinkerJobComponent
from reef.train.tinker_backend.config import SpindleFullConfig
from reef.train.tinker_backend.spindle_export import CheckpointExporter


class SpindleFullTrainingComponent(TinkerJobComponent):
    """Train all the parameters of ``config.base`` on a Spindle deployment's full-parameter engine.

    The engine's ``model`` is the base's repo, which is ``base_model``, and its weights are the base's revision.
    Its artifacts are full weights (kind ``model``):

    - the base: a reference to the base revision (``REFERENCE_FILE``), which ``base/`` holds; a job from it starts a
      new model with the base's own weights (``FreshStart``). ``base/`` is written on first use: before the training
      process runs jobs, or when a new scenario's root snapshot holds the default start (``prepared_base``);
    - a saved checkpoint: a manifest of all the parameters (``lora_rank`` null) of the base model on the
      configured service, whose ``state_path`` a job restores, beside a reference to the commit of
      ``config.release_repo`` that holds the checkpoint's weights, which ``exporter`` made.

    With ``config.score_centering``, its policy-gradient jobs train with score centering on Spindle's ``cispo``
    (``TinkerJobComponent``).

    So every snapshot refers to its weights on the Hub (``artifact_format`` ``"full"``): Spindle's sampler serves a
    manifest's ``sampler_path``, and any other runtime loads the referenced weights. An import takes exactly these
    forms: a manifest that ``validated`` takes beside a reference that resolves to full weights, or files that are
    the base's files (equal content ID), whether Reef keeps them by reference or as a copy. Any other full weights
    are refused, since the engine trains its base only. So every artifact without a manifest has the base's
    weights: ``train-from`` starts it fresh, and Spindle's sampler serves it as the base model.
    """

    def __init__(
        self,
        base_model: str,
        config: SpindleFullConfig,
        client: TinkerClient,
        exporter: CheckpointExporter,
        *,
        sampler: TinkerSampler | None = None,
    ) -> None:
        self._revision = config.base_revision
        self._exporter = exporter
        if base_model != self._revision.repo:
            raise ValueError(f"the base model {base_model} is not the repo of the base {self._revision.uri}")
        #: The content ID of the base's files, read from the Hub at the first import.
        self.base_checksum: str | None = None
        super().__init__(base_model, config, client, sampler=sampler, score_centering=config.score_centering_settings)

    @property
    def name(self) -> str:
        return "spindle_full"

    @property
    def artifact_format(self) -> ArtifactFormat:
        """Full weights: each snapshot refers to a commit on the Hub that holds them."""
        return ArtifactFormat("full")

    def complete_artifact(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Export the checkpoint's weights to the snapshot repo, and refer to the commit beside the manifest."""
        revision = self._exporter.export(checkpoint.state_path, self._revision)
        (directory / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()) + "\n", encoding="utf-8")

    def prepare_base(self) -> None:
        """Write the reference to the configured base when ``base/`` holds no reference or another one.

        The base start is a function of ``base`` alone, so it follows the configuration. A scenario whose root snapshot
        refers to another base cannot train: ``import_start`` refuses that reference.
        """
        if read_reference(self._base) != self._revision:
            self._base.mkdir(parents=True, exist_ok=True)
            (self._base / REFERENCE_FILE).write_text(json.dumps(self._revision.to_dict()) + "\n", encoding="utf-8")

    def validated(self, checkpoint: TinkerCheckpoint) -> TinkerCheckpoint:
        checkpoint.check_compatible(self._model, service=self._config.base_url)
        if checkpoint.lora_rank is not None:
            raise ValueError(f"the Tinker checkpoint is a rank {checkpoint.lora_rank} LoRA, not all the parameters")
        return checkpoint

    def start_at(self, directory: Path) -> TinkerCheckpoint | FreshStart:
        """A manifest that ``validated`` takes, whatever it refers to; else a reference to the base, which starts
        fresh; else ``ValueError``."""
        if (directory / MANIFEST).is_file():
            return super().start_at(directory)
        revision = read_reference(directory)
        if revision != self._revision:
            named = "no Tinker checkpoint manifest" if revision is None else f"a reference to {revision.uri}"
            raise ValueError(f"the start {directory} holds {named}, not the base {self._revision.uri}")
        return FreshStart(self._model)

    def default_start(self) -> DefaultStart:
        """The base's release: the reference to the base revision."""
        inputs = {"model": self._revision.uri, "service": str(self._config.base_url)}
        return DefaultStart(model_files=TinkerBaseFiles(self), inputs=inputs)

    def check_artifact(self, files: ArtifactFiles) -> None:
        """A full-parameter manifest of this component's model and service whose state resolves, or the base's files."""
        if MANIFEST in files.paths():
            self.check_manifest(files)
            if REFERENCE_FILE not in files.paths():
                raise ValueError(f"a {self.name} release holds {REFERENCE_FILE} beside its manifest")
            revision = HuggingFaceRevision.from_dict(json.loads(files.read_bytes(REFERENCE_FILE)))
            weights = HuggingFaceRevisionFiles(revision.repo, revision.commit).paths()
            if "config.json" not in weights or not any(path.endswith(".safetensors") for path in weights):
                raise ValueError(f"{revision.uri} holds no full weights: config.json and safetensors")
        elif files.checksum() != self.base_files_id():
            raise ValueError(
                f"the {self.name} training component trains only its base {self._revision.uri}: an import of full "
                "weights holds the base's files, or a manifest of a full-parameter checkpoint of it"
            )

    def import_start(self, artifact_dir: Path) -> str:
        """A manifest's start, else the base start: every artifact of this component without a manifest has the base's
        files, since its base holds them and ``check_artifact`` admits no other."""
        if (artifact_dir / MANIFEST).exists():
            return self.import_manifest(artifact_dir)
        revision = read_reference(artifact_dir)
        if revision is not None and revision != self._revision:
            raise ValueError(f"the {self.name} training component starts only from its base {self._revision.uri}")
        return self.base_start

    def base_files_id(self) -> str:
        """The content ID of the base's files on the Hub, listed once."""
        if self.base_checksum is None:
            self.base_checksum = HuggingFaceRevisionFiles(self._revision.repo, self._revision.commit).checksum()
        return self.base_checksum
