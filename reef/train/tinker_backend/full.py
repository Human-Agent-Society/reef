"""Spindle's full-parameter training as a training component (RFC #996).

Spindle, the Tinker-compatible server, trains all the parameters of a model on a deployment's full-parameter engine.
The engine's base is a Hugging Face revision; a job starts from those weights or from a saved checkpoint of them.
Each saved checkpoint's weights are exported to a Hugging Face repo, so every release refers to weights on the Hub,
which any serving runtime loads. The component shares Tinker's job loop (``TinkerJobComponent``); its
parameterization, its base and its releases differ.
"""

from __future__ import annotations

import json
from pathlib import Path

from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import BaseArtifactSource
from reef.artifact.sources import (
    REFERENCE_FILE,
    HuggingFaceRevision,
    HuggingFaceRevisionFiles,
    SourceFiles,
    read_reference,
)
from reef.runtime.jobs import ReleaseFormat
from reef.tinker.checkpoint import MANIFEST, FreshStart, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler
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
      process runs jobs, or when a repository imports the base artifact (``prepared_base``);
    - a saved checkpoint: a manifest of all the parameters (``lora_rank`` null) of the base model on the
      configured service, whose ``state_path`` a job restores, beside a reference to the commit of
      ``config.release_repo`` that holds the checkpoint's weights, which ``exporter`` made.

    With ``config.score_centering``, its policy-gradient jobs train with score centering on Spindle's ``cispo``
    (``TinkerJobComponent``).

    So every release refers to its weights on the Hub (``release_format`` ``"full"``): Spindle's sampler serves a
    manifest's ``sampler_path``, and any other runtime loads the referenced weights. An import takes exactly these
    forms: a manifest that ``validated`` takes beside a reference that resolves to full weights; a pinned Hugging
    Face commit that holds full weights, which Reef keeps by reference; or a copy of the base's files (equal
    content ID). A copy of other weights is refused: no runtime could tell it from the base. So an artifact without
    a manifest is a reference to a revision, or has the base's files. ``train-from`` starts the base and the base's
    files fresh, and refuses a reference to another revision, since the engine trains its base only. Spindle's
    sampler serves the base and the base's files as the base model (``checkpoint_sampler_path``).
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
        self.base_content_id: str | None = None
        super().__init__(base_model, config, client, sampler=sampler, score_centering=config.score_centering_settings)

    @property
    def name(self) -> str:
        return "spindle_full"

    @property
    def release_format(self) -> ReleaseFormat:
        """Full weights: each release refers to a commit on the Hub that holds them."""
        return ReleaseFormat("full")

    def complete_release(self, checkpoint: TinkerCheckpoint, directory: Path) -> None:
        """Export the checkpoint's weights to the release repo, and refer to the commit beside the manifest."""
        revision = self._exporter.export(checkpoint.state_path, self._revision)
        (directory / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()) + "\n", encoding="utf-8")

    def prepare_base(self) -> None:
        """Write the reference to the configured base when ``base/`` holds no reference or another one.

        The base start is a function of ``base`` alone, so it follows the configuration. A scenario whose repository
        was made from another base still does not open: its base artifact's inputs differ (``BaseArtifactSource``).
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

    def base_artifact_source(self) -> BaseArtifactSource:
        """The base's release: the reference to the base revision."""
        inputs = {"model": self._revision.uri, "service": str(self._config.base_url)}
        return BaseArtifactSource(model_files=TinkerBaseFiles(self), inputs=inputs)

    def check_release(self, files: ArtifactFiles) -> None:
        """A full-parameter manifest of this component's model and service whose state resolves, full weights that
        an import keeps by reference to a pinned Hugging Face commit, or a copy of the base's files."""
        if MANIFEST in files.paths():
            self.check_manifest(files)
            if REFERENCE_FILE not in files.paths():
                raise ValueError(f"a {self.name} release holds {REFERENCE_FILE} beside its manifest")
            revision = HuggingFaceRevision.from_dict(json.loads(files.read_bytes(REFERENCE_FILE)))
            check_full_weights(revision, HuggingFaceRevisionFiles(revision.repo, revision.commit).paths())
        elif isinstance(files, SourceFiles) and files.reference is not None:
            check_full_weights(
                HuggingFaceRevision.from_dict(json.loads(files.reference[REFERENCE_FILE])), files.paths()
            )
        elif files.content_id() != self.base_files_id():
            raise ValueError(
                f"a {self.name} import of a copy holds the files of its base {self._revision.uri}; import other full "
                "weights by their hf://<repo>@<commit>, which serving reads from the Hub"
            )

    def import_start(self, release: Path) -> str:
        """A manifest's start; else the base start for the base and a copy of the base's files, which
        ``check_release`` admits as the only copy; else ``ValueError`` for a reference to another revision."""
        if (release / MANIFEST).exists():
            return self.import_manifest(release)
        revision = read_reference(release)
        if revision is not None and revision != self._revision:
            raise ValueError(
                f"the {self.name} training component starts only from its base {self._revision.uri} or its own "
                f"checkpoints, not from {revision.uri}"
            )
        return self.base_start

    def base_files_id(self) -> str:
        """The content ID of the base's files on the Hub, listed once."""
        if self.base_content_id is None:
            self.base_content_id = HuggingFaceRevisionFiles(self._revision.repo, self._revision.commit).content_id()
        return self.base_content_id


def check_full_weights(revision: HuggingFaceRevision, paths: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless ``paths``, the files of ``revision``, are full weights: config.json and safetensors."""
    if "config.json" not in paths or not any(path.endswith(".safetensors") for path in paths):
        raise ValueError(f"{revision.uri} holds no full weights: config.json and safetensors")
