"""Slime as a training backend service on a Ray cluster: one Slime run for each job (RFC #1146).

``SlimeBackend`` runs each job in a new Slime training component, in a new work directory, and closes the component
when the job ends, so the Ray cluster's GPUs go back to the cluster. The job's artifact gives the run its start:

- ``release/`` holds the Hugging Face weights, which Slime loads (``--hf-checkpoint``) and keeps as its reference
  model (``--ref-load``); the configured ``base_model`` is the start of the URI ``base:``;
- ``state/`` holds the start that a checkpoint of this backend wrote (``start.json``) and ``megatron/``, the Megatron
  checkpoint of that rollout with its optimizer state. Slime loads it at its start (``--load``), so the job
  continues that training. Without ``state/``, the job starts a new training from the weights.

Each checkpoint becomes an artifact: the rollout's Hugging Face export as ``release/``, and its start with
``megatron/`` (the rollout's ``iter_<n>`` directory and the tracker) as ``state/``.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.train.algos.registry import job_training, resolve_objective
from reef.train.component import SavedCheckpoint, TrainingComponent, build_training_component
from reef.train.remote.components import BASE_URI, run_on_component
from reef.train.remote.contract import RELEASE_DIR, STATE_DIR, BackendInfo, RemoteJob
from reef.train.remote.server import JobRun, TrainingBackend
from reef.train.slime_backend.checkpoints import SlimeStart
from reef.train.slime_backend.reef_adapters.training_job.storage import MEGATRON_TRACKER

#: The directory of the Megatron checkpoint in an artifact's ``state/``.
MEGATRON_DIR = "megatron"


@dataclass(frozen=True)
class SlimeServiceSettings(PluginSettings):
    base_model: str = config_option(help="Local directory of the Hugging Face weights of the URI base:.")
    ray_address: str = config_option(help="Address of the Ray cluster that runs the jobs.")
    options: Mapping[str, Any] = config_option(default_factory=dict, help="Slime options of every job.")
    work_dir: str = config_option("/tmp/reef-slime-jobs", help="Directory of the jobs' work directories.")
    train_timeout_s: float | None = config_option(None, help="Seconds that one Slime call can take; null: no bound.")

    def __post_init__(self) -> None:
        if not self.base_model.strip() or not self.ray_address.strip():
            raise ValueError("base_model and ray_address must be non-empty")
        placed = {"hf-checkpoint", "ref-load", "load", "save", "save-hf", "save-interval", "num-rollout"}
        clash = sorted(placed & set(self.options))
        if clash:
            raise ValueError(f"the Slime backend sets these options itself: {clash}")


class SlimeBackend(TrainingBackend):
    """Slime on a Ray cluster, one Slime run for each job, from the job's artifact.

    Values: ``settings`` are the service's settings. ``slime_component`` builds the component of one job on
    ``settings.ray_address``. ``info`` comes from a component of the base model that never starts.
    """

    def __init__(self, settings: SlimeServiceSettings) -> None:
        self.settings = settings
        with tempfile.TemporaryDirectory() as work:
            probe = self.slime_component(Path(settings.base_model), Path(work) / MEGATRON_DIR, Path(work), "")
            try:
                self._info = BackendInfo(probe.name, probe.release_format, probe.shares_scenarios)
            finally:
                probe.close()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(SlimeServiceSettings.parse(config, plugin_class=cls))

    @property
    def info(self) -> BackendInfo:
        return self._info

    def slime_component(self, weights: Path, megatron: Path, work: Path, loss_family: str) -> TrainingComponent:
        """Slime's component of one job, which starts from ``weights`` and the Megatron checkpoint in ``megatron``,
        saves there, and exports each rollout under ``work/hf``."""
        options = {
            **self.settings.options,
            "hf-checkpoint": str(weights),
            "ref-load": str(weights),
            "load": str(megatron),
            "save": str(megatron),
            # Megatron needs it with --save; the job's save_interval decides when a checkpoint is saved.
            "save-interval": "1",
            "save-hf": str(work / "hf" / "{rollout_id}"),
            "num-rollout": "100000000",
        }
        config = {
            "type": "reef.train.slime_backend.launch:component_factory",
            "loss_family": loss_family,
            "recipe": "the Slime backend service",
            "ray_address": self.settings.ray_address,
            "namespace": "reef",
            "execution": {"execution": {"training": "ray"}},
            "train_timeout_s": self.settings.train_timeout_s,
            "settings": {
                "model_path": str(weights),
                "training_backend_options": options,
                "teacher_model_path": None,
                "teacher_timeout": 300.0,
            },
        }
        return build_training_component(config, model_path=str(weights))

    def run(self, job: RemoteJob, artifact: Path | None, data: Path, run: JobRun) -> Mapping[str, Any]:
        Path(self.settings.work_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.settings.work_dir) as directory:
            work = Path(directory)
            megatron = work / MEGATRON_DIR
            weights, state = self.starts(job, artifact)
            if state is not None and (state / MEGATRON_DIR).is_dir():
                shutil.copytree(state / MEGATRON_DIR, megatron)
            method, _ = job_training(job.parameters)
            component = self.slime_component(weights, megatron, work, resolve_objective(method.objective).loss_family)
            try:
                component.start_jobs()
                start = str(state) if state is not None else str(component.base_start)
                return run_on_component(
                    component,
                    start,
                    job,
                    data,
                    run,
                    add_state=lambda checkpoint, target: add_megatron(checkpoint, megatron, target),
                )
            finally:
                component.close()

    def starts(self, job: RemoteJob, artifact: Path | None) -> tuple[Path, Path | None]:
        """The weights that the job starts from, and the state that it continues, or None for a new training."""
        if job.artifact_uri == BASE_URI:
            return Path(self.settings.base_model), None
        if artifact is None:
            raise ValueError(f"the Slime backend cannot read the artifact {job.artifact_uri!r}")
        if not (artifact / RELEASE_DIR).is_dir():
            return artifact, None
        state = artifact / STATE_DIR
        return artifact / RELEASE_DIR, state if state.is_dir() else None


def add_megatron(checkpoint: SavedCheckpoint, megatron: Path, state: Path) -> None:
    """Add to ``state`` the Megatron checkpoint of ``checkpoint``'s rollout: its ``iter_<n>`` directory and a tracker
    that names it."""
    rollout_id = SlimeStart.read(Path(checkpoint.path)).rollout_id
    if rollout_id is None:
        return
    iteration = f"iter_{rollout_id:07d}"
    target = state / MEGATRON_DIR
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(megatron / iteration, target / iteration)
    (target / MEGATRON_TRACKER).write_text(str(rollout_id), encoding="utf-8")


__all__ = ["MEGATRON_DIR", "SlimeBackend", "SlimeServiceSettings", "add_megatron"]
