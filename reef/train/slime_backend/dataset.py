"""Dataset jobs on Slime: a prompt dataset that Slime rolls out, rewards and trains on its own.

A dataset job has no samples and the job parameter ``dataset``:
``{"path": <JSONL>, "epochs": <positive int, default 1>}``. The JSONL is in
Slime's prompt-data format, whose fields ``training.options`` names
(``input-key``, ``label-key``, ``metadata-key``, ``apply-chat-template``). The
deployment's loss family is ``slime`` (``reef.train.slime_backend.native``).

The job's driver records the job in the save root's ``DATASET_FILE``:
``job_id``, ``path``, ``skip_rollouts`` (the job's rollouts trained before
this driver) and ``token``, new for each driver. The serving session's rollout
manager reads the dataset through ``ReefDatasetSource``
(``reef.train.slime_backend.dataset_source``), which loads it again when the
token changes and skips the prompts of the skipped rollouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The dataset job that the serving session's rollout manager serves, in the save root.
DATASET_FILE = "reef-dataset.json"


@dataclass(frozen=True)
class DatasetSettings:
    """A dataset job's ``dataset`` parameter: an absolute ``path`` to a JSONL file and positive ``epochs``."""

    path: Path
    epochs: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("dataset.path must be an absolute path")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("dataset.epochs must be a positive int")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]) -> DatasetSettings:
        """Read the ``dataset`` parameter of a job's ``parameters``; raise ``ValueError`` for a malformed one."""
        value = parameters.get("dataset")
        if not isinstance(value, Mapping) or set(value) - {"path", "epochs"}:
            raise ValueError("the dataset parameter is an object with path and, optionally, epochs")
        path = value.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("dataset.path must be a non-empty string")
        return cls(Path(path), value.get("epochs", 1))


def save_root(save_hf: str) -> Path:
    """The save root of the Hugging Face checkpoint template ``save_hf``: ``<root>/hf/{rollout_id}``."""
    return Path(save_hf.format(rollout_id=0)).expanduser().resolve().parent.parent


__all__ = ["DATASET_FILE", "DatasetSettings", "save_root"]
