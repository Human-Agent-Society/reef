"""Versioned local manifests referencing durable Tinker checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerCheckpoint:
    base_model: str
    lora_rank: int
    state_path: str
    sampler_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.base_model or self.lora_rank <= 0:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")

    def validate_model(self, base_model: str, lora_rank: int) -> None:
        if (self.base_model, self.lora_rank) != (base_model, lora_rank):
            raise ValueError("Tinker checkpoint model/rank does not match the runtime")

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / MANIFEST, asdict(self))

    @classmethod
    def read(cls, directory: Path) -> TinkerCheckpoint:
        value = json.loads((directory / MANIFEST).read_text())
        if not isinstance(value, dict):
            raise ValueError("Tinker checkpoint manifest must be an object")
        return cls(**value)


def atomic_json(path: Path, value: Any) -> None:
    """Persist a complete manifest before exposing its directory to Reef."""
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tinker-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
