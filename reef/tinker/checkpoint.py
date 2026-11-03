"""The manifest of a Tinker release, as the serving side reads it.

A Tinker release holds only the manifest ``MANIFEST``. The training component
writes the whole manifest (``reef.train.tinker_backend.checkpoint``), and the
serving side reads only the sampler that it names (``TinkerRelease``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerRelease:
    """The sampler weights that a Tinker release serves, on the service that holds them.

    Values: ``base_model`` is non-empty. ``sampler_path`` is a ``tinker://`` path to the weights that the service's
    sampler serves; for a LoRA, it is the adapter that a serving runtime downloads to load into its own engines.
    ``service`` is the base URL of the Tinker-compatible service that holds the path (``TinkerConfig.base_url``), or
    ``None`` for Thinking Machines' hosted Tinker.

    Purpose: ``read`` makes the value from a release's manifest. The Tinker inference runtime and
    ``TinkerSamplerWeights`` read it, and ``check_compatible`` refuses a release of another model or service.
    """

    base_model: str
    sampler_path: str
    service: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, str) or not self.base_model:
            raise ValueError("a Tinker sampler names its base model")
        if not isinstance(self.sampler_path, str) or not self.sampler_path.startswith("tinker://"):
            raise ValueError("Tinker checkpoints require remote tinker:// paths")

    def check_compatible(self, base_model: str, *, service: str | None) -> None:
        """Raise ``ValueError`` unless the sampler is of ``base_model`` and held by ``service``."""
        if self.base_model != base_model:
            raise ValueError(f"the Tinker checkpoint is of base model {self.base_model}, not of {base_model}")
        if self.service != service:
            raise ValueError(
                f"the Tinker checkpoint is held by {self.service or 'hosted Tinker'}, "
                f"not by the configured service {service or 'hosted Tinker'}"
            )

    @classmethod
    def read(cls, directory: Path) -> TinkerRelease:
        """The sampler of the manifest in ``directory``; ``ValueError`` for a manifest of another version."""
        value = json.loads((directory / MANIFEST).read_text())
        if not isinstance(value, dict) or value.get("schema_version", 1) != 1:
            raise ValueError("invalid Tinker checkpoint manifest version")
        return cls(value.get("base_model", ""), value.get("sampler_path", ""), value.get("service"))
