"""Saved weight files: what a training component's snapshots hold and what serving engines load.

``ArtifactFormat`` is what a training component declares about its snapshots.
``ArtifactWeights`` gives a serving runtime the ``WeightFiles`` of one snapshot.
Training and inference both read these values, and neither imports the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reef.core.plugins import ConfiguredPlugin

#: ``"full"`` is a Hugging Face checkpoint of every model weight; ``"adapter"`` is a PEFT adapter directory.
WeightLayout = Literal["full", "adapter"]
WEIGHT_LAYOUTS: tuple[WeightLayout, ...] = ("full", "adapter")
#: The directory of a snapshot in the ``"adapter"`` layout that holds its PEFT adapter.
ADAPTER_DIR = "adapter"


@dataclass(frozen=True)
class WeightFiles:
    """Saved weights as files, for the serving engines to load.

    ``path`` is a directory on the control plane's node. ``layout`` names what
    it holds: ``"full"`` is a Hugging Face checkpoint of every model weight,
    and ``"adapter"`` is a PEFT adapter directory. The serving runtimes in
    ``reef.inference.serving`` make the values from a published snapshot and pass
    them to ``InferenceBackend.load_weight_files``.
    """

    path: Path
    layout: WeightLayout

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("weight files path must be a Path")
        if self.layout not in WEIGHT_LAYOUTS:
            raise ValueError(f"weight files layout must be full or adapter, got {self.layout!r}")


@dataclass(frozen=True)
class ArtifactFormat:
    """The weight files that each snapshot of a training component holds, which a serving runtime loads.

    ``layout`` is ``"full"``, a Hugging Face checkpoint at the snapshot's root,
    or ``"adapter"``, a PEFT adapter in the snapshot's ``adapter/`` directory
    (the layouts of ``WeightFiles``). ``adapter_rank`` is the adapter's
    positive LoRA rank, and ``adapter_target_modules`` the non-empty names of
    the modules it adapts in the serving engine's naming (``("all",)`` for
    every linear module); both are set exactly for ``"adapter"``. A training
    component declares the value (``TrainingComponent.artifact_format``), and
    the serving assembly reads it to configure the engines and to refuse a
    serving runtime that cannot load the snapshots.
    """

    layout: WeightLayout
    adapter_rank: int | None = None
    adapter_target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layout not in WEIGHT_LAYOUTS:
            raise ValueError(f"ArtifactFormat.layout must be 'full' or 'adapter', got {self.layout!r}")
        if self.layout == "full":
            if self.adapter_rank is not None or self.adapter_target_modules:
                raise ValueError("a full-weight ArtifactFormat has no adapter rank or target modules")
            return
        rank = self.adapter_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"an adapter ArtifactFormat needs a positive int adapter_rank, got {rank!r}")
        modules = self.adapter_target_modules
        if (
            not isinstance(modules, tuple)
            or not modules
            or any(not isinstance(name, str) or not name for name in modules)
        ):
            raise ValueError("an adapter ArtifactFormat needs a non-empty tuple of module names")

    def to_dict(self) -> dict[str, Any]:
        """The value as a JSON object, which ``from_dict`` reads back to an equal value."""
        return {
            "layout": self.layout,
            "adapter_rank": self.adapter_rank,
            "adapter_target_modules": list(self.adapter_target_modules),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactFormat:
        """Read a ``to_dict`` value; raise ``ValueError`` for another shape."""
        if (
            not isinstance(value, Mapping)
            or "layout" not in value
            or set(value) - {"layout", "adapter_rank", "adapter_target_modules"}
        ):
            raise ValueError("a artifact format is an object of layout, adapter_rank and adapter_target_modules")
        modules = value.get("adapter_target_modules", ())
        if not isinstance(modules, (list, tuple)):
            raise ValueError("a artifact format's adapter_target_modules is a list of module names")
        return cls(value["layout"], value.get("adapter_rank"), tuple(modules))


class ArtifactWeights(ConfiguredPlugin, ABC):
    """Where a serving runtime gets the weight files of a published snapshot.

    ``weight_files(release, layout)`` takes the local directory ``release`` of
    a materialized snapshot and the ``layout`` that the engines load, and
    returns ``WeightFiles`` of that layout with the snapshot's weights. Equal
    releases give files of equal content. It raises ``ReefError`` when the
    snapshot gives no files of ``layout``.

    Purpose: the serving runtimes of ``reef.inference.serving`` call it at each load. ``ArtifactWeightFiles``,
    the default, reads the snapshot's own files. A training backend whose
    snapshots name weights on a service instead names the plugin that fetches
    them (``TrainingDeployment.engine_weights``).
    """

    @abstractmethod
    def weight_files(self, artifact_dir: Path, layout: WeightLayout) -> WeightFiles: ...
