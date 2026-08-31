"""Megatron-Bridge model bridges for the layouts slime actually builds.

Reef exports adapters through Megatron-Bridge's ``AutoBridge``; Bridge picks
the bridge by the HF architecture name and assumes the Megatron model has the
parameter layout of Bridge's own provider for that architecture. Models that
slime builds through ``slime_plugins`` (Qwen3.5's hybrid GatedDeltaNet layout)
use a different layout, so Reef registers its own bridges for them. Registration
is idempotent and must run before ``AutoBridge.from_hf_pretrained``.
"""

from __future__ import annotations

from reef.train.slime_backend.reef_adapters.megatron.bridges.qwen3_5 import register_slime_qwen35_bridge


def register_reef_model_bridges() -> None:
    """Register every Reef-owned model bridge with Megatron-Bridge's dispatch."""

    register_slime_qwen35_bridge()


__all__ = ["register_reef_model_bridges", "register_slime_qwen35_bridge"]
