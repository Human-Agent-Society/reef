"""Serving capabilities a scenario exposes for its frozen artifact version.

``Surface`` is a frozen composition of optional capabilities — ``loader``,
``inference``, ``files`` — and callers inspect fields, never subclass
identity. The invariant is version fidelity: the consumer observes exactly
the frozen version, and Reef records the exchange against it. The design,
every call site, and the bundled surfaces are documented at
https://reefinfra.ai/docs/developer-guide/surface/.

The package depends only on ``artifact`` and ``core``; it sees runtimes
structurally through ``ServingRuntime`` and ``WeightRuntime`` and never
imports a concrete one.
"""

from reef.surface.adapter import adapter_name, parse_adapter_name
from reef.surface.base import (
    ArtifactActivator,
    ArtifactLoader,
    FileTree,
    InferenceHooks,
    InferenceLease,
    LeasingInferenceHooks,
    ServingRuntime,
    Surface,
    WeightRuntime,
)
from reef.surface.files import TextFileTree
from reef.surface.harnesses import create_harness_surface
from reef.surface.skills import RequestSkillLayer, SkillLayer, create_skill_surface, validate_tree
from reef.surface.weights import WeightInferenceHooks, WeightLoader, WeightVersionMismatch, create_weight_surface

__all__ = [
    "ArtifactActivator",
    "ArtifactLoader",
    "FileTree",
    "InferenceHooks",
    "InferenceLease",
    "LeasingInferenceHooks",
    "RequestSkillLayer",
    "ServingRuntime",
    "SkillLayer",
    "Surface",
    "TextFileTree",
    "WeightInferenceHooks",
    "WeightLoader",
    "WeightRuntime",
    "WeightVersionMismatch",
    "adapter_name",
    "create_harness_surface",
    "create_skill_surface",
    "create_weight_surface",
    "parse_adapter_name",
    "validate_tree",
]
