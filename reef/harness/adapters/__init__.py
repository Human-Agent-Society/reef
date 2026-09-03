"""Adapter registry: bundled descriptors plus entry-point discovery.

Bundled adapters are directories beside this file, each holding a
``descriptor.yaml`` and a ``quirks`` module; external distributions add
theirs through the ``reef.harness_adapters`` entry point group (see
``reef.harness.descriptor``). Descriptors load lazily and cache per process:
loading imports the quirks module, and a registry import must stay cheap.
"""

from __future__ import annotations

from pathlib import Path

from reef.harness.descriptor import AdapterDescriptor, DescriptorError, external_descriptors, load_descriptor

BUILTIN_ADAPTERS = ("claude", "dsh", "hermes", "native", "opencode", "pi")

_cache: dict[str, AdapterDescriptor] = {}


def builtin_descriptor_path(name: str) -> Path:
    return Path(__file__).parent / name / "descriptor.yaml"


def get_adapter(name: str) -> AdapterDescriptor:
    """Resolve one adapter by name: bundled first, then entry points."""
    cached = _cache.get(name)
    if cached is not None:
        return cached
    if name in BUILTIN_ADAPTERS:
        descriptor = load_descriptor(builtin_descriptor_path(name))
    else:
        externals = external_descriptors()
        if name not in externals:
            known = ", ".join(sorted({*BUILTIN_ADAPTERS, *externals}))
            raise DescriptorError(f"unknown harness adapter {name!r}; known adapters: {known}")
        descriptor = externals[name]
    _cache[name] = descriptor
    return descriptor


def available_adapters() -> tuple[str, ...]:
    """Every adapter name ``get_adapter`` can serve."""
    return tuple(sorted({*BUILTIN_ADAPTERS, *external_descriptors()}))
