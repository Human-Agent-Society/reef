"""The engine-side naming contract for scenario-owned adapter revisions."""

from __future__ import annotations

import pytest

from reef.surface import adapter_name, parse_adapter_name


def test_adapter_name_is_lossless_and_scenario_qualified() -> None:
    name = adapter_name("team/math v2", "sha:abc.1")
    assert name.startswith("reef-adapter-")
    assert parse_adapter_name(name) == ("team/math v2", "sha:abc.1")
    # Two scenarios reusing one revision label never collide, and the
    # separator is outside the base64url alphabet so the split is exact.
    assert adapter_name("a", "v1") != adapter_name("b", "v1")
    assert adapter_name("a.b", "v1") != adapter_name("a", "b.v1")


def test_adapter_name_requires_a_scenario_and_a_runtime_load_id() -> None:
    with pytest.raises(ValueError, match="scenario"):
        adapter_name("", "v1")
    with pytest.raises(ValueError, match="runtime_load_id"):
        adapter_name("a", "")


@pytest.mark.parametrize("name", ["lora-1", "reef-adapter-", "reef-adapter-YQ", "reef-adapter-.v1"])
def test_parse_adapter_name_rejects_names_reef_never_minted(name: str) -> None:
    with pytest.raises(ValueError):
        parse_adapter_name(name)
