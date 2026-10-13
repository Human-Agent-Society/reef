"""Routing manifests parse, validate and digest before any process starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reef.service.routing import (
    RoutingManifestError,
    load_routing_manifest,
    parse_routing_manifest,
    routing_digest,
)


def manifest(**overrides) -> dict:
    base: dict = {
        "targets": {
            "large": {"endpoint": "http://127.0.0.1:8300/v1", "model": "Qwen/Qwen3-32B"},
            "small": {"endpoint": "http://127.0.0.1:8301/v1", "model": "Qwen/Qwen3-8B"},
        },
        "scenarios": {
            "coding": {
                "targets": ["large", "small"],
                "keywords": [{"name": "deep", "keywords": ["step by step"], "target": "large"}],
                "default": "small",
            },
            "summarize": {"targets": ["large"], "static": "large"},
        },
        "default_target": "small",
    }
    base.update(overrides)
    return base


def test_manifest_parses_targets_scenarios_and_digest() -> None:
    parsed = parse_routing_manifest(manifest())
    assert set(parsed.targets) == {"large", "small"}
    assert parsed.targets["large"].model == "Qwen/Qwen3-32B"
    assert parsed.default_target == "small"
    coding = parsed.scenarios["coding"]
    assert coding.allowed_targets == ("large", "small")
    assert coding.keywords[0].target == "large" and coding.default == "small" and coding.static is None
    assert parsed.scenarios["summarize"].static == "large"
    assert len(parsed.digest) == 64


def test_manifest_digest_is_canonical_across_key_order() -> None:
    flipped = {
        "scenarios": dict(reversed(list(manifest()["scenarios"].items()))),
        "targets": dict(reversed(list(manifest()["targets"].items()))),
        "default_target": "small",
    }
    assert routing_digest(flipped) == routing_digest(manifest())
    assert routing_digest(manifest()) != routing_digest({**manifest(), "default_target": "large"})


def test_scenario_may_not_route_outside_its_allowed_set() -> None:
    broken = manifest()
    broken["scenarios"]["coding"]["default"] = "undefined"
    with pytest.raises(RoutingManifestError, match="routes to undefined targets"):
        parse_routing_manifest(broken)
    broken = manifest()
    broken["scenarios"]["coding"]["keywords"][0]["target"] = "large"
    broken["scenarios"]["coding"]["targets"] = ["small"]
    with pytest.raises(RoutingManifestError, match="outside its allowed set"):
        parse_routing_manifest(broken)


def test_scenario_allowed_targets_must_exist_and_be_unique() -> None:
    broken = manifest()
    broken["scenarios"]["coding"]["targets"] = ["large", "missing"]
    with pytest.raises(RoutingManifestError, match="allows undefined targets"):
        parse_routing_manifest(broken)
    broken = manifest()
    broken["scenarios"]["coding"]["targets"] = ["large", "large"]
    with pytest.raises(RoutingManifestError, match="repeats a target"):
        parse_routing_manifest(broken)


def test_manifest_requires_targets_scenarios_and_valid_endpoints() -> None:
    with pytest.raises(RoutingManifestError, match="non-empty 'targets'"):
        parse_routing_manifest({"scenarios": {}})
    with pytest.raises(RoutingManifestError, match="non-empty 'scenarios'"):
        parse_routing_manifest({"targets": manifest()["targets"], "scenarios": {}})
    broken = manifest()
    broken["targets"]["large"]["endpoint"] = "not a url"
    with pytest.raises(RoutingManifestError, match="HTTP\\(S\\) URL"):
        parse_routing_manifest(broken)
    broken = manifest()
    del broken["targets"]["small"]["model"]
    with pytest.raises(RoutingManifestError, match="provider model name"):
        parse_routing_manifest(broken)
    broken = manifest()
    broken["default_target"] = "absent"
    with pytest.raises(RoutingManifestError, match="default_target"):
        parse_routing_manifest(broken)
    broken = manifest()
    broken["scenarios"]["coding"]["keywords"] = [{"name": "empty", "keywords": [], "target": "large"}]
    with pytest.raises(RoutingManifestError, match="non-empty list of keyword strings"):
        parse_routing_manifest(broken)


def test_runtime_scenario_and_pin_validation() -> None:
    parsed = parse_routing_manifest(manifest())
    parsed.validate_scenario("coding", "large")
    with pytest.raises(RoutingManifestError, match="not in the routing manifest"):
        parsed.validate_scenario("missing", "large")
    with pytest.raises(RoutingManifestError, match="not allowed for scenario"):
        parsed.validate_scenario("summarize", "small")
    assert parsed.resolve_scenario_target("summarize") == "large"
    assert parsed.resolve_scenario_target("coding", "large") == "large"
    assert parsed.resolve_scenario_target("coding") == "small"  # the keyword misses; default catches
    with pytest.raises(RoutingManifestError, match="not allowed for scenario"):
        parsed.resolve_scenario_target("summarize", "small")


def test_load_routing_manifest_from_file_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(manifest()))
    loaded = load_routing_manifest(path)
    assert loaded.digest == routing_digest(manifest())
    with pytest.raises(RoutingManifestError, match="not found"):
        load_routing_manifest(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text(json.dumps({"targets": {}}))
    with pytest.raises(RoutingManifestError, match="non-empty 'targets'"):
        load_routing_manifest(broken)
