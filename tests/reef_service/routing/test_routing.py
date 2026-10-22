"""Contracts for a scenario-scoped routing manifest and student/teacher selection."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.reef_service.routing.routing_fakes import preview_reply

from reef.inference.http import HttpInferenceHandler
from reef.service.errors import ROUTING_ERRORS
from reef.service.routing import (
    ROUTING_ERROR_STATUSES,
    RoutingAdapter,
    RoutingDecision,
    RoutingRequestError,
    RoutingRule,
    RoutingScenario,
    RoutingSnapshot,
    RoutingTarget,
)
from reef.service.routing_preview import checked_selection

pytestmark = pytest.mark.unit
ROUTER_HASH = "3" * 64


@pytest.fixture
def manifest() -> dict[str, object]:
    """Online OPD: each Scenario serves its student and escalates marked requests to its teacher."""
    return {
        "schema_version": 2,
        "automatic_model": "auto",
        "deployment": {"semantic_router_image": "example/router@sha256:" + "a" * 64},
        "targets": {
            "code-student": {
                "provider_model": "code-student",
                "revision": "r2",
                "url": "http://localhost:8101",
                "role": "student",
                "api_key_env": "STUDENT_TOKEN",
            },
            "code-teacher": {
                "provider_model": "code-teacher",
                "revision": "t1",
                "url": "http://localhost:8102",
                "role": "teacher",
            },
            "math-student": {
                "provider_model": "math-student",
                "revision": "r1",
                "url": "http://localhost:8103",
                "role": "student",
            },
            "math-teacher": {
                "provider_model": "math-teacher",
                "revision": "t1",
                "url": "http://localhost:8104",
                "role": "teacher",
            },
            "retired-teacher": {
                "provider_model": "math-teacher",
                "revision": "t0",
                "url": "http://localhost:8105",
                "role": "teacher",
                "enabled": False,
            },
        },
        "scenarios": {
            "code-opd": {
                "default_target": "code-student",
                "targets": ["code-student", "code-teacher"],
                "rules": [
                    {"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "code-teacher"}
                ],
            },
            "math-opd": {
                "default_target": "math-student",
                "targets": ["math-student", "math-teacher", "retired-teacher"],
                "rules": [
                    {"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "math-teacher"}
                ],
            },
        },
    }


@pytest.fixture
def manifest_path(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def adapter(manifest_path: Path) -> RoutingAdapter:
    return RoutingAdapter.from_manifest(
        manifest_path, router_config_hash=ROUTER_HASH, environ={"STUDENT_TOKEN": "private-test-key"}
    )


CHAT = "/v1/chat/completions"


def client_headers(scenario: str = "code-opd", *extra: tuple[str, str]) -> list[tuple[str, str]]:
    """Received request headers, repeats included, as a client sends them."""
    return [("Authorization", "Bearer client"), ("x-reef-scenario", scenario), *extra]


def automatic_decision(adapter: RoutingAdapter, scenario: str, rule: str, target_id: str) -> RoutingDecision:
    """The decision a valid router reply naming ``rule`` and ``target_id`` makes."""
    reply = preview_reply(scenario, rule, target_id, ROUTER_HASH)
    return checked_selection(adapter, adapter.snapshot.scenarios[scenario], reply, [ROUTER_HASH])


@pytest.mark.parametrize(
    ("scenario", "rule", "target_id", "reason"),
    [
        ("code-opd", "default", "code-student", "default"),
        ("code-opd", "teacher_requested", "code-teacher", "rule"),
        ("math-opd", "default", "math-student", "default"),
        ("math-opd", "teacher_requested", "math-teacher", "rule"),
    ],
)
def test_automatic_selection_maps_each_scenarios_rule_to_its_own_target(
    adapter: RoutingAdapter, scenario: str, rule: str, target_id: str, reason: str
) -> None:
    decision = automatic_decision(adapter, scenario, rule, target_id)
    assert (decision.target.target_id, decision.scenario) == (target_id, scenario)
    assert (decision.reason, decision.rule_name, decision.original_selector) == (reason, rule, "auto")
    assert decision.router_config_hash == ROUTER_HASH
    assert adapter.handler_for(decision) is adapter.handlers[target_id]
    # An automatic request is left to the router.
    assert adapter.explicit_decision(adapter.snapshot.scenarios[scenario], {"model": "auto"}) is None


def test_a_rule_answered_with_another_scenarios_target_is_refused(adapter: RoutingAdapter) -> None:
    reply = preview_reply("code-opd", "teacher_requested", "math-teacher", ROUTER_HASH)
    with pytest.raises(RoutingRequestError, match="selected_model"):
        checked_selection(adapter, adapter.snapshot.scenarios["code-opd"], reply, [ROUTER_HASH])


@pytest.mark.parametrize(("scenario", "model"), [("code-opd", "code-teacher"), ("math-opd", "math-student")])
def test_a_pin_is_decided_locally_and_leaves_the_payload_unchanged(
    adapter: RoutingAdapter, scenario: str, model: str
) -> None:
    payload = {"model": model, "messages": [{"role": "user", "content": "hello"}], "temperature": 0.2}
    before = json.dumps(payload)
    decision = adapter.explicit_decision(adapter.snapshot.scenarios[scenario], payload)
    assert decision == RoutingDecision(
        adapter.snapshot.targets[model], adapter.snapshot.version, scenario, model, "explicit_pin"
    )
    assert decision.rule_name is None and decision.router_config_hash is None
    assert adapter.handler_for(decision) is adapter.handlers[model]
    assert json.dumps(payload) == before


@pytest.mark.parametrize(
    ("scenario", "model", "code", "status"),
    [
        ("code-opd", "math-teacher", "target_not_allowed", 403),
        ("math-opd", "retired-teacher", "target_unavailable", 503),
        ("code-opd", "unknown", "unknown_target", 400),
        ("code-opd", "reef-entrypoint/code-opd", "unknown_target", 400),
        ("code-opd", "Reef-Entrypoint/code-opd", "unknown_target", 400),
        ("code-opd", "", "missing_model", 400),
        ("code-opd", None, "missing_model", 400),
    ],
)
def test_a_pin_outside_the_scenario_disabled_or_unknown_is_refused(
    adapter: RoutingAdapter, scenario: str, model: str | None, code: str, status: int
) -> None:
    with pytest.raises(RoutingRequestError) as captured:
        adapter.explicit_decision(adapter.snapshot.scenarios[scenario], {"model": model})
    assert (captured.value.code, captured.value.status) == (code, status)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "other-opd"},
        {"target_ids": ("code-student", "code-teacher", "math-teacher")},
        {"default_target": "code-teacher"},
        {"rules": ()},
    ],
)
@pytest.mark.parametrize("model", ["code-teacher", "math-teacher", "auto"])
def test_a_scenario_that_is_not_the_snapshots_own_is_refused_before_its_selector(
    adapter: RoutingAdapter, change: dict[str, object], model: str
) -> None:
    foreign = replace(adapter.snapshot.scenarios["code-opd"], **change)
    with pytest.raises(ValueError, match="not part of the routing snapshot"):
        adapter.explicit_decision(foreign, {"model": model})


def test_an_equal_copy_of_the_snapshots_scenario_is_accepted(adapter: RoutingAdapter) -> None:
    scenario = adapter.snapshot.scenarios["code-opd"]
    equal_copy = replace(scenario)
    assert equal_copy is not scenario
    pinned = adapter.explicit_decision(equal_copy, {"model": "code-teacher"})
    assert pinned is not None and (pinned.scenario, pinned.target.target_id) == ("code-opd", "code-teacher")
    assert adapter.explicit_decision(equal_copy, {"model": "auto"}) is None


def test_request_headers_name_one_configured_scenario(adapter: RoutingAdapter) -> None:
    headers = client_headers("math-opd", ("x-reef-tag-session", "s1"), ("Accept", "a"), ("Accept", "b"))
    assert adapter.check_request(headers, CHAT) == adapter.snapshot.scenarios["math-opd"]


@pytest.mark.parametrize(
    ("headers", "path", "code", "status"),
    [
        (client_headers("other-opd"), CHAT, "unknown_scenario", 403),
        (client_headers("default"), CHAT, "unknown_scenario", 403),
        ([("Authorization", "Bearer client")], CHAT, "missing_scenario", 400),
        (client_headers("code-opd", ("X-Reef-Scenario", "math-opd")), CHAT, "duplicate_header", 400),
        (client_headers("code-opd", ("authorization", "Bearer other")), CHAT, "duplicate_header", 400),
        (client_headers("code-opd", ("x-reef-routing-version", "sha256:forged")), CHAT, "reserved_header", 400),
        (client_headers("code-opd", ("X-Selected-Model", "code-teacher")), CHAT, "reserved_header", 400),
        (client_headers("code-opd", ("x-reef-sr-decision", "default")), CHAT, "reserved_header", 400),
        (client_headers("code-opd", ("x-reef-release-id", "reef-release")), CHAT, "release_pin", 400),
        (client_headers(), "/v1/responses", "unsupported_route", 400),
    ],
)
def test_forged_repeated_or_unknown_request_headers_are_refused(
    adapter: RoutingAdapter, headers: list[tuple[str, str]], path: str, code: str, status: int
) -> None:
    with pytest.raises(RoutingRequestError) as captured:
        adapter.check_request(headers, path)
    assert (captured.value.code, captured.value.status) == (code, status)


def test_metadata_does_not_claim_engine_identity_or_expose_credentials(adapter: RoutingAdapter) -> None:
    metadata = automatic_decision(adapter, "code-opd", "default", "code-student").metadata()
    assert metadata["identity_source"] == "deployment_manifest"
    assert metadata["model_revision"] == "r2"
    assert metadata["role"] == "student"
    assert metadata["rule_name"] == "default"
    assert metadata["policy_id"] == "static"
    assert metadata["policy_version"] == metadata["snapshot_version"] == adapter.snapshot.version
    assert metadata["router_config_hash"] == ROUTER_HASH
    assert "runtime_load_id" not in metadata
    assert "artifact_ref" not in metadata
    assert "private-test-key" not in json.dumps(metadata)
    pinned = adapter.explicit_decision(adapter.snapshot.scenarios["code-opd"], {"model": "code-student"})
    assert pinned is not None and pinned.metadata()["router_config_hash"] is None


def test_snapshot_digest_is_order_independent_but_tracks_manifest_changes(manifest: dict[str, object]) -> None:
    original = RoutingSnapshot.from_mapping(manifest)
    reordered = json.loads(json.dumps(manifest, sort_keys=True))
    assert RoutingSnapshot.from_mapping(reordered).version == original.version
    reordered["targets"]["code-teacher"]["revision"] = "t2"
    assert RoutingSnapshot.from_mapping(reordered).version != original.version
    escalation = json.loads(json.dumps(manifest))
    escalation["scenarios"]["code-opd"]["rules"][0]["keywords"].append("hard_problem")
    assert RoutingSnapshot.from_mapping(escalation).version != original.version
    generated = json.loads(json.dumps(manifest))
    generated["deployment"]["configuration_digest"] = "sha256:" + "c" * 64
    assert RoutingSnapshot.from_mapping(generated).version != original.version


def test_snapshot_and_handlers_are_immutable_and_do_not_reload(adapter: RoutingAdapter, manifest_path: Path) -> None:
    decision = automatic_decision(adapter, "code-opd", "default", "code-student")
    with pytest.raises(FrozenInstanceError):
        decision.target.revision = "changed"
    with pytest.raises(TypeError):
        adapter.snapshot.targets["new"] = decision.target
    with pytest.raises(TypeError):
        adapter.snapshot.scenarios["new"] = adapter.snapshot.scenarios["code-opd"]
    with pytest.raises(TypeError):
        adapter.handlers["new"] = adapter.handler_for(decision)
    manifest_path.write_text("invalid replacement", encoding="utf-8")
    assert automatic_decision(adapter, "code-opd", "default", "code-student") == decision
    assert adapter.handlers["code-student"] is not adapter.handlers["code-teacher"]


def test_handler_lookup_rejects_foreign_decision(adapter: RoutingAdapter) -> None:
    decision = automatic_decision(adapter, "code-opd", "default", "code-student")
    with pytest.raises(RoutingRequestError, match="another snapshot"):
        adapter.handler_for(replace(decision, snapshot_version="sha256:" + "0" * 64))
    with pytest.raises(RoutingRequestError, match="another snapshot"):
        adapter.handler_for(replace(decision, target=replace(decision.target, url="http://different")))


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 1),
        (("scenario",), "code-opd"),
        (("automatic_model",), "code-student"),
        (("scenarios",), {}),
        (("scenarios", "default"), {"default_target": "code-student", "targets": ["code-student"]}),
        (("scenarios", "code-opd", "default_target"), "math-student"),
        (("scenarios", "math-opd", "default_target"), "retired-teacher"),
        (("scenarios", "code-opd", "targets"), ["code-student", "unknown-teacher"]),
        (("scenarios", "code-opd", "targets"), ["code-student", "code-student"]),
        # A manifest list stays a list; only direct construction may pass a tuple.
        (("scenarios", "code-opd", "targets"), ("code-student", "code-teacher")),
        (("scenarios", "code-opd", "rules"), [{"name": "hard", "keywords": ("x",), "target": "code-teacher"}]),
        (("scenarios", "code-opd", "tragets"), ["code-student"]),
        (("scenarios", "code-opd", "rules"), [{"name": "default", "keywords": ["x"], "target": "code-teacher"}]),
        (("scenarios", "code-opd", "rules"), [{"name": "hard", "keywords": [], "target": "code-teacher"}]),
        (("scenarios", "code-opd", "rules"), [{"name": "hard", "keywords": ["x"], "target": "math-teacher"}]),
        (("scenarios", "math-opd", "rules"), [{"name": "old", "keywords": ["x"], "target": "retired-teacher"}]),
        (("deployment", "envoy_image"), "envoy:latest"),
        (("deployment", "configuration_digest"), "not-a-digest"),
        (("targets", "code-teacher", "revision"), 1),
        (("targets", "code-teacher", "url"), "file:///tmp/provider"),
        (("targets", "code-teacher", "url"), "http://user:secret@localhost"),
        (("targets", "code-teacher", "url"), "http://localhost?api_key=secret"),
        (("targets", "code-teacher", "enabled"), "true"),
        (("targets", "code-teacher", "api_key"), "inline-secret"),
        (("targets", "code-teacher", "api_key_env"), "invalid-name"),
        # Optional fields may be omitted, never given as null.
        (("targets", "code-teacher", "role"), None),
        (("targets", "code-teacher", "api_key_env"), None),
    ],
)
def test_invalid_manifest_fields_fail_at_load(
    manifest: dict[str, object], keys: tuple[str, ...], value: object
) -> None:
    node = manifest
    for key in keys[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[keys[-1]] = value
    with pytest.raises(ValueError):
        RoutingSnapshot.from_mapping(manifest)


def target(target_id: str = "code-student", **changes: object) -> RoutingTarget:
    fields: dict[str, object] = {"provider_model": "code-student", "revision": "r2", "url": "http://localhost:8101"}
    return RoutingTarget(target_id, **{**fields, **changes})


def automatic(snapshot: RoutingSnapshot, **changes: object) -> RoutingDecision:
    fields: dict[str, object] = {
        "target": snapshot.targets["code-student"],
        "snapshot_version": snapshot.version,
        "scenario": "code-opd",
        "original_selector": "auto",
        "reason": "default",
        "rule_name": "default",
        "router_config_hash": ROUTER_HASH,
    }
    return RoutingDecision(**{**fields, **changes})


def replaced_scenario(snapshot: RoutingSnapshot, scenario: RoutingScenario, version: str = "") -> RoutingSnapshot:
    return RoutingSnapshot(
        "auto", snapshot.targets, {**snapshot.scenarios, scenario.name: scenario}, version or snapshot.version
    )


@pytest.mark.parametrize(
    "make",
    [
        # Values that used to be refused only while a manifest was parsed.
        lambda snapshot: target(url="file:///tmp/provider"),
        lambda snapshot: target(url="http://user:secret@localhost"),
        lambda snapshot: target("reef-entrypoint/code"),
        lambda snapshot: target(provider_model=" padded"),
        lambda snapshot: target(api_key_env="invalid-name"),
        lambda snapshot: target(enabled="true"),
        lambda snapshot: RoutingRule("hard", (), "code-teacher"),
        lambda snapshot: RoutingRule("hard", ("x", "x"), "code-teacher"),
        lambda snapshot: RoutingRule("default", ("x",), "code-teacher"),
        lambda snapshot: RoutingScenario("code-opd", "math-student", ("code-student",)),
        lambda snapshot: RoutingScenario("global", "code-student", ("code-student",)),
        lambda snapshot: RoutingScenario(
            "code-opd", "code-student", ("code-student",), (RoutingRule("hard", ("x",), "math-teacher"),)
        ),
        lambda snapshot: RoutingScenario(
            "code-opd",
            "code-student",
            ("code-student", "code-teacher"),
            (RoutingRule("hard", ("x",), "code-teacher"), RoutingRule("hard", ("y",), "code-student")),
        ),
        # References across the catalog, which only the snapshot can check.
        lambda snapshot: RoutingSnapshot(
            "auto",
            {**snapshot.targets, "other": snapshot.targets["code-student"]},
            snapshot.scenarios,
            snapshot.version,
        ),
        lambda snapshot: replaced_scenario(
            snapshot, RoutingScenario("math-opd", "retired-teacher", ("math-student", "retired-teacher"))
        ),
        lambda snapshot: replaced_scenario(
            snapshot,
            RoutingScenario(
                "math-opd",
                "math-student",
                ("math-student", "retired-teacher"),
                (RoutingRule("old", ("x",), "retired-teacher"),),
            ),
        ),
        lambda snapshot: replaced_scenario(
            snapshot, RoutingScenario("code-opd", "code-student", ("code-student", "unknown-teacher"))
        ),
        lambda snapshot: replaced_scenario(snapshot, snapshot.scenarios["code-opd"], "old-snapshot"),
        lambda snapshot: RoutingSnapshot("code-student", snapshot.targets, snapshot.scenarios, snapshot.version),
        # The three decision reasons and what each one implies.
        lambda snapshot: automatic(snapshot, reason="explicit_pin", original_selector="code-student"),
        lambda snapshot: automatic(snapshot, router_config_hash=None),
        lambda snapshot: automatic(snapshot, rule_name="teacher_requested"),
        lambda snapshot: automatic(snapshot, reason="rule"),
        lambda snapshot: automatic(snapshot, target=snapshot.targets["retired-teacher"]),
        lambda snapshot: RoutingAdapter(snapshot, ROUTER_HASH, {"unknown": HttpInferenceHandler("http://x")}),
        lambda snapshot: RoutingAdapter(snapshot, ROUTER_HASH, {}, {"code-student": ""}),
        lambda snapshot: RoutingAdapter(snapshot, ROUTER_HASH, {}, readiness_timeout_s=11),
        lambda snapshot: RoutingAdapter(snapshot, ROUTER_HASH, {}, readiness_timeout_s=0),
        lambda snapshot: RoutingRequestError("unmapped", status=500),
        lambda snapshot: RoutingRequestError("no code", code=""),
        # A frozen value holds the declared values, not mutable lookalikes.
        lambda snapshot: RoutingScenario(
            "code-opd", "code-student", ("code-student",), (SimpleNamespace(name="hard", target_id="code-student"),)
        ),
        lambda snapshot: RoutingSnapshot(
            "auto",
            {**snapshot.targets, "code-student": SimpleNamespace(target_id="code-student", enabled=True)},
            snapshot.scenarios,
            snapshot.version,
        ),
        lambda snapshot: RoutingSnapshot(
            "auto",
            snapshot.targets,
            {**snapshot.scenarios, "code-opd": SimpleNamespace(**vars(snapshot.scenarios["code-opd"]))},
            snapshot.version,
        ),
        lambda snapshot: automatic(snapshot, target=SimpleNamespace(**vars(snapshot.targets["code-student"]))),
        lambda snapshot: RoutingAdapter(SimpleNamespace(targets=snapshot.targets), ROUTER_HASH, {}),
        lambda snapshot: RoutingAdapter(snapshot, ROUTER_HASH, {"code-student": 42}),
        lambda snapshot: RoutingRequestError("float", status=400.0),
        lambda snapshot: RoutingRequestError("unhashable", status=[400]),
    ],
)
def test_routing_values_refuse_what_their_definitions_exclude(
    manifest: dict[str, object], make: Callable[[RoutingSnapshot], object]
) -> None:
    with pytest.raises(ValueError):
        make(RoutingSnapshot.from_mapping(manifest))


def test_routing_values_keep_their_own_collections(manifest: dict[str, object]) -> None:
    snapshot = RoutingSnapshot.from_mapping(manifest)
    keywords = ["reef_teacher_required"]
    rule = RoutingRule("teacher_requested", keywords, "code-teacher")
    keywords.append("later")
    assert rule.keywords == ("reef_teacher_required",)
    targets = dict(snapshot.targets)
    copied = RoutingSnapshot("auto", targets, snapshot.scenarios, snapshot.version)
    targets.clear()
    assert copied.targets == snapshot.targets

    # Constructors store lists as tuples and accept subclasses of the declared member types.
    class LabelledRule(RoutingRule):
        """A rule subclass, which is still a rule."""

    scenario = RoutingScenario(
        "code-opd", "code-student", ["code-student", "code-teacher"], [LabelledRule("hard", ["x"], "code-teacher")]
    )
    assert scenario.target_ids == ("code-student", "code-teacher")
    assert type(scenario.rules) is tuple and isinstance(scenario.rules[0], LabelledRule)
    # Shared targets, an empty handler map and a pin without a router runtime are all valid.
    assert RoutingAdapter(snapshot, None, {}).handlers == {}
    assert RoutingDecision(
        snapshot.targets["code-teacher"], snapshot.version, "code-opd", "code-teacher", "explicit_pin"
    )


def test_every_routing_error_status_has_an_http_translation() -> None:
    # 413 needs its body limit, so the translation builds it separately.
    assert set(ROUTING_ERRORS) | {413} == ROUTING_ERROR_STATUSES


def test_required_credentials_resolve_only_at_startup(manifest_path: Path) -> None:
    with pytest.raises(ValueError, match="STUDENT_TOKEN"):
        RoutingAdapter.from_manifest(manifest_path, router_config_hash=ROUTER_HASH, environ={})
    first = RoutingAdapter.from_manifest(
        manifest_path, router_config_hash=ROUTER_HASH, environ={"STUDENT_TOKEN": "secret-one"}
    )
    second = RoutingAdapter.from_manifest(
        manifest_path, router_config_hash=ROUTER_HASH, environ={"STUDENT_TOKEN": "secret-two"}
    )
    assert first.snapshot.version == second.snapshot.version
    assert "secret-one" not in repr(first)


@pytest.mark.parametrize("available", [True, False])
def test_readiness_checks_enabled_model_ids_and_reuses_endpoint_probe(
    tmp_path: Path, manifest: dict[str, object], available: bool
) -> None:
    from aiohttp import web

    async def run() -> None:
        calls: list[str | None] = []

        async def models(request: web.Request) -> web.Response:
            calls.append(request.headers.get("Authorization"))
            names = (
                ("code-student", "code-teacher", "math-student", "math-teacher") if available else ("code-student",)
            )
            return web.json_response({"data": [{"id": name} for name in names]})

        app = web.Application()
        app.router.add_get("/v1/models", models)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            url = f"http://127.0.0.1:{runner.addresses[0][1]}"
            targets = manifest["targets"]
            assert isinstance(targets, dict)
            for target in targets.values():
                target["url"] = url
                target["api_key_env"] = "STUDENT_TOKEN"
            path = tmp_path / "ready.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            adapter = RoutingAdapter.from_manifest(
                path, router_config_hash=ROUTER_HASH, environ={"STUDENT_TOKEN": "readiness-secret"}
            )
            if available:
                await adapter.check_ready()
            else:
                with pytest.raises(RoutingRequestError, match="does not expose") as captured:
                    await adapter.check_ready()
                assert captured.value.status == 503
            assert calls == ["Bearer readiness-secret"]
        finally:
            await runner.cleanup()

    asyncio.run(run())
