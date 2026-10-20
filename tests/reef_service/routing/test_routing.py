"""Contracts for a scenario-scoped routing manifest and student/teacher selection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from tests.reef_service.routing.routing_fakes import preview_reply

from reef.service.routing import RoutingAdapter, RoutingDecision, RoutingRequestError, RoutingSnapshot
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
        adapter.handler_for(replace(decision, snapshot_version="old-snapshot"))
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
