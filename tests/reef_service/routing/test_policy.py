"""In-place routing policy updates.

The router side is a small local HTTP server with the upstream management API's
routes, ETag/If-Match rules and response fields; it is not the Semantic Router.
The Reef side uses the real aiohttp service and its private lifecycle control.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import stat
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import (
    OPEN_DEADLINE,
    SELECTION_TOKEN,
    SERVICE_TOKEN,
    FakeControl,
    FakeRuntime,
    routing_dispatcher,
    routing_headers,
    running_router,
    write_domain_classifier,
)

import reef.service.routing as routing_module
from reef.inference.http import HttpInferenceHandler, RequestHeadersFactory
from reef.service.app import CONTROL_APPLICATION, REQUEST_SERVICE_KEY, RoutingPublication, create_app
from reef.service.deploy.routing.deployment_controller import (
    RoutingDeploymentController,
    RoutingDeploymentError,
    RoutingReadinessTimeout,
)
from reef.service.deploy.routing.policy import (
    PolicyState,
    PolicyStateStore,
    RouterManagementClient,
    RouterPolicyError,
    RouterRuntime,
    document_etag,
)
from reef.service.deploy.routing.reef_lifecycle_client import RoutingControlError, RoutingControlTimeout
from reef.service.deploy.routing.renderer import ManagedRoutingRender, generate
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceConfig
from reef.service.lifecycle import REQUEST_ACCEPTANCE, LifecycleSettings
from reef.service.request_service import RequestService
from reef.service.routing import RoutingAdapter, RoutingSnapshot

pytestmark = pytest.mark.unit

MANAGEMENT_TOKEN = "unit-management-token"


def write_source(
    path: Path,
    *,
    keyword: str = "reef_teacher_required",
    rule: str = "teacher_requested",
    url: str = "http://127.0.0.1:9",
    automatic_model: str = "auto",
    scenario: str = "coding",
    domain_classifier: dict[str, object] | None = None,
    domain_target: str | None = None,
    **student_change: str,
) -> Path:
    """A one-Scenario source; ``domain_target`` adds a ``math_request`` domain rule after the keyword rule."""
    targets = {
        "student": {"provider_model": "student-v1", "revision": "r1", "url": url, **student_change},
        "teacher": {"provider_model": "teacher-v2", "revision": "r1", "url": url},
    }
    rules: list[dict[str, object]] = [{"name": rule, "keywords": [keyword], "target": "teacher"}]
    if domain_target is not None:
        rules.append({"name": "math_request", "domains": ["math"], "target": domain_target})
    deployment: dict[str, object] = {"semantic_router_image": "semantic-router@sha256:" + "a" * 64}
    if domain_classifier is not None:
        deployment["domain_classifier"] = domain_classifier
    manifest = {
        "schema_version": 2,
        "automatic_model": automatic_model,
        "targets": targets,
        "scenarios": {scenario: {"default_target": "student", "targets": ["student", "teacher"], "rules": rules}},
        "deployment": deployment,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def etag(document: str) -> str:
    return document_etag(document.encode())


def normalized(document: str) -> str:
    """Semantically equal YAML in another encoding; a stand-in for the router's yaml.Marshal, not its bytes."""
    return yaml.safe_dump(yaml.safe_load(document), sort_keys=True, indent=4)


def runtime(document: str) -> str:
    """A runtime hash in the router's form; here the SHA-256 of the stored document."""
    return hashlib.sha256(document.encode()).hexdigest()


class RouterConfigServer:
    """Upstream management schema: ETag on GET, plan ETags, If-Match PUT, generated/active runtime hashes.

    Like ``config_activation.go``, the status is ``active`` exactly when the
    active runtime hash equals the generated one. ``mode`` scripts the next
    PUT (``ok``, ``other_writer``, ``lost_unknown``, ``pending``, ``stop``,
    ``stop_late``, ``failed``, ``hash_mismatch`` or ``lost``); ``next_mode`` then replaces it.
    """

    def __init__(self, document: str) -> None:
        self.document = document
        self.generated = self.active = runtime(document)
        self.activation_status = "active"
        self.ready = True
        self.mode = "ok"
        #: The mode of the PUT after the next one, such as a rollback's.
        self.next_mode = "ok"
        #: Re-encode stored documents like the router's yaml.Marshal; ETags then hash different bytes.
        self.normalize = False
        #: Another writer changes the document while this numbered plan request is handled.
        self.overwrite_at_plan = 0
        self.plans = 0
        self.puts: list[str] = []
        self.pending_reads = 0
        #: This many ``/ready`` requests lose their reply; another writer may store a document at the first.
        self.lost_ready_replies = 0
        self.writer_at_lost_ready: str | None = None
        #: ``"headers"``: every reply arrives one byte every 0.02 seconds from its status line; ``"body"``: only its
        #: body does.
        self.trickle: str | None = None
        #: Each authorized request as ``"<method> <path>"``, in arrival order.
        self.requests: list[str] = []
        #: When set, each ``/ready`` asks the controller to stop before it answers.
        self.stop_at_ready = False
        self.controller: RoutingDeploymentController | None = None
        state = self

        class Handler(BaseHTTPRequestHandler):
            """Serves the management API from the enclosing server's state to requests with its token."""

            def log_message(self, *arguments: object) -> None:
                pass

            def reply(self, status: int, body: dict[str, object], tag: str | None = None) -> None:
                encoded = json.dumps(body).encode()
                head = (
                    f"HTTP/1.0 {status} {self.responses[status][0]}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(encoded)}\r\n" + ("" if tag is None else f"ETag: {tag}\r\n") + "\r\n"
                ).encode()
                if state.trickle is None:
                    self.wfile.write(head + encoded)
                    return
                fast, slow = (head, encoded) if state.trickle == "body" else (b"", head + encoded)
                # Each byte arrives well within a socket timeout; the whole reply takes seconds.
                with contextlib.suppress(OSError):
                    self.wfile.write(fast)
                    for byte in slow:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)

            def authorized(self) -> bool:
                if self.headers.get("Authorization") == "Bearer " + MANAGEMENT_TOKEN:
                    state.requests.append(f"{self.command} {self.path}")
                    return True
                self.reply(401, {"error": "unauthorized"})
                return False

            def body(self) -> dict[str, object]:
                decoded = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if state.normalize and "yaml" in decoded:
                    decoded["yaml"] = normalized(decoded["yaml"])
                return decoded

            def do_GET(self) -> None:
                if not self.authorized():
                    return
                if self.path == "/api/v1/config":
                    self.reply(200, {"yaml": state.document}, etag(state.document))
                elif self.path == "/api/v1/config/hash":
                    if state.pending_reads > 0:
                        state.pending_reads -= 1
                        if state.pending_reads == 0:
                            state.active, state.activation_status = state.generated, "active"
                    if state.active and state.active == state.generated:
                        status = "active"
                    else:
                        status = "failed" if state.activation_status == "failed" else "pending"
                    self.reply(
                        200,
                        {
                            "source_config_hash": etag(state.document),
                            "generated_runtime_hash": state.generated,
                            "active_runtime_hash": state.active,
                            "activation_status": status,
                        },
                    )
                elif self.path == "/ready":
                    if state.stop_at_ready and state.controller is not None:
                        state.controller.request_stop()
                    if state.lost_ready_replies > 0:
                        state.lost_ready_replies -= 1
                        if state.writer_at_lost_ready is not None:
                            state.document, state.writer_at_lost_ready = state.writer_at_lost_ready, None
                        self.close_connection = True
                        return
                    self.reply(200 if state.ready else 503, {"status": "ready" if state.ready else "starting"})

            def do_POST(self) -> None:
                if self.authorized() and self.path == "/api/v1/config/plan":
                    candidate = self.body()
                    assert candidate["mode"] == "replace"
                    state.plans += 1
                    if state.plans == state.overwrite_at_plan:
                        state.document = "written: elsewhere\n"
                    self.reply(200, {"current_etag": etag(state.document), "candidate_etag": etag(candidate["yaml"])})

            def do_PUT(self) -> None:
                if not self.authorized():
                    return
                document = self.body()["yaml"]
                state.puts.append(document)
                mode, state.mode, state.next_mode = state.mode, state.next_mode, "ok"
                if mode == "other_writer":
                    state.document = "written: elsewhere\n"
                if self.headers.get("If-Match") != etag(state.document):
                    self.reply(412, {"error": "stale"}, etag(state.document))
                    return
                if mode == "lost_unknown":
                    state.document = "written: elsewhere\n"
                    self.close_connection = True
                    return
                state.document = document
                state.generated = runtime(document)
                if mode in {"pending", "stop", "stop_late"}:
                    state.pending_reads = 3 if mode == "pending" else 10**6
                    state.activation_status = "pending"
                    if mode != "pending" and state.controller is not None:
                        state.controller.request_stop()
                    if mode == "stop_late":
                        # This reply, and every later one, outlasts the observation phase.
                        state.trickle = "body"
                    self.reply(
                        202, {"status": "pending", "etag": etag(document), "generated_runtime_hash": state.generated}
                    )
                    return
                if mode == "failed":
                    state.activation_status = "failed"
                    self.reply(503, {"status": "failed", "generated_runtime_hash": state.generated})
                    return
                reported = state.generated
                if mode == "hash_mismatch":
                    state.generated = runtime(state.generated)
                state.active, state.activation_status = state.generated, "active"
                if mode == "lost":
                    self.close_connection = True
                    return
                self.reply(200, {"status": "ok", "etag": etag(document), "generated_runtime_hash": reported})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@dataclass
class PolicyCase:
    """A ready controller whose management client reaches ``router``, serving the policy rendered from ``source``.

    ``startup_version`` is that rendered snapshot's version, and ``actions`` is the log shared by the fakes.
    """

    controller: RoutingDeploymentController
    control: FakeControl
    router: RouterConfigServer
    source: Path
    actions: list[str]
    startup_version: str

    def events(self) -> list[str]:
        path = self.controller.directory / "lifecycle-events.jsonl"
        return [json.loads(line)["state"] for line in path.read_text().splitlines()]

    def state(self) -> dict[str, object]:
        """The journal file's raw JSON, so the assertions also check its unchanged wire form."""
        path = self.controller.registry.directory / "policy-state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@pytest.fixture
def case(tmp_path: Path) -> Iterator[PolicyCase]:
    with started_case(tmp_path, write_source(tmp_path / "source.json")) as started:
        yield started


@pytest.fixture
def domain_case(tmp_path: Path) -> Iterator[PolicyCase]:
    """A ready case whose policy has a stand-in domain classifier and a domain rule for the teacher."""
    classifier = write_domain_classifier(tmp_path / "vela-domain")
    source = write_source(tmp_path / "source.json", domain_classifier=classifier, domain_target="teacher")
    with started_case(tmp_path, source) as started:
        yield started


@contextlib.contextmanager
def started_case(tmp_path: Path, source: Path) -> Iterator[PolicyCase]:
    actions: list[str] = []
    policy = RoutingDeploymentSettings(
        deployment_id="unit-routing",
        ready_timeout_seconds=2,
        drain_timeout_seconds=0.05,
        finalization_timeout_seconds=0.05,
        container_stop_timeout_seconds=0.05,
    )
    settings = ServiceConfig(
        recipe="recipe", host="127.0.0.1", routing_manifest=str(source), tokens=("service",), routing_deployment=policy
    )
    runtime = FakeRuntime(actions)
    controller = RoutingDeploymentController(settings, runtime, registry_root=tmp_path / "registry")
    controller.registry.acquire()
    controller.directory = tmp_path / "generation"
    controller.ownership = controller.registry.prepare("generation-one", tmp_path / "generation")
    startup = generate(
        source,
        controller.directory,
        ManagedRoutingRender(policy, "unit-routing", controller.ownership.owner_id, "generation-one"),
    )
    controller.snapshot_version = startup.version
    control = FakeControl(actions)
    control.snapshot_version = startup.version
    # As ``prepare`` sets it for the startup policy.
    control.open_timeout_seconds = controller.open_timeout_seconds(startup)
    controller.control = control
    controller.event_stream = runtime.events(controller.ownership)
    controller.inspect_ready(runtime.inventory)
    controller.state = "ready"
    router = RouterConfigServer((controller.directory / "router.yaml").read_text())
    router.controller = controller
    controller.management = RouterManagementClient(router.port, MANAGEMENT_TOKEN, timeout_seconds=1)
    # The child already serves the verified startup runtime.
    control.router_config_hash = controller.check_router_policy()
    try:
        yield PolicyCase(controller, control, router, source, actions, startup.version)
    finally:
        router.close()
        controller.registry.release()


def reload(case: PolicyCase) -> None:
    case.actions.clear()
    case.controller.request_reload()
    case.controller.tick()


@pytest.mark.parametrize("mode", ["ok", "pending", "lost"])
def test_policy_update_activates_then_publishes_while_closed(case: PolicyCase, mode: str) -> None:
    committed_etag = case.state()["committed"]["router_etag"]
    write_source(case.source, keyword="escalate", rule="escalated")
    case.router.mode = mode
    reload(case)

    new_version = case.control.snapshot_version
    assert new_version != case.startup_version
    assert case.controller.snapshot_version == new_version
    # The child receives the new manifest together with the runtime the router actually activated.
    [publication] = case.control.publications
    assert publication["router_config_hash"] == case.router.active == runtime(case.router.document)
    assert case.actions.index("control:close") < case.actions.index("control:activate")
    assert case.actions.index("control:activate") < case.actions.index("control:open")
    # A lost reply is inspected, never replayed.
    assert len(case.router.puts) == 1
    committed = case.state()["committed"]
    assert committed["snapshot_version"] == new_version and committed["router_etag"] != committed_etag
    assert committed["runtime_hash"] == case.router.active
    assert "pending" not in case.state()
    assert "escalate" in case.router.document
    assert MANAGEMENT_TOKEN not in json.dumps(case.state())
    assert case.events()[-2:] == ["policy_committed", "ready"]
    # The journal and the committed manifest copy stay owner-only after every replacement.
    for private_file in (case.controller.registry.directory / "policy-state.json", Path(committed["source"])):
        assert stat.S_IMODE(private_file.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("change", "event"),
    [
        ({"automatic_model": "router-auto"}, "policy_rejected"),
        ({"scenario": "review"}, "policy_rejected"),
        ({}, "policy_unchanged"),
    ],
)
def test_incompatible_or_unchanged_sources_leave_the_serving_policy_untouched(
    case: PolicyCase, change: dict[str, str], event: str
) -> None:
    if change:
        write_source(case.source, keyword="escalate", **change)
    reload(case)
    assert case.events()[-1] == event
    if change:
        last = json.loads((case.controller.directory / "lifecycle-events.jsonl").read_text().splitlines()[-1])
        assert last["reason"].endswith("restart the deployment")
    assert case.control.snapshot_version == case.controller.snapshot_version == case.startup_version
    # Nothing reaches the child or the router: not even a candidate check.
    assert case.router.puts == [] and case.actions == []


def edit_source(path: Path, change: str) -> None:
    """Apply one catalog ``change`` to the source that ``write_source`` wrote."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    targets, coding = manifest["targets"], manifest["scenarios"]["coding"]
    if change == "add":
        targets["reviewer"] = {"provider_model": "reviewer-v1", "revision": "r1", "url": "http://127.0.0.1:11"}
        coding["targets"].append("reviewer")
    elif change in ("remove", "disable"):
        # The teacher's rule goes too: a default or rule target must stay an enabled target of its Scenario.
        coding["rules"] = []
        if change == "remove":
            del targets["teacher"]
            coding["targets"] = ["student"]
        else:
            targets["teacher"]["enabled"] = False
    elif change == "provider_model":
        targets["student"]["provider_model"] = "student-v2"
    elif change == "endpoint":
        targets["student"]["url"] = "http://127.0.0.1:12"
    elif change == "remove_referenced":
        del targets["teacher"]
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("change", ["add", "remove", "disable", "provider_model", "endpoint"])
def test_catalog_changes_apply_in_place_after_the_child_checks_the_candidate(case: PolicyCase, change: str) -> None:
    first_allowance = case.control.open_timeout_seconds
    edit_source(case.source, change)
    reload(case)

    # The rendered manifest that the router and the child received; it adds the deployment's configuration digest.
    candidate = RoutingSnapshot.load(case.controller.committed_policy().manifest)
    assert candidate.targets.keys() == json.loads(case.source.read_text())["targets"].keys()
    assert case.events()[-2:] == ["policy_committed", "ready"]
    assert case.control.snapshot_version == case.controller.snapshot_version == candidate.version
    # The child checked the candidate's providers before anything closed, then published it with the router's runtime.
    assert case.control.checked == [candidate.version]
    assert case.actions.index("control:check") < case.actions.index("control:close")
    [publication] = case.control.publications
    assert publication["router_config_hash"] == case.router.active
    models = yaml.safe_load(case.router.document)["providers"]["models"]
    enabled = {name: target.provider_model for name, target in candidate.targets.items() if target.enabled}
    assert {model["name"]: model["provider_model_id"] for model in models} == enabled
    # The next open checks each enabled target, so its allowance follows the committed catalog.
    assert case.control.open_timeout_seconds == case.controller.open_timeout_seconds(candidate)
    assert (case.control.open_timeout_seconds != first_allowance) == (change in ("add", "remove", "disable"))


@pytest.mark.parametrize("mode", ["ok", "lost"])
def test_an_endpoint_change_republishes_the_child_although_the_router_runs_the_same_document(
    case: PolicyCase, mode: str
) -> None:
    """The router never sees endpoints: it keeps its document and runtime; Reef still publishes the new manifest."""
    document, runtime_hash = case.router.document, case.router.active
    edit_source(case.source, "endpoint")
    case.router.mode = mode
    reload(case)
    assert case.router.document == document and case.router.active == runtime_hash
    # The proven transaction stays: one conditional PUT of the same document, inspected rather than replayed.
    assert case.router.puts == [document]
    [publication] = case.control.publications
    assert publication["router_config_hash"] == runtime_hash
    committed = case.controller.committed_policy()
    assert publication["snapshot_version"] == committed.snapshot_version != case.startup_version
    assert RoutingSnapshot.load(committed.manifest).targets["student"].url == "http://127.0.0.1:12"
    assert case.events()[-2:] == ["policy_committed", "ready"]


@pytest.mark.parametrize(
    "error",
    [
        RoutingControlError("REEF candidate check returned HTTP 409: requires environment variable NEW_KEY"),
        RoutingControlError("REEF candidate check returned HTTP 503: target_unavailable"),
        RoutingControlTimeout("REEF candidate check timed out"),
    ],
    ids=["refused", "provider_unavailable", "timeout"],
)
def test_a_failed_candidate_check_rejects_the_update_before_anything_closes(
    case: PolicyCase, error: RoutingControlError
) -> None:
    allowance = case.control.open_timeout_seconds
    edit_source(case.source, "add")
    case.control.check_error = error
    reload(case)
    event = last_event(case)
    assert (event["state"], event["reason"]) == ("policy_rejected", str(error))
    # No drain, journal or router write: the old policy keeps serving, and a later valid source still applies.
    assert case.actions == ["control:check"] and case.router.puts == [] and "pending" not in case.state()
    assert case.control.snapshot_version == case.controller.snapshot_version == case.startup_version
    assert case.control.open_timeout_seconds == allowance and case.controller.state == "ready"
    case.control.check_error = None
    reload(case)
    assert case.events()[-2:] == ["policy_committed", "ready"]


def test_a_stop_during_the_candidate_check_ends_the_update_before_anything_closes(
    case: PolicyCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    check = case.control.check_candidate

    def stopped_during_check(*arguments: str, deadline: float) -> None:
        case.controller.request_stop()
        check(*arguments, deadline=deadline)

    monkeypatch.setattr(case.control, "check_candidate", stopped_during_check)
    edit_source(case.source, "add")
    reload(case)
    # The check passed, but stop intent wins: no drain, journal, router write or publication.
    assert case.actions == ["control:check"] and case.router.puts == [] and "pending" not in case.state()
    assert case.control.publications == [] and case.controller.stop_requested
    assert case.control.snapshot_version == case.controller.snapshot_version == case.startup_version


def test_removing_a_target_that_a_rule_still_names_is_rejected_before_the_child_is_asked(case: PolicyCase) -> None:
    edit_source(case.source, "remove_referenced")
    reload(case)
    event = last_event(case)
    assert event["state"] == "policy_rejected" and "unknown target 'teacher'" in str(event["reason"])
    assert case.actions == [] and case.router.puts == []


def test_a_rolled_back_catalog_change_keeps_the_committed_open_allowance(case: PolicyCase) -> None:
    allowance = case.control.open_timeout_seconds
    edit_source(case.source, "add")
    case.router.mode = "failed"
    reload(case)
    assert "policy_rolled_back" in case.events() and case.control.publications == []
    assert case.control.snapshot_version == case.startup_version
    assert case.control.open_timeout_seconds == allowance and case.actions[-1] == "control:open"


def test_a_source_that_is_not_yaml_is_rejected_and_the_next_valid_source_applies(case: PolicyCase) -> None:
    case.source.write_text("scenarios: [\n", encoding="utf-8")
    reload(case)
    last = json.loads((case.controller.directory / "lifecycle-events.jsonl").read_text().splitlines()[-1])
    assert last["state"] == "policy_rejected" and last["reason"].startswith("invalid routing manifest YAML")
    # The serving policy stays in place: no drain, no router write.
    assert case.controller.state == "ready"
    assert case.control.snapshot_version == case.controller.snapshot_version == case.startup_version
    assert case.router.puts == [] and case.actions == []
    # A new deployment refuses the same source with the same readable error.
    with pytest.raises(ValueError, match="invalid routing manifest YAML"):
        case.controller.validate_candidate()
    write_source(case.source, keyword="escalate", rule="escalated")
    reload(case)
    assert case.events()[-2:] == ["policy_committed", "ready"]
    assert case.control.snapshot_version == case.controller.snapshot_version != case.startup_version


@pytest.mark.parametrize("mode", ["failed", "hash_mismatch"])
def test_failed_activation_restores_the_committed_router_document_and_reopens_the_old_policy(
    case: PolicyCase, mode: str
) -> None:
    committed = case.state()["committed"]
    original = case.router.document
    write_source(case.source, keyword="escalate")
    case.router.mode = mode
    reload(case)
    assert case.router.document == original and len(case.router.puts) == 2
    assert case.control.snapshot_version == case.startup_version
    assert "control:activate" not in case.actions and case.actions[-1] == "control:open"
    assert case.state() == {"committed": committed}
    assert "policy_rolled_back" in case.events()


def test_normalized_update_commits_the_routers_own_identity(case: PolicyCase) -> None:
    case.router.normalize = True
    write_source(case.source, keyword="escalate", rule="escalated")
    reload(case)
    committed = case.state()["committed"]
    assert committed["router_etag"] == etag(case.router.document) == etag(normalized(case.router.puts[0]))
    assert committed["runtime_hash"] == runtime(case.router.document)
    assert "pending" not in case.state() and case.actions[-1] == "control:open"


@pytest.mark.parametrize("rollback_reply", ["ok", "lost"])
def test_normalized_rollback_records_the_restored_identity_for_the_same_snapshot(
    case: PolicyCase, rollback_reply: str
) -> None:
    committed = case.state()["committed"]
    startup_document = case.router.document
    case.router.normalize = True
    write_source(case.source, keyword="escalate", rule="escalated")
    case.router.mode, case.router.next_mode = "failed", rollback_reply
    reload(case)
    # The restored document means the same but has new bytes, ETag and runtime hash.
    restored = normalized(startup_document)
    assert case.router.document == restored != startup_document and len(case.router.puts) == 2
    assert case.router.active == runtime(restored)
    state = case.state()
    assert state == {"committed": {**committed, "router_etag": etag(restored), "runtime_hash": runtime(restored)}}
    assert case.control.snapshot_version == case.controller.snapshot_version == case.startup_version
    # Restoring the policy is not restoring its startup hash: the new runtime is published before reopening,
    # with the snapshot unchanged, or every automatic request would expect the old runtime and be refused.
    [publication] = case.control.publications
    assert publication["router_config_hash"] == runtime(restored) != committed["runtime_hash"]
    assert publication["snapshot_version"] == case.startup_version
    assert case.actions.index("control:activate") < case.actions.index("control:open") == len(case.actions) - 1
    assert case.events()[-3:] == ["policy_rolled_back", "runtime_published", "ready"]


@pytest.mark.parametrize(("lost_replies", "outcome"), [(1, "active"), (10**6, "timeout")])
def test_a_lost_readiness_reply_is_read_again_and_the_put_is_never_sent_again(
    case: PolicyCase, lost_replies: int, outcome: str
) -> None:
    case.controller.policy = replace(case.controller.policy, ready_timeout_seconds=0.5)
    management = case.controller.management
    assert management is not None
    current = case.router.document
    candidate = current + "# candidate\n"
    case.router.lost_ready_replies = lost_replies
    started = time.monotonic()
    assert case.controller.replace_router_document(management, candidate, etag(current), etag(candidate)) == outcome
    assert case.router.puts == [candidate]
    # Lost replies are read again only until the observation deadline.
    assert time.monotonic() - started < 3


def test_an_unknown_writer_is_refused_while_readiness_replies_are_lost(case: PolicyCase) -> None:
    management = case.controller.management
    assert management is not None
    current = case.router.document
    candidate = current + "# candidate\n"
    case.router.lost_ready_replies = 10**6
    case.router.writer_at_lost_ready = "written: elsewhere\n"
    with pytest.raises(RoutingDeploymentError, match="unknown writer"):
        case.controller.replace_router_document(management, candidate, etag(current), etag(candidate))
    assert case.router.puts == [candidate] and case.router.document == "written: elsewhere\n"


@pytest.mark.parametrize("trickle", ["headers", "body"])
def test_a_trickling_management_reply_ends_at_its_deadline(case: PolicyCase, trickle: str) -> None:
    management = case.controller.management
    assert management is not None
    case.router.trickle = trickle
    started = time.monotonic()
    with pytest.raises(RouterPolicyError):
        management.runtime_hashes(deadline=started + 0.2)
    assert time.monotonic() - started < 0.6
    # Without a phase deadline, the client's own one-second timeout bounds the whole exchange.
    started = time.monotonic()
    with pytest.raises(RouterPolicyError):
        management.runtime_hashes()
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("slow_step", ["encode", "decode"])
def test_time_spent_encoding_or_decoding_counts_against_the_call_deadline(
    case: PolicyCase, monkeypatch: pytest.MonkeyPatch, slow_step: str
) -> None:
    import reef.service.deploy.routing.policy as policy_module

    management = case.controller.management
    assert management is not None

    def delayed(function: Callable[..., object]) -> Callable[..., object]:
        def slow(*arguments: object, **options: object) -> object:
            time.sleep(0.1)
            return function(*arguments, **options)

        return slow

    # Only the client's own JSON step is slowed, not the test server's.
    codec: dict[str, Callable[..., object]] = {"dumps": json.dumps, "loads": json.loads}
    step = "dumps" if slow_step == "encode" else "loads"
    codec[step] = delayed(codec[step])
    monkeypatch.setattr(policy_module, "json", SimpleNamespace(**codec))
    current = case.router.document
    candidate = current + "# candidate\n"
    with pytest.raises(RouterPolicyError, match="not sent" if slow_step == "encode" else "after its deadline"):
        management.replace(candidate, etag(current), deadline=time.monotonic() + 0.05)
    # A PUT encoded past its deadline is never sent. One whose reply is decoded too late was still applied.
    assert case.router.puts == ([] if slow_step == "encode" else [candidate])


def test_a_trickling_router_holds_neither_a_reopening_nor_an_update_past_its_phase(case: PolicyCase) -> None:
    management = case.controller.management
    assert management is not None
    case.router.trickle = "body"
    case.actions.clear()
    case.controller.ready_deadline = time.monotonic() + 0.3
    started = time.monotonic()
    with pytest.raises(RoutingReadinessTimeout):
        case.controller.wait_ready(initial=False)
    assert time.monotonic() - started < 1 and "control:open" not in case.actions
    case.controller.policy = replace(case.controller.policy, ready_timeout_seconds=0.3)
    current = case.router.document
    candidate = current + "# candidate\n"
    started = time.monotonic()
    assert case.controller.replace_router_document(management, candidate, etag(current), etag(candidate)) == "timeout"
    assert time.monotonic() - started < 1
    # The PUT was sent once and applied although its reply never arrived: a timeout does not prove it failed.
    assert case.router.puts == [candidate] and case.router.document == candidate


def test_a_writer_during_the_rollback_plan_is_never_overwritten(case: PolicyCase) -> None:
    case.router.normalize = True
    write_source(case.source, keyword="escalate")
    case.router.mode = "failed"
    case.router.overwrite_at_plan = 2
    with pytest.raises(RoutingDeploymentError, match="unknown writer"):
        reload(case)
    assert case.router.document == "written: elsewhere\n" and len(case.router.puts) == 1
    assert "control:open" not in case.actions
    assert case.state()["pending"]["phase"] == "replacing"


@pytest.mark.parametrize("failing_write", [2, 3])
def test_rollback_journal_failures_stay_closed_and_keep_recovery(
    case: PolicyCase, monkeypatch: pytest.MonkeyPatch, failing_write: int
) -> None:
    committed = case.state()["committed"]
    case.router.normalize = True
    write_source(case.source, keyword="escalate")
    case.router.mode = "failed"
    writes = []
    original = PolicyStateStore.write

    def fail_once(store: PolicyStateStore, state: PolicyState) -> None:
        writes.append(state)
        if len(writes) == failing_write:
            raise OSError("disk full")
        original(store, state)

    monkeypatch.setattr(PolicyStateStore, "write", fail_once)
    with pytest.raises(RoutingDeploymentError):
        reload(case)
    assert "control:open" not in case.actions
    state = case.state()
    # The committed snapshot and its manifest copy remain the recovery authority.
    assert state["committed"] == committed and "pending" in state
    assert RoutingSnapshot.load(case.controller.startup_source()).version == case.startup_version
    assert state["pending"]["phase"] == ("replacing" if failing_write == 2 else "restoring")


@pytest.mark.parametrize("mode", ["other_writer", "lost_unknown"])
def test_another_writer_stops_the_update_closed_without_overwriting_it(case: PolicyCase, mode: str) -> None:
    write_source(case.source, keyword="escalate")
    case.router.mode = mode
    with pytest.raises(RoutingDeploymentError, match="writer"):
        reload(case)
    assert case.router.document == "written: elsewhere\n" and len(case.router.puts) == 1
    assert "control:open" not in case.actions and "control:activate" not in case.actions
    assert case.state()["pending"]["snapshot_version"] != case.startup_version


def test_publication_failure_stays_closed_with_the_activated_pair_committed(case: PolicyCase) -> None:
    write_source(case.source, keyword="escalate")
    case.control.activation_error = RoutingControlError("REEF lifecycle returned HTTP 409", status_code=409)
    with pytest.raises(RoutingDeploymentError, match="publication or persistence"):
        reload(case)
    assert "control:open" not in case.actions
    state = case.state()
    # Persisted before the child published: a restart recovers the pair the router actually runs.
    assert state["pending"]["phase"] == "publishing"
    assert state["committed"]["router_etag"] == etag(case.router.document)
    assert state["committed"]["snapshot_version"] == state["pending"]["snapshot_version"]
    assert RoutingSnapshot.load(case.controller.startup_source()).version == state["committed"]["snapshot_version"]


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_journal_write_failures_never_reopen(
    case: PolicyCase, monkeypatch: pytest.MonkeyPatch, failing_write: int
) -> None:
    committed = case.state()["committed"]
    write_source(case.source, keyword="escalate")
    writes = []
    original = PolicyStateStore.write

    def fail_once(store: PolicyStateStore, state: PolicyState) -> None:
        writes.append(state)
        if len(writes) == failing_write:
            raise OSError("disk full")
        original(store, state)

    monkeypatch.setattr(PolicyStateStore, "write", fail_once)
    with pytest.raises(RoutingDeploymentError, match="publication or persistence"):
        reload(case)
    assert "control:open" not in case.actions
    state = case.state()
    if failing_write == 1:
        # The router was never asked to change.
        assert state == {"committed": committed} and case.router.puts == []
    elif failing_write == 2:
        assert state["committed"] == committed and state["pending"]["phase"] == "replacing"
        assert "control:activate" not in case.actions
    else:
        assert state["pending"]["phase"] == "publishing" and "control:activate" in case.actions


def test_stop_during_activation_takes_priority_over_opening(case: PolicyCase) -> None:
    write_source(case.source, keyword="escalate")
    case.router.mode = "stop"
    with pytest.raises(InterruptedError):
        reload(case)
    assert case.controller.stop_requested
    assert "control:open" not in case.actions and "control:activate" not in case.actions


@pytest.mark.parametrize("stop_at", ["expiring_put", "final_ready"])
def test_stop_during_the_last_activation_call_ends_the_update_without_more_router_calls(
    case: PolicyCase, stop_at: str
) -> None:
    write_source(case.source, keyword="escalate")
    if stop_at == "expiring_put":
        # Stop arrives during the PUT, whose reply then outlasts the whole two-second observation phase. The
        # client's own timeout is longer, so the PUT uses up the phase.
        assert case.controller.management is not None
        case.controller.management = replace(case.controller.management, timeout_seconds=5)
        case.router.mode = "stop_late"
    else:
        case.router.stop_at_ready = True
    with pytest.raises(InterruptedError):
        reload(case)
    last_call = "PUT /api/v1/config" if stop_at == "expiring_put" else "GET /ready"
    assert case.router.requests[-1] == last_call and len(case.router.puts) == 1
    assert "control:activate" not in case.actions and "control:open" not in case.actions
    # The uncertain update stays in the journal for the next start to inspect.
    assert "pending" in case.state()


@pytest.mark.parametrize("source", ["candidate", "invalid"])
def test_owner_restart_after_an_interrupted_update_recovers_the_committed_policy(
    case: PolicyCase, source: str
) -> None:
    journal = case.controller.registry.directory / "policy-state.json"
    committed = case.state()["committed"]
    # The previous owner died while candidate B was pending; the source file now holds B (or garbage).
    if source == "candidate":
        write_source(case.source, keyword="escalate", rule="escalated")
    else:
        case.source.write_text("not: [a manifest", encoding="utf-8")
    # An older owner's journal: its pending record names only the candidate ETag.
    journal.write_text(
        json.dumps({"committed": {**committed, "generation": "old-generation"}, "pending": {"to_etag": '"b"'}})
    )

    recovered = case.controller.startup_source()
    assert RoutingSnapshot.load(recovered).version == case.startup_version
    assert case.controller.validate_candidate() == case.startup_version
    # Not retired before this generation's router runs the committed policy.
    case.router.activation_status, case.router.active = "pending", "loading"
    with pytest.raises(RoutingControlError):
        case.controller.check_router_policy()
    assert case.state()["pending"] == {"to_etag": '"b"'}
    case.router.active = case.router.generated
    case.controller.check_router_policy()
    assert "pending" not in case.state()
    assert case.state()["committed"]["generation"] == "generation-one"
    assert case.events()[-1] == "pending_policy_recovered"


@pytest.mark.parametrize(
    "journal", [{"pending": {"to_etag": '"b"'}}, {"pending": {}}], ids=["partial_pending", "empty_pending"]
)
def test_interrupted_update_without_a_committed_copy_stays_closed(
    case: PolicyCase, journal: dict[str, object]
) -> None:
    committed = case.state()["committed"]
    path = case.controller.registry.directory / "policy-state.json"
    path.write_text(json.dumps({"committed": committed, **journal}))
    Path(committed["source"]).unlink()
    with pytest.raises(RoutingDeploymentError, match="recover the deployment explicitly"):
        case.controller.startup_source()
    # A pending update whose committed pair is gone is the same interrupted update, not a schema error.
    path.write_text(json.dumps(journal))
    with pytest.raises(RoutingDeploymentError, match="recover the deployment explicitly"):
        case.controller.startup_source()


@pytest.mark.parametrize("fault", ["pending_runtime", "not_ready", "other_document"])
def test_reopening_requires_the_committed_runtime_active_and_ready(case: PolicyCase, fault: str) -> None:
    committed = case.controller.committed_policy()
    if fault == "pending_runtime":
        case.router.active, case.router.activation_status = "different-runtime", "pending"
    elif fault == "not_ready":
        case.router.ready = False
    else:
        case.router.document = "written: elsewhere\n"
    case.actions.clear()
    case.controller.ready_deadline = time.monotonic() + 0.3
    with pytest.raises(RoutingDeploymentError):
        # Rollback with the committed ETag, then the reopening check used by every path.
        case.controller.restore_committed(case.controller.management, committed, "timeout", '"candidate"')
        case.controller.wait_ready(initial=False)
    assert "control:open" not in case.actions


def test_child_publishes_policy_only_closed_and_drained_and_rejects_the_old_digest(tmp_path: Path) -> None:
    """The new policy also moves both targets to another provider endpoint: only new handlers reach it."""

    def provider_app(received: list[dict[str, object]]) -> web.Application:
        async def models(request: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "student-v1"}, {"id": "teacher-v2"}]})

        async def inference(request: web.Request) -> web.Response:
            payload = await request.json()
            received.append(payload)
            if payload.get("stream"):
                return web.Response(
                    text='data: {"id":"c","choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n',
                    content_type="text/event-stream",
                )
            return web.json_response({"choices": [{"message": {"content": "a"}, "finish_reason": "stop"}]})

        app = web.Application()
        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/chat/completions", inference)
        return app

    async def run() -> None:
        received: list[dict[str, object]] = []
        moved: list[dict[str, object]] = []
        old_hash, new_hash = "4" * 64, "5" * 64
        async with (
            TestServer(provider_app(received)) as provider,
            TestServer(provider_app(moved)) as moved_provider,
            contextlib.AsyncExitStack() as stack,
        ):
            url = str(provider.make_url("")).rstrip("/")
            moved_url = str(moved_provider.make_url("")).rstrip("/")
            old_path = write_source(tmp_path / "old.json", url=url)
            old = RoutingAdapter.from_manifest(old_path, router_config_hash=old_hash, environ={})
            new_path = write_source(tmp_path / "new.json", keyword="escalate", rule="escalated", url=moved_url)
            new_version = RoutingSnapshot.load(new_path).version
            selector_path = write_source(tmp_path / "selector.json", url=url, automatic_model="router-auto")
            scenario_path = write_source(tmp_path / "scenario.json", url=url, scenario="review")
            dispatcher = routing_dispatcher(tmp_path)
            router = await stack.enter_async_context(running_router(old.snapshot, old_hash))
            app = create_app(
                dispatcher,
                tokens=SERVICE_TOKEN,
                routing=old,
                preview=router.settings,
                lifecycle=LifecycleSettings(
                    "deployment",
                    "generation",
                    "control",
                    old.snapshot.version,
                    selection_token=SELECTION_TOKEN,
                    router_config_hash=old_hash,
                ),
            )
            control_headers = {"Authorization": "Bearer control"}
            revision = 0

            async def control(client: TestClient, action: str, **publication: str) -> int:
                nonlocal revision
                body: dict[str, object] = {
                    "deployment_id": "deployment",
                    "generation": "generation",
                    "control_revision": revision,
                    "action": action,
                    **publication,
                }
                if action == "open":
                    body["open_deadline_monotonic"] = OPEN_DEADLINE
                revision += 1
                return (await client.post("/reef/lifecycle", headers=control_headers, json=body)).status

            def activation(path: Path, version: str, expected: str) -> dict[str, str]:
                return {
                    "manifest_path": str(path),
                    "snapshot_version": version,
                    "router_config_hash": new_hash,
                    "expected_snapshot_version": expected,
                }

            def routed(keyword: str, stream: bool) -> dict[str, object]:
                return {"model": "auto", "messages": [{"role": "user", "content": keyword}], "stream": stream}

            try:
                async with (
                    TestClient(TestServer(app)) as client,
                    TestClient(TestServer(app[CONTROL_APPLICATION])) as lifecycle,
                ):
                    # A report that starts while the new child is still closed does not hold its first activation.
                    report_released = asyncio.Event()
                    held_report = asyncio.create_task(app[REQUEST_ACCEPTANCE].writes.run(report_released.wait()))
                    await asyncio.sleep(0)
                    try:
                        republish = {
                            "manifest_path": str(old_path),
                            "snapshot_version": old.snapshot.version,
                            "router_config_hash": old_hash,
                            "expected_snapshot_version": old.snapshot.version,
                        }
                        assert await control(lifecycle, "activate", **republish) == 200
                        state = await (await lifecycle.get("/reef/lifecycle", headers=control_headers)).json()
                        assert state["accepting"] is False
                        assert (state["pending_writes"], state["drain_pending_writes"]) == (1, 0)
                    finally:
                        report_released.set()
                        await held_report
                    assert await control(lifecycle, "open") == 200
                    response = await client.post(
                        "/v1/chat/completions", headers=routing_headers(), json=routed("reef_teacher_required", False)
                    )
                    assert response.status == 200
                    to_new = activation(new_path, new_version, old.snapshot.version)
                    assert await control(lifecycle, "activate", **to_new) == 409
                    assert await control(lifecycle, "close") == 200
                    closed = await client.post(
                        "/v1/chat/completions", headers=routing_headers(), json=routed("escalate", False)
                    )
                    assert closed.status == 503
                    # The child keeps its own check: a selector or Scenario-name change is refused.
                    for incompatible_path in (selector_path, scenario_path):
                        incompatible_version = RoutingSnapshot.load(incompatible_path).version
                        incompatible = activation(incompatible_path, incompatible_version, old.snapshot.version)
                        assert await control(lifecycle, "activate", **incompatible) == 409
                    wrong = activation(new_path, new_version, "sha256:" + "0" * 64)
                    assert await control(lifecycle, "activate", **wrong) == 409
                    assert await control(lifecycle, "activate", **to_new) == 200
                    state = await (await lifecycle.get("/reef/lifecycle", headers=control_headers)).json()
                    assert (state["snapshot_version"], state["router_config_hash"]) == (new_version, new_hash)
                    assert state["accepting"] is False
                    # Opening previews every Scenario first; the router still runs the old runtime, so it stays shut.
                    assert await control(lifecycle, "open") == 503
                    router.snapshot, router.runtime_hash = RoutingSnapshot.load(new_path), new_hash
                    assert await control(lifecycle, "open") == 200
                    for stream in (False, True):
                        response = await client.post(
                            "/v1/chat/completions", headers=routing_headers(), json=routed("escalate", stream)
                        )
                        assert response.status == 200, await response.text()
                        if stream:
                            text = (await response.read()).decode()
                            receipt = next(
                                json.loads(line.removeprefix("data: "))["reef"]["agent_record_id"]
                                for line in text.splitlines()
                                if line.startswith("data: {") and '"reef"' in line
                            )
                        else:
                            receipt = response.headers["x-reef-agent-record-id"]
                        report = await client.post(
                            "/reef/report",
                            headers={"Authorization": "Bearer gateway-test-token", "x-reef-scenario": "coding"},
                            json={"score": 1, "references": [receipt]},
                        )
                        assert report.status == 200
                    records = dispatcher.get_or_create_scenario("coding").records.replay("coding")
                    selections = [
                        (routing["snapshot_version"], routing["router_config_hash"], routing["rule_name"])
                        for routing in (
                            item.payload["metadata"]["routing"]
                            for item in records
                            if "routing" in item.payload.get("metadata", {})
                        )
                    ]
                    assert selections == [
                        (old.snapshot.version, old_hash, "teacher_requested"),
                        (new_version, new_hash, "escalated"),
                        (new_version, new_hash, "escalated"),
                    ]
                    # The call admitted under the old policy used the old endpoint; JSON and SSE after it, the new.
                    assert len(received) == 1 and [payload.get("stream", False) for payload in moved] == [False, True]
                    # The old policy's adapter and handlers are unchanged, for any request that still holds them.
                    assert old.handlers["teacher"] is not app[REQUEST_SERVICE_KEY].routing.handlers["teacher"]
            finally:
                dispatcher.close()

    asyncio.run(run())


def test_a_published_catalog_gets_new_handlers_with_the_startup_credentials_and_full_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, float, bool]] = []

    class RecordingHandler(HttpInferenceHandler):
        """Records each new handler's endpoint, timeout and whether it sends a credential."""

        def __init__(
            self, upstream_url: str, *, request_headers: RequestHeadersFactory | None = None, timeout_s: float = 300.0
        ) -> None:
            built.append((upstream_url, timeout_s, request_headers is not None))
            super().__init__(upstream_url, request_headers=request_headers, timeout_s=timeout_s)

    monkeypatch.setattr(routing_module, "HttpInferenceHandler", RecordingHandler)
    # The child's startup environment: one variable the first policy uses, one it does not.
    environment = {"STUDENT_KEY": "student-secret", "SPARE_KEY": "spare-secret"}
    first = write_source(tmp_path / "first.json", api_key_env="STUDENT_KEY")
    old = RoutingAdapter.from_manifest(first, router_config_hash=None, environ=environment, timeout_s=123.0)
    old_handlers = dict(old.handlers)
    # A variable set after startup never reaches the running child.
    environment["LATE_KEY"] = "late-secret"
    built.clear()
    with contextlib.closing(routing_dispatcher(tmp_path)) as dispatcher:
        publication = RoutingPublication(RequestService(dispatcher, routing=old))
        moved = write_source(tmp_path / "moved.json", url="http://127.0.0.1:12", api_key_env="SPARE_KEY")
        publication.publish(str(moved), RoutingSnapshot.load(moved).version, "5" * 64)
        published = publication.current()
        assert published is not None and published.snapshot.version == RoutingSnapshot.load(moved).version
        # Every enabled target gets a new handler for its new endpoint, with the startup timeout, not the
        # readiness cap; the student's credential comes from the spare variable.
        assert built == [("http://127.0.0.1:12", 123.0, True), ("http://127.0.0.1:12", 123.0, False)]
        assert published.readiness_timeout_s == 10 and "secret" not in repr(published)
        # The old adapter, which an admitted request may still hold, is unchanged.
        assert dict(old.handlers) == old_handlers and old.snapshot.version == RoutingSnapshot.load(first).version
        for variable in ("LATE_KEY", "ABSENT_KEY"):
            refused = write_source(tmp_path / f"{variable}.json", api_key_env=variable)
            with pytest.raises(ValueError, match=f"requires environment variable {variable}") as error:
                publication.publish(str(refused), RoutingSnapshot.load(refused).version, "5" * 64)
            assert "secret" not in str(error.value) and publication.current() is published


@pytest.mark.parametrize("kind", ["rejected", "unchanged"])
def test_diagnostic_policy_outcome_keeps_following_reload_live(case: PolicyCase, kind: str) -> None:
    if kind == "rejected":
        write_source(case.source, automatic_model="unsupported-selector-change")
    reload(case)
    write_source(case.source, keyword="after-diagnostic", rule="updated")
    reload(case)
    assert case.controller.snapshot_version != case.startup_version
    assert case.control.snapshot_version == case.controller.snapshot_version
    assert case.control.router_config_hash == case.router.active
    assert case.controller.state == "ready"


@pytest.mark.parametrize("kind", ["rejected", "unchanged"])
def test_diagnostic_policy_outcome_keeps_dependency_monitoring_live(case: PolicyCase, kind: str) -> None:
    if kind == "rejected":
        write_source(case.source, automatic_model="unsupported-selector-change")
    reload(case)
    case.router.active = "f" * 64
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="fail_stop"):
        case.controller.tick()


def last_event(case: PolicyCase) -> dict[str, object]:
    return json.loads((case.controller.directory / "lifecycle-events.jsonl").read_text().splitlines()[-1])


def test_a_domain_rule_reloads_in_place_while_the_classifier_stays(domain_case: PolicyCase) -> None:
    classifier = json.loads(domain_case.source.read_text())["deployment"]["domain_classifier"]
    write_source(domain_case.source, domain_classifier=classifier, domain_target="student")
    reload(domain_case)
    assert domain_case.events()[-2:] == ["policy_committed", "ready"]
    controller, control = domain_case.controller, domain_case.control
    assert controller.snapshot_version == control.snapshot_version != domain_case.startup_version
    recipe = yaml.safe_load(domain_case.router.document)["recipes"][0]["routing"]
    math_rule = next(decision for decision in recipe["decisions"] if decision["name"] == "math_request")
    assert math_rule["modelRefs"] == [{"model": "student", "use_reasoning": False}]
    assert recipe["signals"]["domains"] == [{"name": "math_request", "mmlu_categories": ["math"]}]


@pytest.mark.parametrize("change", ["threshold", "changed_file", "removed"])
def test_a_classifier_change_is_rejected_and_the_serving_policy_stays(domain_case: PolicyCase, change: str) -> None:
    classifier = json.loads(domain_case.source.read_text())["deployment"]["domain_classifier"]
    if change == "threshold":
        write_source(domain_case.source, domain_classifier={**classifier, "threshold": 0.6}, domain_target="teacher")
    elif change == "changed_file":
        # The host changed a file after startup; the next render checks it again and refuses the update.
        (Path(classifier["model_directory"]) / "model.safetensors").write_text("other weights")
        write_source(domain_case.source, domain_classifier=classifier, domain_target="student")
    else:
        write_source(domain_case.source)
    reload(domain_case)
    event = last_event(domain_case)
    assert event["state"] == "policy_rejected"
    assert "sha256" in str(event["reason"]) if change == "changed_file" else "restart" in str(event["reason"])
    assert domain_case.controller.state == "ready" and domain_case.router.puts == []
    controller, control = domain_case.controller, domain_case.control
    assert control.snapshot_version == controller.snapshot_version == domain_case.startup_version
    assert "control:close" not in domain_case.actions


@pytest.mark.parametrize("fault", ["oversized_threshold", "deep_label_json"])
def test_invalid_classifier_updates_leave_the_active_policy_serving(domain_case: PolicyCase, fault: str) -> None:
    classifier = json.loads(domain_case.source.read_text())["deployment"]["domain_classifier"]
    if fault == "oversized_threshold":
        classifier["threshold"] = 10**1000
    else:
        # Valid JSON can exceed the decoder's nesting limit even in a small file.
        content = '{"padding":' + "[" * 1200 + "0" + "]" * 1200 + "}"
        (Path(classifier["model_directory"]) / "config.json").write_text(content)
        classifier["files"]["config.json"] = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    write_source(domain_case.source, domain_classifier=classifier, domain_target="student")
    reload(domain_case)
    assert last_event(domain_case)["state"] == "policy_rejected"
    assert domain_case.controller.state == "ready" and domain_case.router.puts == []
    assert (
        domain_case.control.snapshot_version == domain_case.controller.snapshot_version == domain_case.startup_version
    )
    assert "control:close" not in domain_case.actions


def test_adding_a_classifier_needs_a_restart(case: PolicyCase, tmp_path: Path) -> None:
    classifier = write_domain_classifier(tmp_path / "vela-domain")
    write_source(case.source, domain_classifier=classifier, domain_target="teacher")
    reload(case)
    assert last_event(case)["state"] == "policy_rejected" and "restart" in str(last_event(case)["reason"])
    assert case.router.puts == [] and case.controller.snapshot_version == case.startup_version


COMMITTED = {
    "generation": "generation-one",
    "snapshot_version": "sha256:" + "2" * 64,
    "source": "/private/committed/source.yaml",
    "manifest": "/private/generation/manifest.yaml",
    "router_document": "/private/generation/router.yaml",
    "router_etag": '"a"',
    "runtime_hash": "6" * 64,
}
FULL_PENDING = {
    "phase": "restoring",
    "from_etag": '"a"',
    "to_etag": '"b"',
    "snapshot_version": "sha256:" + "3" * 64,
    "manifest": "/private/generation/policies/0001/manifest.yaml",
    "router_document": "/private/generation/policies/0001/router.yaml",
    "restore_etag": '"c"',
}


@pytest.mark.parametrize(
    "journal",
    [
        {},
        {"committed": COMMITTED},
        {"committed": COMMITTED, "pending": FULL_PENDING},
        {"committed": COMMITTED, "pending": {"to_etag": '"b"'}},
        {"committed": COMMITTED, "pending": {}},
        {"pending": {"phase": "replacing"}},
    ],
    ids=["empty", "committed", "full_pending", "older_partial_pending", "empty_pending", "pending_without_committed"],
)
def test_the_policy_journal_reads_and_rewrites_its_wire_form_unchanged(
    tmp_path: Path, journal: dict[str, object]
) -> None:
    store = PolicyStateStore(tmp_path / "policy-state.json")
    store.path.write_text(json.dumps(journal))
    state = store.load()
    store.write(state)
    assert json.loads(store.path.read_text()) == journal
    assert store.load() == state
    assert (state.committed is None) is ("committed" not in journal)
    assert (state.pending is None) is ("pending" not in journal)


def test_a_missing_journal_and_null_entries_hold_nothing(tmp_path: Path) -> None:
    store = PolicyStateStore(tmp_path / "policy-state.json")
    assert store.load() == PolicyState()
    store.path.write_text(json.dumps({"committed": COMMITTED, "pending": None}))
    assert store.load().pending is None and store.load().committed is not None
    store.path.write_text(json.dumps({"committed": None, "pending": None}))
    assert store.load() == PolicyState()


@pytest.mark.parametrize(
    "journal",
    [
        [],
        {"other": {}},
        {"committed": []},
        {"committed": {**COMMITTED, "runtime_hash": ""}},
        {"committed": {name: value for name, value in COMMITTED.items() if name != "source"}},
        {"committed": {**COMMITTED, "extra": "x"}},
        {"committed": {**COMMITTED, "generation": 1}},
        {"pending": []},
        {"pending": {"phase": 1}},
        {"pending": {"unknown": "x"}},
    ],
)
def test_a_malformed_policy_journal_is_refused(tmp_path: Path, journal: object) -> None:
    store = PolicyStateStore(tmp_path / "policy-state.json")
    store.path.write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        store.load()


def test_a_runtime_report_keeps_only_non_empty_string_fields() -> None:
    reply = {"activation_status": "pending", "generated_runtime_hash": 7, "active_runtime_hash": ""}
    assert RouterRuntime.from_reply(reply) == RouterRuntime("pending", None, None)
    # Fields the report lacks stay missing; the controller does not mistake that for any hash.
    assert RouterRuntime.from_reply({"source_config_hash": '"a"'}) == RouterRuntime(None, None, None)
    with pytest.raises(ValueError):
        RouterRuntime("", None, None)


@pytest.mark.parametrize(
    ("port", "token", "timeout_seconds"),
    [
        (0, MANAGEMENT_TOKEN, 1.0),
        (True, MANAGEMENT_TOKEN, 1.0),
        (8080, "", 1.0),
        (8080, "two words", 1.0),
        (8080, MANAGEMENT_TOKEN, float("nan")),
        (8080, MANAGEMENT_TOKEN, 0),
    ],
)
def test_a_management_client_needs_a_port_a_header_safe_token_and_a_bounded_timeout(
    port: object, token: object, timeout_seconds: object
) -> None:
    with pytest.raises(ValueError):
        RouterManagementClient(port, token, timeout_seconds)
