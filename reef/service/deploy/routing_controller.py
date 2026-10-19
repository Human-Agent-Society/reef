"""Single-owner state machine for the REEF-managed Semantic Router container and its REEF child.

The child starts closed, with no router runtime. Before every opening the
controller verifies the router's actual runtime (``active == generated``,
``activation_status`` active, ``/ready``, and the committed document), publishes
that verified hash to the child while it is closed, and asks the child to open;
the child previews one message per Scenario first. The router re-encodes the
documents it stores, so a restored policy can have a new runtime hash: the
verified hash is published again whenever it differs from the child's.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import yaml

from reef.service.deploy.routing_containers import (
    ContainerCommandError,
    ContainerEventStream,
    ContainerInventory,
    ContainerRuntime,
)
from reef.service.deploy.routing_control import MAX_SOCKET_PATH_BYTES, ReefLifecycleClient, RoutingControlError
from reef.service.deploy.routing_policy import (
    PolicyStateStore,
    RouterManagementClient,
    RouterPolicyError,
    document_etag,
)
from reef.service.deploy.routing_registry import DeploymentOwnership, DeploymentRegistry
from reef.service.deploy.routing_renderer import (
    MANAGEMENT_TOKEN_ENV,
    SELECTION_TOKEN_ENV,
    ManagedRoutingRender,
    generate,
)
from reef.service.deploy.service_config import ServiceConfig
from reef.service.routing import RoutingSnapshot, check_policy_change
from reef.service.routing_preview import RESPONSE_WRITE_ALLOWANCE_SECONDS

#: Consecutive failed router readiness checks before the router counts as failed.
READINESS_FAILURE_LIMIT = 3
#: The router's activation states (``pkg/apiserver/config_activation.go`` at the pinned commit).
ACTIVATION_ACTIVE = "active"
ACTIVATION_FAILED = "failed"


class RoutingDeploymentError(RuntimeError):
    """The managed routing deployment cannot safely serve requests."""


class RoutingReadinessTimeout(RoutingDeploymentError):
    """A correctly configured replacement did not become ready in time."""


class RoutingDeploymentController:
    """Serialize startup, observation, bounded recovery, policy updates and shutdown mutations.

    Signal handlers only set stop intent. The startup future is joined before
    cleanup; no monitor thread is allowed to mutate the container deployment.
    """

    def __init__(
        self, settings: ServiceConfig, runtime: ContainerRuntime, *, registry_root: Path | None = None
    ) -> None:
        policy = settings.routing_deployment
        self.policy = policy
        self.service_settings = settings
        self.runtime = runtime
        self.registry = DeploymentRegistry(policy.require_identity(), root=registry_root)
        self.ownership: DeploymentOwnership | None = None
        self.snapshot_version = ""
        self.directory = Path()
        #: Short private directory holding the child's control socket, outside long state paths.
        self.control_directory: Path | None = None
        self.control: ReefLifecycleClient | None = None
        self.event_stream: ContainerEventStream | None = None
        self.container_ids: set[str] = set()
        self.start_pool: ThreadPoolExecutor | None = None
        self.start_future: Future[None] | None = None
        self.restart_count = 0
        self.stop_requested = False
        self.force_stop_requested = False
        self.closed = False
        self.failed = False
        self.state = "new"
        self.next_probe = 0.0
        self.next_inspect = 0.0
        self.probe_failures = 0
        self.cleanup_complete = False
        self.ready_deadline = 0.0
        self.reload_requested = False
        self.policy_store = PolicyStateStore(self.registry.directory / "policy-state.json")
        self.management: RouterManagementClient | None = None
        self.policy_count = 0

    def request_stop(self, *, force: bool = False) -> None:
        self.stop_requested = True
        self.force_stop_requested = self.force_stop_requested or force

    def request_reload(self) -> None:
        """Signal-safe: record intent only; ``tick`` applies it between other lifecycle steps."""
        self.reload_requested = True

    def event(self, state: str, **details: str | int | float | bool) -> None:
        # These outcomes leave the committed policy serving. Keep monitoring it
        # and accepting later reloads instead of treating the diagnostic as a phase.
        if state not in {"policy_rejected", "policy_unchanged"}:
            self.state = state
        data = {"monotonic_seconds": time.monotonic(), "state": state, "restarts": self.restart_count, **details}
        if self.ownership is not None:
            with (self.directory / "lifecycle-events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(data, sort_keys=True) + "\n")
            (self.directory / "lifecycle-events.jsonl").chmod(0o600)
        print("[reef-routing] " + json.dumps(data, sort_keys=True), file=sys.stderr, flush=True)

    def render(self, source: Path, output: Path, ownership: DeploymentOwnership) -> RoutingSnapshot:
        return generate(
            source,
            output,
            ManagedRoutingRender(self.policy, ownership.deployment_id, ownership.owner_id, ownership.generation),
        )

    def prepare(self) -> None:
        if sys.platform != "linux":
            raise ValueError("managed routing requires a local Linux Docker host")
        self.validate_candidate()
        self.registry.acquire()
        try:
            self.runtime.ensure_available()
            previous = self.registry.load()
            if previous is not None:
                self.runtime.remove(previous, self.policy.container_stop_timeout_seconds)
                remaining = self.runtime.inspect(previous)
                if remaining.containers or remaining.network_ids:
                    raise RoutingDeploymentError("previous routing generation did not clean up")
            # Decided under the lock: an interrupted update restarts from its committed pair.
            source = self.startup_source()
            generation = uuid.uuid4().hex
            self.directory = self.registry.directory / "generations" / generation
            self.directory.mkdir(mode=0o700, parents=True)
            self.directory.chmod(0o700)
            # Save before rendering or creating resources, including a partial startup.
            self.ownership = self.registry.prepare(generation, self.directory)
            control_directory = self.control_directory = Path(tempfile.mkdtemp(prefix="reef-ctl-"))
            control_directory.chmod(0o700)
            control_socket = control_directory / "control.sock"
            if len(os.fsencode(control_socket)) > MAX_SOCKET_PATH_BYTES:
                raise RoutingDeploymentError("the control socket path is too long; set a shorter TMPDIR")
            # Distinct credentials for every generation; none of them is a client, model or control token.
            management_token, selection_token = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
            self.runtime.use_environment(
                {MANAGEMENT_TOKEN_ENV: management_token, SELECTION_TOKEN_ENV: selection_token}
            )
            self.management = RouterManagementClient(self.policy.sr_management_port, management_token)
            rendered = self.render(source, self.directory, self.ownership)
            self.snapshot_version = rendered.version
            control_token = secrets.token_urlsafe(48)
            control_data = {
                "deployment_id": self.ownership.deployment_id,
                "generation": generation,
                "control_token": control_token,
                "snapshot_version": rendered.version,
                "finalization_timeout_seconds": self.policy.finalization_timeout_seconds,
                "selection_token": selection_token,
                "control_socket": str(control_socket),
            }
            descriptor = os.open(self.directory / "control.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(control_data, stream)
            # An open previews every Scenario, each within the router's deadline and write allowance.
            open_timeout = len(rendered.scenarios) * (
                self.policy.selection_timeout_seconds + RESPONSE_WRITE_ALLOWANCE_SECONDS
            )
            self.control = ReefLifecycleClient(
                control_socket,
                self.ownership.deployment_id,
                generation,
                control_token,
                open_timeout_seconds=open_timeout + 5,
            )
            self.event("prepared")
        except BaseException:
            # The caller's finally invokes shutdown while this owner still
            # holds the lock, including partially rendered generations.
            self.failed = True
            raise

    def validate_candidate(self) -> str:
        """Render a candidate privately before any ownership or Docker mutation."""
        settings = self.service_settings
        if settings.routing_manifest is None or not settings.tokens or settings.host != "127.0.0.1":
            raise ValueError("managed routing requires routing_manifest, tokens and reef.host=127.0.0.1")
        if settings.port == self.policy.sr_management_port:
            raise ValueError("reef.port must differ from routing sr_management_port")
        with tempfile.TemporaryDirectory(prefix="reef-routing-validate-") as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            candidate = generate(
                self.startup_source(),
                directory / "candidate",
                ManagedRoutingRender(self.policy, self.registry.deployment_id, "validation", "validation"),
            )
            return candidate.version

    def require_clients(self) -> tuple[DeploymentOwnership, ReefLifecycleClient]:
        if self.ownership is None or self.control is None:
            raise RoutingDeploymentError("routing deployment has not been prepared")
        return self.ownership, self.control

    def begin_start(self) -> None:
        ownership, _ = self.require_clients()
        self.ready_deadline = time.monotonic() + self.policy.ready_timeout_seconds
        self.event("starting")
        self.event_stream = self.runtime.events(ownership)
        self.start_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reef-routing-start")
        self.start_future = self.start_pool.submit(
            self.runtime.start, self.directory / "compose.yaml", ownership, self.policy.ready_timeout_seconds
        )

    def join_start(self) -> None:
        if self.start_pool is not None:
            self.start_pool.shutdown(wait=True, cancel_futures=True)
            self.start_pool = None
        if self.start_future is not None:
            future = self.start_future
            self.start_future = None
            future.result()

    def check_status(self, status: dict[str, object]) -> None:
        ownership, _ = self.require_clients()
        if (
            status.get("deployment_id") != ownership.deployment_id
            or status.get("generation") != ownership.generation
            or status.get("snapshot_version") != self.snapshot_version
        ):
            raise RoutingDeploymentError("REEF generation or routing snapshot mismatch")
        if status.get("stopping"):
            raise RoutingDeploymentError("REEF is terminally stopping")

    def inspect_ready(self, inventory: ContainerInventory) -> None:
        if len(inventory.containers) != 1 or inventory.containers[0].service != "semantic-router":
            raise RoutingControlError("the routing container is missing or not the only one")
        router = inventory.containers[0]
        if not router.running or router.paused:
            raise RoutingControlError("the routing container is not running")
        if router.health != "healthy":
            raise RoutingControlError("Semantic Router is not healthy")
        self.container_ids = {router.resource_id}

    def committed_manifest(self) -> str:
        committed = self.policy_store.load().get("committed")
        if not isinstance(committed, dict):
            raise RoutingDeploymentError("routing policy state has no committed pair")
        return str(committed["manifest"])

    def wait_ready(self, *, initial: bool = True) -> None:
        """Verify the router, publish its runtime to the closed child when it changed, then open admission."""
        ownership, control = self.require_clients()
        deadline = self.ready_deadline
        while not self.stop_requested and time.monotonic() < deadline:
            if not initial:
                # The existing REEF child must survive every routing replacement.
                try:
                    self.check_status(control.request())
                except RoutingControlError as exc:
                    raise RoutingDeploymentError("REEF lifecycle unavailable during recovery") from exc
            try:
                self.inspect_ready(self.runtime.inspect(ownership))
                status = control.request()
                self.check_status(status)
                if status.get("drain_pending_writes") != 0:
                    raise RoutingControlError("request records are still finalizing")
                # Every open, including startup, rollback and container recovery, checks the router runtime.
                runtime_hash = self.check_router_policy()
                if status.get("router_config_hash") != runtime_hash:
                    control.request(
                        "activate",
                        publication={
                            "manifest_path": self.committed_manifest(),
                            "snapshot_version": self.snapshot_version,
                            "router_config_hash": runtime_hash,
                            "expected_snapshot_version": self.snapshot_version,
                        },
                    )
                    self.event("runtime_published", router_config_hash=runtime_hash)
            except RoutingControlError as exc:
                if not exc.retryable:
                    raise RoutingDeploymentError(str(exc)) from exc
                time.sleep(0.2)
                continue
            if self.stop_requested:
                break
            event_failure = self.observe_events()
            if event_failure is not None:
                raise RoutingDeploymentError("routing observation failed before opening acceptance")
            try:
                # The child previews one message per Scenario under the published runtime before it opens.
                control.request("open")
            except RoutingControlError as exc:
                if not exc.retryable:
                    raise RoutingDeploymentError(str(exc)) from exc
                # A final write submitted after our status snapshot (409), or a readiness preview that failed
                # (503). Recheck identity, the router and terminal state before trying again.
                time.sleep(0.2)
                continue
            if self.stop_requested:
                control.request("close")
                raise InterruptedError("routing stop arrived while opening acceptance")
            self.event("ready")
            self.next_probe = time.monotonic() + 1
            self.next_inspect = time.monotonic() + 5
            self.probe_failures = 0
            return
        if self.stop_requested:
            raise InterruptedError("routing startup stopped")
        raise RoutingReadinessTimeout("routing deployment did not become ready before its deadline")

    def complete_start(self) -> None:
        self.join_start()
        self.wait_ready()

    def observe_events(self) -> str | None:
        if self.event_stream is None:
            return "stream_disconnected"
        while True:
            event = self.event_stream.poll()
            if event is None:
                return None
            if event.action == "stream_disconnected":
                return event.action
            if event.container_id in self.container_ids and event.action in {
                "die",
                "destroy",
                "pause",
                "health_status: unhealthy",
            }:
                return "container_failure"

    def reconnect_events(self) -> None:
        ownership, control = self.require_clients()
        control.request("close")
        self.event("observation_lost")
        if self.event_stream is not None:
            self.event_stream.close()
        self.event_stream = self.runtime.events(ownership)
        # Re-inspection is mandatory; Docker's bounded event history is not a log.
        self.inspect_ready(self.runtime.inspect(ownership))
        self.ready_deadline = time.monotonic() + min(10, self.policy.ready_timeout_seconds)
        self.wait_ready(initial=False)

    def router_failure(self) -> str | None:
        """The router's failure, if its readiness or active runtime no longer matches the committed pair.

        Periodic checks read ``/ready`` and ``/api/v1/config/hash`` only; they never send selection previews.
        """
        management = self.management
        if management is None:
            return None
        committed = self.policy_store.load().get("committed")
        try:
            hashes = management.runtime_hashes()
            ready = management.ready()
        except RouterPolicyError:
            ready, hashes = False, {}
        if hashes and isinstance(committed, dict):
            active = hashes.get("active_runtime_hash")
            if hashes.get("activation_status") != ACTIVATION_ACTIVE or active != committed.get("runtime_hash"):
                return "router_runtime_changed"
        if ready and hashes:
            self.probe_failures = 0
            return None
        self.probe_failures += 1
        return "router_health" if self.probe_failures >= READINESS_FAILURE_LIMIT else None

    def tick(self) -> None:
        if self.stop_requested or self.state != "ready":
            return
        if self.reload_requested:
            self.update_policy()
            return
        ownership, control = self.require_clients()
        event = self.observe_events()
        if event == "stream_disconnected":
            try:
                self.reconnect_events()
            except InterruptedError:
                if not self.stop_requested:
                    raise
            except (RoutingControlError, ContainerCommandError) as exc:
                raise RoutingDeploymentError("routing observation could not be restored") from exc
            return
        failure = event
        now = time.monotonic()
        if now >= self.next_probe and failure is None:
            self.next_probe = now + 1
            try:
                self.check_status(control.request())
            except RoutingControlError as exc:
                # Losing the REEF control boundary cannot be repaired by restarting SR.
                raise RoutingDeploymentError("REEF lifecycle unavailable") from exc
            failure = self.router_failure()
        if now >= self.next_inspect and failure is None:
            self.next_inspect = now + 5
            try:
                self.inspect_ready(self.runtime.inspect(ownership))
            except RoutingControlError:
                failure = "container_health"
        if failure is not None:
            self.recover(failure)

    @staticmethod
    def reef_idle(status: dict[str, object]) -> bool:
        """No admitted request, including an upload in progress, or pending record write remains in REEF."""
        return status.get("active_requests") == 0 and status.get("drain_pending_writes") == 0

    def drain(self, *, recovery: bool) -> None:
        """Close REEF admission and wait for its work; cancel what remains after the drain deadline."""
        _, control = self.require_clients()
        control.request("close")
        self.event("draining", recovery=recovery)
        deadline = time.monotonic() + self.policy.drain_timeout_seconds
        while time.monotonic() < deadline and not self.force_stop_requested:
            status = control.request()
            self.check_status(status)
            if self.reef_idle(status):
                return
            time.sleep(0.05)
        control.request("close", cancel_active=True)
        self.event("finalizing", recovery=recovery)
        deadline = time.monotonic() + self.policy.finalization_timeout_seconds
        while time.monotonic() < deadline:
            status = control.request()
            self.check_status(status)
            if self.reef_idle(status):
                return
            time.sleep(0.05)
        raise RoutingDeploymentError("request or record finalization exceeded its deadline; recovery prohibited")

    def startup_source(self) -> Path:
        """The manifest a new generation renders.

        Normally the configured source. After an owner stopped during an update,
        the committed manifest instead: the source may hold the uncommitted
        candidate. Without that copy, startup stays closed for an operator.
        """
        settings = self.service_settings
        if settings.routing_manifest is None:
            raise ValueError("managed routing requires routing_manifest")
        state = self.policy_store.load()
        if state.get("pending") is None:
            return Path(settings.routing_manifest)
        committed = state.get("committed")
        source = Path(str(committed.get("source", ""))) if isinstance(committed, dict) else None
        if source is None or not source.is_file():
            raise RoutingDeploymentError(
                "an interrupted policy update has no committed manifest copy; recover the deployment explicitly"
            )
        return source

    def check_router_policy(self) -> str:
        """Confirm the router runs the committed policy, and return its verified runtime hash.

        The source ETag alone is not enough: the runtime hash, the router's
        ``active`` status and its readiness must match too. A generation's first
        check records its own committed pair, which ends crash recovery. A pending
        update whose candidate is not running is resolved as not applied.

        Raises:
            RoutingControlError: Retryable while activation or readiness is in
                progress; not retryable when the router runs another document.
        """
        management = self.management
        if management is None:
            raise RoutingDeploymentError("router management is unavailable")
        ownership, _ = self.require_clients()
        try:
            router_etag = management.config_etag()
            hashes = management.runtime_hashes()
            ready = management.ready()
        except RouterPolicyError as exc:
            raise RoutingControlError("router management API is not ready") from exc
        generated, active = hashes.get("generated_runtime_hash"), hashes.get("active_runtime_hash")
        if (
            hashes.get("activation_status") != ACTIVATION_ACTIVE
            or not isinstance(active, str)
            or not active
            or active != generated
            or not ready
        ):
            raise RoutingControlError("router runtime is not active and ready")
        state = self.policy_store.load()
        committed = state.get("committed")
        if not isinstance(committed, dict) or committed.get("generation") != ownership.generation:
            document = self.directory / "router.yaml"
            if router_etag != document_etag(document.read_bytes()):
                raise RoutingControlError("router does not run this generation's configuration", retryable=False)
            self.policy_store.write(
                {
                    "committed": {
                        "generation": ownership.generation,
                        "snapshot_version": self.snapshot_version,
                        "source": str(
                            self.policy_store.keep_source(self.directory / "manifest.yaml", self.snapshot_version)
                        ),
                        "manifest": str(self.directory / "manifest.yaml"),
                        "router_document": str(document),
                        "router_etag": router_etag,
                        "runtime_hash": active,
                    }
                }
            )
            if state.get("pending") is not None:
                self.event("pending_policy_recovered", snapshot_version=self.snapshot_version)
            return active
        if router_etag != committed["router_etag"] or active != committed["runtime_hash"]:
            raise RoutingControlError("router does not run the committed policy", retryable=False)
        if state.get("pending") is not None:
            self.policy_store.write({"committed": committed})
            self.event("policy_rolled_back")
        return active

    def render_policy(self) -> tuple[Path, RoutingSnapshot]:
        """Render the configured source manifest beside this generation without changing anything running."""
        ownership, _ = self.require_clients()
        settings = self.service_settings
        if settings.routing_manifest is None:
            raise ValueError("managed routing has no source manifest")
        current_manifest = Path(self.committed_manifest())
        self.policy_count += 1
        directory = self.directory / "policies" / f"{self.policy_count:04d}"
        directory.parent.mkdir(mode=0o700, exist_ok=True)
        candidate = self.render(Path(settings.routing_manifest), directory, ownership)
        check_policy_change(RoutingSnapshot.load(current_manifest), candidate)
        deployment = yaml.safe_load((directory / "manifest.yaml").read_text())["deployment"]
        if deployment != yaml.safe_load(current_manifest.read_text())["deployment"]:
            raise ValueError("the router image or deployment settings changed; restart the deployment")
        return directory, candidate

    def update_policy(self) -> None:
        """Apply the source manifest's rules and defaults in place, or leave the serving policy unchanged."""
        self.reload_requested = False
        _, control = self.require_clients()
        management = self.management
        if management is None:
            self.event("policy_rejected", reason="router management is unavailable")
            return
        try:
            directory, candidate = self.render_policy()
            document = (directory / "router.yaml").read_text(encoding="utf-8")
            current_etag, candidate_etag = management.plan(document)
        except (ValueError, OSError, KeyError, RouterPolicyError) as exc:
            # Nothing has closed or changed: the serving snapshot stays in place.
            self.event("policy_rejected", reason=str(exc)[:300])
            return
        state = self.policy_store.load()
        committed = state["committed"]
        if not isinstance(committed, dict):
            raise RoutingDeploymentError("routing policy state has no committed pair")
        if candidate.version == self.snapshot_version:
            self.event("policy_unchanged")
            return
        if current_etag != committed["router_etag"]:
            raise RoutingDeploymentError("another writer changed the router configuration; update refused")
        self.event("policy_updating", snapshot_version=candidate.version)
        self.drain(recovery=False)
        if self.stop_requested:
            return
        pending = {
            "phase": "replacing",
            "from_etag": current_etag,
            "to_etag": candidate_etag,
            "snapshot_version": candidate.version,
            "manifest": str(directory / "manifest.yaml"),
            "router_document": str(directory / "router.yaml"),
        }
        try:
            # Journal before the router mutation, so a crash at any point finds the committed pair.
            self.policy_store.write({"committed": committed, "pending": pending})
            outcome = self.replace_router_document(management, document, current_etag, candidate_etag)
            if outcome == "active":
                runtime_hash = management.runtime_hashes().get("active_runtime_hash")
                if not isinstance(runtime_hash, str) or not runtime_hash:
                    raise RoutingDeploymentError("router runtime hash is unavailable after activation")
                activated = {
                    "generation": committed["generation"],
                    "snapshot_version": candidate.version,
                    "source": str(self.policy_store.keep_source(directory / "manifest.yaml", candidate.version)),
                    "manifest": str(directory / "manifest.yaml"),
                    "router_document": str(directory / "router.yaml"),
                    "router_etag": candidate_etag,
                    "runtime_hash": runtime_hash,
                }
                # The router runs the candidate: it is the committed pair before the child publishes it.
                self.policy_store.write({"committed": activated, "pending": {**pending, "phase": "publishing"}})
                control.request(
                    "activate",
                    publication={
                        "manifest_path": str(directory / "manifest.yaml"),
                        "snapshot_version": candidate.version,
                        "router_config_hash": runtime_hash,
                        "expected_snapshot_version": self.snapshot_version,
                    },
                )
                self.policy_store.write({"committed": activated})
                self.snapshot_version = candidate.version
                self.event("policy_committed", snapshot_version=candidate.version)
            else:
                # The restored policy may run under a new runtime hash; wait_ready publishes it before opening.
                self.restore_committed(management, committed, outcome, candidate_etag)
        except InterruptedError:
            # Stop intent wins; InterruptedError is also an OSError.
            raise
        except (OSError, RoutingControlError) as exc:
            raise RoutingDeploymentError("routing policy publication or persistence failed; staying closed") from exc
        if self.stop_requested:
            return
        self.ready_deadline = time.monotonic() + self.policy.ready_timeout_seconds
        self.wait_ready(initial=False)

    def replace_router_document(
        self, management: RouterManagementClient, document: str, current_etag: str, candidate_etag: str
    ) -> str:
        """PUT the candidate once, then inspect: ``active``, ``rejected`` (unchanged), ``failed`` or ``timeout``."""
        if self.stop_requested:
            raise InterruptedError("routing stop arrived before the policy update")
        status: int | None
        try:
            status, body = management.replace(document, current_etag)
        except RouterPolicyError:
            # The reply was lost; the router may or may not have applied it. Inspect, never replay.
            status, body = None, {}
        if status == 412:
            raise RoutingDeploymentError("another writer changed the router configuration during the update")
        expected = body.get("generated_runtime_hash")
        expected_hash = expected if isinstance(expected, str) and expected else None
        deadline = time.monotonic() + self.policy.ready_timeout_seconds
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise InterruptedError("routing stop arrived during the policy update")
            try:
                etag = management.config_etag()
                hashes = management.runtime_hashes()
            except RouterPolicyError:
                time.sleep(0.1)
                continue
            if etag not in {current_etag, candidate_etag}:
                raise RoutingDeploymentError("an unknown writer changed the router configuration")
            if etag == current_etag and status is not None and 400 <= status < 500:
                return "rejected"
            if etag == candidate_etag:
                generated, active = hashes.get("generated_runtime_hash"), hashes.get("active_runtime_hash")
                activation_status = hashes.get("activation_status")
                if expected_hash is not None and generated != expected_hash:
                    return "failed"
                if (
                    activation_status == ACTIVATION_ACTIVE
                    and isinstance(active, str)
                    and active
                    and active == generated
                    and management.ready()
                ):
                    return "active"
                # "pending" and "unknown" keep waiting until the deadline.
                if status == 503 or activation_status == ACTIVATION_FAILED:
                    return "failed"
            time.sleep(0.1)
        return "timeout"

    def restore_committed(
        self, management: RouterManagementClient, committed: dict[str, object], outcome: str, candidate_etag: str
    ) -> None:
        """Return the router to the committed document, which only this owner's candidate may replace.

        The router re-encodes every document it stores, so the restored bytes,
        their ETag and the runtime hash can differ from the startup document's.
        The rollback is planned natively: the plan must start from our
        candidate, and its ETag identifies the only acceptable result. Once that
        exact document is active and ready, its identities are recorded for the
        same committed Reef snapshot. The pending journal stays until
        ``check_router_policy`` confirms it, immediately before reopening.

        Raises:
            RoutingDeploymentError: Another writer changed the document, or the
                rollback could not be confirmed; the journal is kept for recovery.
        """
        committed_etag = str(committed["router_etag"])
        self.event("policy_restoring", reason=outcome)
        if outcome == "rejected":
            return
        etag = management.config_etag()
        if etag == committed_etag:
            # The candidate was never stored; the committed document is unchanged.
            return
        if etag != candidate_etag:
            raise RoutingDeploymentError("an unknown writer changed the router configuration")
        document = Path(str(committed["router_document"])).read_text(encoding="utf-8")
        try:
            planned_from, restore_etag = management.plan(document)
        except RouterPolicyError as exc:
            raise RoutingDeploymentError("routing policy rollback could not be planned; staying closed") from exc
        if planned_from != candidate_etag:
            raise RoutingDeploymentError("an unknown writer changed the router configuration")
        state = self.policy_store.load()
        pending = state.get("pending")
        journal = {**pending} if isinstance(pending, dict) else {}
        # Journal the rollback target first: a lost reply or a crash is inspected against it.
        self.policy_store.write(
            {"committed": committed, "pending": {**journal, "phase": "restoring", "restore_etag": restore_etag}}
        )
        if self.replace_router_document(management, document, candidate_etag, restore_etag) != "active":
            raise RoutingDeploymentError("routing policy rollback could not be confirmed; staying closed")
        runtime_hash = management.runtime_hashes().get("active_runtime_hash")
        if not isinstance(runtime_hash, str) or not runtime_hash:
            raise RoutingDeploymentError("router runtime hash is unavailable after rollback")
        restored = {**committed, "router_etag": restore_etag, "runtime_hash": runtime_hash}
        self.policy_store.write({"committed": restored, "pending": {**journal, "phase": "restored"}})

    def recover(self, reason: str) -> None:
        ownership, control = self.require_clients()
        self.event("unready", reason=reason)
        if self.policy.restart_policy != "bounded":
            raise RoutingDeploymentError("routing dependency failed; fail_stop policy")
        self.drain(recovery=True)
        while not self.stop_requested:
            if self.restart_count >= self.policy.max_restarts:
                raise RoutingDeploymentError("routing restart budget exhausted")
            self.restart_count += 1
            self.event("recovering", reason=reason)
            if self.event_stream is not None:
                self.event_stream.close()
                self.event_stream = None
            self.runtime.remove(ownership, self.policy.container_stop_timeout_seconds)
            self.container_ids.clear()
            backoff = self.policy.restart_backoff_seconds[
                min(self.restart_count - 1, len(self.policy.restart_backoff_seconds) - 1)
            ]
            deadline = time.monotonic() + backoff
            while time.monotonic() < deadline and not self.stop_requested:
                time.sleep(0.05)
            if self.stop_requested:
                return
            try:
                try:
                    self.check_status(control.request())
                except RoutingControlError as exc:
                    raise RoutingDeploymentError("REEF lifecycle unavailable during recovery") from exc
                self.ready_deadline = time.monotonic() + self.policy.ready_timeout_seconds
                self.probe_failures = 0
                self.event_stream = self.runtime.events(ownership)
                self.runtime.start(self.directory / "compose.yaml", ownership, self.policy.ready_timeout_seconds)
                self.wait_ready(initial=False)
                return
            except InterruptedError:
                if not self.stop_requested:
                    raise
                return
            except ContainerCommandError as exc:
                # Unknown mutations must not race a replacement container.
                if exc.outcome_unknown:
                    raise RoutingDeploymentError(
                        "routing mutation outcome unknown; manual reconciliation required"
                    ) from exc
                self.event("recovery_failed")
            except RoutingReadinessTimeout:
                self.event("recovery_failed")
        self.event("stopping")

    def shutdown_failure(self, state: str) -> None:
        self.failed = True
        try:
            self.event(state)
        except Exception:
            # Broken diagnostics must not skip container or native teardown.
            self.state = state

    def release_control_directory(self) -> None:
        """Remove the private control directory once the child has exited, and only its socket inside.

        The child unlinks its own socket when it stops; one left by a crashed
        child is removed only if it is still a socket in this private directory.
        """
        directory = self.control_directory
        if directory is None:
            return
        socket_path = directory / "control.sock"
        try:
            metadata = os.lstat(socket_path)
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if not stat.S_ISSOCK(metadata.st_mode):
                raise RoutingDeploymentError("the control directory holds an unexpected file; left in place")
            socket_path.unlink()
        directory.rmdir()
        self.control_directory = None

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_requested = True
        try:
            try:
                self.join_start()
            except Exception:
                self.shutdown_failure("startup_join_failed")
            if self.ownership is None:
                return
            control = self.control
            if control is not None:
                try:
                    self.drain(recovery=False)
                except Exception:
                    self.shutdown_failure("forced_shutdown")
            if self.event_stream is not None:
                try:
                    self.event_stream.close()
                except Exception:
                    self.shutdown_failure("observer_cleanup_failed")
                self.event_stream = None
            try:
                self.runtime.remove(self.ownership, self.policy.container_stop_timeout_seconds)
                self.cleanup_complete = True
            except Exception:
                self.shutdown_failure("cleanup_incomplete")
            if control is not None:
                try:
                    control.request("stop", cancel_active=True)
                except Exception:
                    self.shutdown_failure("terminal_stop_failed")
            try:
                self.event("stopped", cleanup_complete=self.cleanup_complete)
            except Exception:
                self.shutdown_failure("shutdown_diagnostics_failed")
        finally:
            self.registry.release()
