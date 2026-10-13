"""Routing services ``reef serve`` owns: one more layer beside inference and HTTP.

A deployment with ``reef.routing_manifest`` set gets a ``semantic-router`` and
an ``envoy`` service before the HTTP service, which depends on Envoy, so Reef
accepts requests only while the routing stack is ready — and both processes
live and die with ``reef serve`` under the same supervision as every other
child. The upstream Semantic Router version is pinned where the maintainer
asked for it to live: in the deployment's own ``command`` (and hooks), not in
Reef's source tree; ``third_party/`` stays untouched. Process commands are
config-interpolated, so they can reference ``${reef.routing_manifest}`` and
``${endpoints.*}``.

Without ``reef.routing_manifest`` nothing here runs and nothing changes:
routing stays disabled and the deployment is byte-identical to a non-routing
one. Custom unversioned process stacks wire their own services.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.service.deploy.config_utils import DeployConfigError
from reef.service.routing import ENVOY_SERVICE, SEMANTIC_ROUTER_SERVICE, load_routing_manifest

_ROUTING_SECTION = "routing_deployment"
_MANIFEST_FIELD = "routing_manifest"


@dataclass(frozen=True)
class RoutingProcessSettings:
    """One routing process's launch definition, validated against the service schema."""

    name: str
    command: str | tuple[str, ...]
    ready: str | tuple[str, ...] | None
    ready_timeout: int | None
    endpoint: str | None

    def as_service(self, depends_on: Sequence[str]) -> dict[str, Any]:
        service: dict[str, Any] = {
            "name": self.name,
            # The service schema validates a list; keep strings as given.
            "command": list(self.command) if isinstance(self.command, tuple) else self.command,
        }
        if depends_on:
            service["depends_on"] = list(depends_on)
        if self.endpoint:
            service["endpoint"] = self.endpoint
        # Default readiness is "the process is alive"; a pinned upstream that
        # exposes a health route declares it here instead of Reef guessing.
        if self.ready is None:
            service["ready_timeout"] = self.ready_timeout or 60
        else:
            service["ready"] = list(self.ready) if isinstance(self.ready, tuple) else self.ready
            service["ready_timeout"] = self.ready_timeout or 3600
        return service


def _process_command(value: Any, name: str, field: str) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and value
        and all(isinstance(item, str) and item for item in value)
    ):
        return tuple(value)
    raise ValueError(f"{name}.{field} must be a non-empty string or a list of strings")


def _parse_process(section: Mapping[str, Any], name: str) -> RoutingProcessSettings:
    command = _process_command(section.get("command"), name, "command")
    if command is None:
        raise ValueError(
            f"{name}.command is required: reef serve only hosts a pinned upstream build, "
            "so the deployment names the exact command (and hooks) to run"
        )
    ready = _process_command(section.get("ready"), name, "ready")
    ready_timeout = section.get("ready_timeout")
    if ready_timeout is not None:
        if isinstance(ready_timeout, bool) or not isinstance(ready_timeout, (int, float)) or ready_timeout <= 0:
            raise ValueError(f"{name}.ready_timeout must be a positive number of seconds")
        ready_timeout = int(ready_timeout)
    endpoint = section.get("endpoint")
    if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
        raise ValueError(f"{name}.endpoint must be a string")
    return RoutingProcessSettings(
        name=name,
        command=command,
        ready=ready,
        ready_timeout=ready_timeout,
        endpoint=endpoint.strip() if isinstance(endpoint, str) else None,
    )


def routing_deployment_settings(section: Any) -> tuple[RoutingProcessSettings, RoutingProcessSettings]:
    """Parse the ``reef.routing_deployment`` mapping into the SR and Envoy definitions."""
    section = _require_mapping(section, _ROUTING_SECTION)
    unknown = set(section) - {SEMANTIC_ROUTER_SERVICE, ENVOY_SERVICE}
    if unknown:
        raise ValueError(f"{_ROUTING_SECTION} accepts only '{SEMANTIC_ROUTER_SERVICE}' and '{ENVOY_SERVICE}'")
    router = _parse_process(
        _require_mapping(section.get(SEMANTIC_ROUTER_SERVICE), f"{_ROUTING_SECTION}.{SEMANTIC_ROUTER_SERVICE}"),
        SEMANTIC_ROUTER_SERVICE,
    )
    envoy = _parse_process(
        _require_mapping(section.get(ENVOY_SERVICE), f"{_ROUTING_SECTION}.{ENVOY_SERVICE}"), ENVOY_SERVICE
    )
    return router, envoy


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be an object")
    return value


def routing_manifest_path(config: Mapping[str, Any]) -> str | None:
    """The deployment's routing manifest path, or None while routing is disabled."""
    reef = config.get("reef")
    manifest = reef.get(_MANIFEST_FIELD) if isinstance(reef, Mapping) else None
    if manifest is None:
        return None
    if not isinstance(manifest, str) or not manifest.strip():
        raise DeployConfigError(f"reef.{_MANIFEST_FIELD} must be a non-empty path when present")
    return os.path.expanduser(manifest.strip())


def attach_routing_services(config: dict[str, Any]) -> None:
    """Add the routing stack and make the HTTP service depend on it.

    Loads and validates the routing manifest before any process starts, so a
    broken manifest fails deployment resolution instead of a child. No-op for
    deployments without ``reef.routing_manifest``.
    """
    manifest_path = routing_manifest_path(config)
    if manifest_path is None:
        return
    if not config.get("services"):
        raise DeployConfigError("routing needs the Reef HTTP service in the deployment")
    try:
        manifest = load_routing_manifest(manifest_path)
        router, envoy = routing_deployment_settings(config.get("reef", {}).get(_ROUTING_SECTION))
    except ValueError as exc:
        raise DeployConfigError(f"routing: {exc}") from exc
    # One line in the startup log names the routing configuration this stack
    # runs under: the digest, not a model version.
    print(
        f"[reef] routing: manifest {manifest_path} (sha256 {manifest.digest}), {len(manifest.scenarios)} scenario(s)",
        file=sys.stderr,
    )
    # The readiness default is a live process; operators pinning a build with a
    # health route declare `ready` themselves instead of Reef guessing one.
    router_service = router.as_service(depends_on=())
    envoy_service = envoy.as_service(depends_on=(SEMANTIC_ROUTER_SERVICE,))
    services = config["services"]
    http = next((entry for entry in services if entry.get("name") == "reef"), None)
    if http is None:
        raise DeployConfigError("routing needs the Reef HTTP service in the deployment")
    http["depends_on"] = [*http.get("depends_on", []), ENVOY_SERVICE]
    config["services"] = [router_service, envoy_service, *services]
