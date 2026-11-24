"""Validated settings of the Semantic Router of a routed deployment."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.config import config_arguments, config_option, parse_config_values


@dataclass(frozen=True)
class ShadowLimits:
    """How much shadow work one routed service runs.

    Values: ``max_concurrency`` is an int from 1 to 4096, the most shadow tasks that run at once.
    ``timeout_seconds`` is a finite positive number: how long one shadow call waits for its provider's answer; it
    does not bound the record write that follows. ``max_response_bytes`` is a positive int: the most body bytes,
    success or error, that a shadow call collects before it refuses the body unparsed. The defaults are
    conservative settings, not measured model or GPU capacities.
    Purpose: ``create_app`` gets it from ``reef.routing_deployment``; ``ShadowTasks`` reads ``max_concurrency``
    and each shadow call reads the other two.
    """

    max_concurrency: int = 4
    timeout_seconds: float = 300.0
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4096:
            raise ValueError("shadow max_concurrency must be an integer from 1 to 4096")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("shadow timeout_seconds must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("shadow max_response_bytes must be a positive integer")


@dataclass(frozen=True)
class RoutingDeploymentSettings:
    """The router's lifecycle, identity, resources, preview and recovery settings, and Reef's shadow limits.

    Values: ``lifecycle`` is ``managed`` or ``external``. ``deployment_id`` is None or 1 to 48 lowercase letters,
    digits or hyphens, starting with a letter or digit. ``restart_policy`` is ``fail_stop`` or ``bounded``;
    ``max_restarts`` is an int ≥ 0, and ≥ 1
    when bounded. ``restart_backoff_seconds`` is a non-empty tuple of finite numbers ≥ 0, in order, with
    repeats allowed. The ready, finalization, container stop and image build timeouts and ``sr_cpus`` are finite
    and positive; ``drain_timeout_seconds`` is finite and ≥ 0. ``sr_management_port`` is an int from 1 to 65535,
    ``selection_timeout_seconds`` an int from 1 to 300, ``preview_max_concurrency`` an int from 1 to 4096,
    and ``sr_memory_bytes`` a positive int. ``shadow_max_concurrency``, ``shadow_timeout_seconds`` and
    ``shadow_max_response_bytes`` have the domains of ``ShadowLimits``.
    Purpose: ``routing_deployment_settings`` makes it from ``reef.routing_deployment``; the controller and
    the renderer read it, and each routed service reads its shadow limits (``shadow_limits``). Recovery attempt
    ``n`` (from 1) waits ``restart_backoff_seconds[min(n - 1, len - 1)]``, so the last delay repeats. These settings
    apply only when ``reef.routing_manifest`` is set; ``require_identity`` returns ``deployment_id`` or refuses
    a managed deployment without one. With ``managed``, ``reef serve`` runs the container and reads every
    setting. With ``external``, another system runs the router beside ``reef serve --service inference``;
    only the port, preview, finalization and per-replica shadow settings apply.
    ``image_build_timeout_seconds`` bounds the managed startup's image build and is excluded from the render.
    """

    lifecycle: str = config_option(
        "managed", help="Who runs the router: managed (reef serve) or external (beside --service inference)."
    )
    deployment_id: str | None = config_option(None, help="Stable local deployment identifier.")
    restart_policy: str = config_option("fail_stop", help="Failure policy: fail_stop or bounded.")
    max_restarts: int = config_option(2, help="Recovery attempts allowed during this explicit deployment.")
    restart_backoff_seconds: tuple[float, ...] = (1.0, 2.0)
    ready_timeout_seconds: float = config_option(120.0, help="Complete routing readiness deadline in seconds.")
    drain_timeout_seconds: float = config_option(30.0, help="Seconds allowed for accepted requests to drain.")
    finalization_timeout_seconds: float = config_option(5.0, help="Seconds allowed for pending record writes.")
    container_stop_timeout_seconds: float = config_option(10.0, help="Container stop command deadline in seconds.")
    image_build_timeout_seconds: float = config_option(
        3600.0, help="Seconds a managed startup may spend building the Semantic Router image it has not cached."
    )
    sr_management_port: int = config_option(
        50053, help="Host loopback port of the router's authenticated HTTP API (preview and configuration)."
    )
    selection_timeout_seconds: int = config_option(
        10, help="Whole seconds the router may spend selecting for one request (its preview worker deadline)."
    )
    preview_max_concurrency: int = config_option(
        32, help="Router preview workers; more concurrent selections are refused with 429. Changes need a restart."
    )
    sr_cpus: float = config_option(4.0, help="Semantic Router container CPU limit.")
    sr_memory_bytes: int = config_option(4 * 1024**3, help="Semantic Router container memory limit in bytes.")
    shadow_max_concurrency: int = config_option(
        ShadowLimits.max_concurrency, help="Shadow calls run at once; a request finding none free skips its shadow."
    )
    shadow_timeout_seconds: float = config_option(
        ShadowLimits.timeout_seconds, help="Seconds one shadow call waits for its provider's answer."
    )
    shadow_max_response_bytes: int = config_option(
        ShadowLimits.max_response_bytes, help="Longest provider body, in bytes, that a shadow call reads."
    )

    def __post_init__(self) -> None:
        if self.lifecycle not in {"managed", "external"}:
            raise ValueError("routing lifecycle must be managed or external")
        if self.deployment_id is not None and (
            not isinstance(self.deployment_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", self.deployment_id) is None
        ):
            raise ValueError("routing deployment_id must contain 1-48 lowercase letters, digits or hyphens")
        if self.restart_policy not in {"fail_stop", "bounded"}:
            raise ValueError("routing restart_policy must be fail_stop or bounded")
        if type(self.max_restarts) is not int or self.max_restarts < 0:
            raise ValueError("routing max_restarts must be a nonnegative integer")
        if self.restart_policy == "bounded" and self.max_restarts == 0:
            raise ValueError("bounded routing requires at least one recovery attempt")
        if type(self.restart_backoff_seconds) is not tuple:
            # A frozen value must not keep a list the caller can still change.
            raise ValueError("routing restart_backoff_seconds must be a tuple of numbers")
        if not self.restart_backoff_seconds or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in self.restart_backoff_seconds
        ):
            raise ValueError("routing restart_backoff_seconds must be a nonempty list of finite nonnegative numbers")
        for name, value in (
            ("ready_timeout_seconds", self.ready_timeout_seconds),
            ("finalization_timeout_seconds", self.finalization_timeout_seconds),
            ("container_stop_timeout_seconds", self.container_stop_timeout_seconds),
            ("image_build_timeout_seconds", self.image_build_timeout_seconds),
            ("sr_cpus", self.sr_cpus),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"routing {name} must be finite and positive")
        if (
            type(self.drain_timeout_seconds) not in (int, float)
            or not math.isfinite(self.drain_timeout_seconds)
            or self.drain_timeout_seconds < 0
        ):
            raise ValueError("routing drain_timeout_seconds must be finite and nonnegative")
        if type(self.sr_management_port) is not int or not 1 <= self.sr_management_port <= 65535:
            raise ValueError("routing sr_management_port must be an integer between 1 and 65535")
        if type(self.selection_timeout_seconds) is not int or not 1 <= self.selection_timeout_seconds <= 300:
            raise ValueError("routing selection_timeout_seconds must be a whole number of seconds from 1 to 300")
        if type(self.preview_max_concurrency) is not int or not 1 <= self.preview_max_concurrency <= 4096:
            raise ValueError("routing preview_max_concurrency must be an integer from 1 to 4096")
        if type(self.sr_memory_bytes) is not int or self.sr_memory_bytes <= 0:
            raise ValueError("routing sr_memory_bytes must be a positive integer")
        self.shadow_limits()

    def shadow_limits(self) -> ShadowLimits:
        """The shadow limits these settings name; ``ShadowLimits`` refuses values outside their domains."""
        return ShadowLimits(self.shadow_max_concurrency, self.shadow_timeout_seconds, self.shadow_max_response_bytes)

    def require_identity(self) -> str:
        """The deployment identifier, which an enabled routing deployment must name."""
        if self.deployment_id is None:
            raise ValueError("reef.routing_manifest requires reef.routing_deployment.deployment_id")
        return self.deployment_id


@dataclass(frozen=True)
class RoutedServiceSettings:
    """The settings of the REEF service that the controller of a routing deployment reads.

    Values: ``routing_manifest`` is None or the non-empty path of the source manifest. ``host`` is the HTTP bind
    address of the service, and ``port`` is its HTTP bind port. ``has_tokens`` is whether the service has at least
    one bearer token. ``inference_timeout_seconds`` is the timeout of one provider call, in seconds.
    Purpose: ``ServiceSettings.routed_service`` makes it from the settings that it checked;
    ``RoutingDeploymentController`` reads it to check a managed deployment and to bound the time of one ``open``.
    """

    routing_manifest: str | None
    host: str
    port: int
    has_tokens: bool
    inference_timeout_seconds: float


def routing_deployment_settings(section: Mapping[str, object] | None) -> RoutingDeploymentSettings:
    """Parse the public object and reject unknown or misspelled settings before any process starts."""
    if section is None:
        return RoutingDeploymentSettings()
    if not isinstance(section, Mapping):
        raise ValueError("reef.routing_deployment must be an object")
    folded = {str(key).replace("-", "_"): value for key, value in section.items()}
    if len(folded) != len(section):
        raise ValueError("reef.routing_deployment contains duplicate field spellings")
    backoff = folded.pop("restart_backoff_seconds", (1.0, 2.0))
    if not isinstance(backoff, (list, tuple)):
        raise ValueError("routing restart_backoff_seconds must be a list of numbers")
    if any(type(value) not in (int, float) for value in backoff):
        raise ValueError("routing restart_backoff_seconds must be a list of numbers")
    values = parse_config_values(
        config_arguments(RoutingDeploymentSettings, prefix=("reef", "routing_deployment")), folded
    )
    return RoutingDeploymentSettings(**values, restart_backoff_seconds=tuple(float(value) for value in backoff))
