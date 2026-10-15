"""Local tenant request binding and the scheduler's HTTP control transport."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from reef.core.errors import ReefError

TENANT_HEADER = "x-reef-tenant"
CONTROL_PATH = "/reef/scheduler/control"


class SchedulerUnavailable(ReefError):
    """The scheduler or a controlled service could not be reached."""


class SchedulerResponseError(ReefError):
    """A scheduler HTTP boundary rejected a request with a known status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def object_value(value: object) -> dict[str, object]:
    """Validate an object at a JSON transport boundary."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return dict(value)


def text_value(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def integer_value(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a positive integer within the signed 64-bit range")
    return value


def optional_text(payload: Mapping[str, object], name: str) -> str | None:
    if payload.get(name) is None:
        return None
    return text_value(payload, name)


def tenant_from_headers(headers: Mapping[str, str]) -> int:
    values = {key.lower(): value for key, value in headers.items()}
    tenant = values.get(TENANT_HEADER, "").strip()
    if not tenant or not tenant.isascii() or not tenant.isdecimal():
        raise ValueError(f"{TENANT_HEADER} must be a positive integer")
    return integer_value({"tenant_id": int(tenant)}, "tenant_id")


def validate_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("scheduler endpoints require an HTTP(S) URL without credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("scheduler endpoints cannot include a query or fragment")


@dataclass(frozen=True)
class ServiceSchedulingConfig:
    """Service identity and connection settings for scheduler registration."""

    scheduler_url: str
    service_id: int
    service_name: str
    recipe: str
    standard_execution_units: int  # this is the required scheduler unit for launching a scenario of this service
    control_token: str = field(repr=False)
    scheduler_token: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0  # Each type of service could have their own idle TTL
    service_url: str | None = None
    config_revision: str = ""
    registration_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        validate_endpoint(self.scheduler_url)
        if self.service_url is not None:
            validate_endpoint(self.service_url)
        if not isinstance(self.config_revision, str):
            raise ValueError("config_revision must be a string")
        integer_value({"service_id": self.service_id}, "service_id")
        for name in ("service_name", "recipe", "control_token"):
            text_value(
                {
                    "service_id": self.service_id,
                    "service_name": self.service_name,
                    "recipe": self.recipe,
                    "control_token": self.control_token,
                },
                name,
            )
        integer_value({"standard_execution_units": self.standard_execution_units}, "standard_execution_units")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("scheduler timeout must be positive")
        if not math.isfinite(self.registration_interval_seconds) or self.registration_interval_seconds <= 0:
            raise ValueError("scheduler registration interval must be positive")


class SchedulerHTTPClient:
    """Synchronous transport used only from service worker threads."""

    def __init__(self, endpoint: str, *, token: str | None = None, timeout_seconds: float = 30.0) -> None:
        validate_endpoint(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def request(
        self, method: str, path: str, *, payload: Mapping[str, object] | None = None, tenant_id: int | None = None
    ) -> dict[str, object]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if tenant_id is not None:
            headers[TENANT_HEADER] = str(tenant_id)
        body = None if payload is None else json.dumps(dict(payload)).encode()
        request = Request(f"{self.endpoint}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return object_value(json.load(response))
        except HTTPError as exc:
            # Do not copy remote bodies: a failed backend may include model credentials.
            raise SchedulerResponseError(exc.code, f"scheduler request rejected (HTTP {exc.code})") from exc
        except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
            raise SchedulerUnavailable("scheduler connection failed; lifecycle outcome is unknown") from exc


class ServiceSchedulerClient:
    """Submit names at the boundary; retain service IDs inside scheduling state."""

    def __init__(self, settings: ServiceSchedulingConfig) -> None:
        self.settings = settings
        self.http = SchedulerHTTPClient(
            settings.scheduler_url, token=settings.scheduler_token, timeout_seconds=settings.timeout_seconds
        )

    def register(self) -> dict[str, object]:
        """Register this service, refreshing its in-memory control credential."""
        if self.settings.service_url is None:
            raise ValueError("scheduler registration requires a service_url")
        return self.http.request(
            "POST",
            "/reef/services/register",
            payload={
                "service_id": self.settings.service_id,
                "service_name": self.settings.service_name,
                "service_url": self.settings.service_url,
                "recipe": self.settings.recipe,
                "standard_execution_units": self.settings.standard_execution_units,
                "config_revision": self.settings.config_revision,
                "control_token": self.settings.control_token,
            },
        )

    def submit(
        self,
        tenant_id: int,
        scenario_name: str,
        *,
        release_id: str | None = None,
        model: object = None,
        has_model: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "tenant_id": tenant_id,
            "service_name": self.settings.service_name,
            "recipe": self.settings.recipe,
            "scenario_name": scenario_name,
            "release_id": release_id,
        }
        if has_model:
            payload["model"] = model
        return self.http.request("POST", "/reef/scenarios", payload=payload, tenant_id=tenant_id)

    def list_scenario_admissions(self, tenant_id: int) -> dict[str, object]:
        query = urlencode({"service_id": self.settings.service_id})
        return self.http.request("GET", f"/reef/scenarios?{query}", tenant_id=tenant_id)

    def wake(self, scenario_id: int, tenant_id: int, generation: int, record_id: str) -> dict[str, object]:
        return self.http.request(
            "POST",
            f"/reef/scenarios/{scenario_id}/wake",
            tenant_id=tenant_id,
            payload={"service_id": self.settings.service_id, "generation": generation, "agent_record_id": record_id},
        )

    def delete(self, tenant_id: int, scenario_name: str) -> dict[str, object]:
        return self.http.request(
            "POST",
            "/reef/scenarios/cancel",
            tenant_id=tenant_id,
            payload={"service_name": self.settings.service_name, "scenario_name": scenario_name},
        )
