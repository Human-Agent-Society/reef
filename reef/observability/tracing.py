"""The ``observability.tracing`` configuration for record tracing.

The storage wrappers in :mod:`reef.storage.observer` report every first-time
record insert and every landed commit; :mod:`reef.observability.open_telemetry`
turns those into spans for whatever tracing backend this section points at.
Validating the section needs no tracing SDK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Validated ``observability.tracing`` configuration.

    ``endpoint`` is the full OTLP/HTTP traces URL, for example
    ``http://localhost:4318/v1/traces``. ``authorization`` is the backend
    credential, sent as the ``Authorization`` header: it is the one secret in
    this section, so it is kept out of the dataclass repr and masked in the
    startup report like ``upstream_api_key``. It comes from the YAML value
    (``${LANGFUSE_AUTH}`` interpolates the environment) or, when omitted, from
    ``REEF_TRACING_AUTHORIZATION``; ``headers`` holds non-secret headers only.
    When neither the endpoint nor any header is configured the exporter reads
    the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_HEADERS``
    environment variables. Spans carry the exchange itself (prompt,
    completion, feedback and instruction text) beside identifiers, model
    names, token usage and scores, so choose a backend trusted with the
    scenario's traffic.
    """

    enabled: bool = False
    endpoint: str | None = None
    authorization: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] | None = None
    service_name: str = "reef"

    #: Environment fallback for ``authorization``, read by :func:`reef.observability.build_record_observer`.
    AUTHORIZATION_ENVIRONMENT_VARIABLE = "REEF_TRACING_AUTHORIZATION"

    @classmethod
    def from_mapping(cls, value: object, *, environ: Mapping[str, str] | None = None) -> TracingConfig:
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("observability.tracing must be a mapping")
        allowed = {"enabled", "endpoint", "authorization", "headers", "service_name"}
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(f"unknown observability.tracing settings: {', '.join(unknown)}")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("observability.tracing.enabled must be a boolean")
        endpoint = value.get("endpoint")
        if endpoint is not None and (not isinstance(endpoint, str) or not endpoint.strip()):
            raise ValueError("observability.tracing.endpoint must be a non-empty string")
        service_name = value.get("service_name", "reef")
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError("observability.tracing.service_name must be a non-empty string")
        authorization = value.get("authorization")
        if authorization is not None and not isinstance(authorization, str):
            raise ValueError("observability.tracing.authorization must be a string")
        if authorization is None and environ is not None:
            authorization = environ.get(cls.AUTHORIZATION_ENVIRONMENT_VARIABLE)
        headers = value.get("headers")
        if headers is not None:
            if not isinstance(headers, Mapping) or not all(
                isinstance(name, str) and isinstance(header, str) for name, header in headers.items()
            ):
                raise ValueError("observability.tracing.headers must map header names to strings")
            if any(name.lower() == "authorization" for name in headers):
                raise ValueError(
                    "observability.tracing.headers must not carry the credential; set observability.tracing.authorization"
                )
            headers = dict(headers)
        return cls(
            enabled=enabled,
            endpoint=None if endpoint is None else endpoint.strip(),
            authorization=(authorization or "").strip() or None,
            headers=headers,
            service_name=service_name.strip(),
        )

    def request_headers(self) -> dict[str, str] | None:
        """The OTLP request headers, or ``None`` so the SDK reads ``OTEL_EXPORTER_OTLP_HEADERS``."""
        if self.authorization is None and self.headers is None:
            return None
        headers = dict(self.headers or {})
        if self.authorization is not None:
            headers["Authorization"] = self.authorization
        return headers


__all__ = ["TracingConfig"]
