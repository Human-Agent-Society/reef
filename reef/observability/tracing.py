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
class TracingSettings:
    """The settings of the ``observability.tracing`` section.

    Values: ``enabled`` is a bool. ``endpoint`` is None or a non-empty string without surrounding space.
    ``authorization`` is None or a non-empty string without surrounding space. ``headers`` is None or a mapping
    from header names to strings with no ``Authorization`` name. ``service_name`` is a non-empty string without
    surrounding space. The constructor strips the strings, reads a blank ``authorization`` as None, copies
    ``headers``, and refuses other values with ``ValueError``.
    Purpose: ``from_mapping`` parses it from the section; ``build_record_observer`` and the OpenTelemetry
    observer read it.

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
    completion and feedback text) beside identifiers, model
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

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("observability.tracing.enabled must be a boolean")
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str) or not self.endpoint.strip():
                raise ValueError("observability.tracing.endpoint must be a non-empty string")
            object.__setattr__(self, "endpoint", self.endpoint.strip())
        if not isinstance(self.service_name, str) or not self.service_name.strip():
            raise ValueError("observability.tracing.service_name must be a non-empty string")
        object.__setattr__(self, "service_name", self.service_name.strip())
        if self.authorization is not None and not isinstance(self.authorization, str):
            raise ValueError("observability.tracing.authorization must be a string")
        object.__setattr__(self, "authorization", (self.authorization or "").strip() or None)
        if self.headers is not None:
            if not isinstance(self.headers, Mapping) or not all(
                isinstance(name, str) and isinstance(header, str) for name, header in self.headers.items()
            ):
                raise ValueError("observability.tracing.headers must map header names to strings")
            if any(name.lower() == "authorization" for name in self.headers):
                raise ValueError(
                    "observability.tracing.headers must not carry the credential; set observability.tracing.authorization"
                )
            object.__setattr__(self, "headers", dict(self.headers))

    @classmethod
    def from_mapping(cls, value: object, *, environ: Mapping[str, str] | None = None) -> TracingSettings:
        """Read the ``observability.tracing`` section; ``None`` is an empty section.

        A section without ``authorization`` takes it from ``REEF_TRACING_AUTHORIZATION`` in ``environ``.
        ``ValueError`` when ``value`` is no mapping, names an unknown setting, or holds a value that the
        constructor refuses.
        """
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("observability.tracing must be a mapping")
        allowed = {"enabled", "endpoint", "authorization", "headers", "service_name"}
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(f"unknown observability.tracing settings: {', '.join(unknown)}")
        authorization = value.get("authorization")
        if authorization is None and environ is not None:
            authorization = environ.get(cls.AUTHORIZATION_ENVIRONMENT_VARIABLE)
        return cls(
            enabled=value.get("enabled", False),
            endpoint=value.get("endpoint"),
            authorization=authorization,
            headers=value.get("headers"),
            service_name=value.get("service_name", "reef"),
        )

    def request_headers(self) -> dict[str, str] | None:
        """The OTLP request headers, or ``None`` so the SDK reads ``OTEL_EXPORTER_OTLP_HEADERS``."""
        if self.authorization is None and self.headers is None:
            return None
        headers = dict(self.headers or {})
        if self.authorization is not None:
            headers["Authorization"] = self.authorization
        return headers


__all__ = ["TracingSettings"]
