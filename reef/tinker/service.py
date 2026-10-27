"""SDK sessions on a Tinker-compatible service: the SDK module, and opening and closing a ``ServiceClient``.

This module and the two SDK classes that use it (``reef.tinker.sampling.TinkerSDKSampler`` and
``reef.train.tinker_backend.client.TinkerSDKClient``) are the only code that imports Tinker's SDK.
"""

from __future__ import annotations

import importlib
from typing import Any


def tinker_sdk() -> Any:
    """The installed ``tinker`` SDK module; ``RuntimeError`` naming the extra to install when it is missing."""
    try:
        return importlib.import_module("tinker")
    except ImportError as exc:
        raise RuntimeError("install Tinker support with uv pip install 'reef-infra[tinker]'") from exc


def open_service(sdk: Any, *, api_key: str, project_id: str | None, base_url: str | None) -> Any:
    """A ``ServiceClient`` of the service at ``base_url`` (``None``: hosted Tinker).

    The SDK takes no client-wide timeout: each caller bounds its own call with ``.result(timeout=...)``.
    """
    return sdk.ServiceClient(api_key=api_key, project_id=project_id, base_url=base_url)


def close_service(sdk: Any, service: Any, status: str, *, timeout_s: float | None) -> None:
    """Close the SDK ``ServiceClient`` ``service``, ending its session with ``status`` (``success`` or ``errored``).

    A Tinker-compatible service without the session-finish call, such as Spindle, answers 404 to it and ends
    the session when its heartbeats stop; closing still releases the client's connections.
    """
    try:
        service.close(status).result(timeout=timeout_s)
    except sdk.NotFoundError:
        return
