"""The private HTTP boundary between the routing controller and its REEF child.

The child serves ``/reef/lifecycle`` only on a Unix socket in a private
directory; the public TCP listener has no lifecycle route.
"""

from __future__ import annotations

import http.client
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path

#: AF_UNIX paths are limited to 108 bytes on Linux, including the terminating NUL; keep a margin.
MAX_SOCKET_PATH_BYTES = 103


class RoutingControlError(RuntimeError):
    """A bounded control request failed without exposing credentials or bodies."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise RoutingControlError("control response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RoutingControlError("control response must be an object")
    return value


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout_seconds: float) -> None:
        super().__init__("localhost", timeout=timeout_seconds)
        self.socket_path = socket_path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        try:
            connection.connect(str(self.socket_path))
        except OSError:
            connection.close()
            raise
        self.sock = connection


@dataclass
class ReefLifecycleClient:
    """Owner-generation control requests to the REEF child over its private Unix socket.

    ``timeout_seconds`` bounds ordinary requests; an ``open`` previews every Scenario first and gets
    ``open_timeout_seconds``.
    """

    socket_path: Path
    deployment_id: str
    generation: str
    token: str = field(repr=False)
    revision: int = 0
    timeout_seconds: float = 2.0
    open_timeout_seconds: float = 30.0

    def request(
        self, action: str | None = None, *, cancel_active: bool = False, publication: dict[str, str] | None = None
    ) -> dict[str, object]:
        body = None
        if action is not None:
            self.revision += 1
            body = json.dumps(
                {
                    "deployment_id": self.deployment_id,
                    "generation": self.generation,
                    "control_revision": self.revision,
                    "action": action,
                    "cancel_active": cancel_active,
                    **(publication or {}),
                }
            ).encode()
        timeout = self.open_timeout_seconds if action == "open" else self.timeout_seconds
        connection = UnixHTTPConnection(self.socket_path, timeout)
        try:
            connection.request(
                "GET" if action is None else "POST",
                "/reef/lifecycle",
                body=body,
                headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RoutingControlError(
                    f"REEF lifecycle returned HTTP {response.status}",
                    retryable=response.status >= 500 or response.status == 409,
                    status_code=response.status,
                )
            return json_object(response.read(1024 * 1024))
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise RoutingControlError("REEF lifecycle request failed") from exc
        finally:
            connection.close()
