"""The private HTTP boundary between the routing controller and its REEF child.

The child serves ``/reef/lifecycle`` only on a Unix socket in a private
directory; the public TCP listener has no lifecycle route.
"""

from __future__ import annotations

import http.client
import json
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

#: AF_UNIX paths are limited to 108 bytes on Linux, including the terminating NUL; keep a margin.
MAX_SOCKET_PATH_BYTES = 103


class RoutingControlError(RuntimeError):
    """A bounded control request failed without exposing credentials or bodies.

    Values: ``retryable`` is a bool; ``status_code`` is None when no HTTP reply status exists, or the
    child's HTTP status from 100 to 599. ``retryable`` tells the controller whether waiting and checking
    again can succeed; ``ReefLifecycleClient.request`` itself retries only transport timeouts.
    """

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None) -> None:
        if type(retryable) is not bool:
            raise ValueError("routing control retryable must be a boolean")
        if status_code is not None and (type(status_code) is not int or not 100 <= status_code <= 599):
            raise ValueError("routing control status_code must be None or an HTTP status")
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class RoutingControlTimeout(RoutingControlError):
    """No timely reply: every attempt timed out, or the reply came after the caller's deadline.

    It does not show that the child did nothing: a command may have been applied before its reply was lost.
    """


def json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise RoutingControlError("control response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RoutingControlError("control response must be an object")
    return value


class UnixHTTPConnection(http.client.HTTPConnection):
    """One HTTP exchange over a Unix socket; each blocking step gets only the time left before ``deadline``.

    ``deadline`` is a finite value of the ``time.monotonic`` clock and may already have passed, which makes
    the next step raise ``TimeoutError``. A socket timeout bounds each blocking socket call, not the whole
    exchange. ``ReefLifecycleClient.request`` makes one connection per attempt and closes it, together with
    any response that ``getresponse`` handed the socket to.
    """

    def __init__(self, socket_path: Path, deadline: float) -> None:
        if not math.isfinite(deadline):
            raise ValueError("a Unix HTTP connection deadline must be finite")
        super().__init__("localhost")
        self.socket_path = socket_path
        self.deadline = deadline
        #: Kept after ``getresponse``, which can hand the connection to the response it returns.
        self.connected_socket: socket.socket | None = None

    def remaining_seconds(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("REEF lifecycle attempt deadline passed")
        return remaining

    def connect(self) -> None:
        timeout_seconds = self.remaining_seconds()
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout_seconds)
        try:
            connection.connect(str(self.socket_path))
        except OSError:
            connection.close()
            raise
        self.sock = self.connected_socket = connection

    def limit_next_step(self) -> None:
        """Give the next blocking step only the time left in this attempt."""
        if self.connected_socket is not None:
            self.connected_socket.settimeout(self.remaining_seconds())


#: Attempts for one control request whose reply does not arrive in time. Every attempt sends the same bytes.
CONTROL_ATTEMPTS = 3


@dataclass
class ReefLifecycleClient:
    """Owner-generation control requests to the REEF child over its private Unix socket.

    Values: ``deployment_id``, ``generation`` and ``token`` are non-empty strings; ``revision`` is an int ≥ 0,
    the last allocated command revision. Both timeouts are finite positive numbers of seconds.
    ``socket_path`` is always dialed as a Unix socket, whatever it looks like; it need not exist yet.
    Operations: ``request(None)`` reads the status and leaves ``revision`` unchanged; each command
    increments it once, and its retries resend the same bytes. ``timeout_seconds`` bounds one attempt of an
    ordinary request. An ``open`` checks every provider and previews every Scenario first, so its first
    attempt gets ``open_timeout_seconds``. One controller owns a client, so the child sees one sequence of
    revisions per generation; the client keeps no connection between requests.
    """

    socket_path: Path
    deployment_id: str
    generation: str
    token: str = field(repr=False)
    revision: int = 0
    timeout_seconds: float = 2.0
    open_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if any(not isinstance(text, str) or not text for text in (self.deployment_id, self.generation, self.token)):
            raise ValueError("REEF lifecycle control needs a deployment ID, a generation and a token")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("REEF lifecycle control revision must be a nonnegative integer")
        for seconds in (self.timeout_seconds, self.open_timeout_seconds):
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError("REEF lifecycle control timeouts must be finite positive seconds")

    def request(
        self,
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> dict[str, object]:
        """Send one command, or read the status when ``action`` is None, and return the child's status.

        Only a transport timeout is tried again, on a new connection, at most ``CONTROL_ATTEMPTS`` times in
        all. Every attempt sends the same revision and body: the child applies a revision once and answers
        a repeated one with its status. A later attempt waits at most ``timeout_seconds``.

        An ``open`` carries the end of its first attempt. The child does not open after that time, so a
        reply the owner stopped waiting for cannot open admission later; a retry can only read an open that
        the child already applied.

        Args:
            deadline: The caller's phase end on the ``time.monotonic`` clock. Each blocking step of every
                attempt gets only the time left before it, no attempt starts after it, and a reply that
                completes after it is rejected.

        Raises:
            RoutingControlTimeout: No reply arrived in time.
            RoutingControlError: The request failed, or the child refused it.
        """
        phase_end = math.inf if deadline is None else deadline
        first_attempt_seconds = self.open_timeout_seconds if action == "open" else self.timeout_seconds
        attempt_end = min(time.monotonic() + first_attempt_seconds, phase_end)
        body = None
        if action is not None:
            self.revision += 1
            command: dict[str, object] = {
                "deployment_id": self.deployment_id,
                "generation": self.generation,
                "control_revision": self.revision,
                "action": action,
                "cancel_active": cancel_active,
                **(publication or {}),
            }
            if action == "open":
                command["open_deadline_monotonic"] = attempt_end
            body = json.dumps(command).encode()
        attempt = 1
        while True:
            connection = UnixHTTPConnection(self.socket_path, attempt_end)
            # Closed with the connection: after an HTTP/1.0 or Connection: close reply, it holds the socket alone.
            response: http.client.HTTPResponse | None = None
            try:
                connection.connect()
                connection.limit_next_step()
                connection.request(
                    "GET" if action is None else "POST",
                    "/reef/lifecycle",
                    body=body,
                    headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
                )
                connection.limit_next_step()
                response = connection.getresponse()
                if response.status != 200:
                    raise RoutingControlError(
                        f"REEF lifecycle returned HTTP {response.status}",
                        retryable=response.status >= 500 or response.status == 409,
                        status_code=response.status,
                    )
                connection.limit_next_step()
                reply = response.read(1024 * 1024)
            except TimeoutError as exc:
                if attempt == CONTROL_ATTEMPTS or time.monotonic() >= phase_end:
                    raise RoutingControlTimeout("REEF lifecycle request timed out") from exc
            except (OSError, ValueError, http.client.HTTPException) as exc:
                raise RoutingControlError("REEF lifecycle request failed") from exc
            else:
                if time.monotonic() > phase_end:
                    raise RoutingControlTimeout("REEF lifecycle replied after the caller's deadline")
                return json_object(reply)
            finally:
                if response is not None:
                    response.close()
                connection.close()
            attempt += 1
            attempt_end = min(time.monotonic() + self.timeout_seconds, phase_end)
