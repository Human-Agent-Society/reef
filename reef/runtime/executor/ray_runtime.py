"""Process-wide, reference-counted ownership of a Ray connection/runtime."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any


def _require_ray() -> Any:
    import ray

    return ray


def _validate_local_ray_temp_path() -> None:
    """Fail before Ray constructs a local Unix socket whose path is too long."""
    if sys.platform == "win32":
        return

    temp_root = os.environ.get("RAY_TMPDIR")
    source = "RAY_TMPDIR"
    if temp_root is None and sys.platform.startswith("linux"):
        temp_root = os.environ.get("TMPDIR")
        source = "TMPDIR"
    if temp_root is None:
        temp_root = "/tmp"
        source = "Ray's default temp directory"

    # Ray uses a fixed-width timestamp plus the current PID in each session
    # name. The plasma-store socket is the longer of its two local sockets.
    session_name = f"session_0000-00-00_00-00-00_000000_{os.getpid()}"
    socket_path = os.path.join(temp_root, "ray", session_name, "sockets", "plasma_store")
    max_bytes = 103 if sys.platform.startswith("darwin") else 107
    path_bytes = len(os.fsencode(socket_path))
    if path_bytes > max_bytes:
        raise RuntimeError(
            f"{source} is too long for Ray's local Unix socket "
            f"({path_bytes} bytes; limit {max_bytes}). Set RAY_TMPDIR to a shorter directory, such as /tmp."
        )


class RayRuntimeLease:
    """Keep the shared runtime alive until this owner has stopped its work."""

    def __init__(self, runtime: _RayRuntime, address: str) -> None:
        self.address = address
        self._runtime = runtime
        self._closed = False

    def close(self) -> None:
        self._runtime.release(self)


class _RayRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._users = 0
        self._connected_here = False
        self._ray: Any = None
        self._address = ""

    def acquire(self, address: str | None = None) -> RayRuntimeLease:
        with self._lock:
            ray = _require_ray()
            if self._users:
                if not self._ray.is_initialized():
                    raise RuntimeError("shared Ray runtime was disconnected while executors are still active")
            else:
                self._connected_here = not ray.is_initialized()
                if self._connected_here:
                    # Explicit 'local' avoids silently joining an unrelated
                    # cluster discovered on a shared host. External addresses
                    # (including 'auto') must connect, never fall back locally.
                    target = os.environ.get("RAY_ADDRESS") or address or "local"
                    try:
                        if target == "local":
                            _validate_local_ray_temp_path()
                        ray.init(address=target)
                    except BaseException:
                        ray.shutdown()
                        self._connected_here = False
                        raise
                self._ray = ray
                self._address = ray.get_runtime_context().gcs_address
            self._users += 1
            return RayRuntimeLease(self, self._address)

    def release(self, lease: RayRuntimeLease) -> None:
        with self._lock:
            if lease._closed:
                return
            lease._closed = True
            self._users -= 1
            if self._users == 0:
                try:
                    if self._connected_here:
                        # For an external cluster this disconnects our driver;
                        # only a local cluster started by ray.init is stopped.
                        self._ray.shutdown()
                finally:
                    self._ray = None
                    self._address = ""
                    self._connected_here = False


_runtime = _RayRuntime()


def acquire_ray_runtime(address: str | None = None) -> RayRuntimeLease:
    return _runtime.acquire(address)
