"""Process-wide, reference-counted ownership of a Ray connection/runtime."""

from __future__ import annotations

import os
import threading
from typing import Any


def _require_ray() -> Any:
    import ray

    return ray


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
