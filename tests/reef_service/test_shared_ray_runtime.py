"""Shared Ray ownership is separate from actor ownership and scheduling options."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from reef.runtime.executor import ray_runtime


class FakeRay:
    def __init__(self, initialized=False):
        self.initialized = initialized
        self.init_calls = []
        self.shutdown_calls = 0
        self.fail_init = False

    def is_initialized(self):
        return self.initialized

    def init(self, **options):
        self.init_calls.append(options)
        if self.fail_init:
            raise ConnectionError("external cluster unavailable")
        self.initialized = True

    def shutdown(self):
        self.shutdown_calls += 1
        self.initialized = False

    def get_runtime_context(self):
        return SimpleNamespace(gcs_address="10.0.0.1:12345")


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRay()
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(ray_runtime, "_require_ray", lambda: fake)
    return ray_runtime._RayRuntime(), fake


def test_multiple_owners_start_once_and_last_owner_stops(runtime):
    manager, ray = runtime
    with ThreadPoolExecutor(max_workers=4) as pool:
        leases = list(pool.map(lambda _: manager.acquire(), range(4)))
    assert ray.init_calls == [{"address": "local"}]
    assert {lease.address for lease in leases} == {"10.0.0.1:12345"}
    for lease in leases[:-1]:
        lease.close()
        lease.close()
    assert ray.shutdown_calls == 0
    leases[-1].close()
    assert ray.shutdown_calls == 1
    manager.acquire().close()
    assert len(ray.init_calls) == 2


@pytest.mark.parametrize("address", ["10.0.0.1:12345", "auto"])
def test_external_address_connects_without_starting_local(runtime, address):
    manager, ray = runtime
    manager.acquire(address).close()
    assert ray.init_calls == [{"address": address}]


def test_environment_address_takes_precedence(runtime, monkeypatch):
    manager, ray = runtime
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.2:12345")
    manager.acquire("10.0.0.1:12345").close()
    assert ray.init_calls == [{"address": "10.0.0.2:12345"}]


def test_initialized_runtime_is_borrowed(runtime):
    manager, ray = runtime
    ray.initialized = True
    manager.acquire().close()
    assert ray.init_calls == []
    assert ray.shutdown_calls == 0


def test_failed_connection_does_not_fall_back_or_leak_ownership(runtime):
    manager, ray = runtime
    ray.fail_init = True
    with pytest.raises(ConnectionError):
        manager.acquire("unavailable:12345")
    assert ray.init_calls == [{"address": "unavailable:12345"}]
    assert manager._users == 0
    ray.fail_init = False
    manager.acquire().close()
    assert ray.init_calls[-1] == {"address": "local"}


def test_disconnected_live_runtime_is_not_restarted(runtime):
    manager, ray = runtime
    lease = manager.acquire()
    ray.initialized = False
    with pytest.raises(RuntimeError, match="disconnected"):
        manager.acquire()
    assert len(ray.init_calls) == 1
    lease.close()
