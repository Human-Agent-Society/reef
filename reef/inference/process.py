"""Process, port and Ray actor helpers shared by the native inference integrations."""

from __future__ import annotations

import errno
import logging
import os
import random
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, suppress
from typing import Any

import requests

logger = logging.getLogger(__name__)

#: Startup attempts for one engine while other processes keep holding its probed ports.
PORT_CONFLICT_ATTEMPTS = 3
#: A relaunch skips 1 to ``PORT_CONFLICT_SPREAD - 1`` extra port ranges, chosen at random.
PORT_CONFLICT_SPREAD = 16
#: After a failed engine stops, a port counts as held only if it stays held this long.
PORT_RELEASE_SECONDS = 30.0
#: How often a held port is checked again during ``PORT_RELEASE_SECONDS``.
PORT_RECHECK_SECONDS = 1.0


class PortConflictError(RuntimeError):
    """Another process listens on the port of an engine's server."""


def node_address_and_port(start_port: int = 15000, consecutive: int = 1) -> tuple[str, int]:
    """This node's serving address and the first port of a free range of ``consecutive`` ports."""
    import ray

    address = os.environ.get("REEF_INFERENCE_HOST") or ray.util.get_node_ip_address()
    address = address.strip("[]")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    for port in range(start_port, 65536 - consecutive):
        with suppress(OSError):
            with ExitStack() as stack:
                for offset in range(consecutive):
                    sock = stack.enter_context(socket.socket(family, socket.SOCK_STREAM))
                    sock.bind((address, port + offset))
            return (f"[{address}]" if family == socket.AF_INET6 else address), port
    raise RuntimeError("no free inference port range")


def ports_in_use(address: str, ports: Iterable[int]) -> list[int]:
    """The ``ports`` on ``address`` that a live socket holds.

    The check binds with ``SO_REUSEADDR``, so a port that a closed server left
    in TIME_WAIT counts as free.
    """
    address = address.strip("[]")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    held_ports = []
    for port in ports:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError as error:
                if error.errno != errno.EADDRINUSE:
                    raise
                held_ports.append(port)
    return held_ports


def check_server_owner(address: str, port: int, server_pid: int) -> None:
    """Raise :class:`PortConflictError` if a process outside ``server_pid``'s tree listens on ``port``.

    A probe reserves nothing, so another server can answer the engine's
    readiness check. SGLang binds its HTTP port only after the model loads.
    vLLM binds its port with ``SO_REUSEADDR`` early but listens only after the
    load, and on Linux another server can bind the same port and listen first.
    Call this after the server answers. A listener with no visible process,
    such as one in another container on the host network, counts as another
    process.

    Raises:
        PortConflictError: Another process listens on ``address:port`` or on a wildcard address.
        RuntimeError: Nothing listens on the port, or the server process exited. The engine
            failed, and the port check of :class:`EngineStartup` decides whether to relaunch it.
    """
    import psutil

    resolved = {info[4][0] for info in socket.getaddrinfo(address.strip("[]"), port, type=socket.SOCK_STREAM)}
    addresses = {"0.0.0.0", "::", *resolved}
    try:
        listener_pids = [
            connection.pid
            for connection in psutil.net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN
            and connection.laddr.port == port
            and connection.laddr.ip in addresses
        ]
        # Read the tree after the sockets, so that a process that holds a listener is already in it.
        server_pids = {server_pid, *(child.pid for child in psutil.Process(server_pid).children(recursive=True))}
    except psutil.AccessDenied:
        # A check that cannot run must not fail every startup on this host; keep the unchecked behavior.
        logger.warning("cannot read the socket table to check the owner of port %d; not checked", port)
        return
    except psutil.NoSuchProcess as error:
        raise RuntimeError(f"inference process exited after its server answered on port {port}") from error
    if not listener_pids:
        raise RuntimeError(f"nothing listens on inference port {address}:{port} after the server answered")
    if any(pid not in server_pids for pid in listener_pids):
        raise PortConflictError(f"another process listens on inference port {address}:{port}")


def local_gpu_id(physical_gpu: int) -> int:
    """The index of ``physical_gpu`` within this process's visible devices."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return physical_gpu
    devices = [int(value.strip()) for value in visible.split(",") if value.strip()]
    if physical_gpu in devices:
        return devices.index(physical_gpu)
    if 0 <= physical_gpu < len(devices):
        return physical_gpu
    raise ValueError(f"GPU {physical_gpu} is outside CUDA_VISIBLE_DEVICES")


def wait_ready(url: str, process: Any, timeout: float, *, path: str = "/health_generate") -> None:
    """Poll ``url + path`` until it answers 200, the process dies, or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while process.is_alive() and time.monotonic() < deadline:
        try:
            response = requests.get(url + path, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"inference process did not become ready at {url}")


def retire_engines(engines: Sequence[Any], *, timeout: float = 30) -> None:
    """Ask each engine actor to shut down, then kill it; failures never block retirement."""
    import ray

    pending = []
    for engine in engines:
        with suppress(Exception):
            pending.append(engine.shutdown.remote())
    if pending:
        with suppress(Exception):
            ray.get(pending, timeout=timeout)
    for engine in engines:
        with suppress(Exception):
            ray.kill(engine, no_restart=True)


class EngineGroup(ABC):
    """Engine actors that serve on port ranges probed on their own hosts.

    Each slot holds one actor. An engine spans ``nodes_per_engine`` consecutive
    slots, and its first slot is its root. Only the root waits for readiness.
    """

    #: One actor per slot; ``None`` marks a slot that has no actor yet.
    all_engines: list[Any]
    nodes_per_engine: int
    #: The host and the port range that each slot's latest ``init`` received; the server port comes first.
    port_ranges: dict[int, tuple[str, range]]

    @abstractmethod
    def start_engines(self, cursors: dict[str, int]) -> dict[int, Any]:
        """Launch an actor for every empty slot; return each new slot's pending ``init`` call.

        ``cursors`` holds the next port to probe on each host.
        """

    @abstractmethod
    def init_engines(self, slots: list[int], cursors: dict[str, int]) -> dict[int, Any]:
        """Probe a port range from ``cursors`` for each of ``slots`` and call ``init`` on the slot's actor.

        Returns:
            The pending ``init`` call of each slot in ``slots``.
        """


class EngineStartup:
    """Wait for engines to start, and relaunch an engine whose ports another process holds.

    A probe reserves nothing, so another process on the host can bind a probed
    port before the engine does. When a root's ``init`` fails, the startup stops
    every rank of that engine and checks each rank's port range on its host. A
    port counts as held only if it stays held for ``port_release_seconds``,
    because the engine's own processes can still be exiting. If another process
    holds a port, or the root's ``init`` raised :class:`PortConflictError`, all
    ranks start again on new ranges, at most ``PORT_CONFLICT_ATTEMPTS`` times per
    engine. The startup raises any other failure unchanged.
    """

    def __init__(
        self, random_source: random.Random | None = None, port_release_seconds: float = PORT_RELEASE_SECONDS
    ) -> None:
        #: The next port to probe on each host, shared by every engine this startup launches.
        self.cursors: dict[str, int] = {}
        # A fresh generator seeds from the OS; a trainer may seed the global one identically in every stack.
        self.random_source = random.Random() if random_source is None else random_source
        self.port_release_seconds = port_release_seconds
        self.pending: dict[Any, tuple[EngineGroup, int]] = {}

    def start(self, group: EngineGroup) -> None:
        """Launch the empty slots of ``group``; :meth:`wait` waits for them."""
        for slot, ref in group.start_engines(self.cursors).items():
            self.pending[ref] = (group, slot)

    def wait(self) -> None:
        """Return when every started engine is ready; raise the first failure that is not a port conflict."""
        import ray

        attempts: dict[tuple[EngineGroup, int], int] = {}
        while self.pending:
            ready, _ = ray.wait(list(self.pending), num_returns=1)
            group, slot = self.pending.pop(ready[0])
            try:
                ray.get(ready[0])
            except ray.exceptions.RayTaskError as error:
                # Other ranks return once their process starts, so their failures are not port conflicts.
                if slot % group.nodes_per_engine:
                    raise
                ranks = list(range(slot, slot + group.nodes_per_engine))
                # Stop the engine first: its own sockets must not count as held.
                ray.get([group.all_engines[rank].shutdown.remote() for rank in ranks])
                if isinstance(error, PortConflictError):
                    host, ports = group.port_ranges[slot]
                    held_addresses = [f"{host}:{ports.start}"]
                else:
                    held_addresses = self.held_addresses(group, ranks)
                if not held_addresses:
                    raise
                attempt = attempts.get((group, slot), 1)
                if attempt >= PORT_CONFLICT_ATTEMPTS:
                    raise RuntimeError(
                        f"inference engine ports {', '.join(held_addresses)} were held by other processes "
                        f"after {attempt} startup attempts"
                    ) from error
                attempts[(group, slot)] = attempt + 1
                logger.warning(
                    "inference engine ports %s are held by other processes; starting attempt %d of %d on new ports",
                    ", ".join(held_addresses),
                    attempt + 1,
                    PORT_CONFLICT_ATTEMPTS,
                )
                self.pending = {
                    ref: owner for ref, owner in self.pending.items() if owner[0] is not group or owner[1] not in ranks
                }
                for rank in ranks:
                    host, ports = group.port_ranges[rank]
                    # Skip whole ranges at random so two stacks that collided pick different ranges.
                    skipped = len(ports) * self.random_source.randrange(1, PORT_CONFLICT_SPREAD)
                    self.cursors[host] = max(self.cursors.get(host, 0), ports.stop) + skipped
                for rank, ref in group.init_engines(ranks, self.cursors).items():
                    self.pending[ref] = (group, rank)

    def held_addresses(self, group: EngineGroup, ranks: list[int]) -> list[str]:
        """The ``host:port`` pairs of ``ranks`` that stay held for ``port_release_seconds``.

        A stopped engine's own processes can hold a socket until they finish
        exiting. Another process that uses the port holds it for its whole life.
        """
        import ray

        held_ports = {rank: list(group.port_ranges[rank][1]) for rank in ranks}
        deadline = time.monotonic() + self.port_release_seconds
        while True:
            # Check again only the ports that are still held, so a released port never counts.
            checks = [
                group.all_engines[rank].ports_in_use.remote(group.port_ranges[rank][0], held_ports[rank])
                for rank in ranks
            ]
            held_ports = dict(zip(ranks, ray.get(checks), strict=True))
            if not any(held_ports.values()) or time.monotonic() >= deadline:
                break
            time.sleep(PORT_RECHECK_SECONDS)
        return [f"{group.port_ranges[rank][0]}:{port}" for rank in ranks for port in held_ports[rank]]


class RayHealthProbe:
    """Keep one outstanding RPC; a busy actor is not presumed dead."""

    def __init__(self) -> None:
        self.pending: Any = None

    def poll(self, actor: Any, method: str) -> None:
        import ray

        if self.pending is None:
            self.pending = getattr(actor, method).remote()
        ready, _ = ray.wait([self.pending], timeout=0)
        if not ready:
            return
        pending, self.pending = self.pending, None
        result = ray.get(pending)
        if isinstance(result, dict) and result.get("ok") is False and result.get("recoverable") is not True:
            raise RuntimeError(f"model component failed its health check: {result!r}")


__all__ = [
    "EngineGroup",
    "EngineStartup",
    "PortConflictError",
    "RayHealthProbe",
    "check_server_owner",
    "local_gpu_id",
    "node_address_and_port",
    "ports_in_use",
    "retire_engines",
    "wait_ready",
]
