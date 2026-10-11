"""SGLang and vLLM startup relaunches an engine or an SGLang router whose ports another process holds."""

from __future__ import annotations

import dataclasses
import math
import random
import socket
import sys
from types import SimpleNamespace

import pytest
import ray
from ray.exceptions import RayTaskError

from reef.inference import process
from reef.inference.process import (
    PORT_CONFLICT_SPREAD,
    EngineStartup,
    PortConflictError,
    check_server_owner,
    ports_in_use,
)
from reef.inference.sglang import launch as sglang_launch
from reef.inference.sglang.config import SGLangConfig, SGLangGroupConfig
from reef.inference.sglang.launch import SGLangCluster, SGLangEngineGroup, SGLangModel
from reef.inference.vllm import engine as vllm_engine
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.engine import ReefVLLMEngine
from reef.inference.vllm.launch import VLLMEngineGroup
from reef.inference.vllm.worker import VLLMWorker

NOT_READY = RuntimeError("inference process did not become ready")
#: The tests' release window, in fake seconds; ports are checked again every second.
RELEASE_SECONDS = 3.0
SERVER_PID = 4321


class SkipTwoRanges(random.Random):
    def randrange(self, *args):
        assert args == (1, PORT_CONFLICT_SPREAD)
        return 2


class Pending:
    """A pending ``init`` call; the fake ``ray.get`` raises its error as a failed Ray task."""

    def __init__(self, error):
        self.error = error


class Remote:
    def __init__(self, fn):
        self.remote = fn


class Engines:
    """Fake engine actors on ``hosts``, one per slot, recording each call in ``events``.

    A probe returns its start port. ``failures[slot]`` lists the errors of that
    slot's next ``init`` calls. ``held`` maps each ``(host, port)`` pair that
    another process binds to the fake time when it lets the port go. The fake
    ``time.sleep`` moves the fake clock forward.
    """

    def __init__(self, monkeypatch, hosts):
        self.hosts = hosts
        self.failures = {}
        self.held = {}
        self.events = []
        self.now = 0.0
        monkeypatch.setattr(ray, "get", self.get)
        monkeypatch.setattr(ray, "wait", lambda refs, num_returns=1: (refs[:1], refs[1:]))
        monkeypatch.setattr(process.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(process.time, "sleep", self.sleep)
        monkeypatch.setattr(SGLangEngineGroup, "_launch_actor", lambda group, slot: self.actor(slot))
        monkeypatch.setattr(VLLMEngineGroup, "_launch_actor", lambda group, slot: self.actor(slot))

    def hold(self, host, port, seconds=math.inf):
        self.held[(host, port)] = self.now + seconds

    def sleep(self, seconds):
        # A check loop that ignores its deadline fails here instead of hanging.
        if self.now > 1000:
            raise RuntimeError("the port check never stopped")
        self.now += seconds

    def get(self, value, timeout=None):
        if isinstance(value, list):
            return [self.get(item) for item in value]
        if isinstance(value, Pending) and value.error is not None:
            raise RayTaskError("init", "traceback", value.error).as_instanceof_cause()
        return value

    def actor(self, slot):
        host = self.hosts[slot]

        def init(*args, **kwargs):
            port = kwargs["port"] if kwargs else args[1]
            self.events.append(("init", slot, port, kwargs.get("dist_init_addr")))
            failures = self.failures.get(slot, [])
            return Pending(failures.pop(0) if failures else None)

        def check(address, ports):
            self.events.append(("check", slot))
            return [port for port in ports if self.held.get((address, port), -math.inf) > self.now]

        return SimpleNamespace(
            _get_current_node_ip_and_free_port=Remote(lambda start_port=15000, consecutive=1: (host, start_port)),
            node_address_and_port=Remote(lambda start_port=15000: (host, start_port)),
            init=Remote(init),
            shutdown=Remote(lambda: self.events.append(("shutdown", slot))),
            ports_in_use=Remote(check),
        )


def sglang_group(gpus=2, gpus_per_engine=1, gpus_per_node=2):
    config = SGLangConfig("model", gpus, gpus_per_engine, gpus_per_node)
    return SGLangEngineGroup(config, SGLangGroupConfig("regular", gpus, gpus_per_engine), None, 0, ("router", 3000))


def vllm_group():
    return VLLMEngineGroup(VLLMConfig("model", 2, 1, 2, router_url="http://router"), placement=None)


def start(group):
    startup = EngineStartup(random_source=SkipTwoRanges(), port_release_seconds=RELEASE_SECONDS)
    startup.start(group)
    startup.wait()


def test_port_check_reports_bound_ports_but_not_time_wait_or_free_ports():
    with socket.socket() as listener, socket.socket() as server:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        held = listener.getsockname()[1]
        # Like the servers Reef launches; Linux lets a TIME_WAIT port be reused only if both sockets ask.
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        closed = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", closed))
        connection, _ = server.accept()
        # The server side closes first, so its end of the connection stays in TIME_WAIT.
        connection.close()
        client.close()
        server.close()
        with socket.socket() as plain, pytest.raises(OSError):
            plain.bind(("127.0.0.1", closed))
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            free = unused.getsockname()[1]
        assert ports_in_use("127.0.0.1", [held, closed, free]) == [held]


@pytest.mark.parametrize(
    "make_group,held_port,relaunched",
    [
        # SGLang ranges are 35 ports wide; port + 3 is the rendezvous port that #630 lost.
        (sglang_group, 15003, ("init", 0, 15140, "node-a:15143")),
        (vllm_group, 15000, ("init", 0, 15004, None)),
    ],
)
def test_port_held_through_the_window_relaunches_only_that_engine_past_its_ports(
    monkeypatch, make_group, held_port, relaunched
):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    engines.failures[0] = [NOT_READY]
    engines.hold("node-a", held_port)
    group = make_group()
    start(group)
    # The engine stops before the check, so its own sockets never count; the check repeats each second.
    assert engines.events[2:] == [("shutdown", 0), *[("check", 0)] * 4, relaunched]
    assert [event[:2] for event in engines.events[:2]] == [("init", 0), ("init", 1)]
    assert engines.now == RELEASE_SECONDS
    assert group.num_new_engines == 2


@pytest.mark.parametrize("make_group,held_port", [(sglang_group, 15003), (vllm_group, 15000)])
def test_port_released_within_the_window_raises_the_original_failure(monkeypatch, make_group, held_port):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    engines.failures[0] = [NOT_READY]
    # The engine's own exiting process keeps the port for 2 seconds after shutdown returns.
    engines.hold("node-a", held_port, seconds=2)
    with pytest.raises(RayTaskError) as raised:
        start(make_group())
    assert raised.value.cause is NOT_READY
    assert [event[:2] for event in engines.events] == [
        ("init", 0),
        ("init", 1),
        ("shutdown", 0),
        ("check", 0),
        ("check", 0),
        ("check", 0),
    ]
    assert engines.now == 2


@pytest.mark.parametrize("make_group", [sglang_group, vllm_group])
def test_failure_without_a_held_port_is_raised_without_relaunch(monkeypatch, make_group):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    engines.failures[0] = [NOT_READY]
    with pytest.raises(RayTaskError) as raised:
        start(make_group())
    assert raised.value.cause is NOT_READY
    assert [event[:2] for event in engines.events] == [("init", 0), ("init", 1), ("shutdown", 0), ("check", 0)]
    assert engines.now == 0


@pytest.mark.parametrize(
    "make_group,held_ports",
    [(sglang_group, (15003, 15143, 15248)), (vllm_group, (15000, 15004, 15007))],
)
def test_engine_whose_ports_stay_held_fails_after_three_attempts(monkeypatch, make_group, held_ports):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    engines.failures[0] = [NOT_READY] * 3
    for port in held_ports:
        engines.hold("node-a", port)
    with pytest.raises(RuntimeError, match=f"ports node-a:{held_ports[-1]} were held .* after 3 startup attempts"):
        start(make_group())
    assert len([event for event in engines.events if event[:2] == ("init", 0)]) == 3


@pytest.mark.parametrize(
    "make_group,relaunched,last_server_port",
    [(sglang_group, ("init", 0, 15140, "node-a:15143"), 15245), (vllm_group, ("init", 0, 15004, None), 15007)],
)
def test_server_answered_by_another_process_relaunches_without_a_port_check(
    monkeypatch, make_group, relaunched, last_server_port
):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    conflict = PortConflictError("another process listens on inference port node-a:15000")
    engines.failures[0] = [conflict]
    start(make_group())
    assert engines.events[2:] == [("shutdown", 0), relaunched]
    engines.failures[0] = [conflict] * 3
    with pytest.raises(RuntimeError, match=f"ports node-a:{last_server_port} were held .* after 3 startup attempts"):
        start(make_group())


def test_multinode_engine_relaunches_every_rank_at_a_new_root_address(monkeypatch):
    engines = Engines(monkeypatch, ["node-a", "node-b"])
    engines.failures[0] = [NOT_READY]
    engines.hold("node-a", 15003)
    start(sglang_group(gpus=8, gpus_per_engine=8, gpus_per_node=4))
    assert engines.events == [
        ("init", 0, 15000, "node-a:15003"),
        ("init", 1, 15000, "node-a:15003"),
        ("shutdown", 0),
        ("shutdown", 1),
        *[("check", 0), ("check", 1)] * 4,
        ("init", 0, 15105, "node-a:15108"),
        ("init", 1, 15105, "node-a:15108"),
    ]
    # A rank other than the root returns once its process starts, so its failure is never a port conflict.
    engines.events.clear()
    engines.failures[1] = [NOT_READY]
    engines.hold("node-b", 15003)
    with pytest.raises(RayTaskError):
        start(sglang_group(gpus=8, gpus_per_engine=8, gpus_per_node=4))
    assert [event[0] for event in engines.events] == ["init", "init"]


def test_recovery_raises_an_external_engine_failure_unchanged(monkeypatch):
    engines = Engines(monkeypatch, ["ext"])
    incompatible = ValueError("external SGLang engine has incompatible enable_memory_saver")
    engines.failures[0] = [incompatible]
    external = {"url": "http://ext:30000", "host": "ext", "port": 30000, "worker_type": "regular", "num_gpus": 1}
    config = SGLangConfig("model", 1, 1, 1)
    group = SGLangEngineGroup(config, SGLangGroupConfig("regular", 1, 1), None, 0, ("router", 3000), external)
    with pytest.raises(RayTaskError) as raised:
        SGLangModel([group]).recover()
    assert raised.value.cause is incompatible
    assert [event[0] for event in engines.events] == ["init"]


@pytest.mark.parametrize("site", ["sglang start", "sglang recover", "vllm start", "vllm recover"])
def test_startup_and_recovery_relaunch_on_port_conflicts(monkeypatch, site):
    engines = Engines(monkeypatch, ["node-a", "node-a"])
    engines.failures[0] = [NOT_READY]
    width = 35 if site.startswith("sglang") else 1
    engines.hold("node-a", 15003 if width == 35 else 15000)
    if site == "sglang start":
        monkeypatch.setattr(SGLangCluster, "_router", lambda self, index, pd: ("router", 3000))
        SGLangCluster(SGLangConfig("model", 2, 1, 2), None).start()
    elif site == "sglang recover":
        SGLangModel([sglang_group()]).recover()
    elif site == "vllm start":
        monkeypatch.setattr(VLLMWorker, "_new_rollout_engine_lock", lambda self: None)
        VLLMWorker(VLLMConfig("model", 2, 1, 2, router_url="http://router"), None)
    else:
        vllm_group().recover()
    _, relaunched = [event[2] for event in engines.events if event[:2] == ("init", 0)]
    # Past both engines' ports by a random whole number of ranges, after the default release window.
    skipped = relaunched - (15000 + 2 * width)
    assert skipped % width == 0 and 0 < skipped < width * PORT_CONFLICT_SPREAD
    assert engines.now == process.PORT_RELEASE_SECONDS


class Routers:
    """Fake router processes on node ``10.0.0.5``, recording each launch's options.

    A probe returns its start port, and a relaunch skips two ports. ``failures``
    lists the errors of the next readiness checks: a :class:`PortConflictError`
    comes from the owner check, any other error from ``wait_ready``.
    ``held_ports`` are the ports that another process binds.
    """

    def __init__(self, monkeypatch, held_ports=()):
        self.options = []
        self.processes = []
        self.failures = []
        self.held_ports = held_ports
        router_module = SimpleNamespace(RouterArgs=dataclasses.make_dataclass("RouterArgs", []))
        monkeypatch.setitem(sys.modules, "sglang_router", SimpleNamespace(launch_router=router_module))
        monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_module)
        monkeypatch.setattr(sglang_launch, "node_address_and_port", lambda start_port: ("10.0.0.5", start_port))
        monkeypatch.setattr(sglang_launch, "launch_router", self.launch)
        monkeypatch.setattr(sglang_launch, "wait_ready", self.wait_ready)
        monkeypatch.setattr(sglang_launch, "check_server_owner", self.check_server_owner)
        monkeypatch.setattr(sglang_launch, "ports_in_use", self.ports_in_use)
        # The fake takes no seed, so a seeded generator fails: a shared seed gives every stack the same ports.
        monkeypatch.setattr(sglang_launch, "random", SimpleNamespace(Random=lambda: SkipTwoRanges()))

    def launch(self, options):
        self.options.append(dict(options))
        router = SimpleNamespace(
            pid=SERVER_PID + len(self.processes), ready=False, stopped=False, is_alive=lambda: False
        )
        router.terminate = lambda: setattr(router, "stopped", True)
        router.join = lambda timeout: None
        self.processes.append(router)
        return router

    def wait_ready(self, url, router, timeout, *, path):
        if self.failures and not isinstance(self.failures[0], PortConflictError):
            raise self.failures.pop(0)
        router.ready = True

    def check_server_owner(self, host, port, server_pid):
        router = self.processes[-1]
        assert router.ready and (host, port, server_pid) == ("10.0.0.5", self.options[-1]["port"], router.pid)
        if self.failures:
            raise self.failures.pop(0)

    def ports_in_use(self, host, ports):
        # A router that still runs would hold its own ports.
        assert host == "10.0.0.5" and self.processes[-1].stopped
        return [port for port in ports if port in self.held_ports]


@pytest.mark.parametrize(
    "index, router_port, port, metrics_port",
    [
        (0, None, 3000, 4000),
        # The port above a configured router port can be another stack's router port.
        (0, 30000, 30000, 4000),
        (0, 4000, 4000, 4001),
        # Only the first model's router uses the configured port.
        (1, 30000, 3000, 4000),
    ],
)
def test_router_probes_its_metrics_port_from_4000_on_the_node_address(
    monkeypatch, index, router_port, port, metrics_port
):
    routers = Routers(monkeypatch)
    cluster = SGLangCluster(SGLangConfig("model", 1, 1, 1, router_port=router_port), None)
    assert cluster._router(index, pd=False) == ("10.0.0.5", port)
    (options,) = routers.options
    # Without prometheus_host, the router serves metrics on loopback. The probe cannot see that port, so every stack
    # on the host would pick the same one.
    assert (options["prometheus_host"], options["prometheus_port"]) == ("10.0.0.5", metrics_port)


@pytest.mark.parametrize(
    "router_port, failure, held_ports",
    [
        (None, NOT_READY, [3000]),
        (None, NOT_READY, [4000]),
        # A configured port stays; only the metrics port moves.
        (30000, NOT_READY, [4000]),
        # Another stack's router answered, so its listener holds the port.
        (None, PortConflictError("another process listens on inference port 10.0.0.5:3000"), []),
    ],
)
def test_router_whose_ports_another_process_holds_starts_again_on_new_ports(
    monkeypatch, caplog, router_port, failure, held_ports
):
    routers = Routers(monkeypatch, held_ports)
    routers.failures = [failure]
    cluster = SGLangCluster(SGLangConfig("model", 1, 1, 1, router_port=router_port), None)
    host, port = cluster._router(0, pd=False)
    _, second = routers.options
    assert (host, port) == ("10.0.0.5", second["port"])
    assert second["port"] == (3002 if router_port is None else router_port)
    # The metrics port skips the same number of ports, so routers that collided move their metrics ports apart too.
    assert second["prometheus_port"] == 4002
    assert routers.processes[0].stopped and cluster.routers == routers.processes[1:]
    assert "starting attempt 2 of 3 on new ports" in caplog.text


@pytest.mark.parametrize(
    "router_port, held_ports, failures, message",
    [
        (None, [], [NOT_READY], "did not become ready"),
        # Reef cannot move a configured port that another process holds.
        (3000, [3000], [NOT_READY], "did not become ready"),
        (3000, [], [PortConflictError("another process listens on inference port 10.0.0.5:3000")], "listens"),
        (
            None,
            [],
            [PortConflictError("another process listens")] * 3,
            r"ports 10\.0\.0\.5:3004 were held by other processes after 3 startup attempts",
        ),
    ],
)
def test_router_failure_stops_the_launch(monkeypatch, router_port, held_ports, failures, message):
    routers = Routers(monkeypatch, held_ports)
    routers.failures = list(failures)
    cluster = SGLangCluster(SGLangConfig("model", 1, 1, 1, router_port=router_port), None)
    with pytest.raises(RuntimeError, match=message):
        cluster._router(0, pd=False)
    assert all(router.stopped for router in routers.processes)
    assert len(routers.options) == len(failures) and cluster.routers == []


class AccessDenied(Exception):
    pass


class NoSuchProcess(Exception):
    pass


def install_psutil(monkeypatch, connections, children=(), table_readable=True, server_alive=True):
    """Install a psutil whose socket table lists ``connections`` as ``(ip, port, pid, status)`` entries."""

    def net_connections(kind):
        if not table_readable:
            raise AccessDenied
        return [
            SimpleNamespace(laddr=SimpleNamespace(ip=ip, port=port), pid=pid, status=status)
            for ip, port, pid, status in connections
        ]

    def process(pid):
        if not server_alive:
            raise NoSuchProcess(pid)
        return SimpleNamespace(children=lambda recursive: [SimpleNamespace(pid=child) for child in children])

    psutil = SimpleNamespace(
        CONN_LISTEN="LISTEN",
        AccessDenied=AccessDenied,
        NoSuchProcess=NoSuchProcess,
        net_connections=net_connections,
        Process=process,
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)


def test_server_owner_check_accepts_listeners_of_the_server_tree(monkeypatch):
    install_psutil(
        monkeypatch,
        [
            ("10.0.0.5", 18000, SERVER_PID, "LISTEN"),
            ("0.0.0.0", 18000, 4322, "LISTEN"),  # an API server child of the engine
            ("127.0.0.2", 18000, 900, "LISTEN"),  # another address cannot answer 10.0.0.5
            ("10.0.0.5", 18001, 901, "LISTEN"),
            ("10.0.0.5", 18000, None, "ESTABLISHED"),
        ],
        children=(4322,),
    )
    check_server_owner("10.0.0.5", 18000, SERVER_PID)


@pytest.mark.parametrize(
    "listener",
    [
        ("10.0.0.5", 18000, 900, "LISTEN"),
        # No visible process: for example, a server in another container on the host network.
        ("10.0.0.5", 18000, None, "LISTEN"),
        ("::", 18000, None, "LISTEN"),
    ],
)
def test_server_owner_check_reports_another_listener_as_a_port_conflict(monkeypatch, listener):
    install_psutil(monkeypatch, [("10.0.0.5", 18000, SERVER_PID, "LISTEN"), listener])
    with pytest.raises(PortConflictError, match=r"10\.0\.0\.5:18000"):
        check_server_owner("10.0.0.5", 18000, SERVER_PID)


@pytest.mark.parametrize("server_alive", [True, False])
def test_server_owner_check_without_a_listener_or_server_is_an_engine_failure(monkeypatch, server_alive):
    # The answer came from a server that is gone; the release-window check decides about a relaunch.
    install_psutil(monkeypatch, [], server_alive=server_alive)
    with pytest.raises(RuntimeError) as raised:
        check_server_owner("10.0.0.5", 18000, SERVER_PID)
    assert not isinstance(raised.value, PortConflictError)


def test_server_owner_check_is_skipped_when_the_socket_table_is_unreadable(monkeypatch, caplog):
    install_psutil(monkeypatch, [], table_readable=False)
    check_server_owner("10.0.0.5", 18000, SERVER_PID)
    assert "not checked" in caplog.text


@pytest.mark.parametrize("owner", [SERVER_PID, 900])
def test_vllm_engine_checks_the_listener_once_its_server_answers(monkeypatch, owner):
    install_psutil(monkeypatch, [("10.0.0.5", 18900, owner, "LISTEN")])
    monkeypatch.setattr(vllm_engine, "launch_server", lambda *args: SimpleNamespace(pid=SERVER_PID))
    monkeypatch.setattr(vllm_engine, "wait_ready", lambda *args, **kwargs: None)
    engine = ReefVLLMEngine(VLLMConfig("model", 1, 1, 1), rank=0, gpu_ids=(0,))
    if owner == SERVER_PID:
        engine.init("10.0.0.5", 18900)
    else:
        with pytest.raises(PortConflictError):
            engine.init("10.0.0.5", 18900)
