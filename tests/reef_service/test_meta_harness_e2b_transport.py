"""Fault injection at command start, attachment, archive and cleanup boundaries."""

from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench import e2b_transport as transport
from reef.harness.executor import EpisodeLaunchError, EpisodeTimeout


class Clock:
    now = 0

    def sleep(self, seconds):
        self.now += seconds

    def remaining(self):
        if self.now >= 100:
            raise EpisodeTimeout("deadline")
        return 100 - self.now


class Commands:
    def __init__(self, events):
        self.events = iter(events)
        self.starts = []
        self.connections = []
        self.disconnections = 0

    def handle(self):
        event = next(self.events)

        def wait():
            if isinstance(event, Exception):
                raise event
            return event

        def disconnect():
            self.disconnections += 1

        return SimpleNamespace(pid=2099, wait=wait, disconnect=disconnect)

    def run(self, command, **kwargs):
        self.starts.append((command, kwargs))
        return self.handle()

    def connect(self, pid, **kwargs):
        self.connections.append((pid, kwargs))
        return self.handle()


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(transport.time, "sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("error", [ConnectionError("secret request"), Exception("Command ended without an end event")])
def test_drop_reconnects_same_process_once_without_repeating_start(clock, error):
    result = SimpleNamespace(exit_code=0, stdout="complete", stderr="")
    commands = Commands([error, result])
    diagnostics = {}
    assert (
        transport.run_attached(
            commands, "runner", remaining=clock.remaining, diagnostics=diagnostics, envs={"API_KEY": "secret"}
        )
        is result
    )
    assert len(commands.starts) == 1
    assert commands.starts[0][1]["background"] is True
    assert commands.connections == [(2099, {"timeout": 99})]
    assert commands.disconnections == 2
    assert diagnostics["command_end_received"] is True
    assert diagnostics["reconnects"] == 1
    assert "secret" not in str(diagnostics)


def test_initial_acknowledgement_loss_never_repeats_command(clock):
    class UnknownStart(Commands):
        def run(self, command, **kwargs):
            self.starts.append(command)
            raise ConnectionError("start may have succeeded")

    commands = UnknownStart([])
    diagnostics = {}
    with pytest.raises(ConnectionError):
        transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics=diagnostics)
    assert commands.starts == ["runner"]
    assert not commands.connections
    assert diagnostics["command_pid"] is None


def test_stream_timeout_before_episode_deadline_reconnects(clock):
    from e2b.exceptions import TimeoutException

    result = SimpleNamespace(exit_code=0)
    commands = Commands([TimeoutException("context canceled"), result])
    assert transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics={}) is result
    assert len(commands.starts) == 1


def test_real_deadline_prevents_reconnect(clock):
    clock.now = 99.5
    commands = Commands([ConnectionError("drop")])
    with pytest.raises(EpisodeTimeout):
        transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics={})
    assert not commands.connections
    assert commands.disconnections == 1


def test_repeated_immediate_drops_stop_bounded_without_repeating_work(clock):
    commands = Commands([ConnectionError("drop")] * 6)
    diagnostics = {}
    with pytest.raises(EpisodeLaunchError, match="reconnection failed"):
        transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics=diagnostics)
    assert len(commands.starts) == 1
    assert len(commands.connections) == 5
    assert diagnostics["command_end_received"] is False
    assert commands.disconnections == 6


def test_nonzero_reconnected_result_keeps_diagnostics(clock):
    from e2b.sandbox.commands.command_handle import CommandExitException

    result = CommandExitException(stdout="out", stderr="err", exit_code=7, error="failed")
    commands = Commands([ConnectionError("drop"), result])
    diagnostics = {}
    assert transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics=diagnostics) is result
    assert diagnostics["command_end_received"] is True


@pytest.mark.parametrize("error", [ValueError("bad SDK result"), PermissionError("denied")])
def test_nontransport_failures_do_not_retry(clock, error):
    commands = Commands([error])
    with pytest.raises(type(error)):
        transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics={})
    assert not commands.connections


def test_wrong_reconnected_process_rejected(clock):
    commands = Commands([ConnectionError("drop"), SimpleNamespace(exit_code=0)])
    original = commands.connect

    def connect(pid, **kwargs):
        handle = original(pid, **kwargs)
        handle.pid = 2100
        return handle

    commands.connect = connect
    with pytest.raises(EpisodeLaunchError, match="different command"):
        transport.run_attached(commands, "runner", remaining=clock.remaining, diagnostics={})


def test_exception_chain_retains_detached_safe_diagnostics():
    inner = EpisodeLaunchError("failed")
    inner.e2b_diagnostics = {"sandbox_id": "owned", "command_pid": 2099}
    outer = RuntimeError("wrapper")
    outer.__cause__ = inner
    result = transport.failure_diagnostics(outer)
    result["command_pid"] = 1
    assert inner.e2b_diagnostics["command_pid"] == 2099
