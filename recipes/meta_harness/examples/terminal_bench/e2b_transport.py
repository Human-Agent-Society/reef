"""Reconnect a dropped E2B command stream without repeating execution.

The episode's original deadline bounds every connection and backoff. A lost
start acknowledgement has no trustworthy PID and is never retried. Diagnostics
contain identifiers and exception types, never request bodies or environments.
"""

import time
from copy import deepcopy

from reef.harness.executor import EpisodeLaunchError


def transport_failure(exc):
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from e2b.exceptions import RateLimitException, TimeoutException
    from httpx import TransportError
    from pyqwest import ReadError, StreamError, WriteError

    if isinstance(
        exc,
        (
            TimeoutException,
            RateLimitException,
            ConnectionError,
            TimeoutError,
            TransportError,
            ReadError,
            StreamError,
            WriteError,
        ),
    ):
        return True
    if isinstance(exc, ConnectError):
        return exc.code in (
            Code.CANCELED,
            Code.DEADLINE_EXCEEDED,
            Code.UNAVAILABLE,
            Code.RESOURCE_EXHAUSTED,
        ) or transport_failure(exc.__cause__)
    # E2B 2.46.4 raises this bare Exception when a stream closes without an end
    # event. Do not treat arbitrary candidate, authentication or SDK errors as
    # transport failures.
    return type(exc) is Exception and exc.args == ("Command ended without an end event",)


def failure_diagnostics(exc):
    """Find executor-owned diagnostics through run_episode's exception wrapper."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(value := getattr(exc, "e2b_diagnostics", None), dict):
            return deepcopy(value)
        exc = exc.__cause__
    return None


def failure_evidence(exc):
    """Find separately retained, redacted trial evidence through wrappers."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(value := getattr(exc, "e2b_retained_evidence", None), dict) and value:
            return deepcopy(value)
        exc = exc.__cause__
    return None


def run_attached(commands, command, *, remaining, diagnostics, recover_finished=None, **kwargs):
    """Start once, then wait/reconnect to that PID until completion or deadline."""
    from e2b.sandbox.commands.command_handle import CommandExitException

    diagnostics.update(
        phase="command_start", command_pid=None, reconnects=0, command_end_received=False, connection_failures=[]
    )
    # A failed call here might still have started remote work. No retry is
    # permitted without the start event's process identity.
    handle = commands.run(command, background=True, timeout=remaining(), **kwargs)
    pid = handle.pid
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise EpisodeLaunchError("E2B returned an invalid command process identity")
    diagnostics.update(command_pid=pid, phase="command_wait")
    consecutive = 0
    while True:
        connected_at = time.monotonic()
        try:
            if handle is None:
                diagnostics["phase"] = "command_reconnect"
                diagnostics["reconnects"] += 1
                handle = commands.connect(pid, timeout=remaining())
                if handle.pid != pid:
                    raise EpisodeLaunchError("E2B reconnected to a different command process")
                diagnostics["phase"] = "command_wait"
            try:
                result = handle.wait()
            except CommandExitException as exc:
                result = exc  # Completed nonzero commands still have trial evidence.
            diagnostics.update(
                command_end_received=True,
                phase="command_complete",
                stdout_may_be_partial=bool(diagnostics["reconnects"]),
            )
            return result
        except Exception as exc:
            from e2b.exceptions import NotFoundException

            recoverable = transport_failure(exc) or isinstance(exc, NotFoundException)
            if recover_finished is not None and recoverable:
                try:
                    recovered = recover_finished(pid)
                except Exception as recovery_error:
                    if not transport_failure(recovery_error):
                        raise
                    diagnostics["completion_read_error_type"] = type(recovery_error).__name__
                else:
                    if recovered is not None:
                        diagnostics["completion_recovery_trigger"] = {
                            "phase": diagnostics["phase"],
                            "error_type": type(exc).__name__,
                        }
                        diagnostics.update(
                            phase="command_complete", completion_recovered=True, stdout_may_be_partial=False
                        )
                        return recovered
            if not transport_failure(exc):
                raise
            # A long successful attachment followed by another drop starts a
            # fresh recovery window. Six immediate failures indicate an outage;
            # never turn an eight-hour allowance into a rapid retry storm.
            if time.monotonic() - connected_at >= 30:
                consecutive = 0
            consecutive += 1
            failures = diagnostics["connection_failures"]
            failures.append({"phase": diagnostics["phase"], "error_type": type(exc).__name__})
            del failures[:-128]
            if consecutive >= 6:
                raise EpisodeLaunchError("E2B command reconnection failed repeatedly; usage may be unknown") from exc
            time.sleep(min(2 ** (consecutive - 1), 10, remaining()))
            remaining()  # Never reconnect after the episode deadline.
        finally:
            if handle is not None:
                try:
                    handle.disconnect()
                except Exception as exc:
                    diagnostics["disconnect_error_type"] = type(exc).__name__
            handle = None
