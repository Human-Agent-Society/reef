"""Measure a Reef policy update using the owner's events and model-free probes of the inference route."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import aiohttp

from reef.service.deploy.routing.registry import DeploymentRegistry

PROBE_INTERVAL_SECONDS = 0.025
CHAT = "/v1/chat/completions"


def owner_events(deployment_id: str, pid: int) -> Path:
    """Find this generation and confirm Linux reports that the supplied PID holds its registry lock."""
    if pid <= 1:
        raise ValueError("expected the PID of this tutorial's reef serve process")
    registry = DeploymentRegistry(deployment_id)
    owner = registry.read_owner()
    if owner is None:
        raise ValueError(f"missing deployment {deployment_id}")
    lock = (registry.directory / "lock").stat()
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if fields[1:5] != ["FLOCK", "ADVISORY", "WRITE", str(pid)]:
            continue
        major, minor, inode = fields[5].split(":")
        if (os.makedev(int(major, 16), int(minor, 16)), int(inode)) == (lock.st_dev, lock.st_ino):
            return owner.configuration_directory / "lifecycle-events.jsonl"
    raise ValueError(f"PID {pid} does not own the lock for deployment {deployment_id}")


def send_reload(pidfd: int) -> float:
    """Signal the verified owner's open process descriptor and return the monotonic time of dispatch."""
    if hasattr(signal, "pidfd_send_signal"):
        sent = time.monotonic()
        signal.pidfd_send_signal(pidfd, signal.SIGHUP)
        return sent

    # On Ubuntu a standalone Reef Python may lack this Linux binding. Give the system Python
    # the same process descriptor. A recycled numeric PID cannot redirect the signal.
    helper = (
        "import signal, sys, time; sent = time.monotonic(); "
        "signal.pidfd_send_signal(int(sys.argv[1]), signal.SIGHUP); print(sent)"
    )
    reply = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-c", helper, str(pidfd)],
        pass_fds=(pidfd,),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if reply.returncode:
        raise RuntimeError(f"system Python could not signal the Reef owner: {reply.stderr.strip()}")
    return float(reply.stdout)


async def admission_probe(session: aiohttp.ClientSession) -> tuple[float, float, int]:
    """An unknown pin returns 400 when open, or the admission middleware's specific 503 when closed."""
    start = time.monotonic()
    async with session.post(CHAT, json={"model": "absent-probe-target"}, allow_redirects=False) as response:
        body = await response.text()
        end = time.monotonic()
        if (response.status == 400 and body.startswith("unknown_target:")) or (
            response.status == 503 and body == "routing deployment is not accepting requests"
        ):
            return start, end, response.status
        raise RuntimeError(f"admission probe: unexpected HTTP {response.status}: {body[:200]}")


async def observe_update(
    session: aiohttp.ClientSession, pidfd: int, events: Path, source: Path, candidate: bytes, version: str
) -> dict[str, object]:
    """Replace the source, signal its owner and measure until that snapshot is committed, ready and serving."""
    if (await admission_probe(session))[2] != 400:
        raise RuntimeError("admission was closed before SIGHUP")
    with tempfile.NamedTemporaryFile(dir=source.parent, delete=False) as replacement:
        replacement.write(candidate)
    os.replace(replacement.name, source)
    signal_time = send_reload(pidfd)
    probes = []
    phases: dict[str, float] = {}
    while time.monotonic() - signal_time < 180:
        probes.append(await admission_probe(session))
        for line in events.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue  # The owner may still be writing the last event.
            event = json.loads(line)
            stamp, state = event["monotonic_seconds"], event["state"]
            if stamp < signal_time:
                continue
            phases[state] = stamp
            if state in {"policy_rejected", "policy_unchanged", "policy_restoring", "unready", "stopping", "failed"}:
                raise RuntimeError(f"reload did not apply: {event}")
            if state in {"policy_updating", "policy_committed"} and event["snapshot_version"] != version:
                raise RuntimeError("owner updated a different manifest")
        if "ready" in phases and probes[-1][2] == 400:
            break
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)
    else:
        raise TimeoutError(f"no ready event and open probe for {version}; observed {list(phases)}")

    # Close occurs between policy_updating and draining. Open occurs after commit and by ready.
    updating, draining, committed, ready = (
        phases[s] for s in ("policy_updating", "draining", "policy_committed", "ready")
    )
    if not signal_time <= updating <= draining <= committed <= ready:
        raise RuntimeError(f"out-of-order owner events: {phases}")
    close_min, close_max = updating, draining
    open_min, open_max = committed, ready
    closed = [sample for sample in probes if sample[2] == 503]
    if closed:
        close_max = min(close_max, closed[0][1])
        open_min = max(open_min, closed[-1][0])
        for start, end, status in probes:
            if status == 400 and end <= closed[0][0]:
                close_min = max(close_min, start)
            if status == 400 and start >= closed[-1][1]:
                open_max = min(open_max, end)
    if close_min > close_max or open_min > open_max:
        raise RuntimeError("client probes contradict owner event timestamps")
    return {
        "snapshot": version,
        "signal_monotonic_seconds": signal_time,
        "seconds_from_signal": {
            s: phases[s] - signal_time for s in ("policy_updating", "draining", "policy_committed", "ready")
        },
        "admission_closed_seconds_bounds": [max(0.0, open_min - close_max), open_max - close_min],
        "probe_count": len(probes),
        "admission_503_count": len(closed),
        "probe_sleep_seconds": PROBE_INTERVAL_SECONDS,
        "probe_max_start_gap_seconds": max((b[0] - a[0] for a, b in pairwise(probes)), default=None),
        "first_503_request_seconds_from_signal": [v - signal_time for v in closed[0][:2]] if closed else None,
        "last_503_request_seconds_from_signal": [v - signal_time for v in closed[-1][:2]] if closed else None,
    }
