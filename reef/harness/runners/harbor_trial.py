"""Run one Harbor trial through reef-eval on behalf of a runner Reef ships, and find what the trial wrote.

reef-eval is imported only inside ``run_trial`` and nothing on the render path imports this module, so the adapter
registry stays cheap and a deployment without Harbor can still load every descriptor.

A verifier can reward a task 0 when its own machinery failed rather than the agent.
``REEF_HARBOR_INFRASTRUCTURE_MARKERS`` names such failures declaratively, as a file the verifier writes and the
values one of its keys holds then; the caller of ``run_episode`` sets it (``evolution.infrastructure_markers``),
never the tree, and a runner that finds one records the trial as one that never ran. Only the trial's ``verifier``
directory is read: the agent's directories are the task container's to write.
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from reef.core.errors import ReefError

INFRASTRUCTURE_MARKERS_ENV = "REEF_HARBOR_INFRASTRUCTURE_MARKERS"


class HarborTrialError(ReefError):
    """A Harbor trial cannot start from this process."""


@dataclass(frozen=True)
class TrialResult:
    """One trial: the verifier's rewards (empty when it gave none), Harbor's error, and the trial directory this run
    wrote (``None`` when it cannot be told apart from earlier trials)."""

    rewards: dict[str, float]
    error: str
    trial_path: Path | None


def own_trial(trials_dir: Path, names_before_run: Collection[str], uri: str | None) -> Path | None:
    """The trial directory this run wrote: the one the Lab row's ``file://`` URI names (Harbor's trial URI), else
    the one directory that appeared under ``trials_dir`` during the run. ``None`` when neither says, so a reused trials
    directory never lends this run an earlier trial's steps."""
    if isinstance(uri, str) and uri.startswith("file://"):
        path = Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))
        if path.is_dir():
            return path
    appeared = [path for path in trials_dir.iterdir() if path.is_dir() and path.name not in names_before_run]
    return appeared[0] if len(appeared) == 1 else None


def mount_error(error: str, trials_dir: Path) -> str:
    """Name the mount problem when the task container's writes never reached the trial directory.

    Harbor bind-mounts each trial's ``verifier`` directory into the Docker task container, and the verifier's
    output lands in ``test-stdout.txt`` there before any reward. No reward and no such file on this host means
    Docker wrote into a directory its VM does not share with the host, not that the verifier failed.
    """
    if not error.startswith("No reward file found") or any(trials_dir.rglob("verifier/test-stdout.txt")):
        return error
    return (
        f"{error}. Docker wrote nothing into {trials_dir}, which it bind-mounted into the task container: the "
        "Docker VM does not share that path with this host. Share it with the VM, or use a path it shares "
        "(colima and Docker Desktop share the home directory by default)"
    )


def run_trial(
    task: str,
    agent: Mapping[str, object],
    *,
    trials_path: Path,
    environment: str,
    runner_name: str,
    extra_instruction_paths: Sequence[Path] = (),
) -> TrialResult:
    """Run ``task`` with the Harbor ``agent`` spec in a ``docker`` or ``e2b`` environment, its trial under the
    existing ``trials_path``. ``runner_name`` names the runner in the error a missing reef-eval raises."""
    # Lazy: reef-eval pulls Harbor's dependency tree, which the render path
    # and its tests must never need. Harbor requires Python 3.12, above Reef's
    # own floor, so the extra carries that marker and can be absent on a
    # supported interpreter; say so rather than raising a bare ImportError.
    try:
        from reef_eval import Lab
    except ImportError as exc:
        raise HarborTrialError(
            f"the {runner_name} runner needs reef-eval, a dependency of reef-infra; reinstall reef-infra"
        ) from exc

    names_before_run = {path.name for path in trials_path.iterdir()}
    row = asyncio.run(
        Lab(trials_path).run(
            task,
            dict(agent),
            extra_instruction_paths=list(extra_instruction_paths),
            environment={"type": environment},
        )
    )
    error = str((row.tags or {}).get("error") or "")
    if environment == "docker":
        error = mount_error(error, trials_path)
    return TrialResult(dict(row.rewards or {}), error, own_trial(trials_path, names_before_run, row.uri))


@dataclass(frozen=True)
class InfrastructureMarker:
    """A failure of the verifier's machinery: ``key`` of the JSON file ``file_name`` in the trial names one of
    ``values``."""

    file_name: str
    key: str
    values: tuple[str, ...]


def markers_from(entries: object, source: str) -> tuple[InfrastructureMarker, ...]:
    """``entries`` as markers; a HarborTrialError naming ``source`` when it is not ``[{"file_name", "key",
    "values"}]`` with a plain file name, a key, and a non-empty list of names."""
    shape = 'a JSON list of {"file_name": name, "key": key, "values": [names]}'
    if not isinstance(entries, list):
        raise HarborTrialError(f"{source} must be {shape}")
    markers = []
    for entry in entries:
        file_name = entry.get("file_name") if isinstance(entry, dict) else None
        key = entry.get("key") if isinstance(entry, dict) else None
        values = entry.get("values") if isinstance(entry, dict) else None
        if (
            not isinstance(file_name, str)
            or PurePosixPath(file_name).name != file_name
            or set(file_name) & set("*?[]")
            or not isinstance(key, str)
            or not key
            or not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise HarborTrialError(f"{source} must be {shape}, not {entry!r}")
        markers.append(InfrastructureMarker(file_name, key, tuple(values)))
    return tuple(markers)


def infrastructure_markers(environ: Mapping[str, str]) -> tuple[InfrastructureMarker, ...]:
    """The markers ``REEF_HARBOR_INFRASTRUCTURE_MARKERS`` lists as ``[{"file_name", "key", "values"}]``; none when it
    is unset, and a HarborTrialError naming the variable when it is not that list."""
    text = environ.get(INFRASTRUCTURE_MARKERS_ENV)
    if not text:
        return ()
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HarborTrialError(f"{INFRASTRUCTURE_MARKERS_ENV} must be a JSON list: {exc}") from exc
    return markers_from(entries, INFRASTRUCTURE_MARKERS_ENV)


def markers_text(markers: Sequence[InfrastructureMarker]) -> str:
    """``markers`` as the ``REEF_HARBOR_INFRASTRUCTURE_MARKERS`` value ``infrastructure_markers`` reads back."""
    entries = [{"file_name": marker.file_name, "key": marker.key, "values": list(marker.values)} for marker in markers]
    return json.dumps(entries)


def infrastructure_error(verifier_path: Path | None, markers: Sequence[InfrastructureMarker]) -> str | None:
    """The first marker the trial's verifier directory hits, as the error its row records; None when there is none.

    Each marker reads the first file under ``verifier_path`` named its ``file_name``; the value at its ``key`` hits
    when it is a mapping with one of the ``values`` as a key, a list holding one, or one of them as a string. A file
    that is missing or not JSON (a torn write) is no hit."""
    if verifier_path is None:
        return None
    for marker in markers:
        found = sorted(verifier_path.rglob(marker.file_name))
        if not found:
            continue
        try:
            data = json.loads(found[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        value = data.get(marker.key) if isinstance(data, dict) else None
        if isinstance(value, (dict, list)):
            names = [str(name) for name in value]
        elif isinstance(value, str):
            names = [value]
        else:
            names = []
        hit = next((name for name in marker.values if name in names), None)
        if hit is not None:
            return f"infrastructure failure: {marker.file_name} {marker.key} names {hit}"
    return None
