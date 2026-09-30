"""Run one Harbor trial through reef-eval on behalf of a runner Reef ships, and find what the trial wrote.

reef-eval is imported only inside ``run_trial`` and nothing on the render path imports this module, so the adapter
registry stays cheap and a deployment without Harbor can still load every descriptor.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.core.errors import ReefError


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
