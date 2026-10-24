"""The proposal inbox: patches from clients that a scenario holds until the next automatic step takes one.

A proposal arrives through ``POST /reef/harness/proposals`` once the route
validated its patch against the head tree, and waits as one JSON file
directly under the scenario's inbox directory. ``prepare_step`` claims the
oldest by renaming it into ``claimed/`` before it applies anything, so a
crash between the claim and the settlement never applies it twice. The step
validates the patch again against its own head; a refusal moves the file to
``refused/`` with the reason, and the evaluation's result moves it to
``settled/`` with the result appended. Plain files, so a reader can inspect
every state with ``ls`` and ``cat``.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.harness.tree import TreePatch

CLAIMED_DIR = "claimed"
REFUSED_DIR = "refused"
SETTLED_DIR = "settled"


@dataclass(frozen=True)
class Proposal:
    """One claimed proposal.

    Values: ``id`` is the inbox id (receive time first, so name order is
    age order); ``patch`` is the change the client proposed; ``reason``,
    ``session`` and ``release_id`` are the client's own words, its session
    tag and the release it proposed against, each possibly empty.
    Purpose: ``ProposalInbox.claim`` makes it; the step applies ``patch`` and
    names the rest in its commit metrics.
    """

    id: str
    patch: TreePatch
    reason: str
    session: str
    release_id: str


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` beside ``path`` and rename it over, so a reader never claims a half written file."""
    staging = path.with_name(f".{path.name}.part")
    staging.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(path)


class ProposalInbox:
    """A directory of pending proposals, one JSON file each, with ``claimed/``, ``refused/`` and ``settled/`` beside them."""

    def __init__(self, directory: Path, max_pending: int) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending_proposals must be an integer of at least 1")
        self.directory = Path(directory)
        self.max_pending = max_pending
        # The route thread submits while the training thread claims; the count and the write go together.
        self.lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        """A proposal id that starts with the receive time, so name order is age order."""
        return f"{time.time_ns():020d}-{secrets.token_hex(4)}"

    def pending(self) -> list[str]:
        """The pending proposal ids, oldest first."""
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def submit(self, proposal_id: str, body: Mapping[str, Any]) -> str | None:
        """Store a validated proposal under ``proposal_id``; the refusal when ``max_pending`` already wait."""
        with self.lock:
            if len(self.pending()) >= self.max_pending:
                return "inbox full"
            self.directory.mkdir(parents=True, exist_ok=True)
            write_json(
                self.directory / f"{proposal_id}.json",
                {**body, "proposal_id": proposal_id, "received_at": time.time()},
            )
        return None

    def claim(self) -> Proposal | None:
        """Move the oldest pending proposal into ``claimed/`` and return it; None when nothing waits."""
        with self.lock:
            for proposal_id in self.pending():
                source = self.directory / f"{proposal_id}.json"
                target = self.directory / CLAIMED_DIR / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    source.replace(target)
                except FileNotFoundError:
                    continue
                data = json.loads(target.read_text(encoding="utf-8"))
                return Proposal(
                    id=proposal_id,
                    patch=TreePatch.from_json(data.get("patch") or {}),
                    reason=str(data.get("reason", "")),
                    session=str(data.get("session", "")),
                    release_id=str(data.get("release_id", "")),
                )
        return None

    def refuse(self, proposal_id: str, reason: str) -> None:
        """A claimed proposal the step's validation refused: into ``refused/`` with the reason."""
        self.move(proposal_id, REFUSED_DIR, {"refused": reason})

    def settle(self, proposal_id: str, selection_result: Mapping[str, Any]) -> None:
        """A claimed proposal the evaluation settled: into ``settled/`` with the result.

        Settled again, after the candidate was evaluated once more against a
        newer release, the file carries the latest decision.
        """
        self.move(proposal_id, SETTLED_DIR, {"result": dict(selection_result)})

    def move(self, proposal_id: str, subdir: str, extra: Mapping[str, Any]) -> None:
        """Move a claimed proposal into ``subdir`` with ``extra`` merged into its body."""
        source = self.directory / CLAIMED_DIR / f"{proposal_id}.json"
        target = self.directory / subdir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # Filed before (a second settlement), or an operator moved the claimed file by hand:
            # the earlier filing keeps the proposal's body, and the result still gets filed, not lost.
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                data = {"proposal_id": proposal_id}
        write_json(target, {**data, **extra})
        source.unlink(missing_ok=True)


__all__ = ["CLAIMED_DIR", "REFUSED_DIR", "SETTLED_DIR", "Proposal", "ProposalInbox"]
