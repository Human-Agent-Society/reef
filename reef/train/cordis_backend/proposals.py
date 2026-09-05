"""The proposal inbox: agent proposals a scenario holds until the next evolve step takes one.

A proposal arrives through ``POST /reef/harness/proposals`` once the route
admitted it against the head release's entries, and waits as one JSON file
directly under the scenario's inbox directory. ``prepare_step`` claims the
oldest by renaming it into ``claimed/`` before it applies anything, so a
crash between the claim and the settlement never applies it twice; admission
then runs again against the step's entries, a refusal moves the file to
``refused/`` with the reason, and the gate's verdict moves it to
``settled/`` with the verdict appended. Plain files, so a reader can inspect
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

CLAIMED_DIR = "claimed"
REFUSED_DIR = "refused"
SETTLED_DIR = "settled"


@dataclass(frozen=True)
class Proposal:
    """One claimed proposal: the mutations the step applies and what the commit metrics name."""

    id: str
    mutations: tuple[Mapping[str, Any], ...]
    reason: str
    session: str
    release_id: str


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ProposalInbox:
    """A directory of pending proposals, one JSON file each, with ``claimed/``, ``refused/`` and ``settled/`` beside them."""

    def __init__(self, directory: Path, max_pending: int) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending_proposals must be an integer of at least 1")
        self.directory = Path(directory)
        self.max_pending = max_pending
        # The route thread submits while the training thread claims; the count and the write go together.
        self._lock = threading.Lock()

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
        """Store an admitted proposal under ``proposal_id``; the refusal when ``max_pending`` already wait."""
        with self._lock:
            if len(self.pending()) >= self.max_pending:
                return "inbox full"
            self.directory.mkdir(parents=True, exist_ok=True)
            _write(
                self.directory / f"{proposal_id}.json",
                {**body, "proposal_id": proposal_id, "received_at": time.time()},
            )
        return None

    def claim(self) -> Proposal | None:
        """Move the oldest pending proposal into ``claimed/`` and return it; None when nothing waits."""
        with self._lock:
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
                    mutations=tuple(dict(mutation) for mutation in data.get("mutations") or ()),
                    reason=str(data.get("reason", "")),
                    session=str(data.get("session", "")),
                    release_id=str(data.get("release_id", "")),
                )
        return None

    def refuse(self, proposal_id: str, reason: str) -> None:
        """A claimed proposal the step's admission refused: into ``refused/`` with the reason."""
        self._move(proposal_id, REFUSED_DIR, {"refused": reason})

    def settle(self, proposal_id: str, verdict: Mapping[str, Any]) -> None:
        """A claimed proposal the gate settled: into ``settled/`` with the verdict."""
        self._move(proposal_id, SETTLED_DIR, {"verdict": dict(verdict)})

    def _move(self, proposal_id: str, subdir: str, extra: Mapping[str, Any]) -> None:
        source = self.directory / CLAIMED_DIR / f"{proposal_id}.json"
        target = self.directory / subdir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        _write(target, {**json.loads(source.read_text(encoding="utf-8")), **extra})
        source.unlink()


__all__ = ["CLAIMED_DIR", "REFUSED_DIR", "SETTLED_DIR", "Proposal", "ProposalInbox"]
