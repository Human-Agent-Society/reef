"""Durable proposer-visible state for Meta-Harness search."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reef.harness import AdapterDescriptor

from .composition import CompositionCandidate
from .evaluation import SEARCH_SPLITS, EvaluationResult

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROUND_STATES = {
    None: {"created"},
    "created": {"proposing"},
    "proposing": {"created", "proposed", "proposer_failed"},
    "proposer_failed": {"created"},
    "proposed": {"validating"},
    "validating": {"evaluating", "invalid"},
    "evaluating": {"evaluating", "committed", "failed"},
    "failed": {"evaluating"},
}


class WorkspaceIntegrityError(RuntimeError):
    """The proposer modified immutable history or coordinator state."""


@dataclass(frozen=True)
class WorkspaceProposal:
    name: str
    proposal_id: str
    candidate: CompositionCandidate
    hypothesis: str
    changes: str
    source_path: Path

    @property
    def primary_parent_hash(self) -> str | None:
        return self.candidate.parent_hashes[0] if self.candidate.parent_hashes else None


class EvolutionWorkspace:
    """Persistent population, feedback, proposal, and session filesystem."""

    def __init__(self, root: Path, descriptor: AdapterDescriptor) -> None:
        self.root = Path(root).resolve()
        self.descriptor = descriptor
        self.history_dir = self.root / "history"
        self.proposals_dir = self.root / "proposals"
        self.frozen_dir = self.root / "frozen-proposals"
        self.sessions_dir = self.root / "sessions"
        self.rounds_dir = self.root / "rounds"
        for path in (
            self.root,
            self.history_dir,
            self.proposals_dir,
            self.frozen_dir,
            self.sessions_dir,
            self.rounds_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._write_contract_if_missing()

    def record_candidate(
        self,
        candidate: CompositionCandidate,
        result: EvaluationResult,
        *,
        round_index: int,
        role: str,
        proposal: Mapping[str, Any] | None = None,
    ) -> None:
        """Append train/dev evidence without ever admitting held-out test data."""
        if result.split not in SEARCH_SPLITS:
            raise ValueError("test results are sealed and cannot enter the evolution workspace")
        candidate_dir = self.history_dir / candidate.content_hash
        candidate_dir.mkdir(parents=True, exist_ok=True)
        self._write_once(
            candidate_dir / "candidate.json",
            {"content_hash": candidate.content_hash, "composition": candidate.composition},
        )
        rendered = candidate.render(self.descriptor)
        for relative, text in rendered.items():
            self._write_text_once(candidate_dir / "rendered" / relative, text)
        evaluation_path = candidate_dir / "evaluations" / f"round-{round_index:04d}-{result.split}.json"
        evaluation_payload = result.to_jsonable()
        if result.split == "dev":
            evaluation_payload = {
                "split": "dev",
                "score": result.score,
                "trial_count": len(result.trials),
                "usage": evaluation_payload["usage"],
                "estimated_cost_usd": evaluation_payload["estimated_cost_usd"],
                "wall_time_s": evaluation_payload["wall_time_s"],
            }
        self._write_once(evaluation_path, evaluation_payload)
        self._append_jsonl_once(
            self.root / "evolution-summary.jsonl",
            {
                "candidate_hash": candidate.content_hash,
                "parent_hashes": list(candidate.parent_hashes),
                "round": round_index,
                "role": role,
                "split": result.split,
                "score": result.score,
                "proposal": dict(proposal or {}),
            },
            identity=(candidate.content_hash, round_index, result.split),
        )

    def write_population(self, rows: Sequence[Mapping[str, Any]], incumbent_hash: str) -> None:
        known = {str(row.get("candidate_hash")) for row in rows}
        if incumbent_hash not in known:
            raise ValueError("incumbent must be present in the candidate population")
        self._write_json(
            self.root / "candidate-catalog.json",
            {"incumbent_hash": incumbent_hash, "candidates": [dict(row) for row in rows]},
        )

    def write_search_state(self, state: Mapping[str, Any]) -> None:
        self._write_json(self.root / "search-state.json", dict(state))

    def read_search_state(self) -> dict[str, Any] | None:
        path = self.root / "search-state.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("search-state.json must contain an object")
        return value

    def begin_round(self, round_index: int) -> Path:
        round_dir = self.proposals_dir / f"round-{round_index:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        pending = self.root / "pending_eval.json"
        if pending.exists():
            pending.unlink()
        self.transition_round(round_index, "created")
        return round_dir

    def restart_proposer(self, round_index: int) -> Path:
        """Clear only the failed round's declared outputs before a fresh turn."""
        state = self.round_state(round_index)
        if state not in {"proposing", "proposer_failed"}:
            raise ValueError(f"round {round_index} is not retryable from proposer state {state!r}")
        round_dir = self.proposals_dir / f"round-{round_index:04d}"
        if round_dir.exists():
            shutil.rmtree(round_dir)
        round_dir.mkdir(parents=True)
        (self.root / "pending_eval.json").unlink(missing_ok=True)
        self.transition_round(round_index, "created", retry_from=state)
        return round_dir

    def transition_round(self, round_index: int, state: str, **details: Any) -> None:
        path = self.rounds_dir / f"round-{round_index:04d}.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        previous_state = previous.get("state") if previous else None
        if state not in _ROUND_STATES.get(previous_state, set()):
            raise ValueError(f"invalid round transition {previous_state!r} -> {state!r}")
        history = list(previous.get("history", [])) if previous else []
        history.append(
            {
                "sequence": len(history),
                "state": state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details,
            }
        )
        self._write_json(
            path,
            {"schema_version": 1, "round": round_index, "state": state, "history": history},
        )

    def completed_rounds(self) -> tuple[int, ...]:
        completed: list[int] = []
        for path in sorted(self.rounds_dir.glob("round-*.json")):
            state = json.loads(path.read_text()).get("state")
            if state == "committed":
                completed.append(int(path.stem.removeprefix("round-")))
        return tuple(completed)

    def round_state(self, round_index: int) -> str | None:
        path = self.rounds_dir / f"round-{round_index:04d}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value.get("state")) if value.get("state") else None

    def session_dir(self, round_index: int) -> Path:
        path = self.sessions_dir / f"round-{round_index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_incumbent_surface(
        self,
        round_index: int,
        incumbent: CompositionCandidate,
        population: Sequence[Mapping[str, Any]],
    ) -> Path:
        """Materialize the greedy control's summarized, incumbent-only view."""
        surface = self.root / "incumbent-views" / f"round-{round_index:04d}"
        proposal_dir = surface / "proposals" / f"round-{round_index:04d}"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(surface / "incumbent" / "composition.json", incumbent.composition)
        self._write_json(
            surface / "search-summary.json",
            {
                "incumbent_hash": incumbent.content_hash,
                "candidate_count": len(population),
                "score_history": [
                    {
                        "round": row.get("round"),
                        "candidate_hash": row.get("candidate_hash"),
                        "dev_score": row.get("dev_score"),
                        "outcome": row.get("outcome"),
                    }
                    for row in population
                ],
            },
        )
        (surface / "WORKSPACE.md").write_text(
            "# Incumbent-only control\n\n"
            "Use only `incumbent/composition.json` and the compact score history in "
            "`search-summary.json`. Write one complete candidate under the current `proposals/` "
            "round and declare the incumbent hash as its sole base in `pending_eval.json`.\n",
            encoding="utf-8",
        )
        return surface

    def proposal_surface_snapshot(self, surface: Path, *, writable_round: int) -> dict[str, str]:
        """Hash proposer-visible inputs while excluding its declared outputs."""
        surface = Path(surface).resolve()
        writable = (surface / "proposals" / f"round-{writable_round:04d}").resolve()
        writable_session = (surface / "sessions" / f"round-{writable_round:04d}").resolve()
        paths = []
        for path in surface.rglob("*"):
            if not path.is_file() or path.name == "pending_eval.json":
                continue
            try:
                path.resolve().relative_to(writable)
            except ValueError:
                try:
                    path.resolve().relative_to(writable_session)
                except ValueError:
                    paths.append(path)
        return {str(path.relative_to(surface)): _sha256(path) for path in sorted(paths)}

    def assert_proposal_surface_unchanged(
        self,
        surface: Path,
        before: Mapping[str, str],
        *,
        writable_round: int,
    ) -> None:
        after = self.proposal_surface_snapshot(surface, writable_round=writable_round)
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        if changed:
            raise WorkspaceIntegrityError(f"proposer modified read-only surface: {', '.join(changed[:5])}")

    def collect_proposals(self, round_index: int, *, surface: Path | None = None) -> list[WorkspaceProposal]:
        surface = Path(surface or self.root).resolve()
        pending = surface / "pending_eval.json"
        if not pending.is_file():
            raise ValueError("proposer did not write pending_eval.json")
        raw = json.loads(pending.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("iteration") != round_index:
            raise ValueError("pending_eval.json has the wrong or missing iteration")
        items = raw.get("candidates")
        if not isinstance(items, list) or not items:
            raise ValueError("pending_eval.json requires a non-empty candidates list")

        round_dir = (surface / "proposals" / f"round-{round_index:04d}").resolve()
        known = self._candidate_refs()
        proposals: list[WorkspaceProposal] = []
        seen_names: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError("each pending candidate must be an object")
            name = str(item.get("name") or "")
            if not _SAFE_NAME.fullmatch(name) or name in seen_names:
                raise ValueError(f"invalid or duplicate proposal name {name!r}")
            relative = Path(str(item.get("path") or name))
            source_dir = (round_dir / relative).resolve()
            try:
                source_dir.relative_to(round_dir)
            except ValueError as exc:
                raise ValueError(f"proposal path escapes its round: {relative}") from exc
            if any(path.is_symlink() for path in source_dir.rglob("*")):
                raise ValueError(f"proposal {name!r} contains a symlink")
            source_path = source_dir / "composition.json"
            if not source_path.is_file():
                raise ValueError(f"proposal {name!r} has no composition.json")
            parent_refs = item.get("base_candidates", item.get("parent_hashes", []))
            if isinstance(parent_refs, str):
                parent_refs = [parent_refs]
            if not isinstance(parent_refs, list) or not parent_refs:
                raise ValueError(f"proposal {name!r} must select at least one base candidate")
            try:
                parents = tuple(dict.fromkeys(known[str(ref)] for ref in parent_refs))
            except KeyError as exc:
                raise ValueError(f"proposal {name!r} selects unknown base candidate {exc.args[0]!r}") from exc
            hypothesis = str(item.get("hypothesis") or "").strip()
            changes = str(item.get("changes") or "").strip()
            if not hypothesis or not changes:
                raise ValueError(f"proposal {name!r} requires hypothesis and changes")
            candidate = CompositionCandidate.from_path(
                source_path,
                parent_hashes=parents,
                metadata={"proposal_name": name, "round": round_index},
            )
            frozen_root = self.frozen_dir / f"round-{round_index:04d}" / f"{index:03d}-{name}"
            self._write_text_once(frozen_root / "composition.json", candidate.canonical_json + "\n")
            self._write_once(
                frozen_root / "manifest.json",
                {
                    "proposal_id": f"round-{round_index:04d}:{index:03d}:{name}",
                    "name": name,
                    "candidate_hash": candidate.content_hash,
                    "parent_hashes": list(parents),
                    "hypothesis": hypothesis,
                    "changes": changes,
                },
            )
            proposals.append(
                WorkspaceProposal(
                    name=name,
                    proposal_id=f"round-{round_index:04d}:{index:03d}:{name}",
                    candidate=candidate,
                    hypothesis=hypothesis,
                    changes=changes,
                    source_path=frozen_root / "composition.json",
                )
            )
            seen_names.add(name)
        return proposals

    def resumable_proposal(self, round_index: int) -> WorkspaceProposal | None:
        """Load the one immutable proposal whose evaluation did not commit."""
        if self.round_state(round_index) not in {"evaluating", "failed"}:
            return None
        roots = sorted((self.frozen_dir / f"round-{round_index:04d}").glob("*/manifest.json"))
        if not roots:
            return None
        if len(roots) != 1:
            raise WorkspaceIntegrityError("a resumable round must contain exactly one frozen proposal")
        manifest_path = roots[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        composition_path = manifest_path.parent / "composition.json"
        name = str(manifest.get("name") or manifest_path.parent.name.split("-", 1)[-1])
        candidate = CompositionCandidate.from_path(
            composition_path,
            parent_hashes=tuple(str(value) for value in manifest.get("parent_hashes", ())),
            metadata={"proposal_name": name, "round": round_index},
        )
        if candidate.content_hash != manifest.get("candidate_hash"):
            raise WorkspaceIntegrityError("frozen proposal hash does not match its manifest")
        return WorkspaceProposal(
            name=name,
            proposal_id=str(manifest["proposal_id"]),
            candidate=candidate,
            hypothesis=str(manifest["hypothesis"]),
            changes=str(manifest["changes"]),
            source_path=composition_path,
        )

    def record_attempt(self, row: Mapping[str, Any]) -> None:
        proposal_id = str(row.get("proposal_id") or "")
        if not proposal_id:
            raise ValueError("attempt requires proposal_id")
        self._append_jsonl_once(
            self.root / "proposal-events.jsonl",
            {"schema_version": 1, **dict(row)},
            identity=(proposal_id, str(row.get("status") or "unknown")),
            identity_keys=("proposal_id", "status"),
        )

    def readonly_snapshot(self, *, writable_round: int | None = None) -> dict[str, str]:
        paths: list[Path] = []
        for root in (self.history_dir, self.frozen_dir, self.rounds_dir):
            paths.extend(path for path in root.rglob("*") if path.is_file())
        for round_dir in self.proposals_dir.glob("round-*"):
            if writable_round is not None and round_dir.name == f"round-{writable_round:04d}":
                continue
            paths.extend(path for path in round_dir.rglob("*") if path.is_file())
        for session_dir in self.sessions_dir.glob("round-*"):
            if writable_round is not None and session_dir.name == f"round-{writable_round:04d}":
                continue
            paths.extend(path for path in session_dir.rglob("*") if path.is_file())
        for name in (
            "WORKSPACE.md",
            "candidate-catalog.json",
            "evolution-summary.jsonl",
            "proposal-events.jsonl",
            "search-state.json",
        ):
            path = self.root / name
            if path.is_file():
                paths.append(path)
        return {str(path.relative_to(self.root)): _sha256(path) for path in sorted(set(paths))}

    def assert_readonly_unchanged(self, before: Mapping[str, str], *, writable_round: int) -> None:
        after = self.readonly_snapshot(writable_round=writable_round)
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        if changed:
            raise WorkspaceIntegrityError(f"proposer modified read-only state: {', '.join(changed[:5])}")

    def _candidate_refs(self) -> dict[str, str]:
        catalog_path = self.root / "candidate-catalog.json"
        catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else {"candidates": []}
        refs: dict[str, str] = {}
        for row in catalog.get("candidates", []):
            candidate_hash = str(row.get("candidate_hash") or "")
            if not candidate_hash:
                continue
            refs[candidate_hash] = candidate_hash
            for key in ("alias", "proposal_id", "name"):
                if row.get(key):
                    ref = str(row[key])
                    if ref in refs and refs[ref] != candidate_hash:
                        raise ValueError(f"ambiguous candidate reference {ref!r}")
                    refs[ref] = candidate_hash
        return refs

    def _write_contract_if_missing(self) -> None:
        path = self.root / "WORKSPACE.md"
        if path.exists():
            return
        path.write_text(
            "# Meta-Harness evolution workspace\n\n"
            "Read candidate source and train/dev evidence under `history/`, the population in "
            "`candidate-catalog.json`, and prior session logs under `sessions/`. Historical files "
            "are immutable observations. Write each new complete Reef tree to the current "
            "`proposals/round-NNNN/<name>/composition.json`, then declare its name, path, explicit "
            "base candidate reference(s), hypothesis, and changes in `pending_eval.json`. The outer "
            "loop owns validation and evaluation. Test tasks are never available during search.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def _write_once(cls, path: Path, payload: Any) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        cls._write_text_once(path, text)

    @staticmethod
    def _write_text_once(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise WorkspaceIntegrityError(f"immutable workspace file changed: {path}")
            return
        path.write_text(text, encoding="utf-8")

    @classmethod
    def _append_jsonl_once(
        cls,
        path: Path,
        row: Mapping[str, Any],
        *,
        identity: tuple[Any, ...],
        identity_keys: tuple[str, ...] | None = None,
    ) -> None:
        keys = identity_keys or _identity_keys(len(identity))
        if len(keys) != len(identity):
            raise ValueError("JSONL identity keys and values must have equal length")
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                existing = json.loads(line)
                existing_id = tuple(existing.get(key) for key in keys)
                if existing_id == identity:
                    return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _identity_keys(length: int) -> tuple[str, ...]:
    return ("proposal_id",) if length == 1 else ("candidate_hash", "round", "split")[:length]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
