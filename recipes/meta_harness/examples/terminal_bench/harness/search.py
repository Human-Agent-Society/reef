"""Meta-Harness population search with full-history and greedy controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .composition import CompositionCandidate
from .evaluation import EvaluationResult
from .workspace import EvolutionWorkspace, WorkspaceIntegrityError, WorkspaceProposal

SEARCH_MODES = ("incumbent_only", "full_history")


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: CompositionCandidate, *, split: str, round_index: int) -> EvaluationResult: ...


@dataclass(frozen=True)
class ProposerSession:
    session_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    artifact_dir: str | None = None


class WorkspaceProposer(Protocol):
    def propose(
        self,
        *,
        surface: Path,
        prompt: str,
        session_dir: Path,
        round_index: int,
    ) -> ProposerSession: ...


@dataclass(frozen=True)
class CandidateRecord:
    candidate: CompositionCandidate
    round_index: int
    alias: str
    train_score: float
    dev_score: float
    outcome: str
    proposal_id: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_jsonable(),
            "candidate_hash": self.candidate.content_hash,
            "round": self.round_index,
            "alias": self.alias,
            "train_score": self.train_score,
            "dev_score": self.dev_score,
            "outcome": self.outcome,
            "proposal_id": self.proposal_id,
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> CandidateRecord:
        candidate_value = value.get("candidate")
        if not isinstance(candidate_value, Mapping) or not isinstance(candidate_value.get("composition"), Mapping):
            raise ValueError("search state contains an invalid candidate")
        candidate = CompositionCandidate.from_value(
            candidate_value["composition"],
            parent_hashes=tuple(candidate_value.get("parent_hashes") or ()),
            metadata=candidate_value.get("metadata") if isinstance(candidate_value.get("metadata"), Mapping) else {},
        )
        expected_hash = value.get("candidate_hash")
        if expected_hash != candidate.content_hash:
            raise ValueError("search state candidate hash does not match its composition")
        return cls(
            candidate=candidate,
            round_index=int(value["round"]),
            alias=str(value["alias"]),
            train_score=float(value["train_score"]),
            dev_score=float(value["dev_score"]),
            outcome=str(value["outcome"]),
            proposal_id=str(value["proposal_id"]) if value.get("proposal_id") else None,
        )


class Population:
    """Every unique source remains selectable; only the incumbent is promoted."""

    def __init__(self, records: Sequence[CandidateRecord] = (), *, incumbent_hash: str | None = None) -> None:
        self.records = list(records)
        self._by_hash = {record.candidate.content_hash: record for record in self.records}
        if len(self._by_hash) != len(self.records):
            raise ValueError("population records must contain unique candidate sources")
        self.incumbent_hash = incumbent_hash
        if self.records and incumbent_hash not in self._by_hash:
            raise ValueError("population incumbent is missing")

    @property
    def incumbent(self) -> CandidateRecord:
        if self.incumbent_hash is None:
            raise ValueError("population has no incumbent")
        return self._by_hash[self.incumbent_hash]

    def add(self, record: CandidateRecord) -> tuple[CandidateRecord, bool, bool]:
        existing = self._by_hash.get(record.candidate.content_hash)
        if existing is not None:
            return existing, False, False
        previous_score = self.incumbent.dev_score if self.incumbent_hash else float("-inf")
        moved = record.dev_score > previous_score
        outcome = "selected" if moved else "retained"
        normalized = CandidateRecord(
            candidate=record.candidate,
            round_index=record.round_index,
            alias=record.alias,
            train_score=record.train_score,
            dev_score=record.dev_score,
            outcome=outcome,
            proposal_id=record.proposal_id,
        )
        self.records.append(normalized)
        self._by_hash[normalized.candidate.content_hash] = normalized
        if moved:
            self.incumbent_hash = normalized.candidate.content_hash
        return normalized, True, moved

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "incumbent_hash": self.incumbent_hash,
            "records": [record.to_jsonable() for record in self.records],
        }

    @classmethod
    def from_jsonable(cls, value: Mapping[str, Any]) -> Population:
        rows = value.get("records")
        if not isinstance(rows, list):
            raise ValueError("search state population records must be a list")
        return cls(
            [CandidateRecord.from_jsonable(row) for row in rows if isinstance(row, Mapping)],
            incumbent_hash=str(value["incumbent_hash"]) if value.get("incumbent_hash") else None,
        )

    def catalog_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "candidate_hash": record.candidate.content_hash,
                "alias": record.alias,
                "round": record.round_index,
                "train_score": record.train_score,
                "dev_score": record.dev_score,
                "outcome": record.outcome,
                "proposal_id": record.proposal_id,
            }
            for record in self.records
        ]


@dataclass(frozen=True)
class SearchOutcome:
    selected: CompositionCandidate
    population: tuple[CandidateRecord, ...]
    proposer_sessions: tuple[ProposerSession, ...]
    resumed: bool


class MetaHarnessSearch:
    """Outer loop that owns validation, evaluation, promotion, and restart."""

    def __init__(
        self,
        *,
        workspace: EvolutionWorkspace,
        evaluator: CandidateEvaluator,
        proposer: WorkspaceProposer,
        mode: str,
        rounds: int,
    ) -> None:
        if mode not in SEARCH_MODES:
            raise ValueError(f"search mode must be one of {SEARCH_MODES}")
        if rounds < 1:
            raise ValueError("search rounds must be positive")
        self.workspace = workspace
        self.evaluator = evaluator
        self.proposer = proposer
        self.mode = mode
        self.rounds = rounds

    def run(self, genesis: CompositionCandidate) -> SearchOutcome:
        state = self.workspace.read_search_state()
        resumed = state is not None
        if state is None:
            population = self._initialize(genesis)
            sessions: list[ProposerSession] = []
            completed_rounds = 0
        else:
            completed_rounds = int(state.get("completed_rounds", 0))
            if state.get("mode") != self.mode or self.rounds < completed_rounds:
                raise ValueError("existing search state does not match this run configuration")
            population_value = state.get("population")
            if not isinstance(population_value, Mapping):
                raise ValueError("existing search state has no population")
            population = Population.from_jsonable(population_value)
            sessions = [_session_from_json(row) for row in state.get("proposer_sessions", [])]

        for round_index in range(completed_rounds + 1, self.rounds + 1):
            self._run_round(population, sessions, round_index)
            self._persist(population, sessions, completed_rounds=round_index)
        return SearchOutcome(
            selected=population.incumbent.candidate,
            population=tuple(population.records),
            proposer_sessions=tuple(sessions),
            resumed=resumed,
        )

    def _initialize(self, genesis: CompositionCandidate) -> Population:
        train = self.evaluator.evaluate(genesis, split="train", round_index=0)
        dev = self.evaluator.evaluate(genesis, split="dev", round_index=0)
        self.workspace.record_candidate(genesis, train, round_index=0, role="genesis")
        self.workspace.record_candidate(genesis, dev, round_index=0, role="genesis")
        population = Population()
        population.add(
            CandidateRecord(
                candidate=genesis,
                round_index=0,
                alias="genesis",
                train_score=train.score,
                dev_score=dev.score,
                outcome="selected",
            )
        )
        self._persist(population, [], completed_rounds=0)
        return population

    def _run_round(self, population: Population, sessions: list[ProposerSession], round_index: int) -> None:
        resumable = self.workspace.resumable_proposal(round_index)
        if resumable is not None:
            self._validate_parent(resumable, population)
            self.workspace.transition_round(
                round_index,
                "evaluating",
                proposal_id=resumable.proposal_id,
                resumed=True,
            )
            self._evaluate_round(resumable, population, sessions, round_index)
            return

        state = self.workspace.round_state(round_index)
        if state in {"proposed", "validating"}:
            self._resume_validation(population, sessions, round_index, state)
            return
        if state == "invalid":
            return
        if state in {"proposing", "proposer_failed"}:
            sessions[:] = sessions[: round_index - 1]
            self.workspace.restart_proposer(round_index)
            state = "created"
        if state is None:
            self.workspace.begin_round(round_index)
        elif state != "created":
            raise WorkspaceIntegrityError(f"round {round_index} cannot resume from state {state!r}")
        self.workspace.transition_round(round_index, "proposing")
        surface = self._surface(population, round_index)
        before = self.workspace.readonly_snapshot(writable_round=round_index)
        surface_before = self.workspace.proposal_surface_snapshot(surface, writable_round=round_index)
        try:
            session = self.proposer.propose(
                surface=surface,
                prompt=self._prompt(surface, round_index),
                session_dir=self.workspace.session_dir(round_index),
                round_index=round_index,
            )
            sessions.append(session)
            self.workspace.assert_readonly_unchanged(before, writable_round=round_index)
            self.workspace.assert_proposal_surface_unchanged(
                surface,
                surface_before,
                writable_round=round_index,
            )
            self._persist(population, sessions, completed_rounds=round_index - 1)
            self.workspace.transition_round(round_index, "proposed", session_id=session.session_id)
        except WorkspaceIntegrityError as exc:
            self.workspace.record_attempt(
                {
                    "proposal_id": f"round-{round_index:04d}:proposer",
                    "round": round_index,
                    "status": "integrity_failed",
                    "error": str(exc),
                }
            )
            self.workspace.transition_round(
                round_index,
                "proposer_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        except Exception as exc:
            self.workspace.record_attempt(
                {
                    "proposal_id": f"round-{round_index:04d}:proposer",
                    "round": round_index,
                    "status": f"proposer_failed:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            self.workspace.transition_round(
                round_index,
                "proposer_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        self._resume_validation(population, sessions, round_index, "proposed")

    def _resume_validation(
        self,
        population: Population,
        sessions: list[ProposerSession],
        round_index: int,
        state: str,
    ) -> None:
        surface = self._surface(population, round_index)
        if state == "proposed":
            self.workspace.transition_round(round_index, "validating")
        try:
            proposals = self.workspace.collect_proposals(round_index, surface=surface)
            if len(proposals) != 1:
                raise ValueError("the pinned reproduction requires exactly one candidate per round")
            proposal = proposals[0]
            self._validate_parent(proposal, population)
        except Exception as exc:
            self.workspace.record_attempt(
                {
                    "proposal_id": f"round-{round_index:04d}:invalid",
                    "round": round_index,
                    "status": f"invalid:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            self.workspace.transition_round(round_index, "invalid", error_type=type(exc).__name__, error=str(exc))
            return

        self._persist(population, sessions, completed_rounds=round_index - 1)
        self.workspace.transition_round(round_index, "evaluating", proposal_id=proposal.proposal_id)
        self._evaluate_round(proposal, population, sessions, round_index)

    def _evaluate_round(
        self,
        proposal: WorkspaceProposal,
        population: Population,
        sessions: list[ProposerSession],
        round_index: int,
    ) -> None:
        try:
            self._evaluate_proposal(proposal, population, round_index)
        except Exception as exc:
            self.workspace.record_attempt(
                {
                    "proposal_id": proposal.proposal_id,
                    "round": round_index,
                    "candidate_hash": proposal.candidate.content_hash,
                    "status": f"failed:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            self.workspace.transition_round(round_index, "failed", error_type=type(exc).__name__, error=str(exc))
            raise
        self.workspace.write_population(population.catalog_rows(), population.incumbent.candidate.content_hash)
        self._persist(population, sessions, completed_rounds=round_index)
        self.workspace.transition_round(round_index, "committed", incumbent_hash=population.incumbent_hash)

    def _evaluate_proposal(self, proposal: WorkspaceProposal, population: Population, round_index: int) -> None:
        train = self.evaluator.evaluate(proposal.candidate, split="train", round_index=round_index)
        dev = self.evaluator.evaluate(proposal.candidate, split="dev", round_index=round_index)
        previous = population.incumbent
        record = CandidateRecord(
            candidate=proposal.candidate,
            round_index=round_index,
            alias=proposal.name,
            train_score=train.score,
            dev_score=dev.score,
            outcome="retained",
            proposal_id=proposal.proposal_id,
        )
        normalized, is_new, moved = population.add(record)
        status = "selected" if moved else ("rejected" if is_new else "duplicate")
        retention_status = "retained" if is_new else "existing"
        self.workspace.record_candidate(
            proposal.candidate,
            train,
            round_index=round_index,
            role="candidate",
            proposal={
                "proposal_id": proposal.proposal_id,
                "name": proposal.name,
                "hypothesis": proposal.hypothesis,
                "changes": proposal.changes,
                "status": status,
                "retention_status": retention_status,
            },
        )
        self.workspace.record_candidate(
            proposal.candidate,
            dev,
            round_index=round_index,
            role="candidate",
            proposal={
                "proposal_id": proposal.proposal_id,
                "status": status,
                "retention_status": retention_status,
            },
        )
        self.workspace.record_attempt(
            {
                "proposal_id": proposal.proposal_id,
                "round": round_index,
                "candidate_hash": normalized.candidate.content_hash,
                "parent_hashes": list(proposal.candidate.parent_hashes),
                "hypothesis": proposal.hypothesis,
                "changes": proposal.changes,
                "status": status,
                "retention_status": retention_status,
                "train_score": train.score,
                "dev_score": dev.score,
                "incumbent_before": previous.candidate.content_hash,
                "incumbent_score_before": previous.dev_score,
                "incumbent_after": population.incumbent.candidate.content_hash,
            }
        )

    def _surface(self, population: Population, round_index: int) -> Path:
        if self.mode == "full_history":
            return self.workspace.root
        return self.workspace.build_incumbent_surface(
            round_index,
            population.incumbent.candidate,
            population.catalog_rows(),
        )

    def _validate_parent(self, proposal: WorkspaceProposal, population: Population) -> None:
        known = {record.candidate.content_hash for record in population.records}
        if not proposal.candidate.parent_hashes or not set(proposal.candidate.parent_hashes) <= known:
            raise ValueError("proposal parent selection is not in the retained population")
        if self.mode == "incumbent_only" and proposal.candidate.parent_hashes != (
            population.incumbent.candidate.content_hash,
        ):
            raise ValueError("incumbent-only proposals must branch solely from the current winner")

    def _persist(self, population: Population, sessions: Sequence[ProposerSession], *, completed_rounds: int) -> None:
        self.workspace.write_population(population.catalog_rows(), population.incumbent.candidate.content_hash)
        self.workspace.write_search_state(
            {
                "schema_version": 1,
                "mode": self.mode,
                "round_limit": self.rounds,
                "completed_rounds": completed_rounds,
                "population": population.to_jsonable(),
                "proposer_sessions": [session.__dict__ for session in sessions],
            }
        )

    def _prompt(self, surface: Path, round_index: int) -> str:
        history = (
            "Inspect the complete retained population and read candidate or train-trajectory content under "
            "history/; you may branch from any candidate."
            if self.mode == "full_history"
            else "Use only the incumbent composition and compact score history shown in this control surface."
        )
        return (
            f"Run Meta-Harness iteration {round_index}. Read {surface / 'WORKSPACE.md'}. {history} "
            "Produce exactly one complete, general-purpose Reef composition and the required pending_eval.json. "
            "Do not run Terminal-Bench or access held-out test tasks; the outer loop owns evaluation."
        )


def _session_from_json(value: Any) -> ProposerSession:
    if not isinstance(value, Mapping):
        raise ValueError("search state contains an invalid proposer session")
    artifact_dir = Path(str(value["artifact_dir"])) if value.get("artifact_dir") else None
    if artifact_dir is not None and (artifact_dir.is_absolute() or ".." in artifact_dir.parts):
        raise ValueError("search state proposer artifact path must stay within the workspace")
    return ProposerSession(
        session_id=str(value["session_id"]),
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        estimated_cost_usd=float(value.get("estimated_cost_usd", 0.0)),
        wall_time_s=float(value.get("wall_time_s", 0.0)),
        artifact_dir=artifact_dir.as_posix() if artifact_dir is not None else None,
    )
