"""Shared pieces of the harness service and backend tests: a seeded dispatcher, one step, a stub model.

``harness_dispatcher`` writes the recipe's seed into the repository's
bootstrap directory, as the service's assembly does through
``base_artifact_files``, so a new scenario's base release is the seed tree.
``run_step`` drives one step of a scenario; ``backend_step`` drives one step
of a backend alone, through the phases the trainer runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact import CachedRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import ScoreComparisonPlugin
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.scenario.scenario import Scenario
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.types import JobData, TrainStepResult


def harness_repository(tmp_path: Path, base_files: Mapping[str, str]) -> CachedRepositoryBackendFactory:
    """An in-memory repository whose base release holds ``base_files``."""
    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    for relative, text in base_files.items():
        path = initial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def harness_dispatcher(tmp_path: Path, recipe: Recipe, *, hold_local_cycles: bool = False) -> Dispatcher:
    """A dispatcher over ``recipe`` with a seeded repository and a durable commit history.

    A test that prepares and commits each step itself sets ``hold_local_cycles``, so no local cycle commits the
    same step a second time.
    """
    return Dispatcher(
        recipe,
        harness_repository(tmp_path, recipe.base_artifact_files() or {}),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        hold_local_cycles=hold_local_cycles,
    )


def report_records(scenario: str, suffix: str, *, prompt: str = "q", score: float = 0.0) -> tuple[AgentRecord, ...]:
    """One inference record ``i<suffix>`` and the scored report ``r<suffix>`` on it: one trace sample."""
    return (
        AgentRecord.create(
            scenario=scenario,
            request_type=RequestType.INFERENCE,
            payload={"messages": [{"role": "user", "content": prompt}]},
            agent_record_id=f"i{suffix}",
        ),
        AgentRecord.create(
            scenario=scenario,
            request_type=RequestType.REPORT,
            payload={"score": score, "references": [f"i{suffix}"]},
            agent_record_id=f"r{suffix}",
        ),
    )


def report_once(scenario: Scenario, suffix: str, *, prompt: str = "q", score: float = 0.0) -> None:
    """One trace sample through the record store, without waking the trainer: enough for one step."""
    for record in report_records(scenario.name, suffix, prompt=prompt, score=score):
        scenario.records.append_result(record)


def run_step(scenario: Scenario, suffix: str) -> TrainStepResult:
    """One report, then the step it wakes, committed; the step's result."""
    report_once(scenario, suffix)
    result = scenario.prepare_training_step()
    if result is None:
        raise AssertionError("the report woke no step")
    scenario.commit(result)
    return result


def backend_step(backend: HarnessBackend, data: JobData, state: Mapping[str, Any]) -> TrainStepResult:
    """One step of ``backend`` as the trainer runs it, selected by score comparison; a failure aborts the step."""
    prepared = backend.prepare_step(data, state, 0)
    if prepared.candidate is None:
        return TrainStepResult(prepared.state, prepared.metrics)
    selection = ScoreComparisonPlugin(backend)
    try:
        decision = selection.decide(prepared.candidate, selection.evaluate(prepared.candidate))
        return backend.settle_step(prepared, decision)
    except BaseException:
        backend.abort_step(prepared)
        raise


async def wait_for_rows(scenario: Scenario, count: int, seconds: float = 30.0) -> tuple[dict[str, Any], ...]:
    """The scenario's catalog, newest first, once it holds ``count`` rows; the trainer commits on its own thread."""
    for _ in range(int(seconds / 0.05)):
        rows = scenario.releases()
        if len(rows) >= count:
            return rows
        await asyncio.sleep(0.05)
    raise AssertionError(f"the catalog did not reach {count} rows in {seconds}s: {scenario.releases()}")


class EchoInference(InferenceHandler):
    """An upstream that answers every call with one assistant message."""

    async def inference(self, artifact: Any, path: str, payload: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


__all__ = [
    "EchoInference",
    "backend_step",
    "harness_dispatcher",
    "harness_repository",
    "report_once",
    "report_records",
    "run_step",
    "wait_for_rows",
]
