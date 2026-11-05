"""Shared pieces of the harness service and backend tests: a seeded dispatcher, one step, a stub model.

``harness_dispatcher`` is a dispatcher over an in-memory repository, so a
new scenario's root snapshot is the recipe's seed tree (``default_start``).
``seed_root`` is that root for a trainer that a test builds outside a
scenario.
``run_step`` drives one step of a scenario, which is one job of its harness
training component (``drive_job``); ``backend_step`` drives one step of a
backend alone, through the phases the job runs. ``catalog`` is the
scenario's catalog rows, without the commits that only follow a job.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from tests.support.harness_steps import head_of
from tests.support.samples import data_storage
from tests.support.samples_service import make_samples

from reef.artifact import Artifact, CachedRepositoryBackendFactory, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.evaluation import CandidateEvaluationPlugin
from reef.core.snapshot import RootSource
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import ComparePolicy
from reef.inference.interfaces import InferenceHandler
from reef.recipe import Recipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.scenario.scenario import SNAPSHOT_ROW_KEYS, Scenario
from reef.storage.sql_coordination import memory_coordination
from reef.train.component import ScenarioRoot
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.component import HarnessTrainingComponent
from reef.train.job_queue import JobQueueComponent
from reef.train.trainer import Trainer
from reef.train.types import JobData, TrainingCommit


def harness_repository(tmp_path: Path) -> CachedRepositoryBackendFactory:
    """An in-memory repository under ``tmp_path``; a new scenario's root snapshot holds the recipe's seed tree."""
    return InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")


def seed_root(recipe: HarnessEvolutionRecipe) -> ScenarioRoot:
    """The root snapshot of a scenario created from ``recipe``'s default start, as a local tree under the recipe's
    ``evolution.state_dir``."""
    directory = Path(recipe.evolution.state_dir).expanduser() / ".seed-root"
    if not directory.is_dir():
        recipe.seed.write(directory)
    return ScenarioRoot(Artifact.local(directory), RootSource("default", inputs=recipe.default_start().inputs))


def harness_dispatcher(tmp_path: Path, recipe: Recipe, *, hold_local_cycles: bool = False) -> Dispatcher:
    """A dispatcher over ``recipe`` with a seeded repository, a durable commit history and a data store.

    A test that prepares and commits each step itself sets ``hold_local_cycles``, so no local cycle commits the
    same step a second time.
    """
    storage = data_storage(tmp_path / "agent-record")
    return Dispatcher(
        recipe,
        harness_repository(tmp_path),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=storage,
        record_storage=storage.inner,
        data_store=storage.data,
        hold_local_cycles=hold_local_cycles,
        coordination=memory_coordination(),
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
    """One trace sample through the record store and the samples service, without waking the trainer: enough for
    one step."""
    for record in report_records(scenario.name, suffix, prompt=prompt, score=score):
        scenario.records.append_result(record)
    make_samples(scenario)


def drive_job(scenario: Scenario, *, seconds: float = 30.0) -> TrainingCommit:
    """Commit one training job of ``scenario``'s trainer, from its submission to its end; the step's result.

    The step's result is the commit of the job's checkpoint when the step kept a tree, else the commit of the job's
    end. The training service of the dispatcher runs the job meanwhile. ``AssertionError`` when no job starts or
    the job does not end within ``seconds``.
    """
    deadline = time.monotonic() + seconds
    step: TrainingCommit | None = None
    while time.monotonic() < deadline:
        result = scenario.next_commit()
        if result is None:
            time.sleep(0.02)
            continue
        scenario.commit(result)
        event = result.metrics["training_job"]["event"]
        if event == "checkpoint":
            step = result
        elif event != "submitted":
            return result if step is None else step
    raise AssertionError(f"scenario {scenario.name!r} ran no whole training job in {seconds}s")


def checkpoint_result(scenario: Scenario, *, seconds: float = 30.0) -> TrainingCommit:
    """Commit the submission of ``scenario``'s next job, then wait for the job's checkpoint; the result that publishes
    it, not committed yet.

    ``AssertionError`` when the job submits nothing, ends without a checkpoint, or saves none within ``seconds``.
    """
    submitted = scenario.next_commit()
    if submitted is None or submitted.metrics["training_job"]["event"] != "submitted":
        raise AssertionError(f"scenario {scenario.name!r} submitted no job")
    scenario.commit(submitted)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = scenario.next_commit()
        if result is None:
            time.sleep(0.02)
            continue
        if result.metrics["training_job"]["event"] != "checkpoint":
            raise AssertionError(f"the job of scenario {scenario.name!r} ended without a checkpoint")
        return result
    raise AssertionError(f"the job of scenario {scenario.name!r} saved no checkpoint in {seconds}s")


def drive_trainer_job(trainer: Trainer, *, seconds: float = 30.0) -> TrainingCommit | None:
    """Commit one job of ``trainer``, built alone from a harness recipe, from its submission to its end; the step's
    result as ``drive_job`` gives it, or None when the trainer has no job to submit.

    The trainer's harness component runs its jobs in this process. ``AssertionError`` when the job does not end
    within ``seconds``.
    """
    trainer.job_loop_required().component.start_jobs()
    deadline = time.monotonic() + seconds
    step: TrainingCommit | None = None
    submitted = False
    while time.monotonic() < deadline:
        result = trainer.advance_job()
        if result is None:
            if not submitted:
                return None
            time.sleep(0.02)
            continue
        trainer.settle_commit(trainer.prepare_commit(result))
        event = result.metrics["training_job"]["event"]
        if event == "submitted":
            submitted = True
        elif event == "checkpoint":
            step = result
        else:
            return result if step is None else step
    raise AssertionError(f"the job of trainer {trainer.scenario!r} did not end in {seconds}s")


def job_backend(trainer: Trainer, component: str = "harness") -> HarnessBackend:
    """The backend that ``trainer``'s harness component runs the steps of ``component`` on."""
    harness = trainer.job_loop_required().component
    if isinstance(harness, JobQueueComponent):
        # The dispatcher's trainer submits through the jobs queue; its training worker runs the component itself.
        harness = harness.component
    if not isinstance(harness, HarnessTrainingComponent):
        raise TypeError(f"trainer {trainer.scenario!r} submits no harness jobs")
    return harness.backend_for(trainer.scenario, component)


def run_step(scenario: Scenario, suffix: str) -> TrainingCommit:
    """One report, then the job of the step it wakes, committed; the step's result (``drive_job``)."""
    report_once(scenario, suffix)
    return drive_job(scenario)


def catalog(scenario: Scenario) -> list[dict[str, Any]]:
    """The catalog rows of ``scenario``, newest first, as ``GET /reef/harness/artifacts`` lists them."""
    rows = []
    for row in scenario.harness_snapshot_rows():
        snapshot, selected, parent = row["snapshot"], row["artifact_snapshot"], row["parent_snapshot"]
        listed = {key: value for key, value in row.items() if key not in SNAPSHOT_ROW_KEYS}
        listed["release_id"] = selected.revision
        listed["parent_release_id"] = snapshot.parent_revision if parent is None else parent.revision
        rows.append(listed)
    return rows


#: Each backend's scenario step: the steps ``backend_step`` ran on it.
SCENARIO_STEPS: WeakKeyDictionary[HarnessBackend, int] = WeakKeyDictionary()


def backend_step(
    backend: HarnessBackend,
    data: JobData,
    state: Mapping[str, Any],
    policy: CandidateEvaluationPlugin | None = None,
    *,
    scenario_step: int | None = None,
) -> TrainingCommit:
    """One step of ``backend`` on its ``head_of`` head as a job runs it, decided by ``policy`` (a win-count
    comparison by default).

    The steps of one backend number 1, 2, ... as a scenario numbers its commits: a step that raised keeps its
    number for its retry. ``scenario_step`` sets the scenario step the step follows instead, as a retry after a
    lost commit or a resumed scenario sees it. A failure aborts the step.
    """
    scenario_step = SCENARIO_STEPS.get(backend, 0) if scenario_step is None else scenario_step
    prepared = backend.prepare_step(data, state, scenario_step, head=head_of(backend))
    if prepared.candidate is None:
        result = TrainingCommit(prepared.state, prepared.metrics)
    else:
        selection = ComparePolicy(backend) if policy is None else policy
        try:
            decision = selection.decide(prepared.candidate, selection.evaluate(prepared.candidate))
            result = backend.settle_step(prepared, decision)
        except BaseException:
            backend.abort_step(prepared)
            raise
    SCENARIO_STEPS[backend] = scenario_step + 1
    return result


async def wait_for_rows(scenario: Scenario, count: int, seconds: float = 30.0) -> list[dict[str, Any]]:
    """Return the catalog once it holds ``count`` rows and the current training job has finished committing."""
    for _ in range(int(seconds / 0.05)):
        rows = catalog(scenario)
        if len(rows) >= count and not scenario.trainer.job_running and not scenario.trainer.has_pending_commit:
            return rows
        await asyncio.sleep(0.05)
    raise AssertionError(f"the catalog did not reach {count} rows in {seconds}s: {catalog(scenario)}")


class EchoInference(InferenceHandler):
    """An upstream that answers every call with one assistant message."""

    async def inference(self, artifact: Any, path: str, payload: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


__all__ = [
    "EchoInference",
    "backend_step",
    "catalog",
    "checkpoint_result",
    "drive_job",
    "drive_trainer_job",
    "harness_dispatcher",
    "harness_repository",
    "job_backend",
    "report_once",
    "report_records",
    "run_step",
    "wait_for_rows",
]
