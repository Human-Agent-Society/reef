"""A scenario's committed releases as files: the head a harness backend builds on, and a release by id.

The scenarios are real: a dispatcher over a pi harness recipe whose base
release is the seed, steps committed through the scenario, and rollbacks and
promotes through the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.harness_service import harness_repository, run_step
from tests.support.harness_steps import COMPONENT, PI_SEED, changed_rules, harness_recipe

from reef.artifact import ArtifactNotFound
from reef.dispatcher import Dispatcher
from reef.harness.tree import TreePatch
from reef.scenario.release_files import ScenarioReleaseFiles
from reef.scenario.scenario import Scenario
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.backend import ReleaseFiles

MARKER = TreePatch(changed_rules("marker rules\n"))
MARKED = PI_SEED.apply(MARKER)


def harness_scenario(tmp_path: Path, propose, *, durable: bool = True, **fields) -> tuple[Dispatcher, Scenario]:
    dispatcher = Dispatcher(
        harness_recipe(tmp_path, propose, **fields),
        harness_repository(tmp_path, PI_SEED.files),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record" if durable else None),
    )
    scenario = dispatcher.get_or_create_scenario("demo")
    assert scenario is not None
    return dispatcher, scenario


def release_files(scenario: Scenario) -> ScenarioReleaseFiles:
    return ScenarioReleaseFiles(
        name=scenario.name, store=scenario.store, repository=scenario.repository, surface=scenario.surface
    )


def test_the_head_is_the_latest_publish_and_a_step_that_published_nothing_keeps_it(tmp_path: Path) -> None:
    answers = iter((MARKER, None))
    dispatcher, scenario = harness_scenario(tmp_path, lambda context: next(answers))
    releases = release_files(scenario)
    base = scenario.repository.base_artifact.release_id
    created = scenario.current_artifact_ref().release_id
    try:
        assert isinstance(scenario.trainer.candidate_backend.releases.releases, ScenarioReleaseFiles)
        # Before a publish the head is the release the scenario was created on, the one the catalog lists.
        assert releases.head(COMPONENT) == ReleaseFiles(created, dict(PI_SEED.files))
        assert releases.release(created, COMPONENT).files == PI_SEED.files
        run_step(scenario, "1")
        published = scenario.current_artifact_ref().release_id
        assert published != base
        assert releases.head(COMPONENT) == ReleaseFiles(published, dict(MARKED.files))
        assert run_step(scenario, "2").metrics["skipped"] == "no proposal"
        assert releases.head(COMPONENT).release_id == published
        # Every committed release reads by id, the base included; an id the scenario never committed is not found.
        assert releases.release(base, COMPONENT).files == PI_SEED.files
        assert releases.release(published, COMPONENT).files == MARKED.files
        with pytest.raises(ArtifactNotFound, match="has no release 'nope'"):
            releases.release("nope", COMPONENT)
    finally:
        dispatcher.close()


def test_a_pending_publish_is_the_head_the_next_step_builds_on(tmp_path: Path) -> None:
    trees = []
    dispatcher, scenario = harness_scenario(
        tmp_path, lambda context: trees.append(context.tree) or MARKER, publish="review"
    )
    created = scenario.current_artifact_ref().release_id
    try:
        assert run_step(scenario, "1").pending is True
        # The pending release is not served, but it is the head: under review, steps build on it.
        assert scenario.repository.require_current_artifact().release_id == created
        pending = release_files(scenario).head(COMPONENT)
        assert pending.release_id != created and pending.files == MARKED.files
        # Step 2 proposes on the pending tree, where the same patch changes nothing, and once more on the retry.
        assert run_step(scenario, "2").metrics["skipped"] == "the patch changes nothing"
        assert trees == [PI_SEED, MARKED, MARKED]
    finally:
        dispatcher.close()


def test_a_promote_and_a_rollback_move_the_head(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda context: MARKER, publish="review")
    releases = release_files(scenario)
    created = scenario.current_artifact_ref().release_id
    try:
        run_step(scenario, "1")
        pending = releases.head(COMPONENT).release_id
        promoted = dispatcher.promote("demo", pending).release_id
        assert promoted not in (created, pending)
        assert releases.head(COMPONENT) == ReleaseFiles(promoted, dict(MARKED.files))
        rolled_back = dispatcher.rollback("demo", created).release_id
        assert releases.head(COMPONENT) == ReleaseFiles(rolled_back, dict(PI_SEED.files))
    finally:
        dispatcher.close()


def test_without_a_durable_history_the_head_is_the_served_release(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda context: MARKER, durable=False)
    releases = release_files(scenario)
    try:
        assert not scenario.store.durable
        run_step(scenario, "1")
        served = scenario.repository.require_current_artifact().release_id
        assert releases.head(COMPONENT) == ReleaseFiles(served, dict(MARKED.files))
        assert releases.release(served, COMPONENT).files == MARKED.files
    finally:
        dispatcher.close()
