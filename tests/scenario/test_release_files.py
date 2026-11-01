"""A scenario's committed releases as files: the head a harness backend builds on, and a release by id.

The scenarios are real: a dispatcher over a pi harness recipe whose base
release is the seed, steps committed through the scenario, and activations and
training starts through the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.harness_service import harness_repository, run_step
from tests.support.harness_steps import COMPONENT, PI_SEED, changed_rules, harness_recipe

from reef.artifact import Artifact, ArtifactNotFound
from reef.core.plugins import PluginReference
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import NeverActivate
from reef.harness.tree import TreePatch
from reef.scenario.release_files import ScenarioReleaseFiles
from reef.scenario.scenario import Scenario
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.backend import ReleaseFiles

MARKER = TreePatch(changed_rules("marker rules\n"))
MARKED = PI_SEED.apply(MARKER)


def harness_scenario(tmp_path: Path, method, *, durable: bool = True, **fields) -> tuple[Dispatcher, Scenario]:
    dispatcher = Dispatcher(
        harness_recipe(tmp_path, method, **fields),
        harness_repository(tmp_path),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record" if durable else None),
        # The tests commit each step themselves (``run_step``), so no local cycle commits the same job.
        hold_local_cycles=True,
        coordination=memory_coordination(),
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
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: next(answers))
    releases = release_files(scenario)
    created = scenario.current_artifact_ref().release_id
    try:
        # The release the scenario was created on is its root release.
        assert scenario.repository.start_artifact.release_id == created
        assert isinstance(scenario.trainer.job_loop_required().releases.releases, ScenarioReleaseFiles)
        # Before a publish the head is the release the scenario was created on, the one the catalog lists.
        assert releases.head(COMPONENT) == ReleaseFiles(created, dict(PI_SEED.files))
        assert releases.release(created, COMPONENT).files == PI_SEED.files
        run_step(scenario, "1")
        published = scenario.current_artifact_ref().release_id
        assert published != created
        assert releases.head(COMPONENT) == ReleaseFiles(published, dict(MARKED.files))
        assert run_step(scenario, "2").metrics["skipped"] == "the method left the head tree unchanged"
        assert releases.head(COMPONENT).release_id == published
        # Every committed release reads by id, the root included; an id the scenario never committed is not found.
        assert releases.release(created, COMPONENT).files == PI_SEED.files
        assert releases.release(published, COMPONENT).files == MARKED.files
        with pytest.raises(ArtifactNotFound, match="has no release 'nope'"):
            releases.release("nope", COMPONENT)
    finally:
        dispatcher.close()


def test_an_inactive_release_is_the_head_the_next_step_builds_on(tmp_path: Path) -> None:
    trees = []
    dispatcher, scenario = harness_scenario(
        tmp_path,
        lambda workspace, services: trees.append(workspace.tree()) or MARKER,
        activation=PluginReference(NeverActivate, {}),
    )
    created = scenario.current_artifact_ref().release_id
    try:
        run_step(scenario, "1")
        # The activation rule kept the release inactive, but it is the head: steps build on what training made.
        assert scenario.repository.require_current_artifact().release_id == created
        inactive = release_files(scenario).head(COMPONENT)
        assert inactive.release_id != created and inactive.files == MARKED.files
        # Step 2 starts from the inactive tree, where the same patch changes nothing.
        assert run_step(scenario, "2").metrics["skipped"] == "the method left the head tree unchanged"
        assert trees == [PI_SEED, MARKED]
    finally:
        dispatcher.close()


def test_a_training_start_moves_the_head_and_an_activation_does_not(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(
        tmp_path, lambda workspace, services: MARKER, activation=PluginReference(NeverActivate, {})
    )
    releases = release_files(scenario)
    created = scenario.current_artifact_ref().release_id
    seed_artifact_id = scenario.artifact_status()[COMPONENT]["training_start"]
    try:
        run_step(scenario, "1")
        made = releases.head(COMPONENT).release_id
        assert seed_artifact_id is not None
        dispatcher.train_from("demo", seed_artifact_id)
        assert releases.head(COMPONENT) == ReleaseFiles(created, dict(PI_SEED.files))
        dispatcher.activate("demo", made)
        assert scenario.repository.require_current_artifact().release_id == made
        assert releases.head(COMPONENT) == ReleaseFiles(created, dict(PI_SEED.files))
    finally:
        dispatcher.close()


def test_a_sync_moves_the_head_to_the_persons_tree(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: MARKER)
    releases = release_files(scenario)
    person = PI_SEED.apply(TreePatch(changed_rules("person rules\n")))
    person.write(tmp_path / "person")
    try:
        run_step(scenario, "1")
        active = scenario.current_artifact_ref().release_id
        synced = dispatcher.sync_release(
            "demo", Artifact.local(tmp_path / "person"), component=COMPONENT, expected_release_id=active, metrics={}
        )
        # The synced release serves and is the training start, so the next step builds on the person's tree.
        assert releases.head(COMPONENT) == ReleaseFiles(synced.release_id, dict(person.files))
    finally:
        dispatcher.close()


def test_without_a_durable_history_the_head_is_the_served_release(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: MARKER, durable=False)
    releases = release_files(scenario)
    try:
        assert not scenario.store.durable
        run_step(scenario, "1")
        served = scenario.repository.require_current_artifact().release_id
        assert releases.head(COMPONENT) == ReleaseFiles(served, dict(MARKED.files))
        assert releases.release(served, COMPONENT).files == MARKED.files
    finally:
        dispatcher.close()
