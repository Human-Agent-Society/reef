"""A scenario's committed snapshots as files: the head a harness backend builds on, and a snapshot by id.

The scenarios are real: a dispatcher over a pi harness recipe whose base
snapshot is the seed, steps committed through the scenario, and activations and
training starts through the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.harness_service import harness_repository, run_step
from tests.support.harness_steps import COMPONENT, PI_SEED, changed_rules, harness_recipe

from reef.artifact import Artifact, ArtifactConflict, ArtifactNotFound
from reef.core.plugins import PluginReference
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import NeverActivate
from reef.harness.tree import TreePatch
from reef.scenario.scenario import Scenario
from reef.scenario.snapshot_files import ScenarioArtifactFiles
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.backend import ArtifactTextFiles

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


def snapshot_files(scenario: Scenario) -> ScenarioArtifactFiles:
    return ScenarioArtifactFiles(
        name=scenario.name, store=scenario.store, repository=scenario.repository, components=scenario.components
    )


def test_the_head_is_the_latest_publish_and_a_step_that_published_nothing_keeps_it(tmp_path: Path) -> None:
    answers = iter((MARKER, None))
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: next(answers))
    releases = snapshot_files(scenario)
    created = scenario.artifact_status()[COMPONENT]["active"]
    try:
        # The snapshot the scenario was created on is its root snapshot.
        assert scenario.repository.resolve(scenario.repository.start_artifact).artifact_id == created
        # Before a publish the head is the snapshot the scenario was created on, the one the catalog lists.
        assert releases.head(COMPONENT) == ArtifactTextFiles(created, dict(PI_SEED.files))
        assert releases.artifact(created, COMPONENT).files == PI_SEED.files
        run_step(scenario, "1")
        published = scenario.artifact_status()[COMPONENT]["active"]
        assert published != created
        assert releases.head(COMPONENT) == ArtifactTextFiles(published, dict(MARKED.files))
        assert run_step(scenario, "2").metrics["skipped"] == "the method left the head tree unchanged"
        assert releases.head(COMPONENT).artifact_id == published
        # Every committed snapshot reads by id, the root included; an id the scenario never committed is not found.
        assert releases.artifact(created, COMPONENT).files == PI_SEED.files
        assert releases.artifact(published, COMPONENT).files == MARKED.files
        with pytest.raises(ArtifactNotFound, match="has no artifact 'nope'"):
            releases.artifact("nope", COMPONENT)
    finally:
        dispatcher.close()


def test_an_inactive_release_is_the_head_the_next_step_builds_on(tmp_path: Path) -> None:
    trees = []
    dispatcher, scenario = harness_scenario(
        tmp_path,
        lambda workspace, services: trees.append(workspace.tree()) or MARKER,
        activation=PluginReference(NeverActivate, {}),
    )
    created = scenario.artifact_status()[COMPONENT]["active"]
    try:
        run_step(scenario, "1")
        # The activation rule kept the snapshot inactive, but it is the head: steps build on what training made.
        assert scenario.artifact_status()[COMPONENT]["active"] == created
        inactive = snapshot_files(scenario).head(COMPONENT)
        assert inactive.artifact_id != created and inactive.files == MARKED.files
        # Step 2 starts from the inactive tree, where the same patch changes nothing.
        assert run_step(scenario, "2").metrics["skipped"] == "the method left the head tree unchanged"
        assert trees == [PI_SEED, MARKED]
    finally:
        dispatcher.close()


def test_a_training_start_moves_the_head_and_an_activation_does_not(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(
        tmp_path, lambda workspace, services: MARKER, activation=PluginReference(NeverActivate, {})
    )
    releases = snapshot_files(scenario)
    created = scenario.artifact_status()[COMPONENT]["active"]
    seed_artifact_id = scenario.artifact_status()[COMPONENT]["training_start"]
    try:
        run_step(scenario, "1")
        made = releases.head(COMPONENT).artifact_id
        assert seed_artifact_id is not None
        dispatcher.train_from("demo", seed_artifact_id)
        assert releases.head(COMPONENT) == ArtifactTextFiles(created, dict(PI_SEED.files))
        dispatcher.activate("demo", made)
        assert scenario.artifact_status()[COMPONENT]["active"] == made
        assert releases.head(COMPONENT) == ArtifactTextFiles(created, dict(PI_SEED.files))
    finally:
        dispatcher.close()


def test_a_sync_moves_the_head_to_the_persons_tree(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: MARKER)
    releases = snapshot_files(scenario)
    person = PI_SEED.apply(TreePatch(changed_rules("person rules\n")))
    person.write(tmp_path / "person")
    try:
        run_step(scenario, "1")
        active = scenario.artifact_selection()
        synced = dispatcher.sync_artifact(
            "demo", Artifact.local(tmp_path / "person"), component=COMPONENT, expected_selection=active, metrics={}
        )
        # The synced release serves and is the training start, so the next step builds on the person's tree.
        assert releases.head(COMPONENT) == ArtifactTextFiles(synced.artifact_id, dict(person.files))
        history = scenario.store.history()
        with pytest.raises(ArtifactConflict, match="changed its active artifacts"):
            dispatcher.sync_artifact(
                "demo", Artifact.local(tmp_path / "person"), component=COMPONENT, expected_selection=active, metrics={}
            )
        assert scenario.store.history() == history
        assert scenario.artifact_selection().components[COMPONENT] == synced.artifact_id
    finally:
        dispatcher.close()


def test_without_a_durable_history_the_head_is_the_served_release(tmp_path: Path) -> None:
    dispatcher, scenario = harness_scenario(tmp_path, lambda workspace, services: MARKER, durable=False)
    releases = snapshot_files(scenario)
    try:
        assert not scenario.store.durable
        run_step(scenario, "1")
        served = scenario.artifact_status()[COMPONENT]["active"]
        assert releases.head(COMPONENT) == ArtifactTextFiles(served, dict(MARKED.files))
        assert releases.artifact(served, COMPONENT).files == MARKED.files
    finally:
        dispatcher.close()
