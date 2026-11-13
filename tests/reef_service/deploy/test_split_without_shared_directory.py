"""A split deployment: control and the training service share the store, and no directory.

Each test runs control as a dispatcher without the training service, and the training service as the two workers
that it runs for a component. The training component keeps its state, its checkpoints and its work clone of the
artifact repository under one directory, and control keeps its own under another. While control works, the test
records every file that a control thread opens, lists, moves or removes under the training service's directory:
there is none.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.harness_jobs import changed_rules, harness_recipe
from tests.support.harness_service import report_records
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.records import report, token_inference
from tests.support.runtime_stubs import VERSION_FILE, StubFacts, StubInferenceRuntime, StubTrainingComponent
from tests.support.samples import data_storage

from reef.artifact import GitLFSRepositoryBackend, RepositoryBackendFactory
from reef.dispatcher import Dispatcher
from reef.execution.workers import SERVICE_COMPONENTS
from reef.harness.tree import TreePatch
from reef.recipe import Recipe
from reef.recipe.composite import CompositeRecipe
from reef.scenario.live_objects import LiveObjects
from reef.service.request_service import RequestService
from reef.storage.sql_coordination import memory_coordination
from reef.train.component import TrainingComponent
from reef.train.job_queue import CHECKS_QUEUE, TrainingCheckWorker, TrainingJobWorker, scenario_jobs

#: The file operations that the audit hook sees with a path as their first argument.
FILE_EVENTS = frozenset(
    {"open", "os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rename", "os.rmdir", "shutil.copytree"}
    | {"shutil.move", "shutil.rmtree", "shutil.copyfile"}
)
#: The thread names of the training service of these tests: its two workers, its components' job threads, and the
#: executor threads on which a harness job runs its evaluation runs.
TRAINING_THREADS = ("reef-split-training", "reef-harness-job", "reef-stub-job", "reef-executor")


class ControlFileWatch:
    """Records the file operations under ``root`` of every thread that is no thread of the training service,
    while ``watching`` is open. An audit hook cannot be removed, so the watch of an ended test records nothing."""

    def __init__(self, root: Path) -> None:
        self.root = str(root.resolve())
        self.active = False
        self.seen: list[tuple[str, str, str]] = []
        sys.addaudithook(self.audit)

    def audit(self, event: str, arguments: tuple[object, ...]) -> None:
        if not self.active or event not in FILE_EVENTS or not arguments:
            return
        path, thread = arguments[0], threading.current_thread().name
        if not isinstance(path, str | Path) or not str(path).startswith(self.root):
            return
        # A training job can start threads with no name of its own; a control thread is the test's or has a name.
        if thread == "MainThread" or (thread.startswith("reef-") and not thread.startswith(TRAINING_THREADS)):
            self.seen.append((event, str(path), thread))

    @contextmanager
    def watching(self) -> Iterator[None]:
        self.active = True
        try:
            yield
        finally:
            self.active = False


def git_backends(root: Path) -> RepositoryBackendFactory:
    """The artifact store of one service: its own work and cache directories under ``root``, over the repository
    that both services reach."""
    return GitLFSRepositoryBackend.factory(
        root.parent / "artifacts.git", work_dir=root / "artifact-work", cache_dir=root / "artifact-cache"
    )


@contextmanager
def split(
    tmp_path: Path, recipe: Recipe, live: LiveObjects, components: tuple[TrainingComponent, ...]
) -> Iterator[tuple[Dispatcher, ControlFileWatch]]:
    """Control over ``recipe`` with the facts of ``live`` and no training component, and the training service that
    runs ``components``; the watch of the training service's directory ``tmp_path / "training"``."""
    storage = data_storage(tmp_path / "records")
    coordination = memory_coordination()
    control = Dispatcher(
        recipe,
        git_backends(tmp_path / "control"),
        live=replace(live, training_component=None, harness_component=None, data_store=storage.data),
        local_artifact_dir=tmp_path / "control" / "staged",
        scenario_storage=storage,
        record_storage=storage.inner,
        coordination=coordination,
        components=frozenset(SERVICE_COMPONENTS) - {"training"},
    )
    training_backends = git_backends(tmp_path / "training")
    workers = []
    for component in components:
        for worker_type, name in ((TrainingJobWorker, "jobs"), (TrainingCheckWorker, "checks")):
            worker = worker_type(
                component,
                coordination.work_queue,
                control.job_directory_store,
                training_backends,
                worker_id=f"split-training-{name}-{component.name}",
                local_artifact_dir=tmp_path / "training" / "staged",
                poll_seconds=0.02,
            )
            worker.start()
            workers.append(worker)
    watch = ControlFileWatch(tmp_path / "training")
    try:
        yield control, watch
    finally:
        for worker in reversed(workers):
            worker.stop()
        control.close()


def wait_until(condition, message: str, seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, message
        time.sleep(0.02)


@pytest.mark.integration
def test_a_weight_scenario_trains_serves_and_is_deleted_with_no_directory_that_control_shares(
    tmp_path: Path, fake_git_lfs: None
) -> None:
    component = StubTrainingComponent(tmp_path / "training" / "checkpoints")
    live = LiveObjects(runtime=StubInferenceRuntime(), training_facts=StubFacts(component))
    with split(tmp_path, TestPolicyRecipe(max_staleness=4), live, (component,)) as (control, watch), watch.watching():
        scenario = control.get_or_create_scenario("math")
        root = scenario.serving_artifact_id()
        control.accept_record(token_inference("i1", artifact_id=root))
        control.accept_record(report("r1", "i1"))
        wait_until(lambda: scenario.trainer.state["finished_jobs"] == 1, "the job did not train to a commit")

        # The checkpoint was committed by name, from a directory that only the training service has.
        (job,) = scenario_jobs(control.queue, "math")
        assert job["state"] == "finished" and [entry["content"] for entry in job["checkpoints"]] == [f"{job['id']}:0"]
        assert job["checkpoints"][0]["path"].startswith(str(tmp_path / "training"))
        status = control.build_training_status()["scenarios"]["math"]
        assert status["last_training_commit"]["metrics"]["training_job"]["event"] == "finished"
        trained = scenario.artifact_status()["weights"]["active"]
        assert trained != root
        # Control activates and reads the artifacts through the artifact store.
        control.activate("math", root)
        control.activate("math", trained)
        _, snapshot = scenario.resolve_artifact(trained)
        assert (snapshot.materialize().local_path / VERSION_FILE).read_text() == "w1"

        assert control.delete_scenario("math")["scenario"] == "math"
        wait_until(
            lambda: all(item.final for item in control.queue.items(CHECKS_QUEUE)),
            "the training service did not answer the archive request",
        )
    assert watch.seen == []
    assert not list((tmp_path / "control").rglob("checkpoints"))


@pytest.mark.integration
def test_a_harness_scenario_trains_shows_its_step_and_is_deleted_with_no_directory_that_control_shares(
    tmp_path: Path, fake_git_lfs: None
) -> None:
    def method(workspace, services):
        workspace.write_tree(workspace.tree().apply(TreePatch(changed_rules("marker rules\n"))))
        workspace.write_output("added a rule", {"rules": 1})

    # The two services read one configuration, and each resolves ``evolution.state_dir`` on its own disk.
    for service in ("control", "training"):
        (tmp_path / service).mkdir()
    recipe, control_live = harness_recipe(tmp_path / "control", method)
    _, training_live = harness_recipe(tmp_path / "training", method)
    component = training_live.harness_component
    assert component is not None
    control_state = tmp_path / "control" / "harness"
    training_state = tmp_path / "training" / "harness"
    before = sorted(path.name for path in control_state.iterdir()) if control_state.is_dir() else []
    with split(tmp_path, recipe, control_live, (component,)) as (control, watch), watch.watching():
        scenario = control.get_or_create_scenario("demo")
        for record in report_records("demo", "1"):
            control.accept_record(record)
        wait_until(lambda: scenario.trainer.state["finished_jobs"] == 1, "the step did not train to a commit")

        (job,) = scenario_jobs(control.queue, "demo")
        assert job["state"] == "finished" and len(job["checkpoints"]) == 1
        # The pages of control read the step's record files from the artifact store.
        headers = {"x-reef-scenario": "demo"}
        requests = RequestService(control)
        inventory = requests.harness_step_records(headers, 1, None)
        assert inventory["status"] == "retained"
        assert {"calls.json", "output/notes.json"} <= {entry["path"] for entry in inventory["files"]}
        assert '"rules": 1' in requests.harness_step_records(headers, 1, "output/notes.json")["text"]
        assert requests.step_progress(scenario) is None
        assert requests.harness_manifest(headers)["files"]

        assert (training_state / "demo").is_dir()
        control.delete_scenario("demo")
        wait_until(lambda: not (training_state / "demo").exists(), "the training service did not archive its state")
    assert watch.seen == []
    # The training service moved its own directory aside, and control wrote nothing where it has no state.
    assert [path.name.startswith("demo-") for path in (training_state / "archived").iterdir()] == [True]
    assert sorted(path.name for path in control_state.iterdir()) == before


@pytest.mark.integration
def test_a_composite_scenario_trains_both_components_with_no_directory_that_control_shares(
    tmp_path: Path, fake_git_lfs: None
) -> None:
    def method(workspace, services):
        workspace.write_tree(workspace.tree().apply(TreePatch(changed_rules("marker rules\n"))))
        workspace.write_output("added a rule", {"rules": 1})

    for service in ("control", "training"):
        (tmp_path / service).mkdir()
    harness, control_live = harness_recipe(tmp_path / "control", method)
    _, training_live = harness_recipe(tmp_path / "training", method)
    evolving = training_live.harness_component
    assert evolving is not None
    weights = StubTrainingComponent(tmp_path / "training" / "checkpoints")
    recipe = CompositeRecipe(components={"weights": TestPolicyRecipe(max_staleness=8), "harness": harness})
    live = replace(control_live, runtime=StubInferenceRuntime(), training_facts=StubFacts(weights))
    with split(tmp_path, recipe, live, (weights, evolving)) as (control, watch), watch.watching():
        scenario = control.get_or_create_scenario("agent")
        root = scenario.artifact_status()["weights"]["active"]
        control.accept_record(token_inference("i1", scenario="agent", artifact_id=root))
        control.accept_record(report("r1", "i1", scenario="agent"))

        def trained(component: str) -> bool:
            return scenario.trainer_for(component).state["finished_jobs"] == 1

        wait_until(
            lambda: trained("weights") and trained("harness"),
            f"the two components did not train: {control.build_training_status()}",
        )
        jobs = scenario_jobs(control.queue, "agent")
        assert sorted(len(job["checkpoints"]) for job in jobs) == [1, 1]
        # Each commit carried the other component by its artifact ID, and both artifacts are active.
        status = scenario.artifact_status()
        assert status["weights"]["active"] != root
        _, snapshot = scenario.resolve_artifact(status["harness"]["active"], "harness")
        assert "marker rules" in scenario.components.read_files(snapshot)["plugin/rules.md"]
        control.delete_scenario("agent")
        wait_until(
            lambda: all(item.final for item in control.queue.items(CHECKS_QUEUE)),
            "the training services did not answer the archive requests",
        )
    assert watch.seen == []
