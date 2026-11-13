"""A split deployment: control and the training service share the store, and no directory.

Each test runs control as a dispatcher without the training service, and the training service as the two workers
that it runs for a component. Everything of the training service is under ``tmp_path / "training"``: the state of
its components, its checkpoints and its clone of the artifact repository. Everything of control is under
``tmp_path / "control"``. The artifact repository is a real Git LFS repository beside them.

What the tests prove, and what they do not:

- ``weight_bytes_in`` reads every file under control's directory and looks for the bytes of the weight files. It
  finds bytes that any thread or subprocess of control wrote there, in a checkout, an LFS object or a copy. It does
  not find a file that control only read from the training service's directory.
- Control's ``evolution.state_dir`` is another directory than the training service's, and the tests check that
  control wrote nothing there and still serves the step's records and progress: it has them from the store.
- The tests do not prove that control opens no path of the training service. Both run in one process here, so no
  check of file access can tell them apart.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event

import pytest
from tests.support.harness_jobs import changed_rules, harness_recipe
from tests.support.harness_service import report_records
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.records import report, token_inference
from tests.support.runtime_stubs import VERSION_FILE, StubFacts, StubInferenceRuntime, StubTrainingComponent
from tests.support.samples import data_storage

from reef.artifact import Artifact, ArtifactValidator, GitLFSRepositoryBackend, RepositoryBackendFactory
from reef.artifact.components import ComponentSpec, ScenarioComponents
from reef.artifact.sources import REFERENCE_FILE, DirectorySource, HuggingFaceRevision
from reef.core.snapshot import ItemIdentity
from reef.dispatcher import Dispatcher
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.execution.workers import SERVICE_COMPONENTS
from reef.harness.tree import TreePatch
from reef.recipe import Recipe
from reef.recipe.base import WEIGHTS_COMPONENT
from reef.recipe.composite import CompositeRecipe
from reef.record2sample.service.worker import SAMPLES_QUEUE
from reef.scenario.factory import ItemStart
from reef.scenario.live_objects import LiveObjects, RecipeChecks
from reef.service.request_service import RequestService
from reef.service.scenario_files import artifact_directory
from reef.storage.sql_coordination import open_coordination
from reef.train.component import SavedCheckpoint, TrainingComponent, TrainingJob
from reef.train.job_queue import (
    CHECKS_QUEUE,
    JOBS_QUEUE,
    TrainingCheckWorker,
    TrainingJobWorker,
    discard_ended_content,
    scenario_jobs,
)

#: The bytes that every weight file of these tests holds, which no other file holds.
SENTINEL = b"reef-weight-bytes-that-control-must-not-hold:" + bytes(range(256)) * 64
MODEL = "model.safetensors"
#: The bound of one wait, for a host that runs many tests at once. No test waits a fixed time.
WAIT_SECONDS = 180.0


class SentinelWeights(StubTrainingComponent):
    """A training component whose every checkpoint holds a weight file with ``SENTINEL``, the job's number and
    ``marker``. A job trains only while ``may_train`` is set, which it is unless a test clears it."""

    def __init__(self, checkpoint_root: Path) -> None:
        super().__init__(checkpoint_root)
        self.may_train = Event()
        self.may_train.set()
        self.marker = b""

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        if not self.may_train.wait(WAIT_SECONDS):
            raise RuntimeError("the test did not let the weights train")
        (checkpoint,) = super().train(job)
        (Path(checkpoint.path) / MODEL).write_bytes(SENTINEL + str(len(self.submitted)).encode() + self.marker)
        return (checkpoint,)


#: The bytes in a weight file, or in a reference, that ``SentinelCheck`` refuses.
REFUSED = b"refused-weights"


class SentinelCheck(ArtifactValidator):
    """Refuse weights any file of which holds ``REFUSED``. It reads every file of the artifact that it gets, and
    ``directories`` lists the local directory of each one."""

    def __init__(self) -> None:
        self.directories: list[Path] = []

    def validate(self, artifact: Artifact) -> None:
        directory = artifact.local_path
        assert directory is not None, "a check gets the files of an artifact in a local directory"
        self.directories.append(directory)
        if any(REFUSED in path.read_bytes() for path in directory.rglob("*") if path.is_file()):
            raise ValueError("the check refused the weights")


@dataclass(frozen=True)
class CheckedPolicyRecipe(TestPolicyRecipe):
    """The policy recipe whose weights ``check`` admits."""

    check: SentinelCheck = field(default_factory=SentinelCheck, compare=False)

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents({WEIGHTS_COMPONENT: ComponentSpec(validator=self.check)})


class PinnedWeights(DirectorySource):
    """A directory of weights whose source keeps it fixed, as a Hugging Face commit does: the import stores the
    reference to ``repository``, not the weights."""

    def __init__(self, directory: Path, repository: str) -> None:
        super().__init__(directory, source_version="0" * 40, link=False)
        self.repository = repository

    @property
    def reference(self):
        return {REFERENCE_FILE: json.dumps(HuggingFaceRevision(self.repository, "0" * 40).to_dict()).encode()}


def weight_bytes_in(directory: Path) -> list[str]:
    """The files under ``directory`` that hold ``SENTINEL``, as paths under it."""
    marker = SENTINEL[:64]
    found = []
    for folder, _, names in os.walk(directory):
        for name in names:
            path = Path(folder) / name
            try:
                holds = not path.is_symlink() and marker in path.read_bytes()
            except FileNotFoundError:
                # A service removes its temporary files while this reads the directory.
                continue
            if holds:
                found.append(str(path.relative_to(directory)))
    return sorted(found)


def git_backends(root: Path) -> RepositoryBackendFactory:
    """The artifact store of one service: its own work and cache directories under ``root``, over the repository
    that both services reach."""
    return GitLFSRepositoryBackend.factory(
        root.parent / "artifacts.git", work_dir=root / "artifact-work", cache_dir=root / "artifact-cache"
    )


def stored_content_refs(tmp_path: Path) -> list[str]:
    """The refs of the stored content that the artifact repository holds, of every scenario."""
    return subprocess.run(
        [
            "git",
            "--git-dir",
            str(tmp_path / "artifacts.git"),
            "for-each-ref",
            "--format=%(refname)",
            "refs/reef/contents",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()


@contextmanager
def split(
    tmp_path: Path,
    recipe: Recipe,
    live: LiveObjects,
    components: tuple[TrainingComponent, ...],
    *,
    inference_apart: bool = False,
) -> Iterator[Dispatcher]:
    """Control over ``recipe`` with the facts of ``live`` and no training component, while the training service
    runs ``components`` from the same queue. The control process also runs the other services. With
    ``inference_apart``, the inference service is a dispatcher of its own, with its own directory
    ``tmp_path / "inference"`` and its own runtime, so the weights that it loads are not under control's."""
    storage = data_storage(tmp_path / "records")
    # The work queues live beside the records, so each service opens the coordination store itself.
    coordination = open_coordination("sqlite", agent_record_dir=str(tmp_path / "records"), record_database_url=None)
    control = Dispatcher(
        recipe,
        git_backends(tmp_path / "control"),
        live=replace(live, training_component=None, harness_component=None, data_store=storage.data),
        local_artifact_dir=tmp_path / "control" / "staged",
        scenario_storage=storage,
        record_storage=storage.inner,
        coordination=coordination,
        components=frozenset(SERVICE_COMPONENTS) - {"training"} - ({"inference"} if inference_apart else set()),
    )
    inference = None
    if inference_apart:
        served = data_storage(tmp_path / "records")
        inference = Dispatcher(
            recipe,
            git_backends(tmp_path / "inference"),
            live=replace(live, training_component=None, harness_component=None, data_store=served.data),
            local_artifact_dir=tmp_path / "inference" / "staged",
            scenario_storage=served,
            record_storage=served.inner,
            coordination=open_coordination(
                "sqlite", agent_record_dir=str(tmp_path / "records"), record_database_url=None
            ),
            components=frozenset({"inference"}),
        )
    training_backends = git_backends(tmp_path / "training")
    workers: list[TrainingJobWorker | TrainingCheckWorker] = []
    for component in components:
        workers.append(
            TrainingJobWorker(
                component,
                coordination.work_queue,
                control.job_directory_store,
                training_backends,
                checks=RecipeChecks(recipe),
                worker_id=f"split-training-jobs-{component.name}",
                local_artifact_dir=tmp_path / "training" / "staged",
                poll_seconds=0.02,
            )
        )
        workers.append(
            TrainingCheckWorker(
                component,
                coordination.work_queue,
                control.job_directory_store,
                training_backends,
                worker_id=f"split-training-checks-{component.name}",
                local_artifact_dir=tmp_path / "training" / "staged",
                poll_seconds=0.02,
            )
        )
    for worker in workers:
        worker.start()
    try:
        yield control
    finally:
        for worker in reversed(workers):
            worker.stop()
        if inference is not None:
            inference.close()
        control.close()


def wait_until(condition, message: str, state=None, seconds: float = WAIT_SECONDS) -> None:
    """Wait until ``condition`` holds. The bound is for a host that runs many tests at once: each wait is on a
    condition that a queue round trip between two services makes true, and nothing here waits a fixed time. A
    failure reports ``state()``, read when the wait gave up."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError(message if state is None else f"{message}: {state()}")
        time.sleep(0.02)


def changing_rules(workspace, services) -> None:
    """A harness method that rewrites the rules of the tree, and records a note."""
    workspace.write_tree(workspace.tree().apply(TreePatch(changed_rules("marker rules\n"))))
    workspace.write_output("added a rule", {"rules": 1})


@pytest.fixture
def real_lfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git and Git LFS without the user's configuration, and trainers that look at a running job often: a
    trainer's worker polls a job every ``POLL_SECONDS``, which is 2 seconds in a deployment."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setattr("reef.dispatcher.local_workers.POLL_SECONDS", 0.05)


@pytest.mark.integration
def test_a_weight_scenario_trains_serves_and_is_deleted_with_no_weight_byte_in_control(
    tmp_path: Path, real_lfs: None
) -> None:
    component = SentinelWeights(tmp_path / "training" / "checkpoints")
    live = LiveObjects(runtime=StubInferenceRuntime(), training_facts=StubFacts(component))
    # The inference service of this process is the reader of the weights; control's own steps are checked before
    # anything serves, with the services that load weights left out.
    with split(tmp_path, TestPolicyRecipe(max_staleness=4), live, (component,)) as control:
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
        control.activate("math", root)
        control.activate("math", trained)
        control.train_from("math", root)
        _, snapshot = scenario.resolve_artifact(trained)
        # The serving runtime of this process made the weights local, as the reader of the weights; a snapshot
        # that the committer resolved has no file.
        assert snapshot.local_path is None
        assert (snapshot.materialize().local_path / VERSION_FILE).read_text() == "w1"

        assert control.delete_scenario("math")["scenario"] == "math"
        wait_until(
            lambda: all(item.final for item in control.queue.items(CHECKS_QUEUE)),
            "the training service did not answer the archive request",
        )
    assert not list((tmp_path / "control").rglob("checkpoints"))
    assert stored_content_refs(tmp_path) == []


@pytest.mark.integration
def test_a_check_that_reads_the_weights_runs_where_they_are_and_brings_no_weight_byte_to_control(
    tmp_path: Path, real_lfs: None
) -> None:
    """The weights' check reads their files. The training service runs it on each checkpoint before it stores it,
    and control runs it on the reference that an import stores. An activation and a restore run no check. A
    train-from and a new trainer's root make local only the files that the facts' start checks read
    (``TrainingBackendFacts.start_files``). The inference service runs apart, so every weight byte under control's
    directory would be one that control fetched; after each operation, none is there. A refusal still refuses: a
    refused checkpoint fails its job, and a refused import raises the check's ``ValueError``, as before."""
    component = SentinelWeights(tmp_path / "training" / "checkpoints")
    live = LiveObjects(runtime=StubInferenceRuntime(), training_facts=StubFacts(component))
    recipe = CheckedPolicyRecipe(max_staleness=4)
    control_directory = tmp_path / "control"
    done: list[str] = []

    def checked(operation: str) -> None:
        assert weight_bytes_in(control_directory) == [], f"after {operation} (before it: {done})"
        done.append(operation)

    with split(tmp_path, recipe, live, (component,), inference_apart=True) as control:
        scenario = control.get_or_create_scenario("math")
        root = scenario.serving_artifact_id()
        control.accept_record(token_inference("i1", artifact_id=root))
        control.accept_record(report("r1", "i1"))
        wait_until(lambda: scenario.trainer.state["finished_jobs"] == 1, "the job did not train to a commit")
        trained = scenario.artifact_status()["weights"]["active"]
        assert trained != root
        after_training = scenario.commit_number
        # The check read the checkpoint in the training service's own directory.
        assert [path.is_relative_to(tmp_path / "training") for path in recipe.check.directories] == [True]
        checked("a training commit")

        component.marker = REFUSED
        control.accept_record(token_inference("i2", artifact_id=trained))
        control.accept_record(report("r2", "i2"))
        wait_until(lambda: scenario.trainer.state["finished_jobs"] == 2, "the refused job did not end")
        refused = [job for job in scenario_jobs(control.queue, "math") if job["state"] == "failed"]
        assert len(refused) == 1 and "the check refused the weights" in refused[0]["error"]
        assert refused[0]["checkpoints"] == []
        assert scenario.trainer.state["last_job"]["event"] == "failed"
        assert scenario.artifact_status()["weights"]["active"] == trained
        checked("a refused training commit")

        given = tmp_path / "given"
        given.mkdir()
        (given / "config.json").write_text("{}")
        (given / MODEL).write_bytes(SENTINEL + b"given")
        for work in ("import-work", "refused-import-work"):
            (control_directory / work).mkdir()
        imported = control.import_artifact(
            "math",
            PinnedWeights(given, "org/weights"),
            None,
            source={"uri": f"hf://org/weights@{'0' * 40}", "version": "0" * 40},
            directory=control_directory / "import-work",
        )
        # Control checked what it stored: the reference, in its own import directory.
        assert recipe.check.directories[-1] == control_directory / "import-work"
        checked("an import")
        with pytest.raises(ValueError, match="the check refused the weights"):
            control.import_artifact(
                "math",
                PinnedWeights(given, "org/refused-weights"),
                None,
                source={"uri": f"hf://org/refused-weights@{'0' * 40}", "version": "0" * 40},
                directory=control_directory / "refused-import-work",
            )
        checked("a refused import")

        checks = len(recipe.check.directories)
        control.activate("math", imported.artifact_id)
        checked("activate the import")
        control.activate("math", root)
        checked("activate the root")
        assert control.restore("math", after_training) == {"weights": trained}
        checked("restore")
        # A selection checks nothing: each artifact was admitted when it was stored.
        assert len(recipe.check.directories) == checks
        # The facts' start check of a train-from reads only the files that the facts name, and the stub's none.
        control.train_from("math", trained)
        checked("train from an earlier commit")

        # A scenario whose root is an artifact item: its new trainer's start check reads only those files too. The
        # stub trains one scenario per process, so the first goes first.
        control.delete_scenario("math")
        item = ItemStart(
            ItemIdentity("weights-item"),
            f"hf://org/weights@{'0' * 40}",
            PinnedWeights(given, "org/weights"),
        )
        other, created = control.create_scenario("item", item)
        assert created and other.trainer.state["finished_jobs"] == 0
        checked("create a scenario from an item root")
    assert not list(control_directory.rglob("checkpoints"))


@pytest.mark.integration
def test_a_harness_scenario_trains_shows_its_step_and_is_deleted_with_no_directory_that_control_shares(
    tmp_path: Path, real_lfs: None
) -> None:
    # The two services read one configuration, and each resolves ``evolution.state_dir`` on its own disk.
    for service in ("control", "training"):
        (tmp_path / service).mkdir()
    recipe, control_live = harness_recipe(tmp_path / "control", changing_rules)
    _, training_live = harness_recipe(tmp_path / "training", changing_rules)
    component = training_live.harness_component
    assert component is not None
    control_state = tmp_path / "control" / "harness"
    training_state = tmp_path / "training" / "harness"
    before = sorted(path.name for path in control_state.iterdir()) if control_state.is_dir() else []
    with split(tmp_path, recipe, control_live, (component,)) as control:
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
        # The step's records are stored content, which lives as long as the scenario: the checkpoint's is gone.
        # They are one content for each job, and a harness step is one job.
        assert len(stored_content_refs(tmp_path)) == 1
        control.delete_scenario("demo")
        wait_until(lambda: not (training_state / "demo").exists(), "the training service did not archive its state")
    # The training service moved its own directory aside, and control wrote nothing where it has no state.
    assert [path.name.startswith("demo-") for path in (training_state / "archived").iterdir()] == [True]
    assert sorted(path.name for path in control_state.iterdir()) == before
    # Nothing that a job of the scenario stored outlives the scenario.
    assert stored_content_refs(tmp_path) == []


@pytest.mark.integration
def test_no_operation_of_control_on_a_composite_brings_a_weight_byte_to_control(
    tmp_path: Path, real_lfs: None
) -> None:
    """The inference service runs apart here, with its own directory, so every weight byte under control's
    directory would be one that control fetched. After each operation, none is there."""
    for service in ("control", "training"):
        (tmp_path / service).mkdir()
    harness, control_live = harness_recipe(tmp_path / "control", changing_rules)
    _, training_live = harness_recipe(tmp_path / "training", changing_rules)
    evolving = training_live.harness_component
    assert evolving is not None
    weights = SentinelWeights(tmp_path / "training" / "checkpoints")
    recipe = CompositeRecipe(components={"weights": TestPolicyRecipe(max_staleness=8), "harness": harness})
    runtime = StubInferenceRuntime()
    # The stub answers each call from the artifact that the call names, so it serves an artifact beside the active.
    runtime.answers_each_artifact = True
    live = replace(control_live, runtime=runtime, training_facts=StubFacts(weights))
    control_directory = tmp_path / "control"
    done: list[str] = []

    def checked(operation: str) -> None:
        assert weight_bytes_in(control_directory) == [], f"after {operation} (before it: {done})"
        done.append(operation)

    with split(tmp_path, recipe, live, (weights, evolving), inference_apart=True) as control:
        scenario = control.get_or_create_scenario("agent")
        root = dict(scenario.artifact_selection().components)

        def trained(component: str) -> bool:
            return scenario.trainer_for(component).state["finished_jobs"] >= 1

        def quiet() -> bool:
            """No sample is being made, no job or check waits or runs, and no trainer has anything to train."""
            queues = (SAMPLES_QUEUE, CHECKS_QUEUE, JOBS_QUEUE)
            items = [item for queue in queues for item in control.queue.items(queue, scenario="agent")]
            trainers = [component_trainer.trainer for component_trainer in scenario.trainers]
            return all(item.final for item in items) and not any(
                trainer.holds_untrained_records() or trainer.needs_polling() or trainer.has_pending_commit
                for trainer in trainers
            )

        def state() -> object:
            return control.build_training_status(), scenario_jobs(control.queue, "agent")

        # One record trains both components, and the harness trains first: the weights job waits. Without the
        # order, the test has a race. When the weights commit activates before the harness takes its sample, the
        # sample is of weights that the scenario no longer serves (``max_staleness`` is 0), and the pipeline
        # drops it, so the harness never trains.
        weights.may_train.clear()
        control.accept_record(token_inference("i1", scenario="agent", artifact_id=root["weights"]))
        control.accept_record(report("r1", "i1", scenario="agent"))
        wait_until(lambda: trained("harness"), "the harness did not train", state)
        weights.may_train.set()
        wait_until(lambda: trained("weights") and quiet(), "the weights did not train", state)
        # The training service does hold the weights, so the check can find them.
        assert weight_bytes_in(tmp_path / "training") and weight_bytes_in(tmp_path / "artifacts.git")
        # The inference service is the reader of the weights: it made them local, in its own directory.
        wait_until(lambda: weight_bytes_in(tmp_path / "inference") != [], "the inference service loaded no weights")
        checked("two training commits")
        jobs = scenario_jobs(control.queue, "agent")
        assert sorted(len(job["checkpoints"]) for job in jobs) == [1, 1]
        status = scenario.artifact_status()
        active = {name: status[name]["active"] for name in ("weights", "harness")}
        assert active["weights"] != root["weights"] and active["harness"] != root["harness"]
        after_training = scenario.commit_number

        headers = {"x-reef-scenario": "agent"}
        requests = RequestService(control)
        manifest = requests.harness_manifest(headers)
        assert any("marker rules" in text for text in manifest["files"].values())
        checked("harness manifest")
        assert requests.harness_step_records(headers, 1, None)["status"] == "retained"
        checked("step records")
        assert (artifact_directory(control, "agent", active["harness"], "harness") / "rules.md").is_file()
        checked("artifact files of the harness")
        # An evaluation's rollout reads its candidate: the harness component alone.
        _, candidate = control.candidate("agent", active["harness"], "harness")
        assert candidate.local_path is not None and not list(candidate.local_path.rglob(MODEL))
        checked("an evaluation candidate")

        synced = requests.harness_import(
            headers,
            {
                "base_artifact_id": manifest["artifact_id"],
                "files": {
                    path: text.replace("marker rules", "synced rules") for path, text in manifest["files"].items()
                },
            },
        )
        assert synced["changed"]
        checked("harness sync")
        given = tmp_path / "given"
        given.mkdir()
        (given / "rules.md").write_text("imported rules\n")
        imported = control.import_artifact(
            "agent",
            DirectorySource(given, source_version="v1", link=False),
            "harness",
            source={"uri": "file:///given/", "version": "v1"},
            directory=tmp_path / "control" / "import-work",
        )
        checked("harness import")

        control.activate("agent", root["weights"], "weights")
        checked("activate older weights")
        control.activate("agent", imported.artifact_id, "harness")
        checked("activate the imported harness")
        assert control.restore("agent", after_training) == active
        checked("restore")
        control.set_served("agent", {active["weights"]: 50, root["weights"]: 50}, "weights")
        checked("serve two weights")
        control.train_from("agent", root["weights"], "weights")
        control.train_from("agent", active["weights"], "weights")
        control.train_from("agent", active["harness"], "harness")
        checked("train from")
        EndedWorkSweeper(control.queue, control.backend_factory, retention_seconds=1e-6).sweep()
        checked("sweep")

        # A scenario that is deleted while a job runs or waits in the queue, also a job that its trainer submitted
        # and has not committed yet: the job is cancelled, and nothing that it stored stays.
        control.accept_record(token_inference("i2", scenario="agent", artifact_id=root["weights"]))
        control.accept_record(report("r2", "i2", scenario="agent"))
        wait_until(
            lambda: len(control.queue.items(JOBS_QUEUE, scenario="agent")) > 0, "no further job was queued", state
        )
        control.delete_scenario("agent")
        checked("delete during a job")
        wait_until(
            lambda: all(item.final for queue in (JOBS_QUEUE, CHECKS_QUEUE) for item in control.queue.items(queue)),
            "the training services did not end the jobs and the archive requests",
        )
        # A worker can store a checkpoint after the delete discarded the scenario's content. Its item names it,
        # and the sweep of the ended items discards it.
        discard_ended_content(control.queue, control.backend_factory, ended_before=time.time() + 1, give_up_before=0.0)
        checked("the sweep after the delete")
    assert stored_content_refs(tmp_path) == []
