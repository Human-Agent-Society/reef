from __future__ import annotations

from pathlib import Path

from tests.support._trajectories import policy_trajectory
from tests.support.remote_backend import ended, remote_component, serving

from reef.artifact.repository import DefaultStart
from reef.core.job_data import JobData
from reef.core.weight_layout import ReleaseFormat
from reef.storage.data import MemoryDataStore
from reef.storage.tasks import LocalTaskFileStore
from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob
from reef.train.method import TrainingMethod
from reef.train.remote.components import BASE_URI
from reef.train.remote.contract import RELEASE_DIR, STATE_DIR
from reef.train.slime_backend.checkpoints import SlimeStart
from reef.train.slime_backend.reef_adapters.training_job.storage import MEGATRON_TRACKER
from reef.train.slime_backend.service import MEGATRON_DIR, SlimeBackend, SlimeServiceSettings


class FakeSlime(TrainingComponent):
    """Saves one rollout per job in its Megatron directory, as Slime does: the next rollout after the one that
    ``megatron`` holds at the start, or rollout 0 from the weights. It records what each run started from."""

    runs: list[tuple[Path, int | None, str]] = []

    def __init__(self, weights: Path, megatron: Path, work: Path) -> None:
        self.weights, self.megatron, self.work = weights, megatron, work
        tracker = megatron / MEGATRON_TRACKER
        self.loaded = int(tracker.read_text()) if tracker.is_file() else None
        self.statuses: dict[str, JobStatus] = {}
        self.base_start = str(work / "starts" / "base")
        SlimeStart(None, None, {}, None).write(Path(self.base_start))

    name = "slime"
    release_format = ReleaseFormat("full")

    def default_start(self) -> DefaultStart:
        return DefaultStart()

    def import_start(self, release: Path) -> str:
        raise ValueError("no import")

    def submit(self, job: TrainingJob) -> None:
        held = SlimeStart.read(Path(job.start)).rollout_id
        assert held == self.loaded, "the job continues the rollout that the workers loaded"
        FakeSlime.runs.append((self.weights, self.loaded, job.start))
        rollout = 0 if held is None else held + 1
        iteration = self.megatron / f"iter_{rollout:07d}"
        iteration.mkdir(parents=True)
        (iteration / "optimizer.pt").write_text(f"rollout {rollout}")
        (self.megatron / MEGATRON_TRACKER).write_text(str(rollout))
        hf = self.work / "hf" / str(rollout)
        hf.mkdir(parents=True)
        (hf / "model.txt").write_text(f"rollout {rollout}")
        start = self.work / "starts" / f"rollout_{rollout}"
        SlimeStart(rollout, None, {}, None).write(start)
        self.statuses[job.job_id] = JobStatus(job.job_id, "finished", (SavedCheckpoint(1, str(start), str(hf)),))

    def status(self, job_id: str) -> JobStatus:
        return self.statuses[job_id]

    def cancel(self, job_id: str) -> None:
        return


class FakeSlimeBackend(SlimeBackend):
    """The Slime backend with ``FakeSlime`` for each job's component."""

    def slime_component(self, weights: Path, megatron: Path, work: Path, loss_family: str) -> TrainingComponent:
        return FakeSlime(weights, megatron, work)


def test_each_job_runs_slime_from_its_artifact_and_continues_the_megatron_state(tmp_path):
    FakeSlime.runs.clear()
    base = tmp_path / "base"
    base.mkdir()
    settings = SlimeServiceSettings(base_model=str(base), ray_address="ray://head", work_dir=str(tmp_path / "work"))
    backend = FakeSlimeBackend(settings)
    assert (backend.info.name, backend.info.release_format) == ("slime", ReleaseFormat("full"))
    parameters = {"method": TrainingMethod("sft").to_dict()}
    sample = policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0)
    with serving(backend, tmp_path) as url:
        remote = remote_component(url, tmp_path)

        def run(job_id: str, start: str):
            data = MemoryDataStore(LocalTaskFileStore()).write_job_directory(JobData(job_id, (sample,)))
            remote.submit(TrainingJob(job_id, "slime", start, data, parameters, scenario="s"))
            status = ended(remote, job_id)
            assert status.state == "finished", status.error
            return status.checkpoints[-1]

        first = run("j1", BASE_URI)
        artifact = Path(first.path.removeprefix("file://"))
        assert (artifact / RELEASE_DIR / "model.txt").read_text() == "rollout 0"
        assert (artifact / STATE_DIR / MEGATRON_DIR / "iter_0000000" / "optimizer.pt").read_text() == "rollout 0"
        # The next job loads the release's weights and the Megatron state of rollout 0, and saves rollout 1.
        second = run("j2", first.path)
        assert FakeSlime.runs[0][:2] == (base, None)
        assert FakeSlime.runs[1][:2] == (artifact / RELEASE_DIR, 0)
        assert Path(second.path.removeprefix("file://"), RELEASE_DIR, "model.txt").read_text() == "rollout 1"
