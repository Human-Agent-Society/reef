"""Real Ray scheduling with a logical GPU reservation; no GPU kernels run."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import executor_settings
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.train.cordis_backend.backend import CordisBackend, HarnessCandidate
from reef.train.cordis_backend.strategies import EpisodeScorer, resolve_proposer

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


class GPUScorer(EpisodeScorer):
    def execution_requirements(self):
        return ExecutionRequirements(gpus_per_worker=1)

    def __call__(self, task, result):
        import ray

        gpu_ids = ray.get_gpu_ids()
        assert ray.get_runtime_context().get_assigned_resources()["CPU"] == 2
        if len(gpu_ids) != 1:
            raise RuntimeError(f"scorer did not receive its GPU reservation: {gpu_ids}")
        # CUDA visibility is NVIDIA-specific; this test also runs on Apple GPUs.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != "0":
            raise RuntimeError(f"unexpected worker CUDA visibility: {visible!r}")
        return 1.0


def test_evolution_explicit_ray_places_scorer_on_declared_ray_gpu(tmp_path):
    ray = pytest.importorskip("ray")
    # Ray advertises one logical resource so placement can be tested on CPU CI.
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=1,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}},
    )
    binary = tmp_path / "fake-pi"
    binary.write_text(
        "#!/usr/bin/env python3\nimport os\nfrom pathlib import Path\n"
        "root = Path(os.environ['PI_CODING_AGENT_SESSION_DIR'])\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "(root / 'session.jsonl').write_text('{\"type\":\"agent_end\"}\\n')\n"
    )
    binary.chmod(0o755)
    try:
        backend = CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(lambda *args: None),
            score_episode=GPUScorer(),
            tasks=("one",),
            models=ModelBinding(base_url="http://127.0.0.1:8000", model="unused", api_key="dummy"),
            binary=str(binary),
            worker_executor=executor_settings(
                {}, {"backend": "ray", "workers": 1, "resources": {"cpus_per_worker": 2}}
            ),
        )
        candidate = HarnessCandidate(
            candidate_id="test",
            candidate_files={},
            current_files={},
            candidate_entries=(),
            current_entries=(),
            mutations=(),
            record_dir=tmp_path / "first-record",
        )
        assert backend._worker_selection.settings.backend == "ray"
        result = backend.evaluate(candidate)
        assert result.metrics["candidate_scores"] == (1.0,)
        assert result.metrics["current_scores"] == (1.0,)
        assert (tmp_path / "first-record/episodes/candidate-0/session.jsonl").is_file()
        assert (tmp_path / "first-record/episodes/current-0/episode.json").is_file()
        assert backend.evaluate(replace(candidate, record_dir=tmp_path / "second-record")).metrics[
            "candidate_scores"
        ] == (1.0,)
        assert (tmp_path / "second-record/episodes/candidate-0/session.jsonl").is_file()
        backend.close()
        # Shutdown is asynchronous in Ray: a following actor must be able to
        # acquire the same GPU, proving that backend.close released its workers.
        executor = Executor.create(
            ExecutorConfig(
                backend="ray",
                workers=(WorkerSpec(GPUScorer, options={"num_gpus": 1}),),
                launch_timeout_s=15,
            )
        )
        executor.shutdown()
    finally:
        ray.shutdown()
