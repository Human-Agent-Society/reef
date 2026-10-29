"""Real Ray scheduling with a logical GPU reservation; no GPU kernels run."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import tests
from tests.support.harness_steps import PI_SEED, harness_backend

from reef.evaluation.harness import EpisodeScorer, HarnessCandidate
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import executor_settings
from reef.runtime.executor.requirements import ExecutionRequirements

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


class GPUScorer(EpisodeScorer):
    def execution_requirements(self):
        return ExecutionRequirements(gpus_per_worker=1)

    def score(self, task, result):
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
        runtime_env={"env_vars": {"PYTHONPATH": str(Path(tests.__file__).parent.parent)}},
    )
    try:
        backend, _ = harness_backend(
            tmp_path,
            lambda workspace, services: None,
            score=GPUScorer(),
            tasks=("one",),
            worker_executor=executor_settings(
                {}, {"backend": "ray", "workers": 1, "resources": {"cpus_per_worker": 2}}
            ),
        )
        backend.history.add_candidate(PI_SEED, step=1, origin="seed")
        candidate = HarnessCandidate(
            candidate_id="test",
            candidate_tree=PI_SEED,
            current_tree=PI_SEED,
            evaluation_tasks=("one",),
            step=1,
        )
        assert backend.evaluator.worker_pool.selection.settings.backend == "ray"
        result = backend.evaluate(candidate)
        assert result.metrics["candidate_scores"] == (1.0,)
        assert result.metrics["current_scores"] == (1.0,)
        # A second evaluation on another task runs on the same workers.
        second = backend.evaluate(replace(candidate, evaluation_tasks=("two",)))
        assert second.metrics["candidate_scores"] == (1.0,) and second.metrics["episodes_run"] == 2
        # Each episode's record comes back from its Ray worker into the history.
        history = backend.history
        assert all((history.episode_dir(record.episode) / "episode.json").is_file() for record in history.episodes)
        for evaluation in (result, second):
            (episode,) = evaluation.metrics["candidate_episodes"]
            assert (history.episode_dir(episode["episode"]) / "session.jsonl").is_file()
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
