"""Real Ray scheduling with a logical GPU reservation; no GPU kernels run."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import tests
from tests.support.harness_steps import PI_SEED, harness_backend

from reef.evaluation.harness import HarnessCandidate, RolloutScorer
from reef.harness.tree import TreePatch
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import executor_settings
from reef.runtime.executor.requirements import ExecutionRequirements

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


class GPUScorer(RolloutScorer):
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
            lambda context: None,
            score=GPUScorer(),
            tasks=("one",),
            worker_executor=executor_settings(
                {}, {"backend": "ray", "workers": 1, "resources": {"cpus_per_worker": 2}}
            ),
        )
        candidate = HarnessCandidate(
            candidate_id="test",
            candidate_tree=PI_SEED,
            current_tree=PI_SEED,
            patch=TreePatch({}),
            evaluation_tasks=("one",),
            record_dir=tmp_path / "first-record",
        )
        assert backend.evaluator.worker_selection.settings.backend == "ray"
        result = backend.evaluate(candidate)
        assert result.metrics["candidate_scores"] == (1.0,)
        assert result.metrics["current_scores"] == (1.0,)
        assert (tmp_path / "first-record/rollouts/candidate-0/session.jsonl").is_file()
        assert (tmp_path / "first-record/rollouts/current-0/rollout.json").is_file()
        assert backend.evaluate(replace(candidate, record_dir=tmp_path / "second-record")).metrics[
            "candidate_scores"
        ] == (1.0,)
        assert (tmp_path / "second-record/rollouts/candidate-0/session.jsonl").is_file()
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
