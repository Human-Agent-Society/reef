"""The in-process MLX training runtime and the factory of the ``mlx`` runtime kind.

One process holds one base model, one LoRA adapter and one optimizer.
Serving and training take turns on them: a training step closes request
admission, trains the live adapter, exports the candidate and restores the
served weights. Reef reserves the training samples from recorded traffic,
as it does for the other backends.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.core.batches import StepScheduling
from reef.runtime.deployment import RuntimeBuild, RuntimeFactory
from reef.runtime.interfaces import (
    ModelCandidate,
    PreparedTrainingStep,
    RuntimeContractError,
    TrainingMethod,
    TrainingRuntime,
)
from reef.train.algos.registry import resolve_objective
from reef.train.mlx_backend.config import MLXConfig
from reef.train.mlx_backend.contracts import LossOptions, ModelEngine
from reef.train.mlx_backend.serving import MLXServingRuntime
from reef.train.types import TrainingBatch, trajectories

logger = logging.getLogger(__name__)


class MLXRuntime(TrainingRuntime):
    """Train the serving runtime's adapter in this process, one optimizer step per reserved batch."""

    def __init__(self, serving: MLXServingRuntime, *, checkpoint_dir: str, loss_options: LossOptions) -> None:
        super().__init__()
        self._serving = serving
        self._engine = serving.engine
        self._checkpoint_root = Path(checkpoint_dir)
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._loss_options = loss_options

    @property
    def serving(self) -> MLXServingRuntime:
        return self._serving

    @property
    def engine(self) -> ModelEngine:
        return self._engine

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        """Run the method's objective in this process and check that the engine can train its loss family."""
        objective = resolve_objective(method.objective)
        try:
            self.engine.check_loss(objective.loss_family, self._loss_options)
        except ValueError as exc:
            raise RuntimeContractError(f"the mlx runtime cannot train {method.objective!r}: {exc}") from exc
        if method.learning_rate_schedule is not None:
            raise RuntimeContractError(
                f"the mlx runtime trains at training.options.learning_rate; {method.objective!r} "
                "selects a learning-rate schedule it does not implement"
            )
        # One optimizer step per reserved batch. "configured" names the
        # backend's own batch size, and the reserved batch is that size here.
        unsupported = [
            *([f"epochs={scheduling.epochs}"] if scheduling.epochs != 1 else []),
            *(["shuffle=True"] if scheduling.shuffle else []),
            *(
                [f"batch_size={scheduling.batch_size!r}"]
                if scheduling.batch_size not in ("actual", "configured")
                else []
            ),
        ]
        if unsupported:
            raise RuntimeContractError(
                f"the mlx runtime trains one step per reserved batch; {method.objective!r} asks for "
                f"{', '.join(unsupported)}. Supported: epochs=1, shuffle=False, batch_size='actual'."
            )
        objective.validate_scheduling(scheduling)
        signal = objective.prepare(batch, algorithm_state)
        if signal.action == "skip":
            return PreparedTrainingStep("skip", signal.next_algorithm_state, signal.metrics)
        samples = trajectories(batch)
        if signal.advantages is None or len(signal.advantages) != len(samples):
            raise RuntimeContractError(f"{method.objective!r} must supply one advantage per sample")
        return PreparedTrainingStep(
            "train",
            signal.next_algorithm_state,
            signal.metrics,
            {
                "samples": samples,
                "advantages": tuple(signal.advantages),
                "loss_family": objective.loss_family,
                "method": method.to_dict(),
                "scenario_step": scenario_step,
            },
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train the live adapter, export the candidate, and restore the served weights.

        Requests wait for the whole step, because the step changes the weights
        that serving reads. The served weights are restored before admission
        reopens, whether the step succeeds or fails.
        """
        scenario_step = payload["scenario_step"]
        source_runtime_load_id = self._serving.current_runtime_load_id()
        engine = self.engine
        with self._serving.paused_admission():
            before = engine.adapter_snapshot()
            try:
                try:
                    metrics = engine.train(
                        payload["loss_family"], payload["samples"], payload["advantages"], self._loss_options
                    )
                except ValueError as exc:
                    # The loss family found a sample that it cannot train on.
                    raise RuntimeContractError(f"the mlx training step refused its batch: {exc}") from exc
                after = engine.adapter_snapshot()
                delta, changed = engine.adapter_delta(before, after)
                if changed == 0:
                    raise RuntimeContractError(
                        "the mlx training step left every adapter tensor unchanged; "
                        "refusing to publish an untrained candidate"
                    )
                candidate_id = uuid.uuid4().hex
                destination = self._checkpoint_root / f"candidate-{scenario_step}-{candidate_id[:8]}"
                engine.save_adapter(
                    destination,
                    source={
                        "scenario_step": scenario_step,
                        "training_job_id": candidate_id,
                        "objective": TrainingMethod.from_dict(payload["method"]).objective,
                        "loss_family": payload["loss_family"],
                        "source_runtime_load_id": source_runtime_load_id,
                    },
                )
                self._serving.stage_candidate(candidate_id, after)
            finally:
                engine.apply_adapter(before)
        return ModelCandidate(
            candidate_id=candidate_id,
            training_job_id=candidate_id,
            checkpoint_path=str(destination),
            current_runtime_load_id=source_runtime_load_id,
            training_metrics={**metrics, "adapter_delta_l2": delta, "adapter_tensors_changed": changed},
        )

    def reject_candidate(self, candidate: ModelCandidate) -> None:
        self._serving.discard_candidate(candidate.candidate_id)
        logger.info("rejected mlx candidate %s", candidate.candidate_id)

    def probe_candidate(self, candidate_id: str, prompts: Sequence[Sequence[int]], *, max_tokens: int) -> list[str]:
        """Generate greedily from a staged candidate's weights, then restore the served weights.

        A candidate evaluator uses this before Reef decides to publish the
        candidate. Requests wait for the whole probe, so none is answered by
        weights Reef has not selected.
        """
        snapshot = self._serving.staged_candidate(candidate_id)
        if snapshot is None:
            raise RuntimeContractError(f"no pending mlx candidate {candidate_id!r} to probe")
        engine = self.engine
        with self._serving.paused_admission():
            served = engine.adapter_snapshot()
            engine.apply_adapter(snapshot)
            try:
                return [engine.generate(prompt, max_tokens=max_tokens, temperature=0.0).text for prompt in prompts]
            finally:
                engine.apply_adapter(served)


class MLXRuntimeFactory(RuntimeFactory):
    """Build the MLX training and serving runtimes; MLX is imported only when a deployment selects them."""

    kind = "mlx"

    def config_type(self) -> type:
        return MLXConfig

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> RuntimeBuild:
        settings = MLXConfig(**{key: value for key, value in config.items() if key != "type"})
        try:
            from reef.train.mlx_backend.engine import MLXEngine
        except ImportError as exc:
            raise RuntimeContractError(
                f"the mlx runtime needs the optional MLX dependencies; install reef-infra[mlx] on Apple Silicon ({exc})"
            ) from exc
        return mlx_runtimes(settings, MLXEngine(model_path, settings))


def mlx_runtimes(settings: MLXConfig, engine: ModelEngine) -> tuple[MLXRuntime, MLXServingRuntime]:
    """The training and serving halves around one engine, configured from ``training.options``."""
    serving = MLXServingRuntime(
        engine, adapter_name=settings.adapter_name, inference_timeout_s=settings.inference_timeout_s
    )
    training = MLXRuntime(
        serving,
        checkpoint_dir=settings.checkpoint_dir,
        loss_options=LossOptions(settings.kl_coef, settings.loss_options),
    )
    return training, serving


__all__ = ["MLXRuntime", "MLXRuntimeFactory", "mlx_runtimes"]
