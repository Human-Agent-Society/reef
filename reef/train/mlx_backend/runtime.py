"""An in-process MLX training runtime for a single Apple Silicon host.

This runtime is deliberately not a distributed system. One process holds one
base model, one LoRA adapter and one optimizer; serving and training take
turns on them, coordinated by the inference admission Reef already owns. The
honest description of the topology is: colocated, single host, single process,
synchronous.

What it is NOT: it does not generate rollouts for itself. Reef reserves a
batch of rollouts that real traffic produced and hands it here, exactly as it
does for the Slime path, so the invariant that training data always arrives
from outside the runtime holds here too.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.evaluation import SelectionDecision
from reef.runtime.base import PreparedTrainingStep, RuntimeContractError, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackend
from reef.runtime.registry import RuntimeFactory, register_runtime_kind
from reef.train.algos.registry import resolve_preparer
from reef.train.mlx_backend.rows import DistillationRow, TeacherCandidate, TrainingRow
from reef.train.types import PolicySample, TrainingBatch, policy_samples

logger = logging.getLogger(__name__)

#: How the candidate directory under the checkpoint root is named.
CANDIDATE_DIRNAME = "candidate-{scenario_step}-{candidate}"

#: Loss families this runtime actually implements in MLX. A recipe asking for
#: anything else is refused before training rather than trained under the
#: wrong objective.
SUPPORTED_LOSS_FAMILIES = frozenset({"tttd", "openclawrl"})


class MLXRuntime(TrainingRuntime):
    """Serve and train one LoRA adapter in this process, without Ray or CUDA."""

    def __init__(
        self,
        engine: Any,
        *,
        checkpoint_dir: str,
        base_url: str = "mlx://local",
        inference_timeout_s: float = 300.0,
        kl_coef: float = 0.0,
        adapter_name: str = "reef-mlx",
        openclawrl: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self._engine = engine
        self._checkpoint_root = Path(checkpoint_dir)
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._kl_coef = float(kl_coef)
        self._adapter_name = adapter_name
        #: Objective knobs for the openclawrl loss family. Defaults match
        #: ``recipes/openclawrl/slime``'s ``OpenclawrlSettings``.
        self._openclawrl: dict[str, Any] = {
            "w_rl": 1.0,
            "w_opd": 1.0,
            "eps_lo": 0.2,
            "eps_hi": 0.28,
            "diff_clip": 1.0,
            "kl_coef": 0.0,
            "hint_selection": "sequence_optimal",
            "native_k": 20,
            **dict(openclawrl or {}),
        }
        # The freshly loaded base plus its zero-initialised adapter is already
        # a servable version, and every durable training record must name the
        # weights that answered it. Minting the identity at boot is what lets
        # the very first rollout — served before anything is published — be
        # recorded as trainable.
        self._serving_runtime_load_id: str | None = engine.next_runtime_load_id()
        #: The version Reef has committed. Equal to the serving version except
        #: between an activation and that step's durable commit.
        self._committed_runtime_load_id: str | None = self._serving_runtime_load_id
        self._pending: dict[str, Any] = {}
        self._backend: Any = None

    # ---------------------------------------------------------------- serving

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def inference_backend(self) -> InferenceBackend:
        if self._backend is None:
            from reef.train.mlx_backend.inference import MLXInferenceBackend

            self._backend = MLXInferenceBackend(self)
        return self._backend

    def serving_runtime_load_id(self) -> str | None:
        return self._serving_runtime_load_id

    def serving_adapter_name(self) -> str | None:
        # The adapter is resident in this process rather than addressed by
        # name over a wire, but naming it keeps the served identity explicit
        # in records and in the weight surface.
        return self._adapter_name

    def experiment_config(self) -> Mapping[str, Any]:
        config = self._engine.config
        return {
            "runtime": "mlx",
            "model_path": config.model_path,
            "lora_layers": config.lora_layers,
            "lora_rank": config.lora_rank,
            "learning_rate": config.learning_rate,
            "kl_coef": self._kl_coef,
            "openclawrl": dict(self._openclawrl),
        }

    # --------------------------------------------------------------- training

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep:
        """Run the recipe's registered preparer in this process.

        The Ray path ships the preparer's *name* to a separate backend process
        which resolves it there. Here there is no boundary to cross, so the
        preparer runs directly against the reserved batch.
        """
        preparer = resolve_preparer(step_preparer)
        signal = preparer(batch, algorithm_state)
        if signal.action == "train" and signal.loss_family not in SUPPORTED_LOSS_FAMILIES:
            # The runtime implements one objective. Training a recipe's data
            # under a different loss than it asked for is a silent
            # substitution, so name the mismatch instead.
            raise RuntimeContractError(
                f"the mlx runtime implements {sorted(SUPPORTED_LOSS_FAMILIES)}, but {step_preparer!r} "
                f"asks for loss family {signal.loss_family!r}"
            )
        if signal.action == "skip":
            return PreparedTrainingStep(
                action="skip",
                next_algorithm_state=signal.next_algorithm_state,
                metrics=signal.metrics,
            )
        samples = policy_samples(batch)
        self._require_supported_scheduling(signal, step_preparer)
        advantages = signal.advantages
        if advantages is None or len(advantages) != len(samples):
            raise RuntimeContractError(
                f"{step_preparer!r} must supply one advantage per sample; "
                f"got {0 if advantages is None else len(advantages)} for {len(samples)} samples"
            )
        return PreparedTrainingStep(
            action="train",
            next_algorithm_state=signal.next_algorithm_state,
            metrics=signal.metrics,
            payload={
                "samples": samples,
                "advantages": tuple(advantages),
                "loss_family": signal.loss_family,
                "rollout_id": scenario_step,
            },
        )

    @staticmethod
    def _require_supported_scheduling(signal: Any, step_preparer: str) -> None:
        """Refuse a schedule this runtime would silently mistrain.

        A single-process runtime runs exactly one optimizer step over the
        reserved batch. Multi-epoch, shuffled or sub-batched schedules change
        the objective, so they fail here rather than being ignored.
        """
        scheduling = signal.scheduling
        unsupported = []
        if scheduling.epochs != 1:
            unsupported.append(f"epochs={scheduling.epochs}")
        if scheduling.shuffle:
            unsupported.append("shuffle=True")
        # "configured" names the backend's own batch size, and a single
        # process has none: the reserved batch is the step either way. An
        # explicit integer does mean sub-batching, which this runtime would
        # silently collapse into one step, so that stays refused.
        if scheduling.batch_size not in ("actual", "configured"):
            unsupported.append(f"batch_size={scheduling.batch_size!r}")
        if unsupported:
            raise RuntimeContractError(
                f"the mlx runtime trains one step per reserved batch; {step_preparer!r} asks for "
                f"{', '.join(unsupported)}. Supported: epochs=1, shuffle=False, batch_size='actual'."
            )

    def _distillation_rows(self, samples: Sequence[PolicySample], advantages: Sequence[float]) -> list[Any]:
        """Turn reserved samples into the rows the distillation step consumes.

        Two channels have to be present and cannot be rebuilt afterwards: the
        candidate set captured while generating, and the teacher sequences the
        judge's hints produced. Missing either is a configuration error worth
        naming precisely, because the alternative is training a distillation
        objective on no teacher at all.
        """
        rows = []
        for index, (sample, advantage) in enumerate(zip(samples, advantages, strict=True)):
            if not sample.topk_indices or not sample.topk_log_probs:
                raise RuntimeContractError(
                    f"sample {index} carries no generation top-K; the openclawrl objective distils onto "
                    "the candidates the policy considered, so set reef.runtime_config.capture_topk"
                )
            raw = sample.extras.get("teacher_cands")
            if not raw:
                raise RuntimeContractError(
                    f"sample {index} carries no teacher candidates; the processor attaches them as "
                    "extras['teacher_cands'] once the judge has proposed a hindsight hint"
                )
            candidates = tuple(
                TeacherCandidate(
                    hint=str(entry.get("hint", "")),
                    tokens=tuple(int(token) for token in entry["teacher_tokens"]),
                )
                for entry in raw
            )
            rows.append(
                DistillationRow(
                    tokens=sample.tokens,
                    loss_mask=sample.loss_mask,
                    rollout_log_probs=sample.rollout_log_probs,
                    reward=float(advantage),
                    topk_indices=sample.topk_indices,
                    topk_log_probs=sample.topk_log_probs,
                    candidates=candidates,
                )
            )
        return rows

    def _run_training(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch to the objective the recipe's loss family names."""
        samples: Sequence[PolicySample] = payload["samples"]
        advantages: Sequence[float] = payload["advantages"]
        if payload.get("loss_family") == "openclawrl":
            rows = self._distillation_rows(samples, advantages)
            return dict(
                self._engine.openclawrl_step(
                    rows,
                    w_rl=self._openclawrl["w_rl"],
                    w_opd=self._openclawrl["w_opd"],
                    eps_lo=self._openclawrl["eps_lo"],
                    eps_hi=self._openclawrl["eps_hi"],
                    diff_clip=self._openclawrl["diff_clip"],
                    hint_selection=self._openclawrl["hint_selection"],
                    native_k=self._openclawrl["native_k"],
                    kl_coef=self._openclawrl["kl_coef"],
                )
            )
        rows = [
            TrainingRow(
                tokens=sample.tokens,
                loss_mask=sample.loss_mask,
                rollout_log_probs=sample.rollout_log_probs,
                advantages=(float(advantage),) * len(sample.loss_mask),
            )
            for sample, advantage in zip(samples, advantages, strict=True)
        ]
        if self._kl_coef:
            rows = self._apply_frozen_base_kl(rows)
        return dict(self._engine.train_step(rows))

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train through an exported adapter without changing serving weights.

        Serving is restored to its pre-step parameters before returning, so a
        candidate that Reef later rejects never touched the weights answering
        requests.
        """
        scenario_step = int(payload["rollout_id"])
        current = self.current_runtime_load_id()
        before = self._engine.adapter_snapshot()
        # One admission window covers everything that touches the live
        # parameters. The frozen-base pass zeroes ``lora_b`` in place, so
        # running it while inference is still admitted would let a request
        # generate from the bare base model.
        self._inference_admission.close(wait=True, timeout=self.inference_timeout_s)
        try:
            metrics = self._run_training(payload)
            after = self._engine.adapter_snapshot()
            delta, changed = self._engine.adapter_delta(before, after)
            metrics.update(adapter_delta_l2=delta, adapter_tensors_changed=changed)
            if changed == 0:
                # A step that moved nothing is not a trained candidate. Saying
                # so here is what keeps an unchanged adapter out of the
                # artifact stack with a success record attached.
                self._engine.apply_adapter(before)
                raise RuntimeContractError(
                    "the mlx training step left every adapter tensor unchanged; "
                    "refusing to publish an untrained candidate"
                )
            candidate_id = uuid.uuid4().hex
            destination = self._checkpoint_root / CANDIDATE_DIRNAME.format(
                scenario_step=scenario_step, candidate=candidate_id[:8]
            )
            self._engine.save_adapter(
                destination,
                origin_extra={
                    "scenario_step": scenario_step,
                    "training_job_id": candidate_id,
                    # The objective is whichever family actually trained these
                    # weights, not a constant: an adapter that misnames how it
                    # was produced cannot be reasoned about later.
                    "objective": payload.get("loss_family"),
                    "loss_family": payload.get("loss_family"),
                    "source_runtime_load_id": current,
                },
            )
            # Serving keeps the previous weights until Reef selects this
            # candidate; the trained parameters live only in the export.
            self._pending[candidate_id] = after
            self._engine.apply_adapter(before)
        except BaseException:
            # A failed step may have already applied an optimizer update, or
            # left the adapter zeroed mid-KL-pass. Serving must go back to the
            # weights it was answering with before this step began.
            self._engine.apply_adapter(before)
            raise
        finally:
            self._inference_admission.open()

        return ModelCandidate(
            candidate_id=candidate_id,
            training_job_id=candidate_id,
            checkpoint_path=str(destination),
            current_runtime_load_id=current,
            training_metrics=metrics,
        )

    def _apply_frozen_base_kl(self, rows: Sequence[Any]) -> list[Any]:
        """Add TTT-Discover's centered frozen-base KL term to each advantage.

        A direct port of ``incorporate_kl_penalty``: every response token's
        advantage moves by ``kl_coef * (mean_diff - diff)`` where ``diff`` is
        the rollout-to-base log-probability difference and ``mean_diff`` is its
        batch mean over trained tokens.
        """
        base = self._engine.base_log_probs(rows)
        numerator = 0.0
        denominator = 0
        differences: list[list[float]] = []
        for row, base_row in zip(rows, base, strict=True):
            row_difference = [
                (rollout - reference) * mask
                for rollout, reference, mask in zip(row.rollout_log_probs, base_row, row.loss_mask, strict=True)
            ]
            differences.append(row_difference)
            numerator += sum(row_difference)
            denominator += sum(row.loss_mask)
        if denominator <= 0:
            raise RuntimeContractError("the frozen-base KL term received an empty loss mask")
        average = numerator / denominator
        adjusted = []
        for row, row_difference in zip(rows, differences, strict=True):
            adjusted.append(
                TrainingRow(
                    tokens=row.tokens,
                    loss_mask=row.loss_mask,
                    rollout_log_probs=row.rollout_log_probs,
                    advantages=tuple(
                        advantage + self._kl_coef * mask * (average - difference)
                        for advantage, difference, mask in zip(
                            row.advantages, row_difference, row.loss_mask, strict=True
                        )
                    ),
                )
            )
        return adjusted

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Make a selected candidate the weights that answer new requests."""
        snapshot = self._pending.pop(candidate.candidate_id, None)
        if snapshot is None:
            raise RuntimeContractError(f"no pending mlx candidate {candidate.candidate_id!r} to activate")
        # Admission stays closed past this method. Between swapping the
        # weights and Reef committing the new head, a request would freeze the
        # old artifact and then be answered by the new weights, which the
        # weight surface correctly rejects as a runtime-load mismatch.
        # ``reconcile_training_job`` reopens once the commit is durable.
        self._inference_admission.close(wait=True, timeout=self.inference_timeout_s)
        try:
            self._engine.apply_adapter(snapshot)
            self._serving_runtime_load_id = self._engine.next_runtime_load_id()
        except BaseException:
            self._inference_admission.open()
            raise
        logger.info(
            "activated mlx candidate %s at runtime load ID %s",
            candidate.candidate_id,
            self._serving_runtime_load_id,
        )
        return ActivatedModel(candidate_id=candidate.candidate_id, runtime_load_id=self._serving_runtime_load_id)

    def current_runtime_load_id(self) -> str | None:
        """The version Reef has actually made available to new inference.

        Between activation and Reef's durable commit this deliberately lags
        :meth:`serving_runtime_load_id`: the engine already holds the new
        weights, but no request may be admitted against them yet.
        """
        return self._committed_runtime_load_id

    def reconcile_training_job(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
        scenario: str | None = None,
    ) -> None:
        """Reopen inference once Reef's commit for this step is durable.

        The second half of the activation handshake. Reef calls this after the
        publication is committed, and again at recovery, which is the first
        moment a new request can safely freeze the new head.
        """
        self._committed_runtime_load_id = self._serving_runtime_load_id
        self._inference_admission.open()

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        """Drop a rejected candidate's weights; serving already never saw them."""
        self._pending.pop(candidate.candidate_id, None)
        logger.info("rejected mlx candidate %s: %s", candidate.candidate_id, decision.reason)

    def probe_candidate(
        self,
        candidate_id: str,
        prompts: Sequence[Sequence[int]],
        *,
        max_tokens: int,
    ) -> list[str]:
        """Greedily generate from a pending candidate's weights, then restore serving.

        A candidate evaluator uses this to measure a trained-but-unselected
        candidate on a fixed probe before Reef decides whether to publish it.
        The candidate's parameters are swapped in only for the probe and only
        while inference admission is closed, and the pre-probe serving
        parameters are restored before this returns, so no live request is ever
        answered by weights Reef has not selected.
        """
        snapshot = self._pending.get(candidate_id)
        if snapshot is None:
            raise RuntimeContractError(f"no pending mlx candidate {candidate_id!r} to probe")
        self._inference_admission.close(wait=True, timeout=self.inference_timeout_s)
        try:
            serving = self._engine.adapter_snapshot()
            self._engine.apply_adapter(snapshot)
            try:
                return [
                    self._engine.generate(prompt, max_tokens=max_tokens, temperature=0.0).text for prompt in prompts
                ]
            finally:
                self._engine.apply_adapter(serving)
        finally:
            self._inference_admission.open()

    def restore_checkpoint(self, artifact: Artifact) -> str:
        """Roll serving back to a published adapter."""
        local_path = artifact.local_path
        if local_path is None:
            raise RuntimeContractError("mlx rollback requires a materialized adapter")
        self._inference_admission.close(wait=True, timeout=self.inference_timeout_s)
        try:
            self._engine.load_adapter(Path(local_path))
            self._serving_runtime_load_id = self._engine.next_runtime_load_id()
        finally:
            self._inference_admission.open()
        return self._serving_runtime_load_id


def _template_kwargs(value: Any) -> Mapping[str, Any]:
    """The deployment's chat-template defaults, checked before the engine boots.

    A misspelt key here is silent — chat templates ignore what they do not
    read — so the only thing worth rejecting is a value of the wrong shape,
    which would otherwise surface as a template error on the first request.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeContractError("reef.runtime_config.chat_template_kwargs must be a mapping")
    return {str(key): item for key, item in value.items()}


@register_runtime_kind
class MLXRuntimeFactory(RuntimeFactory):
    """Build the in-process MLX runtime from a deployment's runtime config.

    MLX is imported here, inside the call, so that a deployment which never
    selects this kind — every CUDA deployment — does not need the optional
    dependency installed to import Reef.
    """

    kind = "mlx"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> MLXRuntime:
        # The deployment's contract is checked before MLX is imported: a
        # misconfiguration is reported as itself, not as a missing extra,
        # and the checks hold on any machine.
        max_staleness = config.get("max_staleness")
        if max_staleness:
            raise RuntimeContractError(
                "the mlx runtime trains on exact-version batches only; set reef.max_staleness to 0"
            )
        checkpoint_dir = config.get("checkpoint_dir")
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
            raise RuntimeContractError("the mlx runtime requires reef.runtime_config.checkpoint_dir")
        chat_template_kwargs = _template_kwargs(config.get("chat_template_kwargs"))

        try:
            from reef.train.mlx_backend.engine import MLXEngine, MLXEngineConfig
        except ImportError as exc:
            raise RuntimeContractError(
                "the mlx runtime needs the optional MLX dependencies; install reef-infra[mlx] "
                f"on Apple Silicon ({exc})"
            ) from exc

        engine_config = MLXEngineConfig(
            model_path=model_path,
            lora_layers=int(config.get("lora_layers", 8)),
            lora_rank=int(config.get("lora_rank", 8)),
            lora_scale=float(config.get("lora_scale", 2.0)),
            lora_dropout=float(config.get("lora_dropout", 0.0)),
            lora_keys=tuple(config.get("lora_keys", ("self_attn.q_proj", "self_attn.v_proj"))),
            capture_topk=int(config.get("capture_topk", 0)),
            learning_rate=float(config.get("learning_rate", 1e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            max_tokens=int(config.get("max_tokens", 256)),
            temperature=float(config.get("temperature", 1.0)),
            top_p=float(config.get("top_p", 1.0)),
            seed=int(config.get("seed", 0)),
            micro_batch_size=int(config.get("micro_batch_size", 8)),
            log_probs_chunk_size=int(config.get("log_probs_chunk_size", 0)),
            recurrence_chunk_size=int(config.get("recurrence_chunk_size", 64)),
            checkpoint_layers=bool(config.get("checkpoint_layers", False)),
            prefill_step_size=int(config.get("prefill_step_size", 0)),
            chat_template_kwargs=chat_template_kwargs,
        )
        timeout = config.get("inference_timeout_s")
        return MLXRuntime(
            MLXEngine(engine_config),
            checkpoint_dir=checkpoint_dir,
            kl_coef=float(config.get("kl_coef", 0.0)),
            inference_timeout_s=float(timeout) if timeout else 300.0,
            adapter_name=str(config.get("adapter_name", "reef-mlx")),
            openclawrl=config.get("openclawrl") if isinstance(config.get("openclawrl"), Mapping) else None,
        )


__all__ = ["MLXRuntime", "MLXRuntimeFactory"]
