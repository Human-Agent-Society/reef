"""Opt-in same-forward diagnostics for native SDPO model workers.

Install through ``training.options.custom-megatron-init-path``. Output goes to
``<parent of args.save>/agentcl-worker-capture/sdpo`` for the first two pre-train
calls only. This adapter uses Slime's private ``model.forward_only`` callback
boundary temporarily; it restores that boundary even when native scoring fails.
It does not run models, change teacher weights, or register a new loss family.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist

from reef.train.slime_backend.distill.algorithm import settings_from_args
from reef.train.slime_backend.vocab_parallel import gather_log_probs_at_ids, native_topk_ids, sum_across_vocab_shards

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
TensorRows = dict[str, list[torch.Tensor]]


@dataclass(frozen=True)
class RankIdentity:
    rank: int
    tensor_rank: int
    tensor_world: int
    data_rank: int
    pipeline_rank: int
    pipeline_world: int

    @property
    def writes(self) -> bool:
        return self.tensor_rank == self.data_rank == 0 and self.pipeline_rank == self.pipeline_world - 1


def rank_identity() -> RankIdentity:
    from megatron.core import mpu

    return RankIdentity(
        dist.get_rank() if dist.is_initialized() else 0,
        mpu.get_tensor_model_parallel_rank(),
        mpu.get_tensor_model_parallel_world_size(),
        mpu.get_data_parallel_rank(with_context_parallel=False),
        mpu.get_pipeline_model_parallel_rank(),
        mpu.get_pipeline_model_parallel_world_size(),
    )


def json_value(value: object) -> JsonValue:
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {str(key): json_value(item) for key, item in value.items()}
    raise ValueError("worker diagnostics require JSON-compatible values")


def checksum(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class WorkerCapture:
    """Bounded state for one native worker; all tensor ranks execute scoring."""

    def __init__(self, args: Namespace) -> None:
        self.directory = Path(args.save).resolve().parent / "agentcl-worker-capture" / "sdpo"
        self.pass_count = 0
        self.microbatch_count = 0
        self.active = False
        self.runtime_load_id = ""
        self.written_bytes = 0
        self.teacher_records: dict[int, JsonObject] = {}
        self.runtime: JsonObject = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "slime": importlib.metadata.version("slime"),
            "reef_infra": importlib.metadata.version("reef-infra"),
        }

    def write(self, stage: str, args: Namespace, identity: RankIdentity, samples: list[JsonValue]) -> None:
        if not identity.writes:
            return
        value: JsonObject = {
            "schema_version": 1,
            "method": "sdpo",
            "capture_stage": stage,
            "pre_train_call": self.pass_count,
            "loss_microbatch": self.microbatch_count if stage == "student_loss" else None,
            "runtime_load_id": self.runtime_load_id,
            "rank": json_value(asdict(identity)),
            "runtime": self.runtime,
            "settings": json_value(asdict(settings_from_args(args))),
            "native_settings": {
                "sequence_length": int(args.seq_length),
                "max_tokens_per_gpu": args.max_tokens_per_gpu,
                "dynamic_batch_size": bool(args.use_dynamic_batch_size),
                "calculate_per_token_loss": bool(args.calculate_per_token_loss),
                "context_parallel_size": int(args.context_parallel_size),
                "virtual_pipeline_model_parallel_size": args.virtual_pipeline_model_parallel_size,
            },
            "rollout_temperature": float(args.rollout_temperature),
            "log_probs_chunk_size": int(args.log_probs_chunk_size),
            "samples": samples,
            "sample_order_sha256": checksum(samples),
            "bounds": {"pre_train_calls": 2, "samples_per_pass": 64, "suffix_tokens_per_pass": 32768},
            "private_integration_boundary": "Slime model.forward_only callback; restored in finally",
            "optimizer_execution_verified": False,
        }
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if len(serialized) > 128 * 1024 * 1024 or self.written_bytes + len(serialized) > 256 * 1024 * 1024:
            raise ValueError("worker capture exceeds its private JSON byte budget")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        filename = f"pass-{self.pass_count:02d}-{stage}-{self.microbatch_count:04d}.json"
        destination = self.directory / filename
        if destination.exists():
            raise ValueError("worker capture requires a fresh output directory")
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".capture-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        self.written_bytes += len(serialized)

    def observe_teacher(
        self,
        logits: torch.Tensor,
        args: Namespace,
        tokens: list[torch.Tensor],
        total_lengths: list[int],
        response_lengths: list[int],
        result: TensorRows,
        sample_indices: list[int],
    ) -> None:
        """Read canonical callback arrays and check arithmetic on its exact inputs."""
        from megatron.core import mpu
        from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

        from reef.train.slime_backend.distill.teacher import TEACHER_LOG_PROB_FLOOR

        identity = rank_identity()
        group = mpu.get_tensor_model_parallel_group()
        settings = settings_from_args(args)
        with torch.no_grad():
            _, runtime = get_log_probs_and_entropy(
                logits.detach().float(),
                args=args,
                unconcat_tokens=tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=False,
            )
            tempered = logits.detach().squeeze(0).float()
            if args.rollout_temperature != 1.0:
                tempered = tempered / args.rollout_temperature
            offset = 0
            for index, (sequence, total, length, occurrence) in enumerate(
                zip(tokens, total_lengths, response_lengths, sample_indices, strict=True)
            ):
                start = offset + total - length - 1
                rows = tempered[start : offset + total - 1]
                sampled = sequence[-length:].to(device=rows.device, dtype=torch.long)[:, None]
                actual = gather_log_probs_at_ids(rows, sampled, group, identity.tensor_world, identity.tensor_rank)[
                    :, 0
                ]
                record: JsonObject = {
                    "teacher_tokens": json_value(sequence),
                    "teacher_token_sha256": checksum(json_value(sequence)),
                    "response_ids": json_value(sampled[:, 0]),
                    "teacher_logit_positions": list(range(start, start + length)),
                    "teacher_runtime_sampled_log_probs": json_value(runtime["log_probs"][index]),
                    "teacher_arithmetic_sampled_log_probs": json_value(actual),
                }
                if settings.exact:
                    stored = result["distill_teacher_log_probs"][index]
                    alternatives = native_topk_ids(
                        rows,
                        min(5, rows.size(-1) * identity.tensor_world),
                        group,
                        identity.tensor_world,
                        identity.tensor_rank,
                    )
                    ids = torch.cat([sampled, alternatives], dim=-1)
                    shard_start = identity.tensor_rank * stored.size(-1)
                    owned = (ids >= shard_start) & (ids < shard_start + stored.size(-1))
                    local_ids = (ids - shard_start).clamp(0, stored.size(-1) - 1).to(stored.device)
                    selected = stored.gather(-1, local_ids).float().to(rows.device).masked_fill(~owned, 0)
                    selected = sum_across_vocab_shards(selected, group, identity.tensor_world)
                    # Keep full-vocabulary teacher storage on the host; transfer only row statistics.
                    row_chunk_size = int(args.log_probs_chunk_size) if args.log_probs_chunk_size > 0 else 128
                    mass_parts = [
                        stored[row_start : row_start + row_chunk_size].float().exp().sum(-1)
                        for row_start in range(0, stored.size(0), row_chunk_size)
                    ]
                    mass = sum_across_vocab_shards(torch.cat(mass_parts).to(rows.device), group, identity.tensor_world)
                    record.update(
                        teacher_sampled_log_probs=json_value(selected[:, 0]),
                        teacher_alternative_ids=json_value(alternatives),
                        teacher_alternative_log_probs=json_value(selected[:, 1:]),
                        teacher_stored_row_probability_mass=json_value(mass),
                        teacher_expected_stored_sampled_log_probs=json_value(
                            actual.clamp_min(TEACHER_LOG_PROB_FLOOR).half().float()
                        ),
                        teacher_storage_dtype=str(stored.dtype),
                        teacher_log_prob_floor=TEACHER_LOG_PROB_FLOOR,
                    )
                else:
                    ids = result["distill_teacher_topk_ids"][index].to(device=rows.device, dtype=torch.long)
                    at_ids = gather_log_probs_at_ids(rows, ids, group, identity.tensor_world, identity.tensor_rank)
                    record.update(
                        teacher_topk_ids=json_value(ids),
                        teacher_topk_log_probs=json_value(result["distill_teacher_topk_log_probs"][index]),
                        teacher_arithmetic_topk_log_probs=json_value(at_ids),
                        teacher_sampled_log_probs=json_value(result["distill_teacher_sampled_log_probs"][index]),
                    )
                self.teacher_records[occurrence] = record
                offset += total


capture: WorkerCapture | None = None


def validate_bounds(rollout: dict[str, object]) -> None:
    lengths = cast(list[int], rollout["response_lengths"])
    if len(lengths) > 64 or sum(lengths) > 32768:
        raise ValueError("worker capture exceeds its sample or suffix token budget")


def pre_train(actor: object, rollout: dict[str, object]) -> None:
    """Delegate native pre-train once; observe only its canonical teacher pass."""
    from recipes.sdpo.slime.objective import sdpo_actor_pre_train

    if capture is None:
        sdpo_actor_pre_train(actor, rollout)
        return
    from slime.backends.megatron_utils import model

    from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor

    worker = cast(ReefMegatronTrainRayActor, actor)
    state = capture
    state.pass_count += 1
    state.microbatch_count = 0
    state.active = state.pass_count <= 2
    state.teacher_records.clear()
    if not state.active:
        sdpo_actor_pre_train(actor, rollout)
        return
    state.runtime_load_id = worker.get_runtime_load_id()
    original = model.forward_only
    settings = settings_from_args(worker.args)
    expected_passes = 2 if not settings.exact and settings.top_k_source == "student" else 1
    observed_passes = 0

    def _forward_only(callback, args, modules, iterators, microbatches, *arguments, **options):
        nonlocal observed_passes
        observed_passes += 1
        if observed_passes != expected_passes:
            return original(callback, args, modules, iterators, microbatches, *arguments, **options)
        schedule = iterators[0].micro_batch_indices
        cursor = 0

        def _callback(logits, *, args, unconcat_tokens, total_lengths, response_lengths, with_entropy=False):
            nonlocal cursor
            result = callback(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=with_entropy,
            )
            validate_bounds(rollout)
            if settings.top_k > 100:
                raise ValueError("worker capture supports at most 100 canonical top-K ids")
            state.observe_teacher(
                logits, args, unconcat_tokens, total_lengths, response_lengths, result[1], schedule[cursor]
            )
            cursor += 1
            return result

        return original(_callback, args, modules, iterators, microbatches, *arguments, **options)

    # compute_teacher_rows imports this private Slime boundary inside pre-train.
    model.forward_only = _forward_only
    try:
        sdpo_actor_pre_train(actor, rollout)
    finally:
        model.forward_only = original
    identity = rank_identity()
    samples: list[JsonValue] = []
    for local_index, record in sorted(state.teacher_records.items()):
        sample_index = cast(list[int], rollout["sample_indices"])[local_index]
        record.update(
            local_sample_index=local_index,
            sample_index=sample_index,
            rollout_id=cast(list[int], rollout["rollout_ids"])[local_index],
            producing_runtime_load_id=(
                cast(list[str], rollout["producing_runtime_load_ids"])[local_index]
                if rollout.get("producing_runtime_load_ids") is not None
                else None
            ),
            executing_runtime_load_id=state.runtime_load_id,
            loss_mask=json_value(cast(list[torch.Tensor], rollout["loss_masks"])[local_index]),
        )
        # These arrays are the actual post-pretrain targets, not a diagnostic forward.
        if settings.exact:
            record["canonical_teacher_shape"] = json_value(
                cast(list[torch.Tensor], rollout["distill_teacher_log_probs"])[local_index].shape
            )
        else:
            record["canonical_teacher_topk_ids"] = json_value(
                cast(list[torch.Tensor], rollout["distill_teacher_topk_ids"])[local_index]
            )
            record["canonical_teacher_topk_log_probs"] = json_value(
                cast(list[torch.Tensor], rollout["distill_teacher_topk_log_probs"])[local_index]
            )
            record["canonical_teacher_sampled_log_probs"] = json_value(
                cast(list[torch.Tensor], rollout["distill_teacher_sampled_log_probs"])[local_index]
            )
        samples.append(record)
    if identity.pipeline_rank == identity.pipeline_world - 1 and observed_passes != expected_passes:
        raise ValueError("worker capture observed an unexpected canonical forward-only pass count")
    if identity.pipeline_rank == identity.pipeline_world - 1 and len(samples) != len(
        cast(list[int], rollout["response_lengths"])
    ):
        raise ValueError("worker capture did not observe every canonical teacher suffix")
    state.write("teacher_pre_train", worker.args, identity, samples)


def captured_loss(
    args: Namespace,
    batch: dict[str, object],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return native loss and metrics unchanged after detached same-logit scoring."""
    from recipes.sdpo.slime.objective import sdpo_loss

    result = sdpo_loss(args, batch, logits, sum_of_sample_mean)
    if capture is None or not capture.active:
        return result
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    state = capture
    state.microbatch_count += 1
    validate_bounds(batch)
    with torch.no_grad():
        _, outputs = get_log_probs_and_entropy(
            logits.detach().float(),
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            with_entropy=False,
        )
    samples: list[JsonValue] = []
    for index, student in enumerate(outputs["log_probs"]):
        length = cast(list[int], batch["response_lengths"])[index]
        tokens = cast(list[torch.Tensor], batch["unconcat_tokens"])[index]
        teacher_tokens = cast(list[torch.Tensor], batch["teacher_tokens"])[index]
        mask = cast(list[torch.Tensor], batch["loss_masks"])[index]
        effective_mask = mask.clone()
        effective_mask[: settings_from_args(args).skip_response_tokens] = 0
        sample: JsonObject = {
            "packing_index": index,
            "sample_index": cast(list[int], batch["sample_indices"])[index],
            "rollout_id": cast(list[int], batch["rollout_ids"])[index],
            "producing_runtime_load_id": (
                cast(list[str], batch["producing_runtime_load_ids"])[index]
                if batch.get("producing_runtime_load_ids") is not None
                else None
            ),
            "executing_runtime_load_id": state.runtime_load_id,
            "tokens": json_value(tokens),
            "student_token_sha256": checksum(json_value(tokens)),
            "teacher_token_sha256": checksum(json_value(teacher_tokens)),
            "response_ids": json_value(tokens[-length:]),
            "response_length": length,
            "total_length": cast(list[int], batch["total_lengths"])[index],
            "loss_mask": json_value(mask),
            "effective_loss_mask": json_value(effective_mask),
            "student_log_probs": json_value(student),
            "rollout_log_probs": json_value(cast(list[torch.Tensor], batch["rollout_log_probs"])[index]),
            "distill_sample_weight": float(cast(list[float], batch["distill_sample_weights"])[index]),
        }
        if not torch.equal(tokens[-length:].cpu(), teacher_tokens[-length:].cpu()):
            raise ValueError("worker capture teacher suffix differs from student response ids")
        samples.append(sample)
    state.write("student_loss", args, rank_identity(), samples)
    return result


def initialize_actor(actor: object) -> None:
    """Bind diagnostics after Reef resolves the canonical method hook paths."""
    from reef.train.slime_backend.reef_adapters.megatron.train_actor import ReefMegatronTrainRayActor

    global capture
    worker = cast(ReefMegatronTrainRayActor, actor)
    args = worker.args
    if (
        args.custom_loss_function_path != "recipes.sdpo.slime.objective.sdpo_loss"
        or args.reef_actor_pre_train_hook_path != "recipes.sdpo.slime.objective.sdpo_actor_pre_train"
    ):
        raise ValueError("worker capture requires unmodified native SDPO hooks")
    if args.context_parallel_size != 1 or args.virtual_pipeline_model_parallel_size not in (None, 1):
        raise ValueError("worker capture requires CP1 without virtual pipeline interleaving")
    capture = WorkerCapture(args)
    args.custom_loss_function_path = "recipes.sdpo.examples.agentcl.worker_capture.captured_loss"
    args.reef_actor_pre_train_hook_path = "recipes.sdpo.examples.agentcl.worker_capture.pre_train"


def install(args: Namespace) -> None:
    """Register actor initialization through the existing custom Megatron init hook."""
    from recipes.sdpo.slime import objective as native_module
    from reef.train.slime_backend.algorithm import objective

    if args.loss_family != "sdpo":
        raise ValueError("the SDPO worker capture hook requires the native sdpo loss family")
    args.reef_external_batch_keys = tuple(
        dict.fromkeys(
            (
                *args.reef_external_batch_keys,
                "teacher_tokens",
                "sample_indices",
                "rollout_ids",
                "producing_runtime_load_ids",
            )
        )
    )

    def _initialize(actor: object) -> None:
        initialize_actor(actor)

    # Reef resolves actor-init registrations relative to the native objective module.
    _initialize.__module__ = native_module.__name__
    _initialize.__name__ = "agentcl_initialize_capture"
    native_module.agentcl_initialize_capture = objective("reef_actor_init_hook_path")(_initialize)
