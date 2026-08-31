"""Shared exact-version and synchronized Megatron weight-update behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from itertools import chain
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from slime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import update_weights_from_distributed
from slime.utils.distributed_utils import get_gloo_group
from tqdm import tqdm

from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId, new_runtime_load_id_incarnation


class _TransportRuntimeLoadId(RuntimeLoadId):
    """Numeric formatting used only by raw Slime's disk-delta file layout."""

    def __format__(self, format_spec: str) -> str:
        return format(self.sequence, format_spec) if format_spec else str(self)

    def __sub__(self, other: int) -> int:
        return self.sequence - other


class SynchronizedWeightUpdateMixin:
    """Synchronize rank-local failures and source-only conversion phases."""

    args: Any
    model: Any
    model_name: str
    quantization_config: Any
    rollout_engines: list[Any]
    rollout_engine_lock: Any
    _is_pp_src_rank: bool
    _group_name: str
    _model_update_groups: Any
    runtime_load_id_incarnation: str
    weight_update_sequence: int
    _on_chunk: Callable[[list[tuple[str, torch.Tensor]]], None]

    def _initialize_runtime_load_id(self, incarnation: str | None = None) -> None:
        if incarnation is None:
            shared = [new_runtime_load_id_incarnation() if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(shared, src=0, group=get_gloo_group())
            incarnation = shared[0]
        if not isinstance(incarnation, str) or not incarnation:
            raise ValueError("runtime-load-ID incarnation must be a non-empty string")
        self.runtime_load_id_incarnation = incarnation
        self.weight_update_sequence = 0
        self._source_phase_sequences: dict[str, int] = {}

    @property
    def runtime_load_id(self) -> RuntimeLoadId:
        return _TransportRuntimeLoadId(self.runtime_load_id_incarnation, self.weight_update_sequence)

    @runtime_load_id.setter
    def runtime_load_id(self, value: int | RuntimeLoadId) -> None:
        if isinstance(value, RuntimeLoadId):
            self.runtime_load_id_incarnation = value.incarnation
            self.weight_update_sequence = value.sequence
        else:
            self.weight_update_sequence = int(value)

    @property
    def exact_runtime_load_id(self) -> RuntimeLoadId:
        return self.runtime_load_id

    def restore_exact_runtime_load_id(self, value: str) -> None:
        published = RuntimeLoadId.parse(value)
        if published.sequence < 1:
            raise ValueError("a published checkpoint runtime load ID must have a positive sequence")
        current = self.runtime_load_id
        if current.sequence != 0 and current != published:
            raise RuntimeError(f"cannot republish runtime load ID {published}; updater currently reports {current}")
        self.runtime_load_id_incarnation = published.incarnation
        # The mandatory republication advances once and must recreate the
        # exact durable token tied to the recovered checkpoint.
        self.weight_update_sequence = published.sequence - 1

    def _new_source_phase_prefix(self, kind: str) -> str:
        sequence = self._source_phase_sequences.get(kind, 0)
        self._source_phase_sequences[kind] = sequence + 1
        return f"{self.runtime_load_id_incarnation}:{mpu.get_pipeline_model_parallel_rank()}:{kind}:{sequence}"

    def _reset_source_phases(self) -> None:
        self._source_phase_sequences.clear()
        self._run_rank_zero_action(
            lambda: ray.get(self.rollout_engine_lock.clear_phases.remote()),
            phase="reset weight-update phase outcomes",
        )

    def _run_source_phase(self, phase_id: str, action: Callable[[], Any], *, phase: str) -> Any:
        if self._is_pp_src_rank:
            result: Any = None
            error: BaseException | None = None
            try:
                result = action()
            except BaseException as exc:
                error = exc
            error_text = None if error is None else f"{type(error).__name__}: {error}"
            try:
                ray.get(self.rollout_engine_lock.complete_phase.remote(phase_id, error_text))
            except Exception as exc:
                if error is None:
                    error = RuntimeError(f"{phase} could not publish its phase outcome")
                    error.__cause__ = exc
            if error is not None:
                self._poison_transport()
                raise error
            return result

        deadline = time.monotonic() + self._transport_timeout_s()
        while True:
            try:
                status = ray.get(self.rollout_engine_lock.phase_status.remote(phase_id))
            except Exception as exc:
                raise RuntimeError(f"{phase} could not read its source phase outcome") from exc
            if status is not None:
                if not isinstance(status, Mapping) or (
                    status.get("error") is not None
                    and (not isinstance(status.get("error"), str) or not status["error"])
                ):
                    raise RuntimeError(f"{phase} received a malformed source phase outcome")
                if status.get("error") is not None:
                    self._poison_transport()
                    raise RuntimeError(f"{phase} failed on its source rank: {status['error']}")
                return None
            if time.monotonic() >= deadline:
                self._poison_transport()
                raise TimeoutError(f"{phase} timed out waiting for its source rank")
            time.sleep(0.01)

    def _transport_timeout_s(self) -> float:
        minutes = getattr(self.args, "distributed_timeout_minutes", 10)
        if not isinstance(minutes, int | float) or isinstance(minutes, bool) or minutes <= 0:
            raise ValueError("distributed_timeout_minutes must be a positive number")
        return float(minutes) * 60

    def _raise_synchronized_update_error(self, error: BaseException | None, *, phase: str) -> None:
        local = None if error is None else f"rank {dist.get_rank()} {type(error).__name__}: {error}"
        failures: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(failures, local, group=get_gloo_group())
        failure = next((value for value in failures if value is not None), None)
        if failure is None:
            return
        if error is not None:
            raise error
        raise RuntimeError(f"{phase} failed on another trainer rank: {failure}")

    def _run_rank_zero_action(self, action: Callable[[], object], *, phase: str) -> None:
        error: BaseException | None = None
        if dist.get_rank() == 0:
            try:
                action()
            except BaseException as exc:
                error = exc
        self._raise_synchronized_update_error(error, phase=phase)

    def _poison_transport(self) -> None:
        with suppress(Exception):
            ray.get(self.rollout_engine_lock.poison.remote())

    def _send_weights(self, pbar: tqdm | None) -> None:
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            pass_error: BaseException | None = None
            try:
                phase_prefix = self._new_source_phase_prefix("distributed-fanout")
                for index, hf_chunk in enumerate(chunk_iter):

                    def fan_out(chunk: list[tuple[str, torch.Tensor]] = hf_chunk) -> None:
                        if chunk:
                            self._on_chunk(chunk)
                            self._update_bucket_weights_from_distributed(chunk, pbar=pbar)

                    self._run_source_phase(
                        f"{phase_prefix}:{index}",
                        fan_out,
                        phase="distributed weight fan-out",
                    )
            except BaseException as exc:
                pass_error = exc
            self._raise_synchronized_update_error(pass_error, phase="distributed weight pass")

    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        has_batch = False
        phase_prefix = self._new_source_phase_prefix("non-expert-conversion")
        batch_index = 0
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if has_batch and buffer_size + param_size > self.args.update_weight_buffer_size:
                yield self._convert_non_expert_batch(batch, phase_id=f"{phase_prefix}:{batch_index}")
                batch_index += 1
                batch = []
                buffer_size = 0
                has_batch = False
            if self._is_pp_src_rank:
                batch.append((name, param))
            has_batch = True
            buffer_size += param_size
        if has_batch:
            yield self._convert_non_expert_batch(batch, phase_id=f"{phase_prefix}:{batch_index}")

    def _convert_non_expert_batch(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        *,
        phase_id: str,
    ) -> list[tuple[str, torch.Tensor]]:
        def convert() -> list[tuple[str, torch.Tensor]]:
            converted: list[tuple[str, torch.Tensor]] = []
            for name, param in named_tensors:
                converted.extend(convert_to_hf(self.args, self.model_name, name, param, self.quantization_config))
            return converted

        converted = self._run_source_phase(phase_id, convert, phase="non-expert weight conversion")
        return [] if converted is None else converted

    def _iter_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        params = (
            (name, param) for name, param in named_params_and_buffers(self.args, self.model) if ".experts." in name
        )
        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        phase_prefix = self._new_source_phase_prefix("expert-conversion")
        batch_index = 0
        for name, param in params:
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if (
                batch
                and (buffer_size + param_size) * mpu.get_expert_model_parallel_world_size()
                > self.args.update_weight_buffer_size
            ):
                yield self._ep_gather_and_convert(batch, phase_id=f"{phase_prefix}:{batch_index}")
                batch_index += 1
                batch = []
                buffer_size = 0
            batch.append((name, param))
            buffer_size += param_size
        if batch:
            yield self._ep_gather_and_convert(batch, phase_id=f"{phase_prefix}:{batch_index}")

    def _ep_gather_and_convert(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        *,
        phase_id: str,
    ) -> list[tuple[str, torch.Tensor]]:
        names = [name for name, _ in named_tensors]
        all_names: list[list[str] | None] = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())
        for gathered_names in all_names:
            if gathered_names is None or len(named_tensors) != len(gathered_names):
                raise RuntimeError("expert ranks returned inconsistent tensor names")

        all_gathered_params: list[list[tuple[str, torch.Tensor]]] = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for index, (_, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handles.append(
                dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            )
            for ep_rank, gathered_names in enumerate(all_names):
                if gathered_names is None:
                    raise RuntimeError("expert parameter-name gather returned no names")
                all_gathered_params[ep_rank].append((gathered_names[index], params[ep_rank]))
        for handle in handles:
            handle.wait()
        named_tensors.clear()

        def convert() -> list[tuple[str, torch.Tensor]]:
            converted: list[tuple[str, torch.Tensor]] = []
            for name, param in chain.from_iterable(all_gathered_params):
                converted.extend(convert_to_hf(self.args, self.model_name, name, param, self.quantization_config))
            return converted

        converted = self._run_source_phase(phase_id, convert, phase="expert weight conversion")
        return [] if converted is None else converted

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                str(self.runtime_load_id),
                self.rollout_engines,
                converted_named_tensors,
                load_format=load_format,
            )
            ray.get(refs)
        except BaseException:
            self._poison_transport()
            raise
        else:
            ray.get(self.rollout_engine_lock.release.remote())
        finally:
            converted_named_tensors.clear()
        if pbar is not None:
            pbar.update(1)


__all__ = ["SynchronizedWeightUpdateMixin"]
