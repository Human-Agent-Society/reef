"""Lightweight Reef hooks loaded by Slime worker/role initialization."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.train.slime_backend.algorithm import SlimeAlgorithm, resolve_args_loss_family, resolve_objective_paths
from reef.train.slime_backend.loss_families import UnknownLossFamilyError

_WORKER_METRICS: dict[str, float] = {}
_WORKER_STEP_METRICS: list[dict[str, float]] = []
#: Key under which ``drain_worker_metrics`` returns the per-optimizer-step
#: metric dicts of a training job, in order. Slime logs one dict per step
#: under the ``train/step`` step key; the flat merge above keeps only the
#: last, this keeps them all for experiment trackers that plot every step.
WORKER_STEP_METRICS_KEY = "train_steps"


def _loss_family_spec(args) -> SlimeAlgorithm | None:
    """Resolve the worker's loss-family spec from ``args``, or ``None``.

    Mirrors :func:`resolve_objective_paths`: the driver stamps ``loss_family``
    before workers start, so the spec's wire-surface declarations are readable
    here without the driver forwarding each one.

    ``None`` means a bridge booted without a family, or with one this process
    cannot resolve. The key-set declarations survive that: the driver mirrors
    them onto ``args`` (``configure_reef_loss_args``) and every caller unions
    both sources. ``critic_value_head_zero_init`` has no args-carried mirror,
    so it is the one declaration that needs the spec.
    """
    if not getattr(args, "loss_family", None):
        return None
    try:
        return resolve_args_loss_family(args)
    except UnknownLossFamilyError:
        return None


def resolve_tensor_dtype(name: str):
    """Map a wire-declared dtype name (``rollout_tensor_dtypes``) to a torch dtype."""
    import torch

    dtypes = {"int": torch.int, "long": torch.long, "float32": torch.float32}
    try:
        return dtypes[name]
    except KeyError:
        raise ValueError(f"unknown tensor dtype name {name!r}; expected one of {sorted(dtypes)}") from None


def reef_rollout_env_vars(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that must cross Slime's explicit Ray runtime boundary."""
    source = os.environ if environ is None else environ
    values = {name: value for name, value in source.items() if name.startswith("SGLANG_")}
    host_ip = source.get("SLIME_HOST_IP", "").strip()
    if not host_ip:
        return values

    bypass: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        bypass.extend(part.strip() for part in source.get(name, "").split(",") if part.strip())
    bypass.extend(("127.0.0.1", "localhost", "::1", host_ip.strip("[]")))
    no_proxy = ",".join(dict.fromkeys(bypass))
    values.update(SLIME_HOST_IP=host_ip, NO_PROXY=no_proxy, no_proxy=no_proxy)
    return values


def reef_node_ip_and_free_port(start_port: int = 10000, consecutive: int = 1):
    """Use Slime's deployment address for its train-actor rendezvous."""
    from slime.utils.http_utils import get_host_info
    from slime.utils.misc import get_free_port

    address = get_host_info()[1].strip("[]")
    return address, get_free_port(start_port=start_port, consecutive=consecutive)


def initialize_megatron_objective(args) -> None:
    """Run a user's hook, resolve objectives, and install worker-local adapters."""
    chained_path = getattr(args, "reef_chained_megatron_init_path", None)
    if chained_path:
        from slime.utils.misc import load_function

        load_function(chained_path)(args)

    resolve_objective_paths(args)
    _install_external_batch_keys(args)
    _install_step_batch_size()
    _install_metric_capture()
    _install_rollout_logging()
    _install_critic_metrics(args)
    _install_critic_hf_bootstrap(args)
    _install_lora_hf_bootstrap(args)
    _install_versioned_updaters()
    _install_pg_primitive(args)


def _install_external_batch_keys(args) -> None:
    from slime.backends.megatron_utils import model

    spec = _loss_family_spec(args)
    declared = spec.external_batch_keys if spec is not None else ()
    external_keys = tuple(dict.fromkeys((*declared, *getattr(args, "reef_external_batch_keys", ()))))
    current = model.get_batch
    if getattr(current, "_reef_external_keys", False):
        return

    def get_batch(data_iterator, keys, *call_args, **kwargs):
        extended = [*keys, *(key for key in external_keys if key not in keys)]
        return current(data_iterator, extended, *call_args, **kwargs)

    marked_get_batch: Any = get_batch
    marked_get_batch._reef_external_keys = True
    model.get_batch = get_batch


def _install_step_batch_size() -> None:
    from slime.backends.megatron_utils import loss, model

    current = loss.loss_function
    if getattr(current, "_reef_step_batch_size", False):
        return

    def loss_function(args, batch, num_microbatches, step_global_batch_size, logits):
        batch["step_global_batch_size"] = step_global_batch_size
        return current(args, batch, num_microbatches, step_global_batch_size, logits)

    marked_loss_function: Any = loss_function
    marked_loss_function._reef_step_batch_size = True
    loss.loss_function = loss_function
    model.loss_function = loss_function


def _install_metric_capture() -> None:
    from slime.utils import logging_utils

    current = logging_utils.log
    if getattr(current, "_reef_metric_capture", False):
        return

    def log(args, metrics, step_key):
        record_worker_metrics(metrics)
        if step_key == "train/step":
            record_worker_step(metrics)
        return current(args, metrics, step_key)

    marked_log: Any = log
    marked_log._reef_metric_capture = True
    logging_utils.log = log


def _install_rollout_logging() -> None:
    from slime.backends.megatron_utils import actor, data

    current = data.log_rollout_data
    if getattr(current, "_reef_external_fields", False):
        return
    # The shared wire layer attaches these provenance fields for every loss
    # family, but Slime's rollout logger only aggregates numeric values. Loss
    # families hide their own fields through their declared
    # ``rollout_log_skip_keys``.
    skipped = {
        "producing_runtime_load_ids",
        "producing_runtime_load_spans",
    }

    def log_rollout_data(rollout_id, args, rollout_data):
        spec = _loss_family_spec(args)
        declared = spec.rollout_log_skip_keys if spec is not None else ()
        hidden = skipped | set(declared) | set(getattr(args, "reef_rollout_log_skip_keys", ()) or ())
        prepared = {key: value for key, value in rollout_data.items() if key not in hidden}
        # Integer columns (masks, ids) cannot be averaged as they are; cast the
        # ones the family declared so the logger reports their rate.
        dtypes = dict(spec.rollout_tensor_dtypes) if spec is not None else {}
        dtypes.update(getattr(args, "reef_rollout_tensor_dtypes", None) or {})
        for key, name in dtypes.items():
            if name in ("int", "long") and key in prepared:
                prepared[key] = [value.float() for value in prepared[key]]
        return current(rollout_id, args, prepared)

    marked_log_rollout_data: Any = log_rollout_data
    marked_log_rollout_data._reef_external_fields = True
    data.log_rollout_data = log_rollout_data
    actor.log_rollout_data = log_rollout_data


def _install_critic_metrics(args) -> None:
    import torch
    from slime.backends.megatron_utils import loss

    spec = _loss_family_spec(args)
    mask_key = spec.critic_value_mask_key if spec is not None else None
    current = loss.value_loss_function
    if getattr(current, "_reef_explained_variance", False):
        return

    def value_loss_function(args, batch, logits, sum_of_sample_mean):
        value, metrics = current(args, batch, logits, sum_of_sample_mean)
        with torch.no_grad():
            returns = torch.cat(batch["returns"], dim=0)
            old_values = torch.cat(batch["values"], dim=0)
            masks = batch.get(mask_key) if mask_key else None
            if masks is not None:
                selected = torch.cat(masks, dim=0).to(device=returns.device, dtype=torch.bool)
                if selected.numel() != returns.numel():
                    raise RuntimeError(f"{mask_key}/returns length mismatch: {selected.numel()} vs {returns.numel()}")
                returns = returns[selected]
                old_values = old_values[selected]
            if returns.numel() == 0 or returns.var(unbiased=False) == 0:
                explained_variance = returns.new_tensor(0.0)
            else:
                explained_variance = 1.0 - (returns - old_values).var(unbiased=False) / returns.var(unbiased=False)
        metrics = dict(metrics)
        metrics["explained_variance"] = explained_variance.detach()
        return value, metrics

    marked_value_loss: Any = value_loss_function
    marked_value_loss._reef_explained_variance = True
    loss.value_loss_function = value_loss_function


def _install_critic_hf_bootstrap(args) -> None:
    """Leave a new scalar value-head bias at its local zero initialization.

    Installed for loss families that declare ``critic_value_head_zero_init``:
    their critic's value head has no counterpart tensor in the HF checkpoint,
    so its bias lookup is answered with zeros instead of a failing HF read.
    """
    spec = _loss_family_spec(args)
    if spec is None or not spec.critic_value_head_zero_init:
        return
    import torch
    from slime.backends.megatron_utils import hf_to_megatron

    current = hf_to_megatron.load_model_hf_weights
    if getattr(current, "_reef_critic_value_head", False):
        return

    def load_model_hf_weights(args, model, path, config, get_hf_tensor):
        def critic_hf_tensor(name, reader, hf_config):
            normalized = name
            while normalized.startswith("module."):
                normalized = normalized.removeprefix("module.")
            normalized = normalized.removeprefix("language_model.")
            if normalized == "output_layer.bias":
                return torch.zeros(1)
            return get_hf_tensor(name, reader, hf_config)

        return current(args, model, path, config, critic_hf_tensor)

    marked_loader: Any = load_model_hf_weights
    marked_loader._reef_critic_value_head = True
    hf_to_megatron.load_model_hf_weights = load_model_hf_weights


def _install_lora_hf_bootstrap(args) -> None:
    """Load an HF base model through Bridge wrappers without loading adapters."""
    from reef.train.slime_backend.reef_adapters.megatron.lora import is_megatron_lora_parameter, megatron_lora_enabled

    if not megatron_lora_enabled(args):
        return

    from slime.backends.megatron_utils import hf_to_megatron
    from slime.backends.megatron_utils.update_weight import common as update_common

    current_loader = hf_to_megatron.load_model_hf_weights
    if getattr(current_loader, "_reef_lora_hf_bootstrap", False):
        return
    current_named_tensors = update_common.named_params_and_buffers

    def base_named_tensors(*loader_args, **loader_kwargs):
        for name, parameter in current_named_tensors(*loader_args, **loader_kwargs):
            if is_megatron_lora_parameter(name):
                continue
            yield name.replace(".to_wrap.", "."), parameter

    def load_model_hf_weights(*loader_args, **loader_kwargs):
        previous = update_common.named_params_and_buffers
        update_common.named_params_and_buffers = base_named_tensors
        try:
            return current_loader(*loader_args, **loader_kwargs)
        finally:
            update_common.named_params_and_buffers = previous

    marked_loader: Any = load_model_hf_weights
    marked_loader._reef_lora_hf_bootstrap = True
    hf_to_megatron.load_model_hf_weights = load_model_hf_weights


def _install_versioned_updaters() -> None:
    from slime.backends.megatron_utils import actor

    from reef.train.slime_backend.reef_adapters.weight_updaters import (
        ReefUpdateWeightFromDisk,
        ReefUpdateWeightFromDiskDelta,
        ReefUpdateWeightFromDistributed,
        ReefUpdateWeightFromTensor,
    )

    actor.UpdateWeightFromDisk = ReefUpdateWeightFromDisk
    actor.UpdateWeightFromDistributed = ReefUpdateWeightFromDistributed
    actor.UpdateWeightFromTensor = ReefUpdateWeightFromTensor
    from slime.backends.megatron_utils.update_weight import update_weight_from_disk_delta

    update_weight_from_disk_delta.UpdateWeightFromDiskDelta = ReefUpdateWeightFromDiskDelta


def _install_pg_primitive(args) -> None:
    path = getattr(args, "custom_pg_loss_function_path", None)
    if not path:
        return
    from slime.backends.megatron_utils import loss
    from slime.utils.misc import load_function

    primitive = load_function(path)

    def compute_custom_pg_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high):
        return primitive(args, ppo_kl, log_probs, advantages)

    # ``advantage_estimator`` was already routed to "cispo" driver-side
    # (``configure_reef_loss_args``); the worker only swaps the callsite.
    loss.compute_cispo_loss = compute_custom_pg_loss


def record_worker_metrics(metrics: Mapping[str, Any]) -> None:
    """Merge numeric worker metrics for the bridge's next drain."""
    for name, value in metrics.items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            _WORKER_METRICS[str(name)] = float(value)


def record_worker_step(metrics: Mapping[str, Any]) -> None:
    """Append one optimizer step's numeric metrics for the bridge's next drain."""
    numeric = {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if numeric:
        _WORKER_STEP_METRICS.append(numeric)


def drain_worker_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = dict(_WORKER_METRICS)
    _WORKER_METRICS.clear()
    if _WORKER_STEP_METRICS:
        metrics[WORKER_STEP_METRICS_KEY] = list(_WORKER_STEP_METRICS)
        _WORKER_STEP_METRICS.clear()
    return metrics


def configure_critic_objective(args) -> None:
    """Project the Reef loss family onto a derived Slime critic namespace."""
    chained_path = getattr(args, "reef_chained_critic_args_hook_path", None)
    if chained_path:
        from slime.utils.misc import load_function

        load_function(chained_path)(args)

    # A critic can be requested without a Reef loss family (--use-critic on a
    # driver that names none, or Slime deriving it from --advantage-estimator
    # ppo); there is then no family critic policy to project.
    spec = _loss_family_spec(args)
    if spec is not None:
        spec.configure_critic_args(args)
