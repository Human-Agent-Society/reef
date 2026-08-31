from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.reef_adapters import worker_hooks


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    parent_name, child = name.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(parent, child, module, raising=False)
    return module


@pytest.mark.unit
def test_node_address_honors_deployment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    http_utils = importlib.import_module("slime.utils.http_utils")
    misc = importlib.import_module("slime.utils.misc")
    monkeypatch.setattr(misc, "get_free_port", lambda *, start_port, consecutive: start_port + consecutive)
    monkeypatch.setattr(http_utils, "get_host_info", lambda: ("worker", "[2001:db8::1]"))

    assert worker_hooks.reef_node_ip_and_free_port(12000, 3) == ("2001:db8::1", 12003)


@pytest.mark.unit
def test_rollout_ray_environment_preserves_host_and_local_proxy_bypass() -> None:
    values = worker_hooks.reef_rollout_env_vars(
        {
            "SLIME_HOST_IP": "[2001:db8::1]",
            "NO_PROXY": "metadata.internal,localhost",
            "no_proxy": "metadata.internal",
        }
    )

    assert values["SLIME_HOST_IP"] == "[2001:db8::1]"
    assert values["NO_PROXY"].split(",") == [
        "metadata.internal",
        "localhost",
        "127.0.0.1",
        "::1",
        "2001:db8::1",
    ]
    assert values["no_proxy"] == values["NO_PROXY"]
    assert worker_hooks.reef_rollout_env_vars({}) == {}


@pytest.mark.unit
def test_external_batch_and_step_size_hooks_are_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def get_batch(_iterator, keys, *args, **kwargs):
        seen.append((keys, args, kwargs))
        return "batch"

    model = _stub_module(monkeypatch, "slime.backends.megatron_utils.model", get_batch=get_batch)
    loss = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.loss",
        loss_function=lambda _args, batch, _count, step_size, _logits: (batch, step_size),
    )

    from reef.train.slime_backend.loss_families import LOSS_FAMILIES

    # The forwarded keys come from the family's declaration alone: the spec
    # resolved from loss_family (the direct-start_bridge fallback) unioned
    # with the driver-stamped reef_external_batch_keys.
    args = types.SimpleNamespace(
        loss_family="openclawrl",
        reef_external_batch_keys=LOSS_FAMILIES.resolve("openclawrl").external_batch_keys,
    )
    worker_hooks._install_external_batch_keys(args)
    wrapped_get_batch = model.get_batch
    worker_hooks._install_external_batch_keys(args)
    assert model.get_batch is wrapped_get_batch
    assert model.get_batch(None, ["tokens"], "extra", flag=True) == "batch"
    assert "topk_indices" in seen[0][0]
    assert "action_masks" not in seen[0][0]

    model = _stub_module(monkeypatch, "slime.backends.megatron_utils.model", get_batch=get_batch)
    worker_hooks._install_external_batch_keys(types.SimpleNamespace(loss_family="sao"))
    assert model.get_batch(None, ["tokens"]) == "batch"
    assert "action_masks" in seen[-1][0]

    worker_hooks._install_step_batch_size()
    wrapped_loss = loss.loss_function
    worker_hooks._install_step_batch_size()
    batch: dict[str, object] = {}
    assert loss.loss_function(None, batch, 2, 16, None) == (batch, 16)
    assert batch["step_global_batch_size"] == 16
    assert model.loss_function is wrapped_loss


@pytest.mark.unit
def test_metric_and_rollout_logging_hooks_keep_reef_fields_local(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[object] = []
    logging_utils = _stub_module(
        monkeypatch,
        "slime.utils.logging_utils",
        log=lambda args, metrics, step: logged.append((args, metrics, step)) or "logged",
    )
    worker_hooks.drain_worker_metrics()
    worker_hooks._install_metric_capture()

    assert logging_utils.log("args", {"loss": 1.5, "flag": True, "name": "x"}, "train") == "logged"
    assert worker_hooks.drain_worker_metrics() == {"loss": 1.5}
    assert worker_hooks.drain_worker_metrics() == {}
    # Slime's per-optimizer-step dicts (step key train/step) are kept in
    # order beside the flat last-value merge.
    logging_utils.log("args", {"train/loss": 1.0, "train/step": 0, "name": "x"}, "train/step")
    logging_utils.log("args", {"train/loss": 0.8, "train/step": 1}, "train/step")
    assert worker_hooks.drain_worker_metrics() == {
        "train/loss": 0.8,
        "train/step": 1.0,
        "train_steps": [{"train/loss": 1.0, "train/step": 0.0}, {"train/loss": 0.8, "train/step": 1.0}],
    }
    assert worker_hooks.drain_worker_metrics() == {}

    captured: list[dict[str, object]] = []
    data = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.data",
        log_rollout_data=lambda _rollout_id, _args, values: captured.append(values) or "stored",
    )
    actor = _stub_module(monkeypatch, "slime.backends.megatron_utils.actor")
    worker_hooks._install_rollout_logging()
    values = {
        "tokens": [torch.tensor([1])],
        "topk_log_probs": [torch.tensor([[-0.2]])],
        "producing_runtime_load_ids": ["inc:4"],
        "producing_runtime_load_spans": [[{"start": 0, "end": 1, "runtime_load_id": "inc:4"}]],
    }
    from reef.train.slime_backend.loss_families import LOSS_FAMILIES

    log_args = types.SimpleNamespace(
        loss_family="openclawrl",
        reef_rollout_log_skip_keys=LOSS_FAMILIES.resolve("openclawrl").rollout_log_skip_keys,
    )
    assert data.log_rollout_data(3, log_args, values) == "stored"
    assert "topk_log_probs" not in captured[0]
    assert "producing_runtime_load_ids" not in captured[0]
    assert "producing_runtime_load_spans" not in captured[0]
    assert values["producing_runtime_load_ids"] == ["inc:4"]
    assert actor.log_rollout_data is data.log_rollout_data

    # A family's declared skip keys resolve from loss_family alone, so the
    # direct-start_bridge path (no driver stamping) hides them too.
    # Declared int columns are cast so the numeric logger can average them.
    sao_values = {
        "tokens": [torch.tensor([1])],
        "action_masks": [torch.tensor([1], dtype=torch.int32)],
        "rollout_created_ats": [None],
    }
    assert data.log_rollout_data(4, types.SimpleNamespace(loss_family="sao"), sao_values) == "stored"
    assert "rollout_created_ats" not in captured[1]
    assert "tokens" in captured[1]
    assert captured[1]["action_masks"][0].dtype == torch.float32


@pytest.mark.unit
def test_critic_metric_hook_reports_masked_explained_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    loss = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.loss",
        value_loss_function=lambda *_args: (torch.tensor(2.0), {"value_loss": torch.tensor(2.0)}),
    )
    worker_hooks._install_critic_metrics(SimpleNamespace(loss_family="sao"))

    value, metrics = loss.value_loss_function(
        None,
        {
            "returns": [torch.tensor([1.0, 3.0, 5.0])],
            "values": [torch.tensor([1.0, 2.0, 1.0])],
            "action_masks": [torch.tensor([1, 1, 0])],
        },
        None,
        None,
    )

    assert value.item() == 2.0
    assert metrics["explained_variance"].item() == pytest.approx(0.75)


@pytest.mark.unit
def test_critic_hf_bootstrap_keeps_the_new_value_head_bias_local(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[torch.Tensor] = []

    def load_model_hf_weights(_args, _model, _path, _config, get_hf_tensor):
        loaded.append(get_hf_tensor("module.output_layer.bias", None, None))
        loaded.append(get_hf_tensor("module.decoder.final_layernorm.weight", None, None))

    hf_to_megatron = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.hf_to_megatron",
        load_model_hf_weights=load_model_hf_weights,
    )
    worker_hooks._install_critic_hf_bootstrap(SimpleNamespace(loss_family="sao"))
    wrapped = hf_to_megatron.load_model_hf_weights
    worker_hooks._install_critic_hf_bootstrap(SimpleNamespace(loss_family="sao"))
    assert hf_to_megatron.load_model_hf_weights is wrapped

    wrapped(None, None, None, None, lambda *_args: torch.tensor([3.0]))

    assert loaded[0].tolist() == [0.0]
    assert loaded[1].tolist() == [3.0]

    # Families that do not declare critic_value_head_zero_init keep the plain
    # HF loader — the adapter never keys on a family name.
    plain = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.hf_to_megatron",
        load_model_hf_weights=load_model_hf_weights,
    )
    worker_hooks._install_critic_hf_bootstrap(SimpleNamespace(loss_family="pg"))
    assert plain.load_model_hf_weights is load_model_hf_weights


@pytest.mark.unit
def test_lora_hf_bootstrap_loads_wrapped_base_and_skips_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    named = [
        ("decoder.layers.0.self_attention.linear_proj.to_wrap.weight", parameter),
        ("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight", parameter),
    ]
    update_common = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.update_weight.common",
        named_params_and_buffers=lambda *_args, **_kwargs: iter(named),
    )
    update_weight = _stub_module(monkeypatch, "slime.backends.megatron_utils.update_weight")
    update_weight.common = update_common
    loaded: list[str] = []

    def load_model_hf_weights(*_args, **_kwargs):
        loaded.extend(name for name, _ in update_common.named_params_and_buffers(None, None))

    hf_to_megatron = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.hf_to_megatron",
        load_model_hf_weights=load_model_hf_weights,
    )
    original_named = update_common.named_params_and_buffers
    args = SimpleNamespace(megatron_lora_rank=8)

    worker_hooks._install_lora_hf_bootstrap(args)
    wrapped = hf_to_megatron.load_model_hf_weights
    worker_hooks._install_lora_hf_bootstrap(args)
    assert hf_to_megatron.load_model_hf_weights is wrapped

    wrapped(None, None, None, None, None)

    assert loaded == ["decoder.layers.0.self_attention.linear_proj.weight"]
    assert update_common.named_params_and_buffers is original_named


@pytest.mark.unit
def test_updater_and_policy_gradient_hooks_replace_only_worker_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _stub_module(monkeypatch, "slime.backends.megatron_utils.actor")
    delta = _stub_module(
        monkeypatch,
        "slime.backends.megatron_utils.update_weight.update_weight_from_disk_delta",
    )
    updater_types = {
        name: type(name, (), {})
        for name in (
            "ReefUpdateWeightFromDisk",
            "ReefUpdateWeightFromDiskDelta",
            "ReefUpdateWeightFromDistributed",
            "ReefUpdateWeightFromTensor",
        )
    }
    _stub_module(
        monkeypatch,
        "reef.train.slime_backend.reef_adapters.weight_updaters",
        **updater_types,
    )
    worker_hooks._install_versioned_updaters()
    assert actor.UpdateWeightFromTensor is updater_types["ReefUpdateWeightFromTensor"]
    assert delta.UpdateWeightFromDiskDelta is updater_types["ReefUpdateWeightFromDiskDelta"]

    loaded: list[str] = []
    misc = importlib.import_module("slime.utils.misc")
    monkeypatch.setattr(
        misc,
        "load_function",
        lambda path: (
            loaded.append(path) or (lambda args, ppo_kl, log_probs, advantages: (args, ppo_kl, log_probs, advantages))
        ),
    )
    loss = _stub_module(monkeypatch, "slime.backends.megatron_utils.loss")
    args = SimpleNamespace(custom_pg_loss_function_path="reef.loss", advantage_estimator="grpo")
    worker_hooks._install_pg_primitive(args)
    assert loss.compute_cispo_loss("kl", "logp", "adv", 0.1, 0.2) == (args, "kl", "logp", "adv")
    # Routing the estimator to cispo is the driver's job (configure_reef_loss_args).
    assert args.advantage_estimator == "grpo"
    assert loaded == ["reef.loss"]


@pytest.mark.unit
def test_objective_initializers_chain_user_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    misc = importlib.import_module("slime.utils.misc")
    monkeypatch.setattr(misc, "load_function", lambda path: lambda _args: calls.append(path))
    monkeypatch.setattr(worker_hooks, "resolve_objective_paths", lambda _args: calls.append("resolve"))
    for name in (
        "_install_external_batch_keys",
        "_install_step_batch_size",
        "_install_metric_capture",
        "_install_rollout_logging",
        "_install_critic_metrics",
        "_install_critic_hf_bootstrap",
        "_install_versioned_updaters",
    ):
        monkeypatch.setattr(worker_hooks, name, lambda *_, name=name: calls.append(name))
    monkeypatch.setattr(worker_hooks, "_install_pg_primitive", lambda _args: calls.append("pg"))
    args = SimpleNamespace(
        reef_chained_megatron_init_path="custom.init",
        loss_family="openclawrl",
    )

    worker_hooks.initialize_megatron_objective(args)

    assert calls[0:2] == ["custom.init", "resolve"]
    assert calls[-1] == "pg"

    configured: list[object] = []
    monkeypatch.setattr(
        worker_hooks,
        "resolve_args_loss_family",
        lambda args: SimpleNamespace(configure_critic_args=lambda value: configured.append((args.loss_family, value))),
    )
    critic = SimpleNamespace(loss_family="sao", reef_chained_critic_args_hook_path="custom.critic")
    worker_hooks.configure_critic_objective(critic)
    assert calls[-1] == "custom.critic"
    assert configured == [("sao", critic)]
