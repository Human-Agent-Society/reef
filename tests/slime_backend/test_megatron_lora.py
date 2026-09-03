from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.reef_adapters.megatron.lora import (
    apply_megatron_lora,
    base_checksum_verification_due,
    build_sglang_lora_config,
    collect_lora_train_metrics,
    is_lora_weight_name,
    lora_engine_identity,
    replace_adapter_task_weights_from_backup,
    sglang_lora_target_modules,
    validate_megatron_lora_args,
    validate_sglang_base_checksum_response,
    zero_megatron_lora_adapters,
)
from reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint import save_lora_adapter_to_path


def _args(**overrides):
    values = {
        "megatron_lora_rank": 32,
        "megatron_lora_alpha": None,
        "megatron_lora_dropout": 0.0,
        "megatron_lora_target_modules": None,
        "megatron_to_hf_mode": "bridge",
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "hf_checkpoint": "/models/Qwen3-8B",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_lora_checkpoint_is_readable_by_peft_without_base_weights(tmp_path) -> None:
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    output = tmp_path / "checkpoint-0"
    tensors = [
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 4)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.ones(4, 2)),
    ]

    save_lora_adapter_to_path(_args(megatron_lora_alpha=32), output, tensors)

    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "/models/Qwen3-8B"
    assert config["inference_mode"] is True
    assert config["r"] == 32
    saved = safetensors.load_file(output / "adapter_model.safetensors")
    assert set(saved) == {f"base_model.model.{name}" for name, _ in tensors}
    assert not any(name.endswith("embed_tokens.weight") for name in saved)
    loaded_config = peft.PeftConfig.from_pretrained(output)
    assert loaded_config.r == 32
    from peft.utils.save_and_load import load_peft_weights

    assert set(load_peft_weights(output, device="cpu")) == set(saved)


def test_lora_checkpoint_rejects_base_tensors(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen/base"):
        save_lora_adapter_to_path(
            _args(megatron_lora_alpha=32),
            tmp_path / "checkpoint-0",
            [("model.embed_tokens.weight", torch.ones(2, 2))],
        )


def test_lora_checkpoint_does_not_publish_a_partial_directory(tmp_path, monkeypatch) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    output = tmp_path / "checkpoint-0"

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(safetensors, "save_file", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_lora_adapter_to_path(
            _args(megatron_lora_alpha=32),
            output,
            [("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 4))],
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


class _Adapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = torch.nn.Linear(4, 2, bias=False)
        self.linear_out = torch.nn.Linear(2, 4, bias=False)
        torch.nn.init.zeros_(self.linear_out.weight)


class _FakeLoRA:
    kwargs = None

    def __init__(self, **kwargs) -> None:
        type(self).kwargs = kwargs

    def __call__(self, model, training=True):
        assert training is True
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.adapter = _Adapter()
        return model


def test_validate_megatron_lora_defaults_alpha_to_rank() -> None:
    args = _args()

    validate_megatron_lora_args(args)

    assert args.megatron_lora_alpha == 32


def test_sglang_adapter_config_matches_bridge_default_targets() -> None:
    args = _args(megatron_lora_alpha=32)

    assert sglang_lora_target_modules(args) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert build_sglang_lora_config(args) == {
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    assert is_lora_weight_name("model.layers.0.self_attn.q_proj.lora_A.default.weight")
    assert not is_lora_weight_name("model.embed_tokens.weight")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"verify_lora_base_weights_interval": 0}, "at least 1"),
        ({"megatron_lora_rank": -1}, "non-negative"),
        ({"megatron_lora_dropout": 1.0}, r"\[0, 1\)"),
        ({"megatron_to_hf_mode": "raw"}, "bridge"),
        ({"only_train_params_name_list": ["adapter"]}, "owns actor parameter freezing"),
        ({"colocate": True, "rollout_num_gpus_per_engine": 2}, "requires --rollout-num-gpus-per-engine=1"),
        ({"megatron_lora_rank": 0, "megatron_lora_alpha": 32}, "requires a positive"),
    ],
)
def test_validate_megatron_lora_rejects_incompatible_settings(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_megatron_lora_args(_args(**overrides))


def test_frozen_base_verification_samples_publications() -> None:
    args = _args(verify_lora_base_weights=True, verify_lora_base_weights_interval=16)

    verified = [sequence for sequence in range(1, 34) if base_checksum_verification_due(args, sequence)]

    assert verified == [1, 17, 33], "the first publication and every interval-th one after it"


def test_frozen_base_verification_honors_the_boolean_guard() -> None:
    every = _args(verify_lora_base_weights=True, verify_lora_base_weights_interval=1)
    disabled = _args(verify_lora_base_weights=False, verify_lora_base_weights_interval=1)
    unset = _args(verify_lora_base_weights=True)

    assert all(base_checksum_verification_due(every, sequence) for sequence in range(1, 5))
    assert not any(base_checksum_verification_due(disabled, sequence) for sequence in range(1, 5))
    assert all(base_checksum_verification_due(unset, sequence) for sequence in range(1, 5))


def test_apply_megatron_lora_freezes_base_and_reports_adapter_update() -> None:
    args = _args(megatron_lora_alpha=32)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))

    adapted = apply_megatron_lora(model, args, lora_cls=_FakeLoRA)

    assert _FakeLoRA.kwargs == {"dim": 32, "alpha": 32, "dropout": 0.0}
    assert not adapted[0].weight.requires_grad
    assert adapted.adapter.linear_in.weight.requires_grad
    assert adapted.adapter.linear_out.weight.requires_grad

    adapted.adapter.linear_out.weight.data[0, 0] = 0.25
    metrics = collect_lora_train_metrics([adapted])
    assert metrics["train/lora_trainable_parameters"] == 16
    assert metrics["train/lora_base_trainable_parameters"] == 0
    assert metrics["train/lora_b_nonzero"] == 1
    assert metrics["train/lora_b_l1"] == 0.25


def test_apply_megatron_lora_fails_if_base_parameter_remains_trainable() -> None:
    class BrokenLoRA:
        def __init__(self, **kwargs) -> None:
            pass

        def __call__(self, model, training=True):
            model.adapter = _Adapter()
            return model

    with pytest.raises(RuntimeError, match="non-adapter parameters"):
        apply_megatron_lora(
            torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False)),
            _args(megatron_lora_alpha=32),
            lora_cls=BrokenLoRA,
        )


def test_zero_megatron_lora_adapters_removes_resumed_reference_delta() -> None:
    model = torch.nn.Module()
    model.adapter = _Adapter()
    torch.nn.init.normal_(model.adapter.linear_in.weight)
    torch.nn.init.normal_(model.adapter.linear_out.weight)
    base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    model.register_parameter("base", base)

    assert zero_megatron_lora_adapters([model]) == 2
    assert torch.count_nonzero(model.adapter.linear_in.weight) == 0
    assert torch.count_nonzero(model.adapter.linear_out.weight) == 0
    assert model.base.item() == 1


def test_lora_engine_identity_tracks_actor_incarnations() -> None:
    class ActorId:
        def __init__(self, value: str) -> None:
            self.value = value

        def hex(self) -> str:
            return self.value

    first_handle = SimpleNamespace(_actor_id=ActorId("engine-a"))
    same_actor_handle = SimpleNamespace(_actor_id=ActorId("engine-a"))
    recovered_handle = SimpleNamespace(_actor_id=ActorId("engine-b"))

    assert lora_engine_identity(first_handle) == lora_engine_identity(same_actor_handle)
    assert lora_engine_identity(first_handle) != lora_engine_identity(recovered_handle)


def _load_colocated_module(monkeypatch):
    package_name = "reef.train.slime_backend.reef_adapters.weight_updaters"
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[2] / "reef" / "train" / "slime_backend" / "reef_adapters" / "weight_updaters")
    ]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.delitem(sys.modules, f"{package_name}.colocated", raising=False)
    monkeypatch.delitem(sys.modules, f"{package_name}.lora_transport", raising=False)

    def stub(name, **attributes):
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    stub(
        f"{package_name}.base",
        SynchronizedWeightUpdateMixin=type("SynchronizedWeightUpdateMixin", (), {}),
    )
    stub(
        "slime.backends.megatron_utils.sglang",
        FlattenedTensorBucket=type("FlattenedTensorBucket", (), {}),
        MultiprocessingSerializer=type(
            "MultiprocessingSerializer",
            (),
            {"serialize": staticmethod(lambda *_args, **_kwargs: "")},
        ),
    )
    stub(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        UpdateWeightFromDistributed=type("UpdateWeightFromDistributed", (), {}),
        connect_rollout_engines_from_distributed=lambda *_args, **_kwargs: None,
        disconnect_rollout_engines_from_distributed=lambda *_args, **_kwargs: None,
        post_process_weights=lambda **_kwargs: None,
        update_weights_from_distributed=lambda *_args, **_kwargs: [],
    )
    stub(
        "slime.backends.megatron_utils.update_weight.update_weight_from_tensor",
        UpdateWeightFromTensor=type("UpdateWeightFromTensor", (), {}),
        _send_to_colocated_engine=lambda *_args, **_kwargs: ([], None),
    )
    stub("slime.utils.distributed_utils", get_gloo_group=lambda: None)
    return importlib.import_module(f"{package_name}.colocated")


def _load_distributed_module(monkeypatch):
    _load_colocated_module(monkeypatch)
    name = "reef.train.slime_backend.reef_adapters.weight_updaters.distributed"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_colocated_lora_refresh_uses_tensor_ipc_without_nccl(monkeypatch) -> None:
    colocated = _load_colocated_module(monkeypatch)
    transport = importlib.import_module("reef.train.slime_backend.reef_adapters.weight_updaters.lora_transport")

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, **kwargs):
            events.append((self.name, kwargs))
            return {"success": True}

    class Bucket:
        def __init__(self, named_tensors):
            self.named_tensors = named_tensors

        def get_flattened_tensor(self):
            return torch.ones(1)

        def get_metadata(self):
            return ["metadata"]

    engine = SimpleNamespace(
        unload_lora_adapter=RemoteMethod("unload"),
        load_lora_adapter_from_tensors=RemoteMethod("load"),
    )
    monkeypatch.setattr(transport, "FlattenedTensorBucket", Bucket)
    monkeypatch.setattr(transport.MultiprocessingSerializer, "serialize", lambda *_args, **_kwargs: "payload")
    monkeypatch.setattr(transport.dist, "get_rank", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(transport.dist, "get_world_size", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        transport.dist, "broadcast", lambda *_args, **_kwargs: pytest.fail("colocate must not use NCCL")
    )

    def gather(value, object_gather_list, **_kwargs):
        object_gather_list[0] = value

    monkeypatch.setattr(transport.dist, "gather_object", gather)
    monkeypatch.setattr(transport.ray, "get", lambda value: value)

    refs, keepalive = colocated.send_lora_to_colocated_engine(
        [("layer.lora_A.weight", torch.ones(1))],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
        lora_config={"r": 1},
        lora_name="reef_lora",
        lora_loaded=True,
        expected_checksums={"layer.lora_A.weight": "digest"},
    )

    assert [name for name, _ in events] == ["unload", "load"]
    assert events[1][1]["serialized_named_tensors"] == ["payload"]
    assert events[1][1]["expected_checksums"] == {"layer.lora_A.weight": "digest"}
    assert refs == [{"success": True}]
    assert keepalive


def test_distributed_lora_publication_broadcasts_to_remote_engines(monkeypatch) -> None:
    colocated = _load_colocated_module(monkeypatch)

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, **kwargs):
            events.append((self.name, kwargs))
            return {"success": True}

    engines = tuple(
        SimpleNamespace(load_lora_adapter_from_distributed=RemoteMethod(f"load-{index}")) for index in range(2)
    )

    class Broadcast:
        def wait(self):
            events.append(("wait", {}))

    def broadcast(tensor, source, *, group, async_op):
        events.append(("broadcast", {"tensor": tensor, "source": source, "group": group, "async_op": async_op}))
        return Broadcast()

    monkeypatch.setattr(colocated.dist, "broadcast", broadcast)

    refs = colocated.send_lora_to_distributed_engines(
        [("layer.lora_A.weight", torch.ones(2, dtype=torch.float16))],
        rollout_engines=engines,
        model_update_group="nccl-group",
        group_name="reef-lora",
        lora_config={"r": 1},
        lora_name="reef_lora",
    )

    assert [name for name, _ in events] == ["load-0", "load-1", "broadcast", "wait"]
    assert events[0][1]["upsert"] is True
    assert events[0][1]["names"] == ["layer.lora_A.weight"]
    assert events[0][1]["dtypes"] == [torch.float16]
    assert events[0][1]["shapes"] == [torch.Size([2])]
    assert events[0][1]["group_name"] == "reef-lora"
    assert events[2][1]["group"] == "nccl-group"
    assert refs == [{"success": True}, {"success": True}]


def test_lora_connect_preserves_slime_colocated_and_remote_partition(monkeypatch) -> None:
    colocated = _load_colocated_module(monkeypatch)
    updater = object.__new__(colocated.ReefUpdateWeightFromTensor)
    updater.is_lora = True
    updater._lora_rollout_engines = ()

    engines = (object(), object(), object())

    def base_connect(self, rollout_engines, *_args, **_kwargs):
        self.rollout_engines = rollout_engines[:1]
        self.distributed_rollout_engines = rollout_engines[1:]
        self.use_distribute = True
        self._group_name = "slime"
        self._model_update_groups = "slime-group"
        self._ipc_engine = rollout_engines[0]

    monkeypatch.setattr(
        colocated.UpdateWeightFromTensor,
        "connect_rollout_engines",
        base_connect,
        raising=False,
    )
    updater.connect_rollout_engines(engines, object(), engine_gpu_counts=[1, 1, 1])

    assert updater._lora_rollout_engines == engines
    assert updater.rollout_engines == engines[:1]
    assert updater.distributed_rollout_engines == engines[1:]
    assert updater._model_update_groups == "slime-group"


def test_mixed_lora_publication_uses_ipc_locally_and_nccl_remotely(monkeypatch) -> None:
    colocated = _load_colocated_module(monkeypatch)
    updater = object.__new__(colocated.ReefUpdateWeightFromTensor)
    local_engine = object()
    remote_engines = (object(), object())
    updater._ipc_engine = local_engine
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = "gloo-group"
    updater._lora_config = {"r": 1}
    updater._lora_loaded_engine_ids = set()
    updater.use_distribute = True
    updater._is_distributed_src_rank = True
    updater.distributed_rollout_engines = remote_engines
    updater._model_update_groups = "nccl-group"
    updater._group_name = "slime"

    events = []
    monkeypatch.setattr(
        colocated,
        "send_lora_to_colocated_engine",
        lambda tensors, **kwargs: events.append(("ipc", tensors, kwargs)) or (["ipc-ref"], ["keepalive"]),
    )
    monkeypatch.setattr(
        colocated,
        "send_lora_to_distributed_engines",
        lambda tensors, **kwargs: events.append(("nccl", tensors, kwargs)) or ["nccl-ref"],
    )

    tensors = [("layer.lora_A.weight", torch.ones(1))]
    refs, keepalive, engine_id = updater._send_lora_params(
        tensors, expected_checksums={"tensor": "digest"}, lora_name="reef-adapter-bWF0aA.deployment:3"
    )

    assert [event[0] for event in events] == ["ipc", "nccl"]
    assert events[0][2]["ipc_engine"] is local_engine
    assert events[0][2]["lora_name"] == events[1][2]["lora_name"] == "reef-adapter-bWF0aA.deployment:3"
    assert events[0][2]["lora_loaded"] is False, "versioned names are never reloaded in place"
    assert events[1][2]["rollout_engines"] == remote_engines
    assert events[1][2]["model_update_group"] == "nccl-group"
    assert refs == ["ipc-ref", "nccl-ref"]
    assert keepalive == ["keepalive"]
    assert engine_id == lora_engine_identity(local_engine)


def test_disjoint_updater_publishes_lora_through_distributed_receiver(monkeypatch) -> None:
    distributed = _load_distributed_module(monkeypatch)
    updater = object.__new__(distributed.ReefUpdateWeightFromDistributed)
    updater.is_lora = True
    updater.args = SimpleNamespace(
        check_lora_weight_equal=False,
        verify_lora_base_weights=False,
    )
    updater._lora_weights_getter = lambda: {"adapter": torch.ones(1)}
    updater._lora_hf_weight_iterator = SimpleNamespace(
        get_hf_weight_chunks=lambda *_args, **_kwargs: [
            [("model.layers.0.q_proj.lora_A.weight", torch.ones(2, dtype=torch.float16))]
        ]
    )
    updater._model_update_groups = "slime-group"
    updater._group_name = "slime-pp_0"
    updater._lora_config = {"r": 1}
    updater.runtime_load_id = "deployment:3"
    updater._base_checksums = None
    # ``update_weights`` stamps the publication sequence before delegating here.
    updater.weight_update_sequence = 1
    updater.tie_word_embeddings = False
    updater.active_scenario = "math"

    events = []

    class RemoteMethod:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))
            return {"success": True}

    updater.rollout_engines = [SimpleNamespace(set_runtime_load_id=RemoteMethod("version"))]
    updater._run_rank_zero_action = lambda action, **_kwargs: action()

    def raise_synchronized(error, **_kwargs):
        if error is not None:
            raise error

    updater._raise_synchronized_update_error = raise_synchronized
    monkeypatch.setattr(distributed.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(distributed.ray, "get", lambda value: value)

    sent = []
    monkeypatch.setattr(
        distributed,
        "send_lora_to_distributed_engines",
        lambda tensors, **kwargs: sent.append((tensors, kwargs)) or [{"success": True}],
    )

    updater._update_lora_weights(manage_generation=False)

    assert sent[0][1]["rollout_engines"] is updater.rollout_engines
    assert sent[0][1]["model_update_group"] == "slime-group"
    assert sent[0][1]["group_name"] == "slime-pp_0"
    # Every publication is scenario-qualified and versioned; which revisions
    # stay resident is the bridge's residency manager's decision, not the
    # updater's.
    assert sent[0][1]["lora_name"] == "reef-adapter-bWF0aA.deployment:3"
    assert events == [("version", ("deployment:3",), {})]


def test_disjoint_updater_selects_lora_instead_of_full_weight_fanout(monkeypatch) -> None:
    distributed = _load_distributed_module(monkeypatch)
    updater = object.__new__(distributed.ReefUpdateWeightFromDistributed)
    updater.is_lora = True
    updater.weight_update_sequence = 4
    events = []
    updater._reset_source_phases = lambda: events.append("reset")
    updater._update_lora_weights = lambda **kwargs: events.append(("lora", kwargs))

    updater.update_weights(manage_generation=False)

    assert updater.weight_update_sequence == 5
    assert events == ["reset", ("lora", {"manage_generation": False})]


def test_base_checksum_validation_respects_embedding_tying() -> None:
    tied = {
        "success": True,
        "per_engine_checksum": "base-checksum",
        "ranks": {
            "checksums": {
                "model.embed_tokens.weight": "shared",
                "lm_head.weight": "shared",
            }
        },
    }
    assert (
        validate_sglang_base_checksum_response(
            tied,
            engine_index=0,
            tie_word_embeddings=True,
        )
        == "base-checksum"
    )

    with pytest.raises(RuntimeError, match="identical untied"):
        validate_sglang_base_checksum_response(
            tied,
            engine_index=0,
            tie_word_embeddings=False,
        )


def test_tied_base_checksum_allows_deduplicated_lm_head() -> None:
    response = {
        "success": True,
        "per_engine_checksum": "base-checksum",
        "ranks": {"checksums": {"model.embed_tokens.weight": "shared"}},
    }

    assert (
        validate_sglang_base_checksum_response(
            response,
            engine_index=1,
            tie_word_embeddings=True,
        )
        == "base-checksum"
    )


def test_adapter_merge_tasks_use_cpu_actor_backup_instead_of_paused_model() -> None:
    class BackupWeight:
        def __init__(self, value):
            self.value = value

        def cuda(self):
            return f"cuda:{self.value}"

    @dataclass(frozen=True)
    class Task:
        vp_stage: int
        param_name: str
        param_weight: object

    @dataclass(frozen=True)
    class AdapterTask:
        linear_in_task: Task
        linear_out_task: Task

    paused_weight = object()
    adapter_tasks = [
        AdapterTask(
            linear_in_task=Task(0, "decoder.layer.adapter.linear_in.weight", paused_weight),
            linear_out_task=Task(0, "decoder.layer.adapter.linear_out.weight", paused_weight),
        )
    ]
    backup = {
        "vp_stages.0.decoder.layer.adapter.linear_in.weight": BackupWeight("A"),
        "vp_stages.0.decoder.layer.adapter.linear_out.weight": BackupWeight("B"),
    }

    replaced = replace_adapter_task_weights_from_backup(adapter_tasks, backup)

    assert replaced[0].linear_in_task.param_weight == "cuda:A"
    assert replaced[0].linear_out_task.param_weight == "cuda:B"
    assert adapter_tasks[0].linear_in_task.param_weight is paused_weight
