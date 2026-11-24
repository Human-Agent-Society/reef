from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.reef_adapters.megatron.lora import (
    apply_megatron_lora,
    build_sglang_lora_config,
    collect_lora_train_metrics,
    is_lora_weight_name,
    replace_adapter_task_weights_from_backup,
    sglang_lora_target_modules,
    validate_megatron_lora_args,
    zero_megatron_lora_adapters,
)
from reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint import (
    export_lora_adapter_tensors,
    save_lora_adapter_to_path,
)


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

    save_lora_adapter_to_path(_args(megatron_lora_alpha=32), output, tensors, rollout_id=0)

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


def test_lora_checkpoint_records_auditable_metadata(tmp_path) -> None:
    peft = pytest.importorskip("peft")
    base = tmp_path / "Qwen3-8B"
    base.mkdir()
    (base / "tokenizer_config.json").write_text('{"chat_template": "x"}', encoding="utf-8")
    output = tmp_path / "checkpoint-4"
    tensors = [
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.ones(4, 2, dtype=torch.bfloat16)),
    ]

    save_lora_adapter_to_path(
        _args(megatron_lora_alpha=32, hf_checkpoint=str(base)),
        output,
        tensors,
        rollout_id=4,
    )

    document = json.loads((output / "reef-adapter.json").read_text(encoding="utf-8"))
    assert document["schema"] == 1
    assert document["source"] == {"rollout_id": 4}
    assert document["dtype"] == "bfloat16"
    assert document["peft"]["r"] == 32
    assert document["base_model"]["name_or_path"] == str(base)
    # The tokenizer belongs to the base checkpoint. The adapter records its
    # checksum so that swapping the base later is visible.
    assert set(document["base_model"]["tokenizer"]) == {"tokenizer_config.json"}

    config_bytes = (output / "adapter_config.json").read_bytes()
    assert document["files"]["adapter_config.json"] == hashlib.sha256(config_bytes).hexdigest()
    weight_bytes = (output / "adapter_model.safetensors").read_bytes()
    assert document["files"]["adapter_model.safetensors"] == hashlib.sha256(weight_bytes).hexdigest()

    # The extra file must not break portability: PEFT ignores files it does not know.
    assert peft.PeftConfig.from_pretrained(output).r == 32


def test_lora_checkpoint_metadata_passes_reefs_own_validator(tmp_path) -> None:
    """Export and validation are one contract, so what Reef writes it must accept."""
    pytest.importorskip("safetensors.torch")
    from reef.artifact import Artifact, PEFTValidator

    base = tmp_path / "Qwen3-8B"
    base.mkdir()
    output = tmp_path / "checkpoint-0"
    save_lora_adapter_to_path(
        _args(megatron_lora_alpha=32, hf_checkpoint=str(base)),
        output,
        [("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 4))],
        rollout_id=0,
    )

    PEFTValidator(str(base), require_metadata=True).validate(Artifact.local(output))


def test_lora_checkpoint_records_no_dtype_for_a_mixed_export(tmp_path) -> None:
    """Recording one dtype here would be wrong, so record none."""
    pytest.importorskip("safetensors.torch")
    output = tmp_path / "checkpoint-0"
    save_lora_adapter_to_path(
        _args(megatron_lora_alpha=32),
        output,
        [
            ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.float32)),
            ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.ones(4, 2, dtype=torch.bfloat16)),
        ],
        rollout_id=0,
    )

    assert json.loads((output / "reef-adapter.json").read_text(encoding="utf-8"))["dtype"] is None


def test_lora_checkpoint_rejects_base_tensors(tmp_path) -> None:
    with pytest.raises(ValueError, match="frozen/base"):
        save_lora_adapter_to_path(
            _args(megatron_lora_alpha=32),
            tmp_path / "checkpoint-0",
            [("model.embed_tokens.weight", torch.ones(2, 2))],
            rollout_id=0,
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
            rollout_id=0,
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
        ({"megatron_lora_rank": -1}, "non-negative"),
        ({"megatron_lora_dropout": 1.0}, r"\[0, 1\)"),
        ({"megatron_to_hf_mode": "raw"}, "bridge"),
        ({"only_train_params_name_list": ["adapter"]}, "owns actor parameter freezing"),
        ({"megatron_lora_rank": 0, "megatron_lora_alpha": 32}, "requires a positive"),
    ],
)
def test_validate_megatron_lora_rejects_incompatible_settings(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_megatron_lora_args(_args(**overrides))


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


def test_lora_export_gathers_the_adapter_chunks_of_the_actor_weights() -> None:
    calls = []
    adapter = [("layer.q_proj.lora_A.weight", torch.ones(1)), ("layer.q_proj.lora_B.weight", torch.zeros(1))]

    class WeightIterator:
        def get_hf_weight_chunks(self, weights, *, weight_type):
            calls.append((weights, weight_type))
            return [adapter[:1], adapter[1:]]

    weights = {"vp_stages.0.layer.adapter.linear_in.weight": torch.ones(1)}
    assert export_lora_adapter_tensors(WeightIterator(), weights) == adapter
    assert calls == [(weights, "lora")]
