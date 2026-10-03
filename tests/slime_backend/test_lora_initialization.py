"""A PEFT SFT adapter must retain its function across TP import."""

import json

import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

from reef.train.slime_backend.reef_adapters.megatron.lora_initialization import adapter_shard, load_initial_adapter


def test_row_parallel_adapter_shards_reconstruct_the_dense_update() -> None:
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(4, 12, generator=generator)
    b = torch.randn(8, 4, generator=generator)
    inputs = torch.randn(3, 12, generator=generator)
    a_shards, b_shards = [], []
    for rank in range(4):
        a_parameter = torch.nn.Parameter(torch.empty(4, 3))
        b_parameter = torch.nn.Parameter(torch.empty(2, 4))
        a_parameter.tensor_model_parallel = b_parameter.tensor_model_parallel = True
        a_parameter.partition_dim, b_parameter.partition_dim = 1, 0
        a_shards.append(adapter_shard(a, a_parameter, rank, 4))
        b_shards.append(adapter_shard(b, b_parameter, rank, 4))
    hidden = sum(part @ shard.T for part, shard in zip(inputs.chunk(4, dim=1), a_shards, strict=True))
    result = torch.cat([hidden @ shard.T for shard in b_shards], dim=1)
    torch.testing.assert_close(result, inputs @ a.T @ b.T)


def test_adapter_shard_rejects_unexplained_shape_change() -> None:
    with pytest.raises(ValueError, match="layout"):
        adapter_shard(torch.ones(4, 8), torch.zeros(4, 4), 0, 2)


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.linear_fc2 = torch.nn.Module()
        layer.mlp.linear_fc2.adapter = torch.nn.Module()
        layer.mlp.linear_fc2.adapter.linear_in = torch.nn.Linear(8, 2, bias=False)
        layer.mlp.linear_fc2.adapter.linear_out = torch.nn.Linear(2, 4, bias=False)
        self.decoder.layers = torch.nn.ModuleList([layer])


@pytest.fixture(params=["model.layers", "model.language_model.layers"])
def initial_adapter(tmp_path, monkeypatch, request):
    parallel_state = pytest.importorskip("megatron.core.parallel_state")

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 2, "bias": "none"}))
    prefix = f"base_model.model.{request.param}.0.mlp.down_proj"
    tensors = {f"{prefix}.lora_A.weight": torch.ones(2, 8), f"{prefix}.lora_B.weight": torch.full((4, 2), 3.0)}
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    return tmp_path, tensors


def test_initial_adapter_preserves_sft_function(initial_adapter) -> None:
    path, tensors = initial_adapter
    model = ToyActor()
    load_initial_adapter(model, str(path), rank=2, alpha=2)
    adapter = model.decoder.layers[0].mlp.linear_fc2.adapter
    inputs = torch.arange(8.0).unsqueeze(0)
    expected = inputs @ next(v for k, v in tensors.items() if "lora_A" in k).T
    expected = expected @ next(v for k, v in tensors.items() if "lora_B" in k).T
    torch.testing.assert_close(adapter.linear_out(adapter.linear_in(inputs)), expected)


@pytest.mark.parametrize("failure", ["missing", "extra", "nonfinite", "scale", "mixed_prefix", "unknown_prefix"])
def test_initial_adapter_rejects_bad_input_without_partial_mutation(initial_adapter, failure) -> None:
    path, tensors = initial_adapter
    if failure == "missing":
        tensors.pop(next(k for k in tensors if "lora_B" in k))
    elif failure == "extra":
        tensors["unrelated.weight"] = torch.ones(1)
    elif failure == "nonfinite":
        tensors[next(k for k in tensors if "lora_B" in k)][0, 0] = float("nan")
    elif failure == "mixed_prefix":
        key = next(iter(tensors))
        alternate = (
            key.replace("model.language_model.layers", "model.layers")
            if "language_model" in key
            else key.replace("model.layers", "model.language_model.layers")
        )
        tensors[alternate] = tensors[key].clone()
    elif failure == "unknown_prefix":
        tensors = {key.replace("base_model.model.", "unsupported."): value for key, value in tensors.items()}
    else:
        (path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    save_file(tensors, str(path / "adapter_model.safetensors"))
    model = ToyActor()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_initial_adapter(model, str(path), rank=2, alpha=2)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
