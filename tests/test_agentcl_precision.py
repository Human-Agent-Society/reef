"""CPU checks using collected native function bodies, not a Megatron installation.

Set AGENTCL_NATIVE_BOUNDARY_SOURCE to the parent's collected source directory.
Collectives and CUDA allocation are simulated; these tests do not qualify TP4.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import io
import json
import os
import sys
import warnings
from argparse import Namespace
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch


class CPUGroup:
    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0


class CPUHandle:
    def wait(self) -> None:
        return None


class CPUMemoryBuffer:
    def get_tensor(self, shape: list[int], dtype: torch.dtype, name: str) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)


@pytest.fixture(params=["sdft", "sdpo"])
def native(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    source_root = os.environ.get("AGENTCL_NATIVE_BOUNDARY_SOURCE")
    if not source_root:
        pytest.skip("set AGENTCL_NATIVE_BOUNDARY_SOURCE for collected native-body CPU tests")
    root = Path(source_root)
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest:
        path = root / record["project"] / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    layer_path = root / "megatron/megatron/core/tensor_parallel/layers.py"
    ddp_path = root / "megatron/megatron/core/distributed/distributed_data_parallel.py"
    group = CPUGroup()
    events: list[tuple[str, tuple[int, ...], torch.dtype, object]] = []
    memory = CPUMemoryBuffer()

    def _gather(output, input, *, group, async_op=False):
        events.append(("gather", tuple(input.shape), input.dtype, group))
        output.copy_(torch.cat((input, input), dim=0))
        return CPUHandle()

    def _scatter(output, input, *, group, async_op=False):
        events.append(("scatter", tuple(input.shape), input.dtype, group))
        output.copy_(input.chunk(2, dim=0)[0] * 2)
        return CPUHandle()

    def _allreduce(input, *, group, async_op=False):
        events.append(("allreduce", tuple(input.shape), input.dtype, group))
        input.mul_(2)
        return CPUHandle()

    monkeypatch.setattr(torch.distributed, "all_reduce", _allreduce)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    class CopyRegion(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, group):
            ctx.group = group
            return input

        @staticmethod
        def backward(ctx, grad):
            _allreduce(grad, group=ctx.group)
            return grad, None

    namespace = {
        "torch": torch,
        "os": os,
        "warnings": warnings,
        "Optional": Optional,
        "List": list,
        "Tuple": tuple,
        "Callable": Callable,
        "ModelParallelConfig": object,
        "Parameter": torch.nn.Parameter,
        "ShardedStateDict": dict,
        "custom_fwd": partial(torch.amp.custom_fwd, device_type="cuda"),
        "custom_bwd": partial(torch.amp.custom_bwd, device_type="cuda"),
        "get_global_memory_buffer": lambda: memory,
        "get_tensor_model_parallel_group_if_none": lambda value: value,
        "dist_all_gather_func": _gather,
        "dist_reduce_scatter_func": _scatter,
        "prepare_input_tensors_for_wgrad_compute": lambda grad, input: (
            grad.contiguous().view(-1, grad.shape[-1]),
            input.contiguous().view(-1, input.shape[-1]),
        ),
        "copy_to_tensor_model_parallel_region": lambda input, group: CopyRegion.apply(input, group),
        "HAVE_TE": False,
    }
    selected = {
        "LinearWithFrozenWeight",
        "linear_with_frozen_weight",
        "LinearWithGradAccumulationAndAsyncCommunication",
        "linear_with_grad_accumulation_and_async_allreduce",
        "ColumnParallelLinear",
    }
    tree = ast.parse(layer_path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in selected]
    assert len(nodes) == len(selected)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(layer_path), "exec"), namespace)
    namespace["linear_with_grad_accumulation_and_async_allreduce"].warned = True
    column = namespace["ColumnParallelLinear"]
    ddp_tree = ast.parse(ddp_path.read_text())
    ddp_class = next(
        node for node in ddp_tree.body if isinstance(node, ast.ClassDef) and node.name == "DistributedDataParallel"
    )
    hook = next(
        node
        for node in ddp_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_backward_post_hook"
    )
    ddp_node = ast.ClassDef(name="NativeDDP", bases=[], keywords=[], body=[hook], decorator_list=[])
    ddp_module = ast.fix_missing_locations(ast.Module(body=[ddp_node], type_ignores=[]))
    namespace["is_graph_capturing"] = lambda: False
    exec(compile(ddp_module, str(ddp_path), "exec"), namespace)

    class CPUModel(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.output_layer = head
            self.post_process = True
            self.share_embeddings_and_output_weights = False
            self.fuse_linear_cross_entropy = False
            self.config = head.config
            self.pg_collection = SimpleNamespace(tp=group)

    layers = ModuleType("megatron.core.tensor_parallel.layers")
    layers.ColumnParallelLinear = column
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_data_parallel_world_size=lambda: 1)
    gpt = ModuleType("megatron.core.models.gpt")
    gpt.GPTModel = CPUModel
    for name, module in (
        ("megatron", ModuleType("megatron")),
        (core.__name__, core),
        (layers.__name__, layers),
        (gpt.__name__, gpt),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    precision = importlib.import_module(f"recipes.{request.param}.examples.agentcl.precision")
    head = column.__new__(column)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.arange(88, dtype=torch.float32).reshape(11, 8).div(37).to(torch.bfloat16))
    head.register_parameter("bias", None)
    head.config = SimpleNamespace(
        defer_embedding_wgrad_compute=False,
        cpu_offloading=False,
        fp8=None,
        fp4=None,
        cuda_graph_impl="none",
        _cpu_offloading_context=None,
    )
    head.tp_group = group
    head.sequence_parallel = False
    head.allreduce_dgrad = True
    head.gradient_accumulation_fusion = True
    head.skip_bias_add = False
    head.explicit_expert_comm = False
    head.disable_grad_reduce = False
    head.is_expert = False
    head.embedding_activation_buffer = None
    head.grad_output_buffer = None
    head.gather_output = False
    head.input_size = 8
    head.output_size_per_partition = 11
    head.weight.main_grad = torch.zeros_like(head.weight, dtype=torch.float32)
    head.weight.grad_added_to_main_grad = False
    ddp = namespace["NativeDDP"]()
    ddp.ddp_config = SimpleNamespace(overlap_grad_reduce=False)
    ddp.param_to_bucket_group = {head.weight: object()}
    accumulator = head.weight.expand_as(head.weight).grad_fn.next_functions[0][0]
    accumulator.register_hook(ddp._make_backward_post_hook(head.weight))
    args = Namespace(
        loss_family=request.param,
        use_critic=False,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        offload_train=False,
        megatron_lora_rank=0,
        bf16=True,
        untie_embeddings_and_output_weights=True,
        custom_model_provider_path=f"recipes.{request.param}.examples.agentcl.precision.provide_actor_model",
        reef_external_batch_keys=("rollout_log_probs",),
    )
    return SimpleNamespace(
        precision=precision,
        head=head,
        model=CPUModel(head),
        args=args,
        events=events,
        group=group,
        namespace=namespace,
        accumulator=accumulator,
        monkeypatch=monkeypatch,
    )


def test_native_forward_gradients_and_ddp_accumulation(native: SimpleNamespace) -> None:
    native.precision.bind_model(native.model)
    head = native.head
    weight_id = id(head.weight)
    main_grad_id = id(head.weight.main_grad)
    expected_accumulation = torch.zeros_like(head.weight.main_grad)
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        hidden = torch.randn(6, 2, 8, generator=generator).to(torch.bfloat16).requires_grad_()
        reference_hidden = hidden.detach().float().requires_grad_()
        reference_weight = head.weight.detach().float().requires_grad_()
        reference = torch.matmul(reference_hidden, reference_weight.t())
        logits, bias = head(hidden)
        assert bias is None
        assert torch.equal(logits, reference)
        assert not torch.equal(torch.matmul(hidden.detach(), head.weight.detach().t()).float(), reference.detach())
        logits.square().sum().backward()
        reference.square().sum().backward()
        assert torch.equal(hidden.grad, (reference_hidden.grad * 2).to(torch.bfloat16))
        expected_accumulation.add_(reference_weight.grad.to(torch.bfloat16).float())
        assert torch.equal(head.weight.main_grad, expected_accumulation)
        assert head.weight.grad is None
        assert not head.weight.grad_added_to_main_grad
    assert id(head.weight) == weight_id
    assert id(head.weight.main_grad) == main_grad_id
    assert head.gradient_accumulation_fusion
    assert len(native.events) == 20
    assert all(
        event[0] == "allreduce" and event[2] == torch.float32 and event[3] is native.group for event in native.events
    )


def test_native_sequence_parallel_dimensions(native: SimpleNamespace) -> None:
    native.head.sequence_parallel = True
    native.head.allreduce_dgrad = False
    native.precision.bind_model(native.model)
    hidden = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8).div(29).to(torch.bfloat16).requires_grad_()
    full_hidden = torch.cat((hidden.detach(), hidden.detach()), dim=0).float().requires_grad_()
    reference_weight = native.head.weight.detach().float().requires_grad_()
    reference = torch.matmul(full_hidden, reference_weight.t())
    logits, _ = native.head(hidden)
    assert logits.shape == (6, 2, 11)
    assert torch.equal(logits, reference)
    logits.square().sum().backward()
    reference.square().sum().backward()
    assert torch.equal(hidden.grad, (full_hidden.grad[:3] * 2).to(torch.bfloat16))
    assert torch.equal(native.head.weight.main_grad, reference_weight.grad.to(torch.bfloat16).float())
    assert [event[0] for event in native.events] == ["gather", "gather", "scatter"]
    assert [event[1] for event in native.events] == [(3, 2, 8), (3, 2, 8), (6, 2, 8)]
    assert all(event[2] == torch.float32 and event[3] is native.group for event in native.events)


def test_state_parameter_and_batch_identity(native: SimpleNamespace) -> None:
    before = dict(native.model.named_parameters())
    state = native.model.state_dict()
    original_weight = native.head.weight.detach().clone()
    batch = {
        "tokens": torch.tensor([2, 3]),
        "loss_mask": torch.tensor([1, 0]),
        "teacher_tokens": torch.tensor([1, 2, 3]),
    }
    batch_ids = {name: id(value) for name, value in batch.items()}
    assert native.precision.bind_model(native.model) is native.model
    bound = native.head._forward_impl
    assert native.precision.bind_model(native.model) is native.model
    assert native.head._forward_impl is bound
    assert {name: id(value) for name, value in native.model.named_parameters()} == {
        name: id(value) for name, value in before.items()
    }
    assert tuple(native.model.state_dict()) == tuple(state)
    assert torch.equal(native.head.weight, original_weight)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    native.model.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(native.head.weight, original_weight)
    assert {name: id(value) for name, value in batch.items()} == batch_ids


def test_provider_delegates_once_without_argument_mutation(native: SimpleNamespace) -> None:
    global_vars = ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: native.args
    provider_module = ModuleType("slime.backends.megatron_utils.model_provider")
    calls = []

    def _factory(args, role):
        assert args is not native.args
        assert args.custom_model_provider_path is None
        assert role == "actor"

        def _provider(pre_process, post_process, vp_stage):
            calls.append((pre_process, post_process, vp_stage))
            native.model.post_process = post_process
            return native.model

        return _provider

    provider_module.get_model_provider_func = _factory
    native.monkeypatch.setitem(sys.modules, global_vars.__name__, global_vars)
    native.monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    original_args = vars(native.args).copy()
    assert native.precision.provide_actor_model(False, True, 0) is native.model
    assert calls == [(False, True, 0)]
    assert vars(native.args) == original_args


def test_non_output_chunk_is_unchanged(native: SimpleNamespace) -> None:
    native.model.post_process = False
    original = native.head._forward_impl.__func__
    assert native.precision.bind_model(native.model) is native.model
    assert native.head._forward_impl.__func__ is original


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_critic", True),
        ("megatron_lora_rank", 4),
        ("context_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("untie_embeddings_and_output_weights", False),
    ],
)
def test_unsupported_options_fail_before_binding(native: SimpleNamespace, field: str, value: object) -> None:
    vars(native.args)[field] = value
    original = native.head._forward_impl.__func__
    with pytest.raises(ValueError):
        native.precision.validate_options(native.args)
    assert native.head._forward_impl.__func__ is original


@pytest.mark.parametrize(
    "field,value",
    [("defer_embedding_wgrad_compute", True), ("cpu_offloading", True), ("cuda_graph_impl", "local"), ("fp8", "e4m3")],
)
def test_unsupported_native_configs_fail_before_binding(native: SimpleNamespace, field: str, value: object) -> None:
    vars(native.model.config)[field] = value
    original = native.head._forward_impl.__func__
    with pytest.raises(ValueError):
        native.precision.bind_model(native.model)
    assert native.head._forward_impl.__func__ is original


def test_custom_implementation_is_not_overwritten(native: SimpleNamespace) -> None:
    def _custom(head, *args, **kwargs):
        raise RuntimeError("custom head")

    native.head._forward_impl = MethodType(_custom, native.head)
    with pytest.raises(ValueError, match="existing custom"):
        native.precision.bind_model(native.model)
    assert native.head._forward_impl.__func__ is _custom


def test_autocast_does_not_round_head_output(native: SimpleNamespace) -> None:
    native.precision.bind_model(native.model)
    hidden = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, _ = native.head(hidden)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.matmul(hidden.float(), native.head.weight.detach().float().t()))


def test_post_init_keeps_model_wrappers_and_optimizer_state(native: SimpleNamespace) -> None:
    class CPUWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model.bfloat16()

    class CPUDDP(torch.nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.module = wrapper
            self.ddp_config = SimpleNamespace(delay_wgrad_compute=False)

    distributed = ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = CPUDDP
    transformer = ModuleType("megatron.core.transformer.module")
    transformer.Float16Module = CPUWrapper
    native.monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    native.monkeypatch.setitem(sys.modules, transformer.__name__, transformer)
    optimizer = torch.optim.SGD(native.model.parameters(), lr=0.1, momentum=0.9)
    wrapper = CPUWrapper(native.model)
    ddp = CPUDDP(wrapper)
    lifecycle = []
    native.args.offload_train = True
    actor = SimpleNamespace(
        args=native.args,
        role="actor",
        model=[ddp],
        optimizer=optimizer,
        wake_up=lambda: lifecycle.append("wake"),
        sleep=lambda: lifecycle.append("sleep"),
    )
    native.monkeypatch.setattr(
        sys.modules["megatron.core"].mpu,
        "get_data_parallel_world_size",
        lambda: 1 if lifecycle and lifecycle[-1] == "wake" else (_ for _ in ()).throw(RuntimeError("asleep")),
    )
    parameter = native.head.weight
    before = optimizer.state_dict()
    native.precision.initialize_actor(actor)
    assert lifecycle == ["wake", "sleep"]
    assert actor.model[0] is ddp
    assert ddp.module is wrapper
    assert wrapper.module is native.model
    assert native.head.weight is parameter
    assert optimizer.state_dict() == before
    assert optimizer.param_groups[0]["params"][0] is parameter
    actor.role = "critic"
    with pytest.raises(ValueError, match="actor model chunk"):
        native.precision.initialize_actor(actor)


def test_runtime_override_rejections(native: SimpleNamespace) -> None:
    native.precision.bind_model(native.model)
    hidden = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    original_weight = native.head.weight
    with pytest.raises(ValueError, match="original untied"):
        native.head(hidden, weight=original_weight.detach().clone())
    with pytest.raises(ValueError, match="deferred"):
        native.head._forward_impl(
            hidden, original_weight, None, True, True, False, grad_output_buffer=[], tp_group=native.group
        )
    original_weight.grad_added_to_main_grad = True
    with pytest.raises(ValueError, match="reset"):
        native.head(hidden)


def test_frozen_head_uses_canonical_frozen_autograd(native: SimpleNamespace) -> None:
    native.head.weight.requires_grad_(False)
    native.precision.bind_model(native.model)
    hidden = torch.ones(3, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().float().requires_grad_()
    reference = torch.matmul(reference_hidden, native.head.weight.detach().float().t())
    logits, _ = native.head(hidden)
    assert torch.equal(logits, reference)
    logits.square().sum().backward()
    reference.square().sum().backward()
    assert torch.equal(hidden.grad, (reference_hidden.grad * 2).to(torch.bfloat16))
    assert torch.count_nonzero(native.head.weight.main_grad) == 0


def test_hook_installation_composes_capture_and_precision_once(native: SimpleNamespace) -> None:
    from reef.train.slime_backend import algorithm

    method = native.args.loss_family
    objective_module = importlib.import_module(f"recipes.{method}.slime.objective")
    native.monkeypatch.setattr(
        algorithm,
        "_objective_registry",
        {
            objective_module.__name__: {
                "custom_loss_function_path": f"{method}_loss",
                "reef_actor_pre_train_hook_path": f"{method}_actor_pre_train",
            }
        },
    )
    native.monkeypatch.setattr(objective_module, "agentcl_initialize_precision", None, raising=False)
    calls = []
    capture = importlib.import_module(f"recipes.{method}.examples.agentcl.worker_capture")
    native.monkeypatch.setattr(capture, "initialize_actor", lambda actor: calls.append("capture"))
    native.monkeypatch.setattr(native.precision, "initialize_actor", lambda actor: calls.append("precision"))
    native.precision.install(native.args)
    initialized = objective_module.agentcl_initialize_precision
    initialized(object())
    assert calls == ["capture", "precision"]
    assert native.args.reef_external_batch_keys == (
        "rollout_log_probs",
        "teacher_tokens",
        "sample_indices",
        "rollout_ids",
        "producing_runtime_load_ids",
    )
    assert algorithm._objective_registry[objective_module.__name__] == {
        "custom_loss_function_path": f"{method}_loss",
        "reef_actor_pre_train_hook_path": f"{method}_actor_pre_train",
        "reef_actor_init_hook_path": "agentcl_initialize_precision",
    }


def test_between_step_training_offload_does_not_change_binding(native: SimpleNamespace) -> None:
    native.args.offload_train = True
    native.precision.validate_options(native.args)
    parameter = native.head.weight
    native.precision.bind_model(native.model)
    native.head.weight.data = native.head.weight.detach().clone()
    native.precision.bind_model(native.model)
    assert native.head.weight is parameter
    assert native.head._forward_impl.__func__ is native.precision.fp32_head_forward
