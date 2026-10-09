"""CPU tests of the optional worker hooks with native loss and teacher callbacks."""

from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from recipes.sdft.slime import SdftAlgorithm, SdftSettings
from recipes.sdpo.slime import SdpoAlgorithm, SdpoSettings
from reef.train.slime_backend.distill import teacher


class CPUBackuper:
    def __init__(self) -> None:
        self.values = {"actor": {"weight": torch.tensor([1.0, 2.0])}}
        self.backup_tags = ["actor"]
        self.weight = torch.tensor([1.0, 2.0])

    def backup(self, tag: str) -> None:
        self.values[tag] = {"weight": self.weight.clone()}
        if tag not in self.backup_tags:
            self.backup_tags.append(tag)

    def get(self, tag: str) -> dict[str, torch.Tensor]:
        return self.values[tag]


class CPUActor:
    """Explicit stand-in for the private Slime actor weight-switch contract."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.weights_backuper = CPUBackuper()
        self.model = [self]
        self.switches: list[str] = []
        self.forward_calls = 0
        self.callback_calls = 0
        self.callback_results: list[dict[str, list[torch.Tensor]]] = []
        self.raw_teacher_logits: dict[int, torch.Tensor] = {}

    def _switch_model(self, tag: str) -> None:
        self.switches.append(tag)
        self.weights_backuper.weight.copy_(self.weights_backuper.get(tag)["weight"])

    def get_runtime_load_id(self) -> str:
        return "runtime-pre-update"


@pytest.fixture(params=["sdft", "sdpo"])
def case(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    method = request.param
    module = importlib.import_module(f"recipes.{method}.examples.agentcl.worker_capture")
    native = importlib.import_module(f"recipes.{method}.slime.objective")
    monkeypatch.setattr(module, "capture", None)
    args = Namespace(
        loss_family=method,
        save=str(tmp_path / "checkpoint" / "megatron"),
        custom_loss_function_path=f"recipes.{method}.slime.objective.{method}_loss",
        reef_actor_pre_train_hook_path=f"recipes.{method}.slime.objective.{method}_actor_pre_train",
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        reef_external_batch_keys=("rollout_log_probs",),
        rollout_temperature=0.7,
        log_probs_chunk_size=2,
        calculate_per_token_loss=False,
        score_centering=False,
        max_tokens_per_gpu=15,
        seq_length=32,
        use_dynamic_batch_size=True,
    )
    if method == "sdft":
        SdftAlgorithm().apply_driver_options(args, SdftSettings(skip_response_tokens=1))
    else:
        SdpoAlgorithm().apply_driver_options(args, SdpoSettings(top_k=3, top_k_source="student"))

    def _responses(logits, *, args, unconcat_tokens, total_lengths, response_lengths):
        tempered = logits.squeeze(0) / args.rollout_temperature
        offset = 0
        for sequence, total, length in zip(unconcat_tokens, total_lengths, response_lengths, strict=True):
            yield tempered[offset + total - length - 1 : offset + total - 1], sequence[-length:]
            offset += total

    score_calls: list[torch.Tensor] = []

    def _log_probs(logits, *, with_entropy, **options):
        score_calls.append(logits)
        values = [
            rows.log_softmax(-1).gather(-1, sampled[:, None])[:, 0] for rows, sampled in _responses(logits, **options)
        ]
        return torch.empty(0), {"log_probs": values}

    def _reducer(total_lengths, response_lengths, masks, packed, per_token):
        def _reduce(values):
            return sum(
                (part * mask).sum() / mask.sum().clamp_min(1)
                for part, mask in zip(values.split(response_lengths), masks, strict=True)
            )

        return _reduce

    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda **options: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_virtual_pipeline_model_parallel_world_size=lambda: None,
    )
    core = ModuleType("megatron.core")
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, core.__name__, core)
    loss_module = ModuleType("slime.backends.megatron_utils.loss")
    loss_module.get_responses = _responses
    loss_module.get_log_probs_and_entropy = _log_probs
    monkeypatch.setitem(sys.modules, loss_module.__name__, loss_module)
    cp_module = ModuleType("slime.backends.megatron_utils.cp_utils")
    cp_module.get_sum_of_sample_mean = _reducer
    monkeypatch.setitem(sys.modules, cp_module.__name__, cp_module)
    data_module = ModuleType("slime.backends.megatron_utils.data")

    class DataIterator:
        def __init__(self, view, schedule):
            self.rollout_data = view
            self.micro_batch_indices = schedule

    data_module.DataIterator = DataIterator
    monkeypatch.setitem(sys.modules, data_module.__name__, data_module)
    model_module = ModuleType("slime.backends.megatron_utils.model")

    def _forward_only(callback, args, models, iterators, microbatches):
        actor = models[0]
        actor.forward_calls += 1
        view = iterators[0].rollout_data
        collected = {}
        for indices in iterators[0].micro_batch_indices:
            tokens = [view["tokens"][index] for index in indices]
            totals = [view["total_lengths"][index] for index in indices]
            lengths = [view["response_lengths"][index] for index in indices]
            rows = []
            for index, sequence in zip(indices, tokens, strict=True):
                raw = torch.arange(sequence.numel() * 11, dtype=torch.float32).reshape(sequence.numel(), 11)
                raw = (raw / 9).sin()
                raw[:, 2] = -20000  # Exercise the native exact teacher storage floor.
                if sequence.numel() >= 7:
                    actor.raw_teacher_logits[index] = raw.clone()
                rows.append(raw)
            logits = torch.cat(rows).unsqueeze(0)
            result = callback(
                logits,
                args=args,
                unconcat_tokens=tokens,
                total_lengths=totals,
                response_lengths=lengths,
                with_entropy=False,
            )
            actor.callback_calls += 1
            actor.callback_results.append(result[1])
            for key, values in result[1].items():
                collected.setdefault(key, []).extend(values)
        return collected

    model_module.forward_only = _forward_only
    parent_module = ModuleType("slime.backends.megatron_utils")
    parent_module.model = model_module
    monkeypatch.setitem(sys.modules, parent_module.__name__, parent_module)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    actor_module = ModuleType("reef.train.slime_backend.reef_adapters.megatron.train_actor")
    actor_module.ReefMegatronTrainRayActor = CPUActor
    monkeypatch.setitem(sys.modules, actor_module.__name__, actor_module)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(teacher, "_TEACHER", teacher.MovingCopy(0.01))
    suffixes = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    rollout = {
        "tokens": [torch.cat([torch.tensor([0, 0]), suffix]) for suffix in suffixes],
        "teacher_tokens": [torch.cat([torch.tensor([0, 0, 0, 0]), suffix]) for suffix in suffixes],
        "response_lengths": [3, 3, 4],
        "loss_masks": [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 1.0]), torch.ones(4)],
        "rollout_log_probs": [torch.full((length,), -2.0) for length in (3, 3, 4)],
        "distill_sample_weights": [1.0, 0.5, 1.5],
        "sample_indices": [12, 13, 14],
        "rollout_ids": [8, 8, 8],
        "producing_runtime_load_ids": ["runtime-pre-update"] * 3,
    }
    return SimpleNamespace(
        method=method,
        module=module,
        native=native,
        args=args,
        actor=CPUActor(args),
        rollout=rollout,
        reducer=_reducer,
        score_calls=score_calls,
        model_module=model_module,
        mpu=mpu,
    )


def training_batch(case: SimpleNamespace, order: list[int]) -> dict[str, object]:
    rollout = case.rollout
    keys = [
        "tokens",
        "teacher_tokens",
        "response_lengths",
        "loss_masks",
        "rollout_log_probs",
        "distill_sample_weights",
        "sample_indices",
        "rollout_ids",
        "producing_runtime_load_ids",
    ]
    keys.extend(key for key in rollout if key.startswith("distill_teacher_"))
    batch = {key: [rollout[key][index] for index in order] for key in keys}
    batch["unconcat_tokens"] = batch.pop("tokens")
    batch["total_lengths"] = [int(sequence.numel()) for sequence in batch["unconcat_tokens"]]
    return batch


@pytest.mark.unit
@pytest.mark.parametrize("cap", [0.0, 2.0])
def test_loss_capture_on_off_returns_original_results_and_identical_gradients(case, monkeypatch, cap):
    from reef.train.slime_backend.distill.objective import distill_loss

    case.args.distill_importance_sampling_cap = cap
    teacher.compute_teacher_rows(case.actor, case.rollout)
    batch = training_batch(case, [2, 0, 1])
    total = sum(batch["total_lengths"])
    generator = torch.Generator().manual_seed(77)
    plain_logits = torch.randn(1, total, 11, generator=generator).requires_grad_()
    captured_logits = plain_logits.detach().clone().requires_grad_()
    masks = [mask.clone() for mask in batch["loss_masks"]]
    reduce = case.reducer(batch["total_lengths"], batch["response_lengths"], masks, None, False)
    plain_result = case.module.captured_loss(case.args, batch, plain_logits, reduce)
    plain_result[0].backward()
    results = []

    def _native_loss(*arguments):
        result = distill_loss(*arguments)
        results.append(result)
        return result

    monkeypatch.setattr(case.native, f"{case.method}_loss", _native_loss)
    case.module.capture = case.module.WorkerCapture(case.args)
    case.module.capture.active = True
    case.module.capture.pass_count = 1
    case.module.capture.runtime_load_id = "runtime-pre-update"
    scored_before = len(case.score_calls)
    captured_result = case.module.captured_loss(case.args, batch, captured_logits, reduce)
    assert captured_result is results[0]
    assert len(results) == 1
    assert len(case.score_calls) == scored_before + (2 if cap else 1)
    captured_result[0].backward()
    assert torch.equal(captured_result[0], plain_result[0])
    assert torch.equal(captured_logits.grad, plain_logits.grad)
    for name in plain_result[1]:
        assert torch.equal(captured_result[1][name], plain_result[1][name])
    for original, current in zip(masks, batch["loss_masks"], strict=True):
        assert torch.equal(original, current)
    paths = list(case.module.capture.directory.glob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["optimizer_execution_verified"] is False
    assert record["settings"]["importance_sampling_cap"] == cap
    assert [sample["sample_index"] for sample in record["samples"]] == [14, 12, 13]
    assert record["samples"][1]["student_token_sha256"] == record["samples"][2]["student_token_sha256"]
    offset = 0
    for sample, total_length, response_length in zip(
        record["samples"], batch["total_lengths"], batch["response_lengths"], strict=True
    ):
        expected = (
            captured_logits.detach()[0, offset + total_length - response_length - 1 : offset + total_length - 1] / 0.7
        ).log_softmax(-1)
        sampled = torch.tensor(sample["response_ids"])
        torch.testing.assert_close(
            torch.tensor(sample["student_log_probs"]), expected.gather(-1, sampled[:, None])[:, 0]
        )
        assert sample["teacher_token_sha256"] == case.module.checksum(
            case.module.json_value(batch["teacher_tokens"][sample["packing_index"]])
        )
        offset += total_length
    assert paths[0].stat().st_mode & 0o777 == 0o600
    assert paths[0].parent.stat().st_mode & 0o777 == 0o700
    assert not list(paths[0].parent.glob(".capture-*"))


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [False, True])
def test_canonical_teacher_once_exact_suffix_and_identical_sibling_order(case, monkeypatch, enabled):
    counts = {"pre_train": 0, "mix": 0}
    original_mix = teacher.mix_teacher_weights
    boundary = case.model_module.forward_only

    def _mix(*arguments):
        counts["mix"] += 1
        return original_mix(*arguments)

    def _native_pre_train(actor, rollout):
        counts["pre_train"] += 1
        teacher.compute_teacher_rows(actor, rollout)

    monkeypatch.setattr(teacher, "mix_teacher_weights", _mix)
    monkeypatch.setattr(case.native, f"{case.method}_actor_pre_train", _native_pre_train)
    if enabled:
        case.module.capture = case.module.WorkerCapture(case.args)
    case.module.pre_train(case.actor, case.rollout)
    assert counts == {"pre_train": 1, "mix": 1}
    assert case.actor.switches == ["distill_teacher", "actor"]
    assert case.actor.forward_calls == (1 if case.method == "sdft" else 2)
    assert case.model_module.forward_only is boundary
    if not enabled:
        assert not (Path(case.args.save).parent / "agentcl-worker-capture").exists()
        return
    paths = list(case.module.capture.directory.glob("*.json"))
    record = json.loads(paths[0].read_text())
    assert record["runtime_load_id"] == "runtime-pre-update"
    assert [sample["sample_index"] for sample in record["samples"]] == [12, 13, 14]
    assert record["samples"][0]["teacher_token_sha256"] == record["samples"][1]["teacher_token_sha256"]
    for index, sample in enumerate(record["samples"]):
        sequence = case.rollout["teacher_tokens"][index]
        length = case.rollout["response_lengths"][index]
        raw = case.actor.raw_teacher_logits[index]
        expected = (raw[sequence.numel() - length - 1 : -1] / 0.7).log_softmax(-1)
        sampled = sequence[-length:]
        selected = expected.gather(-1, sampled[:, None])[:, 0]
        torch.testing.assert_close(torch.tensor(sample["teacher_runtime_sampled_log_probs"]), selected)
        torch.testing.assert_close(torch.tensor(sample["teacher_arithmetic_sampled_log_probs"]), selected)
        assert sample["response_ids"] == sampled.tolist()
        assert len(sample["teacher_logit_positions"]) == length
        assert sample["loss_mask"] == case.rollout["loss_masks"][index].tolist()
        if case.method == "sdft":
            assert sample["canonical_teacher_shape"] == [length, 11]
            assert torch.equal(
                torch.tensor(sample["teacher_sampled_log_probs"]),
                selected.clamp_min(teacher.TEACHER_LOG_PROB_FLOOR).half().float(),
            )
            assert sample["teacher_sampled_log_probs"] == sample["teacher_expected_stored_sampled_log_probs"]
            assert "distill_teacher_log_probs" not in sample
        else:
            ids = case.rollout["distill_teacher_topk_ids"][index]
            assert sample["canonical_teacher_topk_ids"] == ids.tolist()
            assert (
                sample["canonical_teacher_topk_log_probs"]
                == case.rollout["distill_teacher_topk_log_probs"][index].tolist()
            )
            assert sample["canonical_teacher_sampled_log_probs"] == sample["teacher_sampled_log_probs"]
            torch.testing.assert_close(
                torch.tensor(sample["teacher_arithmetic_topk_log_probs"]), expected.gather(-1, ids)
            )
    assert record["samples"][0]["teacher_logit_positions"] == [3, 4, 5]
    assert record["samples"][1]["teacher_logit_positions"] == [10, 11, 12]
    assert record["samples"][2]["teacher_logit_positions"] == [3, 4, 5, 6]
    for key, rows in case.actor.callback_results[-1].items():
        assert rows[0] is case.rollout[key][-1]


@pytest.mark.unit
def test_strict_same_version_admission_omits_optional_producing_version_array(case):
    case.rollout.pop("producing_runtime_load_ids")
    case.module.capture = case.module.WorkerCapture(case.args)
    case.module.pre_train(case.actor, case.rollout)
    batch = dict(case.rollout)
    # Slime's minibatch builder materializes an omitted optional column as None.
    batch["producing_runtime_load_ids"] = None
    batch["unconcat_tokens"] = batch.pop("tokens")
    batch["total_lengths"] = [len(tokens) for tokens in batch["unconcat_tokens"]]
    logits = torch.zeros(1, sum(batch["total_lengths"]), 11, requires_grad=True)
    reduce = case.reducer(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], None, False)
    loss, _ = case.module.captured_loss(case.args, batch, logits, reduce)
    loss.backward()
    files = list(case.module.capture.directory.glob("*.json"))
    assert len(files) == 2
    for path in files:
        record = json.loads(path.read_text())
        assert record["runtime_load_id"] == "runtime-pre-update"
        for sample in record["samples"]:
            assert sample["producing_runtime_load_id"] is None
            assert sample["executing_runtime_load_id"] == "runtime-pre-update"
    assert case.actor.switches == ["distill_teacher", "actor"]


@pytest.mark.unit
def test_capture_stops_after_two_pretrain_calls_without_skipping_native_calls(case, monkeypatch):
    calls = []

    def _pre_train(actor, rollout):
        calls.append(actor)
        teacher.compute_teacher_rows(actor, rollout)

    monkeypatch.setattr(case.native, f"{case.method}_actor_pre_train", _pre_train)
    case.module.capture = case.module.WorkerCapture(case.args)
    for _ in range(3):
        case.module.pre_train(case.actor, case.rollout)
    assert len(calls) == 3
    assert case.module.capture.active is False
    assert len(list(case.module.capture.directory.glob("*.json"))) == 2
    before = len(case.score_calls)
    batch = training_batch(case, [0, 1, 2])
    logits = torch.zeros(1, sum(batch["total_lengths"]), 11, requires_grad=True)
    reduce = case.reducer(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], None, False)
    case.module.captured_loss(case.args, batch, logits, reduce)
    assert len(case.score_calls) == before + 1  # Only the canonical cap-2 correction.


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [False, True])
def test_original_loss_and_pretrain_exceptions_propagate_unchanged(case, monkeypatch, enabled):
    error = RuntimeError("native failure")
    calls = []

    def _fail(*arguments):
        calls.append(arguments)
        raise error

    monkeypatch.setattr(case.native, f"{case.method}_loss", _fail)
    monkeypatch.setattr(case.native, f"{case.method}_actor_pre_train", _fail)
    if enabled:
        case.module.capture = case.module.WorkerCapture(case.args)
        case.module.capture.active = True
    with pytest.raises(RuntimeError) as raised:
        case.module.captured_loss(case.args, {}, torch.empty(0), torch.sum)
    assert raised.value is error
    assert len(calls) == 1
    boundary = case.model_module.forward_only
    with pytest.raises(RuntimeError) as raised:
        case.module.pre_train(case.actor, case.rollout)
    assert raised.value is error
    assert len(calls) == 2
    assert case.model_module.forward_only is boundary
    assert not (Path(case.args.save).parent / "agentcl-worker-capture").exists()


@pytest.mark.unit
def test_observer_exception_restores_boundary_and_actor_weights(case, monkeypatch):
    error = ValueError("diagnostic failure")
    case.module.capture = case.module.WorkerCapture(case.args)

    def _observe(*arguments):
        raise error

    def _pre_train(actor, rollout):
        teacher.compute_teacher_rows(actor, rollout)

    monkeypatch.setattr(case.module.capture, "observe_teacher", _observe)
    monkeypatch.setattr(case.native, f"{case.method}_actor_pre_train", _pre_train)
    boundary = case.model_module.forward_only
    with pytest.raises(ValueError) as raised:
        case.module.pre_train(case.actor, case.rollout)
    assert raised.value is error
    assert case.model_module.forward_only is boundary
    assert case.actor.switches == ["distill_teacher", "actor"]


@pytest.mark.unit
def test_install_registers_after_resolution_without_loss_registry_mutation(case, monkeypatch):
    from reef.train.slime_backend import algorithm

    monkeypatch.setattr(
        algorithm, "_objective_registry", {key: dict(value) for key, value in algorithm._objective_registry.items()}
    )
    monkeypatch.setattr(case.native, "agentcl_initialize_capture", None, raising=False)
    loss_before = case.args.custom_loss_function_path
    case.module.install(case.args)
    assert case.args.custom_loss_function_path == loss_before
    assert "sample_indices" in case.args.reef_external_batch_keys
    assert "producing_runtime_load_ids" in case.args.reef_external_batch_keys
    registry = algorithm._objective_registry[case.native.__name__]
    assert registry["custom_loss_function_path"] == f"{case.method}_loss"
    assert registry["reef_actor_pre_train_hook_path"] == f"{case.method}_actor_pre_train"
    assert registry["reef_actor_init_hook_path"] == "agentcl_initialize_capture"
    case.native.agentcl_initialize_capture(case.actor)
    assert case.args.custom_loss_function_path == f"{case.module.__name__}.captured_loss"
    assert case.args.reef_actor_pre_train_hook_path == f"{case.module.__name__}.pre_train"
    with pytest.raises(ValueError, match="unmodified native"):
        case.module.initialize_actor(case.actor)


@pytest.mark.unit
def test_nonwriter_still_enters_teacher_and_student_collectives(case, monkeypatch):
    monkeypatch.setattr(case.mpu, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(case.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    collective_calls = []

    def _reduce(tensor, **options):
        collective_calls.append(tuple(tensor.shape))

    def _gather(outputs, tensor, **options):
        collective_calls.append(tuple(tensor.shape))
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(torch.distributed, "all_reduce", _reduce)
    monkeypatch.setattr(torch.distributed, "all_gather", _gather)
    case.module.capture = case.module.WorkerCapture(case.args)

    def _pre_train(actor, rollout):
        teacher.compute_teacher_rows(actor, rollout)

    monkeypatch.setattr(case.native, f"{case.method}_actor_pre_train", _pre_train)
    case.module.pre_train(case.actor, case.rollout)
    assert collective_calls
    assert len(case.score_calls) == 2  # Two teacher microbatches, despite not writing.
    before = len(case.score_calls)
    batch = training_batch(case, [0, 1, 2])
    logits = torch.zeros(1, sum(batch["total_lengths"]), 11, requires_grad=True)
    reduce = case.reducer(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"], None, False)
    case.module.captured_loss(case.args, batch, logits, reduce)
    assert len(case.score_calls) == before + 2
    assert not case.module.capture.directory.exists()


@pytest.mark.unit
def test_private_output_bounds_and_nonfinite_values_fail_explicitly(case):
    state = case.module.WorkerCapture(case.args)
    identity = case.module.RankIdentity(0, 0, 1, 0, 0, 1)
    with pytest.raises(ValueError, match="sample or suffix"):
        case.module.validate_bounds({"response_lengths": [1] * 65})
    state.written_bytes = 256 * 1024 * 1024
    with pytest.raises(ValueError, match="byte budget"):
        state.write("student_loss", case.args, identity, [])
    state.written_bytes = 0
    with pytest.raises(ValueError, match="JSON compliant"):
        state.write("student_loss", case.args, identity, [{"log_prob": float("nan")}])
    assert not state.directory.exists()
    state.write("student_loss", case.args, identity, [])
    with pytest.raises(ValueError, match="fresh output"):
        state.write("student_loss", case.args, identity, [])
