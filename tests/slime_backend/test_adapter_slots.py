"""One Megatron LoRA slot, time-sliced between scenarios (rank-local snapshots)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reef.train.slime_backend.reef_adapters.megatron.adapter_slots import AdapterSlotSwitcher, scenario_key


class _Adapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = torch.nn.Linear(4, 2, bias=False)
        self.linear_out = torch.nn.Linear(2, 4, bias=False)


class _Wrapped(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 4, bias=False)
        self.base.weight.requires_grad_(False)
        self.adapter = _Adapter()


class _Chunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = _Wrapped()


class _Scheduler:
    def __init__(self) -> None:
        self.num_steps = 0
        self.stepped: list[int] = []

    def step(self, increment: int) -> None:
        self.stepped.append(increment)
        self.num_steps += increment


class _MegatronLikeOptimizer:
    """Mirror MegatronOptimizer: fp32 main copies stepped by an inner torch optimizer."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.main = [p.detach().clone().float().requires_grad_(True) for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.main, lr=0.1)


def _build(tmp_path: Path | None = None, rank: int = 0):
    torch.manual_seed(0)
    model = [_Chunk()]
    optimizer = _MegatronLikeOptimizer(model[0])
    scheduler = _Scheduler()
    slots = AdapterSlotSwitcher(model, optimizer, scheduler, store_dir=tmp_path, rank=rank)
    return model, optimizer, scheduler, slots


def _adapter_values(model) -> list[torch.Tensor]:
    return [p.detach().clone() for n, p in model[0].named_parameters() if ".adapter." in n]


def _train_step(model, optimizer, scheduler) -> None:
    for main, param in zip(optimizer.main, [p for p in model[0].parameters() if p.requires_grad], strict=True):
        main.grad = torch.randn_like(main)
        param.data.copy_(main.detach() - 0.1 * main.grad)  # keep the model in step with main params
    optimizer.optimizer.step()
    scheduler.step(1)


def test_restore_puts_each_scenarios_state_back_exactly() -> None:
    model, optimizer, scheduler, slots = _build()
    pristine = _adapter_values(model)
    assert slots.restore("a") is False
    _train_step(model, optimizer, scheduler)
    slots.capture("a")
    a_values, a_steps = _adapter_values(model), scheduler.num_steps
    a_moments = [optimizer.optimizer.state[p]["exp_avg"].clone() for p in optimizer.main]

    assert slots.restore("b") is False, "a first-time scenario starts pristine"
    for got, want in zip(_adapter_values(model), pristine, strict=True):
        assert torch.equal(got, want)
    assert scheduler.num_steps == 0
    assert all(p not in optimizer.optimizer.state for p in optimizer.main)

    _train_step(model, optimizer, scheduler)
    _train_step(model, optimizer, scheduler)
    slots.capture("b")
    assert slots.restore("a") is True
    for got, want in zip(_adapter_values(model), a_values, strict=True):
        assert torch.equal(got, want)
    assert scheduler.num_steps == a_steps
    for p, want in zip(optimizer.main, a_moments, strict=True):
        assert torch.equal(optimizer.optimizer.state[p]["exp_avg"], want)
    assert slots.scenarios == ("a", "b") and slots.active == "a"


def test_persisted_snapshots_survive_a_restart(tmp_path: Path) -> None:
    model, optimizer, scheduler, slots = _build(tmp_path)
    slots.restore("math")
    _train_step(model, optimizer, scheduler)
    slots.capture("math")
    path = slots.persist("math")
    assert path == tmp_path / scenario_key("math") / "rank_00000.pt" and path.is_file()
    expected = _adapter_values(model)

    fresh_model, fresh_optimizer, fresh_scheduler, fresh_slots = _build(tmp_path)
    assert fresh_slots.restore("math") is True
    for got, want in zip(_adapter_values(fresh_model), expected, strict=True):
        assert torch.equal(got, want)
    assert fresh_scheduler.num_steps == 1
    assert all(p in fresh_optimizer.optimizer.state for p in fresh_optimizer.main)


def test_persist_after_restore_writes_the_slot_without_recapture(tmp_path: Path) -> None:
    """A save can persist the active scenario without capturing it again.

    The training actor relies on this: ``train_actor`` captures the slot on
    its way out and ``restore`` records what it put back, so the snapshot the
    switcher holds for the active scenario is already the slot's state.
    """
    model, optimizer, scheduler, slots = _build(tmp_path)
    slots.restore("math")
    _train_step(model, optimizer, scheduler)
    slots.capture("math")
    trained = _adapter_values(model)
    assert slots.restore("code") is False, "the slot moves to another scenario"
    assert slots.restore("math") is True

    slots.persist("math")

    fresh_model, _, fresh_scheduler, fresh_slots = _build(tmp_path)
    assert fresh_slots.restore("math") is True
    for got, want in zip(_adapter_values(fresh_model), trained, strict=True):
        assert torch.equal(got, want)
    assert fresh_scheduler.num_steps == 1


def test_snapshot_layout_mismatch_fails_closed(tmp_path: Path) -> None:
    _, _, _, slots = _build(tmp_path)
    slots.restore("a")
    slots.capture("a")
    slots.persist("a")
    loaded = torch.load(slots.snapshot_path("a"), weights_only=False)
    loaded["adapter"] = {name: torch.zeros(1) for name in loaded["adapter"]}
    torch.save(loaded, slots.snapshot_path("a"))
    _, _, _, fresh = _build(tmp_path)
    with pytest.raises(RuntimeError, match="LoRA layout"):
        fresh.restore("a")


def test_persist_without_a_store_is_a_no_op_and_requires_capture(tmp_path: Path) -> None:
    _, _, _, slots = _build()
    slots.restore("a")
    assert slots.persist("a") is None
    _, _, _, stored = _build(tmp_path)
    with pytest.raises(RuntimeError, match="no captured adapter state"):
        stored.persist("never")


def test_scenario_key_is_filesystem_safe_and_distinct() -> None:
    assert scenario_key("a/b:c") != scenario_key("a-b-c")
    assert "/" not in scenario_key("a/b:c")
    with pytest.raises(ValueError):
        scenario_key("")


def test_requires_lora_parameters() -> None:
    with pytest.raises(RuntimeError, match="LoRA parameters"):
        AdapterSlotSwitcher([torch.nn.Linear(2, 2)], _MegatronLikeOptimizer(torch.nn.Linear(2, 2)))
