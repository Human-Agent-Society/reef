from __future__ import annotations

import importlib.util
import sys
import threading
import types
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


class _StubTimer:
    """Slime's timer as the adapter meets it: a singleton that refuses a
    second start under one name (``assert name not in self.start_time``)."""

    _instance: _StubTimer | None = None
    started: list[str] = []
    _open: set[str] = set()

    def __new__(cls) -> _StubTimer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls.started = []
        cls._open = set()

    @contextmanager
    def context(self, name: str):
        assert name not in self._open, f"Timer {name} already started."
        self._open.add(name)
        self.started.append(name)
        try:
            yield
        finally:
            self._open.discard(name)


def _stub_timer_decorator(function):
    """Slime's ``@timer``: time the call under the function's own name."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        with _StubTimer().context(function.__name__):
            return function(*args, **kwargs)

    return wrapper


def _install_updater_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    ray = types.ModuleType("ray")
    ray.get = lambda value, **_kwargs: value  # type: ignore[attr-defined]
    ray.ObjectRef = object  # type: ignore[attr-defined]
    ray_actor = types.ModuleType("ray.actor")
    ray_actor.ActorHandle = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)

    megatron = types.ModuleType("megatron")
    megatron.__path__ = []  # type: ignore[attr-defined]
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = types.SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    distributed_utils = types.ModuleType("slime.utils.distributed_utils")
    distributed_utils.get_gloo_group = lambda: None  # type: ignore[attr-defined]
    distributed_utils.init_process_group = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils.distributed_utils", distributed_utils)

    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda value: value  # type: ignore[attr-defined]
    http_utils.is_port_available = lambda _port: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", http_utils)

    megatron_to_hf = types.ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf.convert_to_hf = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.megatron_to_hf",
        megatron_to_hf,
    )

    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_param = lambda _name, value: value  # type: ignore[attr-defined]
    common.named_params_and_buffers = lambda *_args: []  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.common",
        common,
    )
    distributed = types.ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    distributed.UpdateWeightFromDistributed = type("UpdateWeightFromDistributed", (), {})  # type: ignore[attr-defined]
    distributed.post_process_weights = lambda **_kwargs: None  # type: ignore[attr-defined]
    distributed.update_weights_from_distributed = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        distributed,
    )
    upstream_checkpoint = types.ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_disk")
    upstream_checkpoint.UpdateWeightFromDisk = type("UpdateWeightFromDisk", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_disk",
        upstream_checkpoint,
    )
    upstream_delta = types.ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_disk_delta")
    upstream_delta.UpdateWeightFromDiskDelta = type(  # type: ignore[attr-defined]
        "UpdateWeightFromDiskDelta",
        (),
        {},
    )
    upstream_delta._atomic_write = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.megatron_utils.update_weight.update_weight_from_disk_delta",
        upstream_delta,
    )

    safetensors = types.ModuleType("safetensors")
    safetensors.__path__ = []  # type: ignore[attr-defined]
    safetensors_numpy = types.ModuleType("safetensors.numpy")
    safetensors_numpy.save = lambda *_args, **_kwargs: b""  # type: ignore[attr-defined]
    safetensors.numpy = safetensors_numpy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.numpy", safetensors_numpy)

    disk_delta = types.ModuleType("slime.utils.disk_delta")
    disk_delta.NUM_WORKERS = 1  # type: ignore[attr-defined]
    disk_delta.checksum = lambda *_args: "digest"  # type: ignore[attr-defined]
    disk_delta.make_tensor_reader = lambda *_args: lambda _name: b""  # type: ignore[attr-defined]
    disk_delta.overwrite_encode = lambda new, _mask: new  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils.disk_delta", disk_delta)

    zstandard = types.ModuleType("zstandard")
    zstandard.ZstdCompressor = type(  # type: ignore[attr-defined]
        "ZstdCompressor", (), {"__init__": lambda self, **_kwargs: None, "compress": lambda self, value: value}
    )
    monkeypatch.setitem(sys.modules, "zstandard", zstandard)

    hf_saver = types.ModuleType("slime.backends.megatron_utils.hf_checkpoint_saver")
    hf_saver.save_hf_model_to_path = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.hf_checkpoint_saver", hf_saver)


def _load_module(monkeypatch: pytest.MonkeyPatch, filename: str):
    _install_updater_stubs(monkeypatch)
    package = "reef.train.slime_backend.reef_adapters.weight_updaters"
    root = Path(__file__).parents[2] / "reef" / "train" / "slime_backend" / "reef_adapters" / "weight_updaters"
    package_module = types.ModuleType(package)
    package_module.__path__ = [str(root)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, package, package_module)

    base_name = f"{package}.base"
    base_spec = importlib.util.spec_from_file_location(base_name, root / "base.py")
    assert base_spec is not None and base_spec.loader is not None
    base = importlib.util.module_from_spec(base_spec)
    monkeypatch.setitem(sys.modules, base_name, base)
    base_spec.loader.exec_module(base)

    distributed_name = f"{package}.distributed"
    distributed_spec = importlib.util.spec_from_file_location(distributed_name, root / "distributed.py")
    assert distributed_spec is not None and distributed_spec.loader is not None
    distributed = importlib.util.module_from_spec(distributed_spec)
    monkeypatch.setitem(sys.modules, distributed_name, distributed)
    distributed_spec.loader.exec_module(distributed)
    if filename == "distributed.py":
        return distributed

    name = f"{package}.{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, root / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_reef_train_actor_adapter(monkeypatch: pytest.MonkeyPatch):
    _install_updater_stubs(monkeypatch)
    actor_module = types.ModuleType("slime.backends.megatron_utils.actor")

    class MegatronTrainRayActor:
        def update_weights(self):
            return self.weight_updater.update_weights()

    actor_module.MegatronTrainRayActor = MegatronTrainRayActor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.actor", actor_module)
    memory_utils = types.ModuleType("slime.utils.memory_utils")
    memory_utils.print_memory = lambda *_args: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils.memory_utils", memory_utils)
    reloadable = types.ModuleType("slime.utils.reloadable_process_group")
    reloadable.destroy_process_groups = lambda: None  # type: ignore[attr-defined]
    reloadable.reload_process_groups = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils.reloadable_process_group", reloadable)
    timer = types.ModuleType("slime.utils.timer")
    timer.timer = _stub_timer_decorator  # type: ignore[attr-defined]
    timer.Timer = _StubTimer  # type: ignore[attr-defined]
    _StubTimer.reset()
    monkeypatch.setitem(sys.modules, "slime.utils.timer", timer)
    memory_saver = types.ModuleType("torch_memory_saver")
    memory_saver.torch_memory_saver = types.SimpleNamespace(disable=lambda: nullcontext())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_memory_saver", memory_saver)
    lora = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.lora")
    lora.collect_lora_train_metrics = lambda _model: {}  # type: ignore[attr-defined]
    lora.megatron_lora_enabled = lambda _args: False  # type: ignore[attr-defined]
    lora.lora_per_scenario_enabled = lambda _args: False  # type: ignore[attr-defined]
    lora.zero_megatron_lora_adapters = lambda _model: None  # type: ignore[attr-defined]
    lora.is_megatron_lora_parameter = lambda name: ".adapter." in name  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reef.train.slime_backend.reef_adapters.megatron.lora", lora)
    lora_checkpoint = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint")
    lora_checkpoint.save_lora_adapter_to_path = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "reef.train.slime_backend.reef_adapters.megatron.lora_checkpoint",
        lora_checkpoint,
    )
    hooks = types.ModuleType("reef.train.slime_backend.reef_adapters.worker_hooks")
    hooks.drain_worker_metrics = dict  # type: ignore[attr-defined]
    hooks.record_worker_metrics = lambda _metrics: None  # type: ignore[attr-defined]
    hooks.reef_node_ip_and_free_port = lambda: ("127.0.0.1", 1234)  # type: ignore[attr-defined]
    hooks._loss_family_spec = lambda _args: None  # type: ignore[attr-defined]
    hooks.resolve_tensor_dtype = lambda name: name  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reef.train.slime_backend.reef_adapters.worker_hooks", hooks)

    name = "reef.train.slime_backend.reef_adapters._train_actor_recovery_test"
    path = (
        Path(__file__).parents[2]
        / "reef"
        / "train"
        / "slime_backend"
        / "reef_adapters"
        / "megatron"
        / "train_actor.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_reef_actor_adapter_passes_generation_policy_without_widening_slime_actor_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_reef_train_actor_adapter(monkeypatch)
    calls = []

    class Updater:
        def update_weights(self, *, manage_generation=True, force_full=False):
            calls.append((manage_generation, force_full))
            return "updated"

    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.weight_updater = Updater()
    actor.args = object()

    assert actor.update_weights(manage_generation=False, force_full=True) == "updated"
    assert calls == [(False, True)]
    assert "update_weights" not in vars(actor.weight_updater)


@pytest.mark.unit
def test_reef_actor_adapter_remains_compatible_with_unextended_updater(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_reef_train_actor_adapter(monkeypatch)
    calls = []

    class Updater:
        def update_weights(self):
            calls.append("updated")

    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.weight_updater = Updater()
    actor.args = object()

    actor.update_weights(manage_generation=False, force_full=True)

    assert calls == ["updated"]


@pytest.mark.unit
def test_full_weight_save_delegates_without_reopening_slimes_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Slime times its own save_model. A full-weight save delegates to it, so
    # the adapter must not hold a timer of the same name open across the call:
    # Slime's Timer is a singleton and asserts "Timer save_model already
    # started", which failed every non-LoRA save after a training step.
    module = _load_reef_train_actor_adapter(monkeypatch)
    saved: list[tuple[int, bool]] = []
    base = module.ReefMegatronTrainRayActor.__mro__[1]

    def base_save_model(self, rollout_id, force_sync=False):
        with _StubTimer().context("save_model"):
            saved.append((rollout_id, force_sync))

    monkeypatch.setattr(base, "save_model", base_save_model, raising=False)
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = object()

    actor.save_model(7, force_sync=True)

    assert saved == [(7, True)]
    assert _StubTimer.started == ["save_model"]


@pytest.mark.unit
def test_lora_save_times_the_branch_that_does_not_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    # The LoRA branch saves the adapter itself, so it keeps the timing the
    # decorator used to provide.
    module = _load_reef_train_actor_adapter(monkeypatch)
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda _args: True)
    saved: list[tuple[int, bool]] = []
    base = module.ReefMegatronTrainRayActor.__mro__[1]
    monkeypatch.setattr(
        base,
        "save_model",
        lambda *_args, **_kwargs: pytest.fail("the LoRA branch must not delegate"),
        raising=False,
    )
    actor = object.__new__(module.ReefMegatronTrainRayActor)
    actor.args = object()
    monkeypatch.setattr(
        actor,
        "_save_lora_model",
        lambda rollout_id, force_sync=False: saved.append((rollout_id, force_sync)),
        raising=False,
    )

    actor.save_model(3)

    assert saved == [(3, False)]
    assert _StubTimer.started == ["save_model"]


@pytest.mark.unit
def test_failed_distributed_fanout_poisons_the_rollout_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch, "distributed.py")
    events: list[str] = []

    class RemoteMethod:
        def __init__(self, value: str) -> None:
            self.value = value

        def remote(self):
            events.append(self.value)
            return self.value

    updater = object.__new__(module.ReefUpdateWeightFromDistributed)
    updater.rollout_engine_lock = types.SimpleNamespace(
        acquire=RemoteMethod("acquire"),
        release=RemoteMethod("release"),
        poison=RemoteMethod("poison"),
    )
    updater._group_name = "group"
    updater._model_update_groups = object()
    updater.runtime_load_id_incarnation = "engine"
    updater.weight_update_sequence = 4
    updater.rollout_engines = [object()]
    tensors = [("weight", object())]

    def get(value, **_kwargs):
        if value == "acquire":
            return True
        if value == ["failed-update"]:
            raise RuntimeError("fanout failed")
        return value

    monkeypatch.setattr(module.ray, "get", get)
    base = sys.modules["reef.train.slime_backend.reef_adapters.weight_updaters.base"]
    monkeypatch.setattr(base, "update_weights_from_distributed", lambda *_args, **_kwargs: ["failed-update"])

    with pytest.raises(RuntimeError, match="fanout failed"):
        updater._update_bucket_weights_from_distributed(tensors)

    assert tensors == []
    assert events == ["acquire", "poison"]


@pytest.mark.unit
def test_force_full_uses_complete_disk_checkpoint_and_rebuilds_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module(monkeypatch, "checkpoint.py")
    events: list[object] = []

    class RemoteMethod:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))
            return self.name

    updater = object.__new__(module.ReefUpdateWeightFromDiskDelta)
    updater.args = types.SimpleNamespace(
        update_weight_disk_dir=str(tmp_path),
        update_weight_local_checkpoint_dir="/local/weights",
    )
    updater.delta_dir = str(tmp_path)
    updater.model = []
    updater.model_name = "model"
    updater.quantization_config = None
    updater.runtime_load_id_incarnation = "engine"
    updater.weight_update_sequence = 6
    updater.rollout_engines = [
        types.SimpleNamespace(
            pull_weights=RemoteMethod("pull"),
            update_weights_from_disk=RemoteMethod("activate"),
        )
    ]
    updater.rollout_engine_lock = types.SimpleNamespace(poison=RemoteMethod("poison"))
    updater._post_write_hook = None
    updater._snapshot = {"old": object()}
    updater._baseline_captured = True
    updater._run_rank_zero_action = lambda action, **_kwargs: action()
    updater._raise_synchronized_update_error = lambda error, **_kwargs: (_ for _ in ()).throw(error) if error else None
    updater._iter_hf_tensors = lambda: iter([])
    monkeypatch.setattr(module, "save_hf_model_to_path", lambda *_args, **_kwargs: events.append("save-full"))

    updater._republish_full_weights(manage_generation=False)

    assert updater.weight_update_sequence == 7
    assert events[0] == "save-full"
    assert events[1][0] == "pull"
    assert events[2] == (
        "activate",
        (),
        {"model_path": "/local/weights", "runtime_load_id": "engine:7"},
    )
    assert updater._snapshot == {}
    assert updater._baseline_captured is True


@pytest.mark.unit
def test_rank_zero_delta_weight_update_failure_poisons_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "checkpoint.py")
    events: list[str] = []

    class RemoteMethod:
        def __init__(self, event: str) -> None:
            self.event = event

        def remote(self, *_args, **_kwargs):
            events.append(self.event)
            return self.event

    updater = object.__new__(module.ReefUpdateWeightFromDiskDelta)
    updater._post_write_hook = None
    updater._version_dir = "/weights/v7"
    updater.runtime_load_id_incarnation = "engine"
    updater.weight_update_sequence = 7
    updater.args = types.SimpleNamespace(update_weight_local_checkpoint_dir="/local/weights")
    updater.rollout_engines = [types.SimpleNamespace(pull_weights=RemoteMethod("pull"))]
    updater.rollout_engine_lock = types.SimpleNamespace(poison=RemoteMethod("poison"))
    updater._raise_synchronized_update_error = lambda error, **_kwargs: (_ for _ in ()).throw(error) if error else None
    updater._run_rank_zero_action = lambda _action, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("receiver apply failed")
    )

    with pytest.raises(RuntimeError, match="receiver apply failed"):
        updater._reload_engines(manage_generation=False)

    assert events == ["poison"]


@pytest.mark.unit
def test_source_phase_publishes_conversion_failure_and_poisons_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "distributed.py")
    events: list[str] = []

    class RemoteMethod:
        def __init__(self, event: str, result=None) -> None:
            self.event = event
            self.result = result

        def remote(self, *_args):
            events.append(self.event)
            return self.result

    def fail_conversion():
        raise RuntimeError("conversion failed")

    updater = object.__new__(module.ReefUpdateWeightFromDistributed)
    updater.args = types.SimpleNamespace(distributed_timeout_minutes=1)
    updater._is_pp_src_rank = True
    updater.rollout_engine_lock = types.SimpleNamespace(
        poison=RemoteMethod("poison"),
        complete_phase=RemoteMethod("complete"),
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        updater._run_source_phase(
            "conversion:0",
            fail_conversion,
            phase="weight conversion",
        )

    assert events == ["complete", "poison"]


@pytest.mark.unit
def test_peer_waits_for_source_phase_completion_before_continuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch, "distributed.py")
    outcomes: dict[str, dict[str, str | None]] = {}
    peer_polled = threading.Event()

    class RemoteMethod:
        def __init__(self, function) -> None:
            self.function = function

        def remote(self, *args):
            return self.function(*args)

    lock = types.SimpleNamespace(
        complete_phase=RemoteMethod(lambda phase_id, error: outcomes.__setitem__(phase_id, {"error": error})),
        phase_status=RemoteMethod(lambda phase_id: (peer_polled.set(), outcomes.get(phase_id))[1]),
        poison=RemoteMethod(lambda: None),
    )
    source = object.__new__(module.ReefUpdateWeightFromDistributed)
    source.args = types.SimpleNamespace(distributed_timeout_minutes=1)
    source._is_pp_src_rank = True
    source.rollout_engine_lock = lock
    peer = object.__new__(module.ReefUpdateWeightFromDistributed)
    peer.args = source.args
    peer._is_pp_src_rank = False
    peer.rollout_engine_lock = lock
    peer_result = []

    thread = threading.Thread(
        target=lambda: peer_result.append(peer._run_source_phase("phase:0", lambda: None, phase="conversion"))
    )
    thread.start()
    assert peer_polled.wait(timeout=1)
    assert thread.is_alive()

    assert source._run_source_phase("phase:0", lambda: "converted", phase="conversion") == "converted"
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert peer_result == [None]
