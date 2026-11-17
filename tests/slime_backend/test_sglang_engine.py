from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
import types
from concurrent.futures import Future
from pathlib import Path

import pytest

from reef.core.weight_layout import ArtifactFormat
from reef.inference.remote.runtime import serving_engine_config
from reef.inference.sglang import plugin as sglang_plugin
from reef.inference.sglang.config import SGLangSettings
from reef.inference.sglang.lora_schema import adapter_scoped_prefix_cache_supported
from reef.inference.sglang.plugin import (
    REEF_SGLANG_PLUGIN_ENV,
    install_colocated_retract_offload,
    install_runtime_load_id_updates,
)
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars


def _load_sglang_engine_module(monkeypatch: pytest.MonkeyPatch):
    requests = types.ModuleType("requests")
    requests.get = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    requests.post = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    requests.exceptions = types.SimpleNamespace(  # type: ignore[attr-defined]
        HTTPError=type("HTTPError", (Exception,), {}),
    )
    urllib3 = types.ModuleType("urllib3")
    urllib3.__path__ = []  # type: ignore[attr-defined]
    urllib3_exceptions = types.ModuleType("urllib3.exceptions")
    urllib3_exceptions.NewConnectionError = type("NewConnectionError", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "urllib3", urllib3)
    monkeypatch.setitem(sys.modules, "urllib3.exceptions", urllib3_exceptions)

    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__version__ = "0.3.0"  # type: ignore[attr-defined]
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    sglang_utils = types.ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda _pid: None  # type: ignore[attr-defined]
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []  # type: ignore[attr-defined]
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_srt.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    lora = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.lora")
    lora.LORA_ADAPTER_NAME = "reef_lora"  # type: ignore[attr-defined]
    lora.megatron_lora_enabled = lambda _args: False  # type: ignore[attr-defined]
    lora.sglang_lora_target_modules = lambda _args: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reef.train.slime_backend.reef_adapters.megatron.lora", lora)

    external = types.ModuleType("slime.backends.sglang_utils.external")
    external.get_server_info = lambda _url: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", external)

    class BaseEngine:
        def __init__(self, args, rank, base_gpu_id=None):
            self.args = args
            self.rank = rank
            self.base_gpu_id = base_gpu_id

        def _register_to_router(self, server_args_dict):
            self.registered = server_args_dict
            return "registered"

    raw_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    raw_engine.SGLangEngine = BaseEngine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_engine", raw_engine)

    path = Path(__file__).parents[2] / "reef" / "inference" / "sglang" / "engine.py"
    spec = importlib.util.spec_from_file_location("_reef_test_sglang_engine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_engine_extension_preserves_version_and_adapter_unload_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []
    engine = object.__new__(module.ReefSGLangEngine)

    def request(route, payload=None, **_kwargs):
        calls.append((route, payload or {}))
        return {"success": True}

    engine._make_request = request

    assert engine.set_runtime_load_id("deployment:2") == {"success": True}
    assert engine.load_lora_adapter_from_disk("adapter", "/release/adapter") == {"success": True}
    assert engine.unload_lora_adapter("adapter") == {"success": True}
    assert calls == [
        ("update_weight_version", {"new_version": "deployment:2", "abort_all_requests": False}),
        ("load_lora_adapter", {"lora_name": "adapter", "lora_path": "/release/adapter"}),
        ("unload_lora_adapter", {"lora_name": "adapter"}),
    ]


@pytest.mark.unit
def test_get_runtime_load_id_uses_model_info(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    requested: list[tuple[str, float]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"weight_version": "training-incarnation:3"}

    def get(url: str, *, timeout: float):
        requested.append((url, timeout))
        return Response()

    monkeypatch.setattr(module.requests, "get", get)
    engine = object.__new__(module.ReefSGLangEngine)
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.config = SGLangSettings("model", 1, 1, 1)

    assert engine.get_runtime_load_id() == "training-incarnation:3"
    assert requested == [("http://127.0.0.1:30000/model_info", 30.0)]


@pytest.mark.unit
def test_engine_gives_the_scheduler_its_runtime_load_id_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    events = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine.get_runtime_load_id = lambda: "engine:3"
    engine._make_request = lambda endpoint, payload, **_kwargs: events.append((endpoint, payload)) or {"success": True}

    engine.router_ip = None
    assert engine._register_to_router() is None
    assert events == [("update_weight_version", {"new_version": "engine:3", "abort_all_requests": False})]
    # An engine that does not take the version fails registration.
    engine._make_request = lambda endpoint, payload, **_kwargs: {"success": False, "message": "refused"}
    with pytest.raises(RuntimeError, match="did not accept the runtime-load-ID"):
        engine._register_to_router()


@pytest.mark.unit
def test_pause_generation_forwards_in_place_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    calls = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine._make_request = lambda endpoint, payload=None: calls.append((endpoint, payload))

    engine.pause_generation("in_place")
    engine.continue_generation()

    assert calls == [("pause_generation", {"mode": "in_place"}), ("continue_generation", None)]


@pytest.mark.unit
def test_disk_weight_update_does_not_implicitly_flush_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    requested = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine._make_request = lambda endpoint, payload: requested.append((endpoint, payload)) or {"success": True}

    engine.update_weights_from_disk("/weights/v2", runtime_load_id="engine:2")

    assert requested == [
        (
            "update_weights_from_disk",
            {"model_path": "/weights/v2", "flush_cache": False, "weight_version": "engine:2"},
        )
    ]


def test_inference_engine_does_not_inherit_or_patch_slime(monkeypatch):
    from reef.inference.interfaces import InferenceEngine

    module = _load_sglang_engine_module(monkeypatch)
    assert module.ReefSGLangEngine.__bases__ == (InferenceEngine,)
    assert not hasattr(module, "install_sglang_extensions")


def test_reef_engine_carries_lora_server_args_across_ray_process(monkeypatch):
    module = _load_sglang_engine_module(monkeypatch)
    import pickle

    values = serving_engine_config(
        "sglang",
        model_path="model",
        num_gpus=1,
        tensor_parallel_size=1,
        options={},
        artifact_format=ArtifactFormat("adapter", 8, ("q_proj",)),
        adapter_capacity=1,
        prefix_sharing=False,
        executor="auto",
        executor_options={},
        environ={},
    )
    config = pickle.loads(pickle.dumps(SGLangSettings(**values)))
    engine = module.ReefSGLangEngine(config, 0)
    assert engine.config.options["enable_lora"] is True
    assert engine.config.options["lora_target_modules"] == ["q_proj"]
    assert engine.config.options["max_lora_rank"] == 8
    assert engine.config.options["max_loaded_loras"] == 1
    assert engine.config.options["tokenizer_worker_num"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["dataclass", "struct"])
def test_server_options_keep_only_the_fields_of_either_server_args_kind(monkeypatch, kind):
    """SGLang 0.5.21 made ServerArgs a msgspec Struct; earlier versions declare a dataclass. Both name their fields."""
    module = _load_sglang_engine_module(monkeypatch)
    names = (
        "model_path",
        "host",
        "port",
        "nccl_port",
        "dist_init_addr",
        "tp_size",
        "pp_size",
        "base_gpu_id",
        "gpu_id_step",
        "mem_fraction_static",
    )
    if kind == "dataclass":
        server_args = dataclasses.make_dataclass(
            "ServerArgs", [(name, object, dataclasses.field(default=None)) for name in names]
        )
    else:
        server_args = type("ServerArgs", (), {"__struct_fields__": names})
    monkeypatch.setattr(sys.modules["sglang.srt.server_args"], "ServerArgs", server_args)
    monkeypatch.setattr(module, "local_gpu_id", lambda gpu: gpu)
    config = SGLangSettings("model", 1, 1, 1, options={"mem_fraction_static": 0.5, "not_a_server_arg": 1})
    options = module.ReefSGLangEngine(config, 0)._server_options("10.0.0.1:5000", 30000, None, "10.0.0.1")
    assert set(options) <= set(names)
    assert "not_a_server_arg" not in options
    assert (options["mem_fraction_static"], options["tp_size"], options["port"]) == (0.5, 1, 30000)


@pytest.mark.unit
@pytest.mark.parametrize("broken", [None, "request", "radix", "request_id", "unsupported", "unforeseen"])
def test_adapter_prefix_check_exercises_actual_key_isolation(
    monkeypatch: pytest.MonkeyPatch, broken: str | None
) -> None:
    requests = types.ModuleType("sglang.srt.managers.schedule_batch")
    radix = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    sampling = types.ModuleType("sglang.srt.sampling.sampling_params")

    class Req:
        def __init__(self, *, rid, origin_input_text, origin_input_ids, sampling_params, lora_id):
            if broken == "unsupported":
                raise TypeError("unsupported request API")
            if broken == "unforeseen":
                raise ValueError("a later pin rejects this request shape")
            self.extra_key = None if broken == "request" else lora_id
            if broken == "request_id":
                self.extra_key = rid

    class RadixKey:
        def __init__(self, *, token_ids, extra_key):
            self.token_ids = token_ids
            self.extra_key = extra_key

        def child_key(self):
            if broken == "radix":
                return tuple(self.token_ids)
            return self.extra_key, tuple(self.token_ids)

    requests.Req = Req  # type: ignore[attr-defined]
    radix.RadixKey = RadixKey  # type: ignore[attr-defined]
    sampling.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    for module in (requests, radix, sampling):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter_scoped_prefix_cache_supported.cache_clear()

    try:
        assert adapter_scoped_prefix_cache_supported() is (broken is None)
    finally:
        adapter_scoped_prefix_cache_supported.cache_clear()


@pytest.mark.unit
def test_sglang_plugin_environment_crosses_ray_actor_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "SGLANG_PLUGINS": "telemetry,reef",
        REEF_SGLANG_PLUGIN_ENV: "1",
        "UNRELATED": "ignored",
    }

    assert reef_rollout_env_vars(env) == {
        "SGLANG_PLUGINS": "telemetry,reef",
        REEF_SGLANG_PLUGIN_ENV: "1",
    }


@pytest.mark.unit
def test_native_sglang_plugin_is_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = []
    monkeypatch.setattr(
        sglang_plugin,
        "install_runtime_load_id_updates",
        lambda: installed.append("runtime_load_ids"),
    )
    monkeypatch.setattr(
        sglang_plugin,
        "install_colocated_retract_offload",
        lambda: installed.append("colocated"),
    )
    monkeypatch.delenv(REEF_SGLANG_PLUGIN_ENV, raising=False)

    sglang_plugin.install_sglang_plugin()
    assert installed == []

    monkeypatch.setenv(REEF_SGLANG_PLUGIN_ENV, "1")
    sglang_plugin.install_sglang_plugin()
    assert installed == ["runtime_load_ids", "colocated"]


@pytest.mark.unit
def test_scheduler_refuses_stale_deltas_and_opens_a_session_for_each_weight_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_args = types.SimpleNamespace(weight_version="engine:6")
    runtime_context = types.ModuleType("sglang.srt.runtime_context")
    runtime_context.get_serving = lambda: current_args  # type: ignore[attr-defined]

    components = types.ModuleType("sglang.srt.managers.scheduler_components")
    components.__path__ = []  # type: ignore[attr-defined]
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class UpdateResult:
        def __init__(self, *, success=True, message="updated", num_paused_requests=0):
            self.success = success
            self.message = message
            self.num_paused_requests = num_paused_requests

    io_struct.UpdateWeightFromDiskReqOutput = UpdateResult  # type: ignore[attr-defined]

    class BeginWeightUpdateReqInput:
        def __init__(self, *, selector="all"):
            self.selector = selector

    class EndWeightUpdateReqInput:
        pass

    io_struct.BeginWeightUpdateReqInput = BeginWeightUpdateReqInput  # type: ignore[attr-defined]
    io_struct.EndWeightUpdateReqInput = EndWeightUpdateReqInput  # type: ignore[attr-defined]
    weight_updater = types.ModuleType("sglang.srt.managers.scheduler_components.weight_updater")

    class SchedulerWeightUpdaterManager:
        """The session semantics SGLang's RL weight path enforces."""

        calls = []
        _weight_update_in_progress = False

        def begin_weight_update(self, recv_req):
            assert not self._weight_update_in_progress, "session already open"
            self._weight_update_in_progress = True
            self.calls.append(("begin", recv_req.selector))
            return UpdateResult()

        def end_weight_update(self, recv_req):
            assert self._weight_update_in_progress, "end without begin"
            self._weight_update_in_progress = False
            self.calls.append(("end", None))
            return UpdateResult()

        def update_weights_from_disk(self, recv_req):
            self.calls.append(("disk", recv_req.weight_version))
            return self.recorded(recv_req)

        def recorded(self, recv_req):
            # SGLang's updater records the version on the scheduler after a successful load.
            current_args.weight_version = recv_req.weight_version
            return UpdateResult()

        def update_weights_from_distributed(self, recv_req):
            assert self._weight_update_in_progress, "requires an open begin_weight_update session"
            self.calls.append(("distributed", recv_req.weight_version))
            return self.recorded(recv_req)

        def update_weights_from_tensor(self, recv_req):
            assert self._weight_update_in_progress, "requires an open begin_weight_update session"
            self.calls.append(("tensor", recv_req.weight_version))
            return self.recorded(recv_req)

    weight_updater.SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", runtime_context)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler_components", components)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler_components.weight_updater", weight_updater)

    install_runtime_load_id_updates()

    updater = SchedulerWeightUpdaterManager()
    current_args.weight_version = "engine:7"
    stale = updater.update_weights_from_disk(types.SimpleNamespace(load_format="delta", weight_version="engine:6"))
    cross_incarnation = updater.update_weights_from_disk(
        types.SimpleNamespace(load_format="delta", weight_version="previous:99")
    )
    assert stale.success is False
    assert "refusing stale delta" in stale.message
    assert cross_incarnation.success is False
    assert SchedulerWeightUpdaterManager.calls == []

    current_batch = updater.update_weights_from_disk(
        types.SimpleNamespace(load_format="delta", weight_version="engine:7")
    )
    next_tensor = updater.update_weights_from_tensor(types.SimpleNamespace(weight_version="engine:8"))
    synced = updater.update_weights_from_distributed(types.SimpleNamespace(weight_version="engine:9"))
    assert current_batch.success is True
    assert next_tensor.success is True
    assert synced.success is True
    # The disk load needs no session. The two updates that assert one are
    # wrapped in begin -> update -> end, the sequence Slime's single-call form
    # omits and SGLang refuses without.
    assert SchedulerWeightUpdaterManager.calls == [
        ("disk", "engine:7"),
        ("begin", "all"),
        ("tensor", "engine:8"),
        ("end", None),
        ("begin", "all"),
        ("distributed", "engine:9"),
        ("end", None),
    ]
    assert updater._weight_update_in_progress is False
    assert current_args.weight_version == "engine:9"


@pytest.mark.unit
def test_weight_update_session_defers_to_the_caller_and_to_older_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class BeginWeightUpdateReqInput:
        def __init__(self, *, selector="all"):
            self.selector = selector

    io_struct.BeginWeightUpdateReqInput = BeginWeightUpdateReqInput  # type: ignore[attr-defined]
    io_struct.EndWeightUpdateReqInput = type("EndWeightUpdateReqInput", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    calls: list[str] = []

    class SessionUpdater:
        _weight_update_in_progress = False

        def begin_weight_update(self, recv_req):
            calls.append("begin")
            self._weight_update_in_progress = True

        def end_weight_update(self, recv_req):
            calls.append("end")
            self._weight_update_in_progress = False

    # A caller that opened its own session keeps it: opening a second one is
    # what SGLang asserts against, and closing someone else's would end it early.
    owned = SessionUpdater()
    owned._weight_update_in_progress = True
    with sglang_plugin._weight_update_session(owned):
        calls.append("update")
    assert calls == ["update"]
    assert owned._weight_update_in_progress is True

    # An SGLang predating the two-phase protocol has neither method.
    calls.clear()
    with sglang_plugin._weight_update_session(object()):
        calls.append("update")
    assert calls == ["update"]

    # A failing update still closes the session it opened.
    calls.clear()
    updater = SessionUpdater()
    with pytest.raises(RuntimeError, match="broadcast failed"), sglang_plugin._weight_update_session(updater):
        raise RuntimeError("broadcast failed")
    assert calls == ["begin", "end"]
    assert updater._weight_update_in_progress is False


@pytest.mark.unit
def test_colocated_retract_queue_is_idle_only_for_paused_gpu_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def pause_generation(self, recv_req):
            if getattr(self, "pause_error", False):
                raise RuntimeError("pause failed")

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            del for_health_check
            return (ignore_waiting or not self.waiting_queue) and not self.gpu_busy

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()

    scheduler = Scheduler()
    suspended = [object(), object()]
    scheduler.waiting_queue = suspended
    scheduler.grammar_manager = types.SimpleNamespace(grammar_queue=[])
    scheduler.gpu_busy = False

    scheduler.pause_generation(types.SimpleNamespace(mode="in_place"))
    assert scheduler.is_fully_idle() is False

    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    assert scheduler.is_fully_idle() is True
    assert scheduler.waiting_queue is suspended
    assert scheduler.is_fully_idle(for_health_check=True) is False

    scheduler.continue_generation(types.SimpleNamespace())
    assert scheduler.is_fully_idle() is False

    scheduler.pause_error = True
    with pytest.raises(RuntimeError, match="pause failed"):
        scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    assert scheduler.is_fully_idle() is False

    scheduler.pause_error = False
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    scheduler.gpu_busy = True
    assert scheduler.is_fully_idle() is False
    assert scheduler.waiting_queue is suspended


@pytest.mark.unit
def test_colocated_retract_uses_pinned_ignore_waiting_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def __init__(self):
            self.calls = []
            self.waiting_queue = []
            self.grammar_manager = types.SimpleNamespace(grammar_queue=[])

        def pause_generation(self, recv_req):
            return None

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            self.calls.append((for_health_check, ignore_waiting))
            return ignore_waiting and not for_health_check

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()

    scheduler = Scheduler()
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))

    assert scheduler.is_fully_idle(ignore_waiting=False) is True
    assert scheduler.calls[-1] == (False, True)
    assert scheduler.is_fully_idle(for_health_check=True) is False
    assert scheduler.calls[-1] == (True, False)
    assert scheduler.is_fully_idle(True, False) is False
    assert scheduler.calls[-1] == (True, False)


@pytest.mark.unit
def test_retract_flush_preserves_pending_grammar_and_gpu_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")
    future: Future[str] = Future()
    request = types.SimpleNamespace(grammar=future)
    grammar_queue = [request]

    class Scheduler:
        def __init__(self):
            self.waiting_queue = []
            self.grammar_manager = types.SimpleNamespace(grammar_queue=grammar_queue)
            self.gpu_busy = False
            self.idle_error = False
            self.cache = {"old-prefix": "old-kv"}

        def pause_generation(self, recv_req):
            return None

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            if self.idle_error:
                raise RuntimeError("idle check failed")
            return (
                not self.gpu_busy
                and (ignore_waiting or not self.waiting_queue)
                and (for_health_check or not self.grammar_manager.grammar_queue)
            )

        def flush_cache(self):
            if not self.is_fully_idle():
                return False
            self.cache.clear()
            return True

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()
    scheduler = Scheduler()
    scheduler.pause_generation(types.SimpleNamespace(mode="in_place"))
    assert scheduler.flush_cache() is False
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    scheduler.gpu_busy = True
    assert scheduler.flush_cache() is False
    assert scheduler.cache
    scheduler.gpu_busy = False
    assert scheduler.flush_cache() is True
    assert not scheduler.cache
    assert scheduler.grammar_manager.grammar_queue is grammar_queue
    assert not future.done()

    scheduler.idle_error = True
    with pytest.raises(RuntimeError, match="idle check failed"):
        scheduler.is_fully_idle()
    assert scheduler.grammar_manager.grammar_queue is grammar_queue
    scheduler.idle_error = False
    scheduler.continue_generation(types.SimpleNamespace())
    assert scheduler.is_fully_idle() is False
    future.set_result("compiled grammar")
    assert scheduler.grammar_manager.grammar_queue.pop().grammar.result() == "compiled grammar"
