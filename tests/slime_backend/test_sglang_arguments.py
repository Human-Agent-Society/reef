from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_arguments_module(monkeypatch: pytest.MonkeyPatch):
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    launch_router = types.ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = type("RouterArgs", (), {})
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang_router", types.ModuleType("sglang_router"))
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)
    http_utils = types.ModuleType("slime.utils.http_utils")
    http_utils._wrap_ipv6 = lambda value: value  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "slime.utils.http_utils",
        http_utils,
    )

    installed = importlib.util.find_spec("slime.backends.sglang_utils.arguments")
    assert installed is not None and installed.origin is not None
    spec = importlib.util.spec_from_file_location("_reef_test_sglang_arguments", Path(installed.origin))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("parallel_sizes", "expected"),
    [
        (
            {
                "sglang_data_parallel_size": 2,
                "sglang_pipeline_parallel_size": 2,
                "sglang_expert_parallel_size": 4,
                "sglang_moe_data_parallel_size": 1,
            },
            (2, 2, 4, 4),
        ),
        (
            {
                "sglang_dp_size": 2,
                "sglang_pp_size": 2,
                "sglang_ep_size": 4,
                "sglang_moe_dp_size": 1,
            },
            (2, 2, 4, 4),
        ),
    ],
)
def test_validate_args_accepts_both_sglang_parallel_size_apis(
    monkeypatch: pytest.MonkeyPatch,
    parallel_sizes: dict[str, int],
    expected: tuple[int, int, int, int],
) -> None:
    arguments = _load_arguments_module(monkeypatch)
    args = SimpleNamespace(
        **parallel_sizes,
        rollout_num_gpus_per_engine=8,
        sglang_enable_dp_attention=True,
        sglang_router_ip=None,
        prefill_num_servers=None,
        rollout_external=False,
        sglang_config=None,
    )

    arguments.validate_args(args)

    assert (
        args.sglang_dp_size,
        args.sglang_pp_size,
        args.sglang_ep_size,
        args.sglang_tp_size,
    ) == expected
