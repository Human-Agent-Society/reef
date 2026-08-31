from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

if not __package__:
    # Direct execution (`python test_plugin_runtime_hook_contracts.py ...`)
    # imports this file as a top-level module; put tests/ on sys.path so the
    # canonical package import below works in that mode too. Under pytest the
    # file is always imported as part of the plugin_contracts package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin_contracts._shared import get_contract_path, install_paths, run_contract_test_for_file, stubbed_modules

install_paths()

NUM_GPUS = 0

# The vendored utils module imports torch at module scope; stub it only for
# the duration of this import so a fake torch never leaks into the rest of
# the test session.
with stubbed_modules(with_torch=True):
    from slime.utils.misc import load_function


def run_contract_test_file() -> None:
    run_contract_test_for_file(__file__, path_args=["rollout-data-postprocess-path"])


def reference_rollout_data_postprocess(args, rollout_id, rollout_data) -> None:
    args.rollout_data_postprocess_called = True


def test_rollout_data_postprocess_callsite_is_stable() -> None:
    spec = importlib.util.find_spec("slime.backends.megatron_utils.actor")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text()
    assert "self.rollout_data_postprocess(self.args, rollout_id, rollout_data)" in source


def test_rollout_data_postprocess_path_aligns_with_expected_format() -> None:
    path = get_contract_path(
        "ROLLOUT_DATA_POSTPROCESS_PATH",
        "plugin_contracts.test_plugin_runtime_hook_contracts.reference_rollout_data_postprocess",
    )
    # A user-supplied hook module may itself import torch; keep the stub
    # scoped to loading and exercising the hook.
    with stubbed_modules(with_torch=True):
        fn = load_function(path)
        assert tuple(inspect.signature(fn).parameters) == ("args", "rollout_id", "rollout_data")

        args = type("Args", (), {})()
        assert fn(args, 0, {}) is None
        assert args.rollout_data_postprocess_called is True


if __name__ == "__main__":
    run_contract_test_file()
