"""Training drains must fail, rather than report success with unfinished steps."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/sao/examples/imo_answerbench"


@pytest.mark.parametrize("filename", ["run.py", "stream.py"])
def test_training_drain_timeout(filename, monkeypatch):
    if filename == "run.py":
        # run.py drives reef-eval's Lab at import; the source suite does not install reef-eval.
        pytest.importorskip("reef_eval")
    monkeypatch.syspath_prepend(str(EXAMPLE))
    # Both example harnesses use the self-contained package name `harness`.
    for name in list(sys.modules):
        if name == "harness" or name.startswith("harness."):
            monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("imo_drain", EXAMPLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = iter([0, 0, 2])
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: next(clock), sleep=lambda _: None))
    monkeypatch.setattr(module, "TRAIN_DRAIN_TIMEOUT_S", 1)
    if filename == "stream.py":
        monkeypatch.setattr(module, "releases", list)
    else:
        monkeypatch.setattr(module, "training_release_count", lambda: 0)
    with pytest.raises(TimeoutError, match="0/1"):
        module.wait_for_training(1)
