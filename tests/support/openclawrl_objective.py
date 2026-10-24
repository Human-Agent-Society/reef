"""Load OpenClaw-RL's Megatron objective with its device glue stubbed, for CPU tests.

The tensor kernel and Slime's PPO helpers stay real. Megatron is not
available on a CPU host, so the stubs supply the two worker seams the
objective reads: the tensor-parallel group and the runtime's response slicing.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace

import pytest

OBJECTIVE_MODULE = "recipes.openclawrl.slime.objective"


def load_topk_objective(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Yield the objective module, and remove it from ``sys.modules`` afterwards; skip without torch or Slime.

    Use it from a fixture: ``yield from load_topk_objective(monkeypatch)``.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("slime")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    loss_module = ModuleType("slime.backends.megatron_utils.loss")

    def get_log_probs_and_entropy(logits, **kwargs):
        log_probs = logits.log_softmax(dim=-1)[:, 0]
        output = {"log_probs": [log_probs]}
        if kwargs.get("with_entropy"):
            probabilities = logits.softmax(dim=-1)
            output["entropy"] = [-(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)]
        return torch.empty(0), output

    def get_responses(logits, **kwargs):
        yield logits, torch.empty(0, dtype=torch.long)

    loss_module.get_log_probs_and_entropy = get_log_probs_and_entropy
    loss_module.get_responses = get_responses
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", loss_module)

    previous = sys.modules.pop(OBJECTIVE_MODULE, None)
    try:
        yield importlib.import_module(OBJECTIVE_MODULE)
    finally:
        sys.modules.pop(OBJECTIVE_MODULE, None)
        if previous is not None:
            sys.modules[OBJECTIVE_MODULE] = previous


__all__ = ["load_topk_objective"]
