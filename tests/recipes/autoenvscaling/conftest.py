"""The fixtures of the AutoEnvScaling tests."""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``import e2b`` succeed where the package is missing.

    Building the agent Designer checks that the package imports; no test here opens a sandbox.
    """
    if importlib.util.find_spec("e2b") is None:
        monkeypatch.setitem(sys.modules, "e2b", types.ModuleType("e2b"))
