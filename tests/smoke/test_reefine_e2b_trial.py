"""Reefine's scripted trial in an E2B sandbox: the trial driver runs with node in the shared session."""

from __future__ import annotations

import json
import os

import pytest

from recipes.beta.reefine.agent import launch_pi
from reef.execution.sandbox.config import build_executor
from reef.execution.sandbox.e2b import E2BExecutor
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.vendor_install import AdapterProgram

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_E2B") != "1", reason="opt-in paid E2B/OpenRouter test")


def test_reefine_scripted_trial_uses_node_in_the_shared_session(tmp_path):
    executor, _ = build_executor({"executor": "e2b"}).for_program(AdapterProgram(get_adapter("pi")))
    assert isinstance(executor, E2BExecutor)
    session = executor.open()
    root = tmp_path / "trial"
    root.mkdir()
    try:
        outcome = launch_pi(
            get_adapter("pi"),
            "/host-only/pi",
            session,
            {},
            "",
            {},
            root=root,
            timeout=60,
            script={"fixture_tools": [], "steps": [{"prompt": "hello", "expect": {"model_called": True}}]},
        )[0]
        report = json.loads((root / "sessions/trial-result.json").read_text())
        assert outcome.exit_code == 0, (outcome.stderr, report)
        assert report["passed"]
        assert report["steps"][0]["model_requests"] == 1
    finally:
        session.close()
