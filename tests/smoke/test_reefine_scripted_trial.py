"""Scripted trials against the installed pi, without a paid model or external network."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.tree.render import render_composition
from reef.recipe.reefine.agent import AGENT_TOOLS, launch_pi
from reef.recipe.reefine.trial import trial_script
from reef.train.cordis_backend.backend import _StepCalls
from reef.train.cordis_backend.strategies import AgentHost

REAL_PI = os.environ.get("REEF_REAL_PI_BINARY", "")
pytestmark = pytest.mark.skipif(not REAL_PI, reason="REEF_REAL_PI_BINARY does not name a real pi binary")

CHAT_EXTENSION = """
export default function(pi) {
  let active = false;
  let previous;
  pi.registerCommand('chat', {description: 'Toggle chat', handler: async () => {
    active = !active;
    if (active) { previous = pi.getActiveTools(); pi.setActiveTools(['web_search']); }
    else pi.setActiveTools(previous);
  }});
  pi.on('before_agent_start', event => active ? {systemPrompt: 'Chat only.'} : undefined);
  pi.on('tool_call', event => active && event.toolName !== 'web_search'
    ? {block: true, reason: 'chat mode'} : undefined);
}
"""


@pytest.mark.parametrize("broken", [False, True])
def test_script_observes_mode_switching_tool_execution_and_fresh_sessions(tmp_path: Path, broken: bool) -> None:
    extension = CHAT_EXTENSION
    if broken:
        extension = extension.replace("['web_search']", "['web_search', 'file_search_query']")
    descriptor = get_adapter("pi")
    host = AgentHost(descriptor, REAL_PI, LocalExecutor(), None, _StepCalls(0, []), 60.0, 30.0)
    files = render_composition(
        (
            ("rules", {"text": "PRIVATE_AGENT_MARKER"}),
            ("skill", {"name": "private", "text": "---\nname: private\ndescription: PRIVATE_SKILL_MARKER\n---\nbody"}),
            ("code_extension", {"name": "chat", "code": extension}),
        ),
        descriptor,
    )
    plan = trial_script(
        {
            "fixture_tools": ["web_search", "file_search_query"],
            "steps": [
                {"prompt": "normal", "expect": {"system_contains": ["PRIVATE_AGENT_MARKER", "PRIVATE_SKILL_MARKER"]}},
                {"prompt": "/chat", "expect": {"model_called": False}},
                {
                    "prompt": "try local search",
                    "tool_call": {"name": "file_search_query", "arguments": {}},
                    "expect": {
                        "tools": ["web_search"],
                        "executed_tools": [],
                        "tool_errors": ["file_search_query"],
                        "system_excludes": ["PRIVATE_AGENT_MARKER", "PRIVATE_SKILL_MARKER"],
                    },
                },
                {
                    "prompt": "search web",
                    "tool_call": {"name": "web_search", "arguments": {}},
                    "expect": {"executed_tools": ["web_search"], "tool_errors": []},
                },
                {"prompt": "/chat"},
                {
                    "prompt": "restored",
                    "tool_call": {"name": "file_search_query", "arguments": {}},
                    "expect": {"executed_tools": ["file_search_query"], "system_contains": ["PRIVATE_AGENT_MARKER"]},
                },
                {"prompt": "/chat"},
                {"new_session": True},
                {"prompt": "fresh", "expect": {"system_contains": ["PRIVATE_AGENT_MARKER"]}},
            ],
        }
    )
    outcome, _ = launch_pi(host, host.executor, files, "", {}, root=tmp_path, timeout=45, script=plan)
    report = json.loads((tmp_path / "sessions/trial-result.json").read_text())
    assert report["passed"] is not broken, (report, outcome.stderr)
    assert outcome.exit_code == int(broken)
    assert not report["errors"]
    if broken:
        assert any(not check["passed"] and check["name"] == "tools" for check in report["steps"][2]["checks"])


def test_script_checks_provider_payload_after_extension_hooks_and_requires_a_request(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    host = AgentHost(descriptor, REAL_PI, LocalExecutor(), None, _StepCalls(0, []), 60.0, 30.0)
    files = render_composition(
        (
            (
                "code_extension",
                {
                    "name": "payload",
                    "code": """
export default function(pi) {
  pi.registerCommand('noop', {description: 'Do nothing', handler: async () => {}});
  pi.on('before_provider_request', event => ({...event.payload, tools: []}));
}
""",
                },
            ),
        ),
        descriptor,
    )
    plan = trial_script(
        {
            "steps": [
                {"prompt": "hello", "expect": {"tools": []}},
                {"prompt": "/noop", "expect": {"system_excludes": ["absent-marker"]}},
            ]
        }
    )
    outcome, _ = launch_pi(host, host.executor, files, "", {}, root=tmp_path, timeout=45, script=plan)
    report = json.loads((tmp_path / "sessions/trial-result.json").read_text())
    assert not report["passed"] and outcome.exit_code == 1, (report, outcome.stderr)
    assert report["steps"][0]["passed"]
    assert not report["steps"][1]["passed"] and report["steps"][1]["model_requests"] == 0


def test_proposer_restores_saved_progress_and_deadline_without_old_history(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    host = AgentHost(descriptor, REAL_PI, LocalExecutor(), None, _StepCalls(0, []), 60.0, 30.0)
    files = render_composition(
        (("code_extension", {"name": "proposer", "code": AGENT_TOOLS.read_text()}),), descriptor
    )
    plan = trial_script(
        {
            "steps": [
                {
                    "prompt": "Record the finding",
                    "tool_call": {
                        "name": "harness_progress",
                        "arguments": {"text": "Confirmed API_MARKER; next verify restoration."},
                    },
                    "expect": {"tool_errors": []},
                },
                {"new_session": True},
                {
                    "prompt": "Continue",
                    "expect": {
                        "messages_contains": ["API_MARKER", "Remaining execution time", "Finish now"],
                        "messages_excludes": ["Record the finding"],
                    },
                },
            ]
        }
    )
    env = {
        "REEF_PROPOSER_URL": "http://127.0.0.1:1",
        "REEF_PROPOSER_DEADLINE_MS": str(int(time.time() * 1000) + 30000),
        "REEF_PROPOSER_FINISH_SECONDS": "60",
    }
    outcome, _ = launch_pi(host, host.executor, files, "", env, root=tmp_path, timeout=45, script=plan)
    report = json.loads((tmp_path / "sessions/trial-result.json").read_text())
    assert report["passed"] and outcome.exit_code == 0, (report, outcome.stderr)
    assert "API_MARKER" in (tmp_path / "workspace/progress.md").read_text()


def test_script_reports_extension_errors_even_when_payload_checks_pass(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    host = AgentHost(descriptor, REAL_PI, LocalExecutor(), None, _StepCalls(0, []), 60.0, 30.0)
    files = render_composition(
        (
            (
                "code_extension",
                {
                    "name": "broken",
                    "code": """
export default function(pi) {
  pi.on('before_agent_start', () => { throw new Error('BROKEN_EXTENSION'); });
}
""",
                },
            ),
        ),
        descriptor,
    )
    plan = trial_script({"steps": [{"prompt": "hello", "expect": {"model_called": True}}]})
    outcome, _ = launch_pi(host, host.executor, files, "", {}, root=tmp_path, timeout=45, script=plan)
    report = json.loads((tmp_path / "sessions/trial-result.json").read_text())
    assert outcome.exit_code == 1 and not report["passed"]
    assert any("BROKEN_EXTENSION" in error for error in report["errors"])


def test_script_checks_an_unrelated_verbosity_command(tmp_path: Path) -> None:
    """The same trial mechanism checks a preference unrelated to tool restrictions."""
    descriptor = get_adapter("pi")
    host = AgentHost(descriptor, REAL_PI, LocalExecutor(), None, _StepCalls(0, []), 60.0, 30.0)
    files = render_composition(
        (
            (
                "code_extension",
                {
                    "name": "verbosity",
                    "code": """
export default function(pi) {
  let concise = false;
  pi.registerCommand('verbosity', {description: 'Set answer verbosity', handler: async args => {
    concise = args.trim() === 'concise';
  }});
  pi.on('before_agent_start', event => concise
    ? {systemPrompt: event.systemPrompt + '\\nAnswer concisely.'} : undefined);
}
""",
                },
            ),
        ),
        descriptor,
    )
    plan = trial_script(
        {
            "steps": [
                {"prompt": "/verbosity concise", "expect": {"model_called": False}},
                {"prompt": "Explain a term", "expect": {"system_contains": ["Answer concisely."]}},
                {"prompt": "/verbosity normal", "expect": {"model_called": False}},
                {"prompt": "Explain a term", "expect": {"system_excludes": ["Answer concisely."]}},
                {"prompt": "/verbosity concise"},
                {"new_session": True},
                {"prompt": "Explain a term", "expect": {"system_excludes": ["Answer concisely."]}},
            ]
        }
    )
    outcome, _ = launch_pi(host, host.executor, files, "", {}, root=tmp_path, timeout=45, script=plan)
    report = json.loads((tmp_path / "sessions/trial-result.json").read_text())
    assert report["passed"] and outcome.exit_code == 0, (report, outcome.stderr)
