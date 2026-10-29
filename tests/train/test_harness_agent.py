"""The deployment's coding agent in a harness step, and ``AgentMethod``, a method that is a folder of prompts.

The agent is a stand-in for pi: a script that reads its prompt and rules,
calls the gateway's tools and model route, edits the workspace and writes
what it saw to ``output/notes.json``. The gateway relays its model call to a
local stand-in upstream. Rollouts run the fake pi of ``harness_steps``.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_recipe

from reef.core.plugins import PluginConfigError
from reef.execution.sandbox.e2b import E2BExecutor
from reef.execution.sandbox.modal import ModalExecutor
from reef.execution.sandbox.process import LocalExecutor, SandboxExecutor, SandboxUnavailable
from reef.execution.sandbox.remote import deployment_owner
from reef.harness.rollouts.model_binding import ModelBinding
from reef.recipe.errors import RecipeConfigError
from reef.train.backend import ComponentReleases
from reef.train.harness_backend import AgentMethod
from reef.train.harness_backend.agent.settings import AgentSettings, agent_settings
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData

FAKE_AGENT = textwrap.dedent(
    """\
    #!{python}
    import json, os, pathlib, sys, urllib.error, urllib.request

    def call(route, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(os.environ["REEF_AGENT_URL"] + route, data=data,
                                         headers={{"content-type": "application/json"}})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
        try:
            with opener.open(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    mode = pathlib.Path(sys.argv[0]).with_name("mode").read_text()
    if mode == "fail":
        sys.exit(3)
    seen = {{
        "prompt": sys.argv[sys.argv.index("-p") + 1],
        "rules": pathlib.Path(os.environ["PI_CODING_AGENT_DIR"], "AGENTS.md").read_text(),
        "skill": pathlib.Path(os.environ["PI_CODING_AGENT_DIR"], "skills/plan/SKILL.md").is_file(),
        "tools": sorted(tool["name"] for tool in call("/tools")[1]["tools"]),
        "outside": call("/check", {{"tree": "../.."}}),
    }}
    pathlib.Path("harness/rules.md").write_text("marker rules\\n")
    seen["check"] = call("/check", {{"tree": "harness"}})
    seen["evaluate"] = call("/evaluate", {{"tasks": ["task one"]}})
    seen["model"] = call("/v1/chat/completions", {{"model": "any", "messages": [{{"role": "user", "content": "hi"}}]}})
    pathlib.Path("state/memory.md").write_text("tried the marker\\n")
    pathlib.Path("output/summary.md").write_text("Added the marker to the rules.\\n")
    pathlib.Path("output/notes.json").write_text(json.dumps({{"seen": seen, "agent": "dropped"}}))
    sessions = pathlib.Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
    sessions.mkdir(parents=True, exist_ok=True)
    event = {{"type": "message", "message": {{"role": "assistant", "content": [{{"type": "text", "text": "done"}}]}}}}
    (sessions / "s.jsonl").write_text(json.dumps(event) + "\\n")
    """
)


class Upstream:
    """A stand-in provider that answers every chat completion with ``ok``, keeping each request it got."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                upstream.requests.append({"path": self.path, "body": body})
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    served = Upstream()
    yield served
    served.server.shutdown()
    served.server.server_close()


def method_folder(tmp_path: Path, *, resume: bool = True) -> Path:
    folder = tmp_path / "method"
    (folder / "skills" / "plan").mkdir(parents=True)
    (folder / "START.md").write_text("Step {step}: improve the rules. The history is at {history}.\n")
    if resume:
        (folder / "RESUME.md").write_text("Step {step}: continue from state/.\n")
    (folder / "rules.md").write_text("Prefer small edits.\n")
    (folder / "skills" / "plan" / "SKILL.md").write_text("---\nname: plan\ndescription: Plan a change.\n---\nPlan.\n")
    (folder / "README.md").write_text("For people, not for the agent.\n")
    return folder


def agent_backend(tmp_path: Path, upstream: Upstream, mode: str = ""):
    """A pi backend whose method is ``AgentMethod`` over ``method_folder``, its agent the fake one on ``upstream``."""
    binary = tmp_path / "bin" / "agent"
    binary.parent.mkdir()
    binary.write_text(FAKE_AGENT.format(python=sys.executable))
    binary.chmod(0o755)
    binary.with_name("mode").write_text(mode)
    method = AgentMethod.from_config({"prompts": str(method_folder(tmp_path)), "model": "proposer"})
    recipe = harness_recipe(
        tmp_path,
        method,
        agent=AgentSettings("pi", LocalExecutor(), 60.0, str(binary)),
        models={"proposer": ModelBinding(base_url=upstream.base_url, model="proposer-model", api_key="k")},
    )
    releases = TreeReleases(recipe.seed)
    return recipe.build_backend("demo", ComponentReleases(releases, COMPONENT), None)


def test_the_agent_edits_the_workspace_through_its_tools_and_its_notes_become_the_steps(
    tmp_path: Path, upstream: Upstream
) -> None:
    backend = agent_backend(tmp_path, upstream)
    first = backend_step(backend, JobData("demo:1", ()), backend.initial_state())
    notes = first.metrics["method_notes"]
    seen = notes["seen"]
    assert first.artifact is not None and first.metrics["method_summary"] == "Added the marker to the rules."
    assert notes["agent"]["exit_code"] == 0 and "dropped" not in json.dumps(notes["agent"])
    # Reef's protocol rules come before the method's own, with the step and the history filled in.
    assert seen["prompt"] == "Step 1: improve the rules. The history is at ../../history.\n"
    assert "Reef runs you in step 1" in seen["rules"] and seen["rules"].endswith("Prefer small edits.\n")
    assert seen["skill"] and seen["tools"] == ["check", "evaluate"]
    assert seen["outside"][0] == 400 and "outside the step's workspace" in seen["outside"][1]["error"]
    assert seen["check"] == [200, {"passed": True, "problems": []}]
    (rollout,) = seen["evaluate"][1]["rollouts"]
    assert seen["evaluate"][0] == 200 and rollout["score"] == 1.0 and rollout["directory"].startswith("../../history/")
    # The model call went to the method's binding, metered and recorded with the step.
    assert seen["model"][0] == 200 and upstream.requests[0]["body"]["model"] == "proposer-model"
    step_dir = Path(first.metrics["step_record"])
    calls = json.loads((step_dir / "calls.json").read_text())
    assert [call["source"] for call in calls] == ["agent"] and calls[0]["model"] == "proposer-model"
    assert (step_dir / "output" / "agent-session.jsonl").is_file()
    assert HarnessState.from_json(first.state).method_files == {"memory.md": "tried the marker\n"}
    # The next step continues from state/, with the resume prompt.
    second = backend_step(backend, JobData("demo:2", ()), first.state)
    assert second.metrics["method_notes"]["seen"]["prompt"] == "Step 2: continue from state/.\n"


def test_an_agent_that_fails_leaves_the_head_tree_so_the_step_skips(tmp_path: Path, upstream: Upstream) -> None:
    backend = agent_backend(tmp_path, upstream, mode="fail")
    result = backend_step(backend, JobData("demo:1", ()), backend.initial_state())
    assert result.artifact is None and result.metrics["method_notes"]["agent"]["exit_code"] == 3
    assert result.metrics["method_summary"].startswith("the agent exited 3")


def test_a_method_folder_needs_a_start_prompt_and_only_tree_paths(tmp_path: Path) -> None:
    folder = method_folder(tmp_path)
    (folder / "START.md").unlink()
    with pytest.raises(PluginConfigError, match=r"holds no START\.md"):
        AgentMethod.from_config({"prompts": str(folder)})
    (folder / "START.md").write_text("go\n")
    (folder / "notes.txt").write_text("stray\n")
    with pytest.raises(PluginConfigError, match=r"notes\.txt"):
        AgentMethod.from_config({"prompts": str(folder)})


def test_a_method_that_runs_an_agent_needs_evolution_agent(tmp_path: Path) -> None:
    method = AgentMethod.from_config({"prompts": str(method_folder(tmp_path))})
    with pytest.raises(RecipeConfigError, match=r"AgentMethod runs a coding agent: set evolution\.agent"):
        harness_recipe(tmp_path, method)


def test_agent_settings_choose_the_adapter_and_the_isolation(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    assert agent_settings(None, {}, state_dir=state_dir) is None
    settings = agent_settings({"sandbox": "none", "timeout_s": 60}, {}, state_dir=state_dir)
    assert settings is not None and isinstance(settings.executor, LocalExecutor)
    assert (settings.adapter, settings.binary, settings.timeout_s) == ("pi", None, 60.0)
    claude = agent_settings({"sandbox": "none", "adapter": "claude", "binary": "/opt/claude"}, {}, state_dir=state_dir)
    assert claude is not None and (claude.adapter, claude.binary) == ("claude", "/opt/claude")
    unset = agent_settings({}, {"REEF_AGENT_SANDBOX": "none"}, state_dir=state_dir)
    assert unset is not None and isinstance(unset.executor, LocalExecutor)
    # The variable's old name would be ignored, and the agent run with another isolation: it is refused.
    with pytest.raises(ValueError, match="set REEF_AGENT_SANDBOX instead"):
        agent_settings({}, {"REEF_PROPOSER_SANDBOX": "none"}, state_dir=state_dir)
    with pytest.raises(ValueError, match="one of 'bwrap', 'e2b', 'modal', 'none'"):
        agent_settings({"sandbox": "docker"}, {}, state_dir=state_dir)
    with pytest.raises(ValueError, match="positive number"):
        agent_settings({"timeout_s": 0}, {}, state_dir=state_dir)
    # E2B: the key from the section, else the environment; none at all is refused at startup.
    with pytest.raises(ValueError, match=r"sandbox is e2b, but .*E2B_API_KEY"):
        agent_settings({"sandbox": "e2b"}, {}, state_dir=state_dir)
    monkeypatch.setattr(E2BExecutor, "preflight", lambda self: None)  # the e2b package is an extra
    remote = agent_settings({"sandbox": "e2b", "timeout_s": 900}, {"E2B_API_KEY": "e2b-env"}, state_dir=state_dir)
    assert remote is not None and isinstance(remote.executor, E2BExecutor)
    assert (remote.executor.api_key, remote.executor.timeout_s, remote.executor.template) == ("e2b-env", 900.0, "")
    assert "e2b-env" not in repr(remote)
    # The deployment's state directory names its sandboxes, so a later start can stop the ones a stopped run left.
    assert remote.executor.owner == deployment_owner(state_dir)
    configured = agent_settings(
        {"sandbox": "e2b", "e2b_api_key": "e2b-cfg", "e2b_template": "mine"},
        {"E2B_API_KEY": "e2b-env"},
        state_dir=state_dir,
    )
    assert configured is not None and isinstance(configured.executor, E2BExecutor)
    assert (configured.executor.api_key, configured.executor.template) == ("e2b-cfg", "mine")
    # Modal: the provider's options from the section, the deployment that names its sandboxes, and the timeout.
    monkeypatch.setattr(ModalExecutor, "preflight", lambda self: None)
    modal = agent_settings({"sandbox": "modal", "modal_app": "reef-agents", "timeout_s": 90}, {}, state_dir=state_dir)
    assert modal is not None and isinstance(modal.executor, ModalExecutor)
    assert (modal.executor.app_name, modal.executor.timeout_s) == ("reef-agents", 90.0)
    assert modal.executor.owner == deployment_owner(state_dir)
    monkeypatch.undo()

    def unavailable(self) -> None:
        raise SandboxUnavailable("no bwrap here")

    monkeypatch.setattr(SandboxExecutor, "preflight", unavailable)
    # Unset on a host that cannot isolate: the agent is off and the served model answers in text.
    assert agent_settings({}, {}, state_dir=state_dir) is None
    with pytest.raises(ValueError, match="no bwrap here"):
        agent_settings({"sandbox": "bwrap"}, {}, state_dir=state_dir)
