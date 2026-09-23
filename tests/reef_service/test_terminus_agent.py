"""The terminus adapter against the real Harbor package.

Skipped unless the ``terminus`` extra is installed. These are the contract
checks no hermetic mapping can make: that Harbor accepts the agent spec and
the trial overrides the runner builds, that the constructor arguments the
render quirk admits are real Terminus 2 parameters, that Terminus 2 sends the
bound model name to the bound endpoint and reads that model's context limit,
and that Harbor loads the reefine health task. They need Harbor, but not
Docker and not a model (a local endpoint stands in), so they run in any job
that installs the extra.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.adapters.terminus.quirks import _ALLOWED_KNOBS, _BINDING_KNOBS
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.runners.terminus import instruction_paths, skill_roots
from reef.harness.runners.terminus.runner import AGENT_NAME, agent_spec
from reef.harness.tree.render import render_composition
from reef.recipe.reefine.evolution import HEALTH_TASK_DIRECTORY

pytest.importorskip("harbor", reason="harbor is not installed")

NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize."}),
    ("config", {"data": {"model_name": "openai/stub", "max_turns": 12}}),
]


def bound_config(base_url: str, served: str, knobs: dict[str, str] | None = None) -> dict[str, Any]:
    """The terminus config a tree with ``knobs`` renders under the model binding for ``served`` at ``base_url``."""
    descriptor = get_adapter("terminus")
    binding = ModelBinding(base_url=base_url, model=served, api_key="k-1")
    nodes = [("config", {"data": knobs or {}}), *binding.compose_nodes(descriptor)]
    return json.loads(render_composition(nodes, descriptor)["terminus/config.json"])


@pytest.fixture
def endpoint() -> Iterator[tuple[str, list[tuple[str, dict[str, Any], str]]]]:
    """A local Chat Completions endpoint that records the path, body and Authorization header of every request."""
    seen: list[tuple[str, dict[str, Any], str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, body, self.headers.get("authorization", "")))
            message = {"role": "assistant", "content": "pong"}
            answer = json.dumps(
                {
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()


def _rendered(root: Path) -> dict[str, str]:
    files = render_composition(NODES, get_adapter("terminus"))
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return files


@pytest.mark.unit
def test_harbor_validates_the_trial_the_runner_builds(tmp_path: Path) -> None:
    from harbor.models.trial.config import TaskConfig, TrialConfig

    files = _rendered(tmp_path / "root")
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"path": Path("recipes/basic/harbor")}).model_dump(),
            "trials_dir": tmp_path / "trials",
            "agent": agent_spec(str(tmp_path / "root"), files),
            "extra_instruction_paths": instruction_paths(tmp_path / "root", files),
        }
    )
    # Harbor's own agent, configured rather than subclassed.
    assert config.agent.name == AGENT_NAME
    assert config.agent.model_name == "openai/stub"
    assert config.agent.kwargs == {"max_turns": 12}
    assert config.extra_instruction_paths == [tmp_path / "root" / "terminus/AGENTS.md"]


@pytest.mark.unit
def test_harbor_resolves_both_skill_roots_the_tree_renders(tmp_path: Path) -> None:
    from harbor.skills import resolve_skills

    files = _rendered(tmp_path / "root")
    roots = skill_roots(tmp_path / "root", files)
    resolved = resolve_skills([str(path) for path in roots])
    # One skill and one command, each discovered as its own skill directory,
    # so Harbor keeps progressive loading instead of pasting bodies inline.
    assert len(resolved) == 2


@pytest.mark.unit
def test_every_admitted_knob_is_a_real_terminus_2_argument() -> None:
    import inspect

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    parameters = set(inspect.signature(Terminus2.__init__).parameters)
    unknown = sorted((_ALLOWED_KNOBS | _BINDING_KNOBS) - parameters)
    assert unknown == [], f"the quirk admits arguments Terminus 2 does not take: {unknown}"


@pytest.mark.unit
@pytest.mark.parametrize("served", ["qwen/qwen3-coder", "deepseek/deepseek-chat", "openai/gpt-4o-mini", "gemma4:26b"])
def test_terminus_2_sends_the_served_model_to_the_bound_endpoint(tmp_path: Path, endpoint, served: str) -> None:
    """Terminus 2 built from the bound config posts the served name unchanged to api_base. Without the binding's
    custom_openai, a vendor prefix litellm does not know has no route, and one it knows picks that vendor's client
    and drops the prefix. The tree's reasoning_effort, which litellm drops for custom_openai unless allowed, and the
    bound key arrive too."""
    from harbor.agents.terminus_2.terminus_2 import Terminus2

    base_url, seen = endpoint
    agent = Terminus2(logs_dir=tmp_path, **bound_config(base_url, served, {"reasoning_effort": "low"}))
    asyncio.run(agent._llm.call("ping"))
    [(path, body, authorization)] = seen
    assert (path, body["model"], body["reasoning_effort"]) == ("/v1/chat/completions", served, "low")
    assert authorization == "Bearer k-1"
    assert not {"custom_llm_provider", "allowed_openai_params"} & set(body)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("served", "listed"), [("openai/gpt-4o-mini", True), ("deepseek/deepseek-chat", True), ("qwen/qwen3-coder", False)]
)
def test_terminus_2_reads_the_context_limit_under_the_served_name(tmp_path: Path, served: str, listed: bool) -> None:
    """Harbor looks the context limit up under model_name, so the binding keeps the served name there. A provider
    prefix would miss litellm's model table and fall back to Harbor's 1,000,000 tokens, and proactive summarization
    would never start. litellm lists qwen3-coder only as openrouter/qwen/qwen3-coder, so that name gets the
    fallback with or without a binding."""
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.lite_llm import LiteLLM
    from litellm import get_model_info

    agent = Terminus2(logs_dir=tmp_path, **bound_config("http://127.0.0.1:9", served))
    limit = agent._llm.get_model_context_limit()
    assert limit == LiteLLM(model_name=served).get_model_context_limit()
    if listed:
        assert limit == get_model_info(served)["max_input_tokens"] < 1_000_000
    else:
        with pytest.raises(Exception, match="isn't mapped yet"):
            get_model_info(served)


@pytest.mark.unit
def test_harbor_loads_the_reefine_health_task_directory() -> None:
    from harbor.models.task.task import Task

    assert Task.is_valid_dir(HEALTH_TASK_DIRECTORY)
    task = Task(HEALTH_TASK_DIRECTORY)
    assert "echo reef-ok" in task.instruction
    assert task.paths.test_path.is_file() and (task.paths.environment_dir / "Dockerfile").is_file()


@pytest.mark.unit
def test_extension_must_remain_a_terminus_agent(monkeypatch) -> None:
    from reef.harness.episodes.executor import ISOLATION_ENV
    from reef.harness.runners.terminus.tree import ENVIRONMENT_ENV, TerminusTreeError

    monkeypatch.setenv(ISOLATION_ENV, "bwrap")
    monkeypatch.setenv(ENVIRONMENT_ENV, "e2b")
    files = render_composition(
        [*NODES, ("code_extension", {"name": "agent", "code": "class Agent: pass\n"})], get_adapter("terminus")
    )
    with pytest.raises(TerminusTreeError, match="must inherit Harbor's Terminus2"):
        agent_spec("/root", files)
