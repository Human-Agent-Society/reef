"""Opt-in cloud acceptance: REEF_TEST_E2B=1, E2B_API_KEY and OPENROUTER_API_KEY.

Only a short synthetic prompt is sent to OpenRouter. The provider key stays in
the test process; the sandbox reaches a temporary model endpoint over Reef's tunnel.
"""

import os
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reef.evaluation.harness import EpisodeEvaluationWorker, EpisodeScorer
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import run_episode
from reef.harness.episodes.vendor_install import AdapterProgram
from reef.harness.tree import HarnessTree, render_tree
from reef.sandbox.config import build_executor
from reef.sandbox.e2b import E2BExecutor
from reef.sandbox.process import EpisodeTimeout

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_E2B") != "1", reason="opt-in paid E2B/OpenRouter test")


def bound_files(binding: ModelBinding, adapter: str) -> dict[str, str]:
    """The empty tree rendered for ``adapter`` with ``binding`` as its model: an episode with the harness's own
    defaults."""
    descriptor = get_adapter(adapter)
    return render_tree(HarnessTree({}), descriptor, binding.config_overlays(descriptor))


class MarkerScorer(EpisodeScorer):
    def score(self, task, result):
        for event in result.trajectory:
            message = event.get("message", event)
            if message.get("role") == "assistant" and "REEF_E2B_OK" in str(message.get("content", "")):
                return 1.0
        return 0.0


@pytest.fixture(scope="module")
def model_endpoint():
    key = os.environ["OPENROUTER_API_KEY"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if (
                self.path not in ("/v1/chat/completions", "/v1/responses")
                or self.headers.get("Authorization") != "Bearer episode-test"
            ):
                self.send_error(403)
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = urllib.request.Request(
                f"https://openrouter.ai/api{self.path}",
                data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            try:
                response = urllib.request.urlopen(request, timeout=90)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("adapter,api", [("pi", "openai"), ("codex", "responses")])
def test_real_agent_episode_through_generic_executor(adapter, api, model_endpoint, tmp_path):
    descriptor = get_adapter(adapter)
    configured = build_executor({"executor": "e2b", "sandbox": {"forward_ports": [model_endpoint]}})
    executor, _ = configured.for_program(AdapterProgram(descriptor))
    binding = ModelBinding(
        base_url=f"http://127.0.0.1:{model_endpoint}",
        model=os.environ.get("REEF_E2B_TEST_MODEL", "openai/gpt-4.1-mini"),
        api_key="episode-test",
        api=api,
        max_output_tokens=256,
    )
    result = run_episode(
        descriptor,
        bound_files(binding, adapter),
        "Reply exactly REEF_E2B_OK. Do not use tools.",
        executor=executor,
        timeout=180,
        keep_dir=tmp_path / "trajectory",
    )
    assert result.exit_code == 0, result.stderr[-2000:]
    assert result.trajectory, result.stdout[-2000:]
    assert "REEF_E2B_OK" in result.stdout
    assert not result.residue
    assert any((tmp_path / "trajectory").rglob("*.jsonl"))


def test_remote_inputs_are_readonly_and_timeout_retires_children(tmp_path):
    executor, _ = build_executor({"executor": "e2b"}).for_program(AdapterProgram(get_adapter("pi")))
    assert isinstance(executor, E2BExecutor)
    session = replace(executor, binary="python3", timeout_s=90).open()
    root = tmp_path / "episode"
    workspace = root / "workspace"
    state = root / "state"
    workspace.mkdir(parents=True)
    state.mkdir()
    config = state / "input.json"
    config.write_text("original")
    script = root / "check.py"
    script.write_text(
        "import os, pathlib, subprocess\n"
        "path = pathlib.Path('../state/input.json')\n"
        "for operation in (lambda: path.write_text('changed'), lambda: path.unlink(), lambda: path.chmod(0o777)):\n"
        "    try: operation()\n"
        "    except OSError: pass\n"
        "    else: raise RuntimeError('input was mutable')\n"
        "assert subprocess.run(['sudo', '-n', 'true'], capture_output=True).returncode != 0\n"
        "pathlib.Path('answer.txt').write_text('ok')\n"
        "pathlib.Path('../state/log.txt').write_text('state')\n"
        "print('protected')\n"
    )
    try:
        outcome = session.launch(
            ["python3", str(script)],
            root=root,
            workspace=workspace,
            env={},
            timeout=20,
            writable_paths=(state,),
            readonly_paths=(config, script),
        )
        assert outcome.exit_code == 0, outcome.stderr
        assert config.read_text() == "original" and (workspace / "answer.txt").read_text() == "ok"
        with pytest.raises(EpisodeTimeout):
            session.launch(
                ["python3", "-c", "import subprocess,time; subprocess.Popen(['sleep','120']); time.sleep(120)"],
                root=root,
                workspace=workspace,
                env={},
                timeout=2,
            )
        processes = session.sandbox.commands.run("ps -eo args").stdout
        assert "sleep 120" not in processes
    finally:
        session.close()


def test_the_evaluation_worker_scores_a_real_remote_episode(model_endpoint, tmp_path):
    """The harness backend's evaluation episodes and a method's trial episodes run through this same worker."""
    descriptor = get_adapter("pi")
    configured = build_executor({"executor": "e2b", "sandbox": {"forward_ports": [model_endpoint]}})
    executor, _ = configured.for_program(AdapterProgram(descriptor))
    binding = ModelBinding(
        base_url=f"http://127.0.0.1:{model_endpoint}",
        model=os.environ.get("REEF_E2B_TEST_MODEL", "openai/gpt-4.1-mini"),
        api_key="episode-test",
        max_output_tokens=256,
    )
    scorer = MarkerScorer()
    task = "Reply exactly REEF_E2B_OK. Do not use tools."
    worker = EpisodeEvaluationWorker(descriptor, scorer, None, 180, executor, True)
    scored = worker.run(bound_files(binding, "pi"), task, models=ModelBindings(binding))
    assert scored.score == 1.0 and scored.failure is None
    assert scored.reply is not None and "REEF_E2B_OK" in scored.reply
