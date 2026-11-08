"""Run the generated pi and native readers against a local embedding endpoint."""

import json
import runpy
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from recipes.reasoning_bank.reading import ReadNote, ReadSettings, runtime_files
from recipes.reasoning_bank.runtime.native import embedding_vectors
from reef.harness.adapters import get_adapter
from reef.harness.tree import HarnessTree, validate_tree

NOTES = (
    ReadNote("fruit", "Banana", "Use for cooking", "Peel the banana."),
    ReadNote("code", "Parser", "Use for parsing", "Check the parser input."),
)

PI_DRIVER = """\
import install from './reader.mjs';
const hooks = {}, tools = {}, events = [];
const pi = {
  on(name, handler) { hooks[name] = handler; },
  registerTool(tool) { tools[tool.name] = tool; },
  appendEntry(name, data) { events.push({name, data}); },
};
install(pi);
let result;
if (tools.memory_search) {
  result = await tools.memory_search.execute('call', {query: 'parser task'});
} else {
  result = await hooks.before_agent_start({prompt: 'parser task', systemPrompt: 'Keep the original rules.'}, {hasUI: false});
}
process.stdout.write(JSON.stringify({result, events}));
"""


@pytest.fixture
def embeddings():
    """A deterministic OpenAI-compatible endpoint that reverses response rows to test index handling."""
    requests: list[dict[str, object]] = []

    class EmbeddingHandler(BaseHTTPRequestHandler):
        """One test endpoint: POST embeds parser text along x and other text along y; logging is disabled."""

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"body": body, "authorization": self.headers.get("Authorization")})
            rows = [
                {"index": index, "embedding": [1, 0] if "parser" in text.lower() else [0, 1]}
                for index, text in enumerate(body["input"])
            ]
            encoded = json.dumps({"data": list(reversed(rows))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/embeddings", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("adapter", ["pi", "native"])
@pytest.mark.parametrize("inject", ["system", "message", "tool"])
@pytest.mark.parametrize("reader", ["all", "top_k"])
def test_runtime_selects_injects_and_records_the_same_notes(
    tmp_path, monkeypatch, embeddings, adapter, inject, reader
):
    endpoint, requests = embeddings
    monkeypatch.setenv("REEF_MEMORY_EMBEDDING_API_KEY", "test-bearer-key")
    settings = ReadSettings(
        reader=reader,
        inject=inject,
        embedding_url=endpoint if reader == "top_k" else "",
        embedding_model="test-embedding" if reader == "top_k" else "",
    )
    files = runtime_files(NOTES, settings, adapter)
    assert validate_tree(HarnessTree(files), get_adapter(adapter)).passed
    source = next(iter(files.values()))
    assert "test-bearer-key" not in source
    if adapter == "pi":
        node = shutil.which("node")
        if node is None:
            pytest.skip("pi runtime checks need Node.js")
        (tmp_path / "reader.mjs").write_text(source)
        (tmp_path / "driver.mjs").write_text(PI_DRIVER)
        completed = subprocess.run([node, str(tmp_path / "driver.mjs")], capture_output=True, text=True, check=True)
        output = json.loads(completed.stdout)
        result, record = output["result"], output["events"][0]["data"]
        if inject == "system":
            text = result["systemPrompt"]
            assert text.startswith("Keep the original rules.")
        elif inject == "message":
            text = result["message"]["content"]
        else:
            text = result["content"][0]["text"]
    else:
        script = tmp_path / "reader.py"
        script.write_text(source)
        module = runpy.run_path(str(script))
        if inject == "tool":
            result = module["run"]({"query": "parser task"}, str(tmp_path))
            text = result["text"]
        else:
            result = module["listen"]({"step": 1, "task": "parser task"}, lambda: {"kind": "enter", "messages": []})
            if inject == "system":
                text = result["system_suffix"]
            else:
                text = result["messages"][0]
                repeated = module["listen"](
                    {"step": 2, "task": "parser task"}, lambda: {"kind": "enter", "messages": []}
                )
                assert repeated == {"kind": "enter", "messages": []}
        record = result["memory_read"]
    expected = ["code"] if reader == "top_k" else ["fruit", "code"]
    assert record == {"event": "memory/read", "entries": expected, "reader": reader, "inject": inject}
    assert "untrusted data" in text and "not instructions" in text
    assert [note["id"] for note in json.loads(text.rsplit("\n", 1)[1])] == expected
    if reader == "top_k":
        assert len(requests) == 1
        assert requests[0]["authorization"] == "Bearer test-bearer-key"
        assert requests[0]["body"]["model"] == "test-embedding"
    else:
        assert requests == []


@pytest.mark.parametrize(
    "response",
    [
        {"data": []},
        {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}]},
        {"data": [{"index": 0, "embedding": [0]}, {"index": 1, "embedding": [1]}]},
        {"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [1, 2]}]},
        {"data": [{"index": 0, "embedding": [float("nan")]}, {"index": 1, "embedding": [1]}]},
    ],
)
def test_bad_embedding_responses_are_refused(response):
    with pytest.raises(ValueError, match="embedding"):
        embedding_vectors(response, 2)


def test_runtime_settings_refuse_invalid_configuration():
    for values in ({"reader": "unknown"}, {"reader": "top_k"}, {"top_k": 0}, {"timeout_seconds": 0}):
        with pytest.raises(ValueError):
            ReadSettings(**values)
    with pytest.raises(ValueError, match="credentials"):
        ReadSettings(
            reader="top_k", embedding_url="https://user:secret@example.com/v1/embeddings", embedding_model="x"
        )
    with pytest.raises(ValueError, match="pi and native"):
        runtime_files(NOTES, ReadSettings(reader="all"), "codex")
