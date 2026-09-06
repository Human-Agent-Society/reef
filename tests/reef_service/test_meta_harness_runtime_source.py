from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench import runtime_source
from recipes.meta_harness.examples.terminal_bench.history import HistoryBinding


def test_frozen_source_cannot_read_host_paths_or_drift(monkeypatch):
    bundle = {"harbor/agent.py": "class Agent:\n    pass\n"}
    with pytest.raises(ValueError, match="unknown"):
        runtime_source.read_source(bundle, "../../.env")
    assert runtime_source.read_source(bundle, "harbor/agent.py", 1, 1)["lines"] == ["    pass"]
    assert runtime_source.search_source(bundle, "Agent")["matches"][0]["line_offset"] == 0
    manifest = runtime_source.source_manifest(bundle)
    monkeypatch.setattr(runtime_source, "source_bundle", lambda: bundle)
    assert runtime_source.verified_source(manifest) == bundle
    bundle["harbor/agent.py"] += "changed\n"
    with pytest.raises(ValueError, match="committed"):
        runtime_source.verified_source(manifest)


def test_proposer_can_inspect_source_through_audited_tool_turns():
    bodies = []

    def complete(body):
        bodies.append(body)
        if len(bodies) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "source",
                                    "function": {
                                        "name": "read_runtime_source",
                                        "arguments": '{"path":"harbor/agent.py"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "composition"}}]}

    history = HistoryBinding(
        SimpleNamespace(api="openai", complete=complete),
        {},
        executable=True,
        sources={"harbor/agent.py": "class Agent: pass"},
    )
    assert history.chat([]) == "composition"
    assert "class Agent: pass" in bodies[1]["messages"][-1]["content"]
    assert len(history.audit) == 2
