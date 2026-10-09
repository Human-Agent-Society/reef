"""Real HTTP receipt read-back with synthetic model tensors and no GPU or provider calls."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer
from reef_client import ReefClient

from recipes.opd.examples.agentcl.harness import episode as opd_episode
from recipes.sdft.examples.agentcl.harness import episode as sdft_episode
from recipes.sdpo.examples.agentcl.harness import episode as sdpo_episode
from reef.artifact import Artifact
from reef.dispatcher import build_default_dispatcher
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage


class SyntheticAgentHandler(InferenceHandler):
    """Emit exact synthetic append-only token histories through the real service."""

    async def inference(self, artifact: Artifact, path: str, payload: dict) -> dict:
        assert path == "/v1/chat/completions"
        turn = sum(message["role"] == "assistant" for message in payload["messages"])
        outputs = (
            "```python\nvalue = 2\n```",
            "```python\nprint(value + 1)\n```",
            "FINAL\n```python\ndef solution():\n    return 3\n```",
        )
        prompt = [10, 11]
        for index in range(turn):
            prompt.extend([20 + index, 30 + index])
        return {
            "choices": [{"message": {"role": "assistant", "content": outputs[turn]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1},
            "training": {
                "tokens": [*prompt, 20 + turn],
                "loss_mask": [1],
                "rollout_log_probs": [-0.2],
                "runtime_load_id": "synthetic-runtime-1",
                "request_messages": copy.deepcopy(payload["messages"]),
                "request_tools": None,
            },
        }


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_agentcl_reads_private_tensors_from_real_authenticated_record_route(method: str, tmp_path: Path) -> None:
    episode = {"sdft": sdft_episode, "sdpo": sdpo_episode, "opd": opd_episode}[method]

    class SyntheticSandbox(sdft_episode.EpisodeSandbox, sdpo_episode.EpisodeSandbox, opd_episode.EpisodeSandbox):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.answer: str | None = None
            self.closed = False

        async def start(self) -> None:
            return None

        async def execute(self, code: str, timeout_seconds: int, output_limit: int) -> dict:
            self.calls.append(code)
            return {"status": "ok", "output": "3"}

        async def submit(self, code: str) -> None:
            self.answer = code

        async def close(self) -> None:
            self.closed = True

    async def exercise() -> None:
        dispatcher = build_default_dispatcher(
            local_artifact_dir=tmp_path / "artifacts", scenario_storage=SQLiteScenarioStorage()
        )
        scenario = dispatcher.get_or_create_scenario("agentcl-record-test")
        assert scenario is not None
        release = scenario.repository.require_current_artifact().release_id
        server = TestServer(
            create_app(dispatcher, tokens="synthetic-test-token", inference_handler=SyntheticAgentHandler())
        )
        await server.start_server()
        try:
            client = ReefClient(str(server.make_url("")).rstrip("/"), token="synthetic-test-token", timeout_s=5)
            sandbox = SyntheticSandbox()
            settings = episode.EpisodeSettings(
                episode_id="synthetic-service-episode",
                model_name="reef",
                expected_release=release,
                expected_runtime_load_id="synthetic-runtime-1",
                phase="train",
                max_turns=3,
            )
            runner = episode.EpisodeRunner(
                episode.ReefRecordedModel(client, "agentcl-record-test"), sandbox, settings, tmp_path / "logs"
            )
            result = await runner.run("SYNTHETIC FIXTURE: implement solution().")
            assert result["outcome"] == "completed"
            assert len(result["references"]) == 3 and len(set(result["references"])) == 3
            assert len(sandbox.calls) == 2 and sandbox.answer is not None and sandbox.closed
            for turn in result["turns"]:
                assert "training" not in turn["response"]
                assert turn["record"]["payload"]["response"]["training"]["loss_mask"] == [1]
                assert turn["record"]["artifact_ref"]["release_id"] == release
                assert turn["runtime_load_id"] == "synthetic-runtime-1"
            records = scenario.records.replay("agentcl-record-test")
            assert len(records) == 3 and all(record.request_type.value == "inference" for record in records)
        finally:
            await server.close()
            dispatcher.close()

    asyncio.run(exercise())
