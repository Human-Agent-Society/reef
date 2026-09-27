"""CPU checks for the reef-eval/Harbor feedback-group example's handoff."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from recipes.sdpo.examples.feedback_group.harness import agent
from recipes.sdpo.preparer import SDPOAttempt

EXAMPLE = Path(__file__).resolve().parents[1] / "recipes" / "sdpo" / "examples" / "feedback_group"


@pytest.mark.unit
def test_example_samples_one_version_and_saves_the_full_graded_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts_path = tmp_path / "attempts.json"
    for name, value in {
        "REEF_SERVICE_URL": "http://reef.test",
        "REEF_SCENARIO": "sdpo-test",
        "REEF_TOKEN": "local",
        "SDPO_ATTEMPTS_PATH": str(attempts_path),
    }.items():
        monkeypatch.setenv(name, value)

    class FakeClient:
        count = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def inference_with_record(self, scenario: str, path: str, payload: dict) -> tuple[dict, str]:
            assert scenario == "sdpo-test" and path == "/v1/chat/completions"
            assert payload["messages"] == [{"role": "user", "content": "Question?"}]
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            self.count += 1
            return (
                {
                    "choices": [
                        {
                            "message": {"content": f"<answer>{self.count}</answer>"},
                            "meta_info": {"runtime_load_id": "v1"},
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 5},
                },
                f"receipt-{self.count}",
            )

    class FakeEnvironment:
        async def exec(self, command: str) -> SimpleNamespace:
            assert shlex.split(command)[:2] == ["python3", "/opt/grade.py"]
            return SimpleNamespace(return_code=0, stdout=json.dumps({"score": 0, "feedback": "too low"}), stderr="")

    monkeypatch.setattr(agent, "ReefClient", FakeClient)
    context = SimpleNamespace(metadata={}, n_input_tokens=0, n_output_tokens=0)
    asyncio.run(agent.HarborAgent.run(object(), "Question?", FakeEnvironment(), context))

    saved = json.loads(attempts_path.read_text(encoding="utf-8"))
    assert len(saved) == 8
    assert {entry["artifact_version"] for entry in saved} == {"v1"}
    assert [entry["inference_id"] for entry in saved] == [f"receipt-{index}" for index in range(1, 9)]
    assert context.metadata["reef"]["agent_record_ids"] == [entry["inference_id"] for entry in saved]
    assert (context.n_input_tokens, context.n_output_tokens) == (32, 40)


@pytest.mark.unit
def test_example_config_uses_student_topk_tail_and_reference_sequence_mean() -> None:
    config = yaml.safe_load((EXAMPLE / "serve.yaml").read_text(encoding="utf-8"))
    options = config["training"]["options"]
    assert config["recipe"]["implementation"] == "recipes.sdpo.recipe:SDPORecipe"
    assert config["recipe"]["config"]["batch-size"] == config["training"]["config"]["global_batch_size"] == 8
    assert options["sdpo-top-k-tail"] is True
    assert not options.get("calculate-per-token-loss", False)
    assert config["recipe"]["config"]["enable-thinking"] is False
    assert options["sdpo-importance-sampling-mode"] == "token"
    assert (EXAMPLE / "harbor" / "arithmetic" / "tests" / "test.sh").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("post_version", ["v1", "v2"])
def test_smoke_checks_the_served_version_and_saves_run_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, post_version: str
) -> None:
    monkeypatch.setitem(sys.modules, "reef_eval", SimpleNamespace(Lab=lambda path: object()))
    spec = importlib.util.spec_from_file_location("sdpo_smoke_runner", EXAMPLE / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name, value in {
        "SDPO_WORK": str(tmp_path),
        "REEF_SERVICE_URL": "http://reef.test",
        "REEF_TOKEN": "local",
        "REEF_SCENARIO": "sdpo-test",
    }.items():
        monkeypatch.setenv(name, value)
    reports = []
    releases = []
    client = SimpleNamespace(report=lambda scenario, payload: reports.append(payload))
    monkeypatch.setattr(runner, "ReefClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "training_releases", lambda: 0)
    monkeypatch.setattr(runner, "wait_for_release", lambda target, **kwargs: releases.append(target))

    async def trial(lab, key, path):
        version = "v1" if key.endswith("before") else post_version
        return SimpleNamespace(rewards={"reward": 0}), [
            SDPOAttempt("q1", f"{key}-{index}", version, "wrong", 0, "test failed") for index in range(8)
        ]

    monkeypatch.setattr(runner, "trial", trial)
    summary_path = tmp_path / "check-summary.json"
    if post_version == "v1":
        with pytest.raises(RuntimeError, match="pre-training artifact version"):
            asyncio.run(runner.run("check"))
        assert not summary_path.exists()
    else:
        asyncio.run(runner.run("check"))
        summary = json.loads(summary_path.read_text())
        assert (summary["pre_artifact_version"], summary["post_artifact_version"]) == ("v1", "v2")
        assert summary["active_responses"] == 8
    assert len(reports) == 8
    assert releases == [1]
