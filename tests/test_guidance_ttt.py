from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.guidance_ttt import (
    ExecutionBackend,
    OpenAICompatibleExecutionClient,
    ReefGuidanceTTTHarness,
    TaskSpec,
    gpt_oss_120b_backend,
    openrouter_glm_5_2_backend,
    prepare_library,
)
from examples.guidance_ttt.deployments.qwen3_8b_lora.cluster import build_parser
from examples.guidance_ttt.deployments.qwen3_8b_lora.runner import _validate_settings
from examples.guidance_ttt.library import GuidanceLibrary
from examples.guidance_ttt.prompts import build_execution_prompt, build_guidance_prompt, extract_strict_guidance
from examples.guidance_ttt.puct import archive_puct_score
from examples.guidance_ttt.state import LibraryEntry, LLMResponse, VerificationResult, make_root_node
from examples.guidance_ttt.tasks.polyomino import POLYOMINO_PROBLEM_PROMPT
from examples.guidance_ttt.verifier.frontiercs_adapter import FrontierCSResult, evaluate_cpp_solution
from examples.guidance_ttt.verifier.polyomino import extract_cpp_solution_code

SEED = REPO_ROOT / "examples" / "guidance_ttt" / "seeds" / "polyomino_packing" / "gpt_oss_120b_bootstrap_library.json"


def _task() -> TaskSpec:
    def verify(text, *, timeout_s, config):
        _ = timeout_s, config
        code = extract_cpp_solution_code(text)
        if not code:
            return VerificationResult(0.0, None, False, "parse_error", "missing code")
        score = float(len(code))
        return VerificationResult(score, score, True, "valid", "accepted", {"code": code})

    return TaskSpec(
        task_id="polyomino_packing",
        problem_prompt=POLYOMINO_PROBLEM_PROMPT,
        solution_language="cpp",
        execution_solution_contract="Return one complete C++17 solution.",
        guidance_mechanism_constraint="Use only self-contained C++17 mechanisms.",
        score_direction="max",
        raw_score_label="FrontierCS score",
        create_root_node=lambda: None,
        verifier=verify,
        solution_extractor=extract_cpp_solution_code,
        guidance_objective=lambda _: "Improve the FrontierCS score.",
    )


class _ReefClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inferences: list[dict] = []
        self.reports: list[dict] = []

    def inference_with_record(self, scenario, path, payload, *, recipe=None, artifact_version=None):
        assert (scenario, path, recipe, artifact_version) == (
            "guidance-smoke",
            "/v1/chat/completions",
            "tttd",
            None,
        )
        with self._lock:
            index = len(self.inferences)
            self.inferences.append(payload)
        content = "malformed guidance" if index == 0 else f"<guidance>improve strategy {index}</guidance>"
        return {"choices": [{"message": {"content": content}}]}, f"receipt-{index}"

    def report(self, scenario, payload, *, recipe=None, artifact_version=None):
        assert (scenario, recipe, artifact_version) == ("guidance-smoke", "tttd", None)
        with self._lock:
            self.reports.append(payload)
        return {}


class _ExecutionClient:
    backend = ExecutionBackend(
        name="test",
        model="test-executor",
        base_url="http://executor.invalid/v1",
        concurrency=4,
        max_retries=0,
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = []

    def complete(self, request):
        with self._lock:
            self.requests.append(request)
            index = len(self.requests)
        text = f"""<solution>
```cpp
#include <iostream>
int main() {{ return {index}; }}
```
</solution>
<summary>Candidate {index} applies the requested strategy while preserving the parent contract.</summary>"""
        return LLMResponse(text, request.model, "stop", reasoning="tested the guidance")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<guidance>one direction</guidance>", "one direction"),
        ("thinking first\n<guidance>one direction</guidance>", "one direction"),
        ("<guidance></guidance>", None),
        ("<guidance>one", None),
        ("<guidance>one</guidance> trailing", None),
        ("<guidance>one</guidance><guidance>two</guidance>", None),
    ],
)
def test_strict_guidance_parser_never_repairs_or_retries(text: str, expected: str | None) -> None:
    guidance, error = extract_strict_guidance(text)

    assert guidance == expected
    assert (error is None) is (expected is not None)


def test_summary_only_policy_prompt_hides_code_but_execution_prompt_contains_it(tmp_path: Path) -> None:
    library = prepare_library(
        seed_path=SEED,
        run_path=tmp_path / "library.json",
        groups_per_step=1,
        rollouts_per_group=2,
    )
    node = library.acquire_group("1:0", visible_timestep_exclusive=1, require_solution=True)
    entry = library.context_for_node(node, visible_timestep_exclusive=1)["selected_entry"]
    assert entry is not None

    guidance_prompt = build_guidance_prompt(
        problem_prompt="problem",
        selected_node=node,
        selected_entry=entry,
        objective_text="improve",
        mechanism_constraint="stay executable",
        raw_score_label="score",
    )
    execution_prompt = build_execution_prompt(
        problem_prompt="problem",
        selected_entry=entry,
        guidance="improve packing",
        solution_language="cpp",
        solution_contract="return C++",
        score_direction="max",
        raw_score_label="score",
    )

    assert entry.solution not in guidance_prompt.user
    assert "<selected_summary>" in guidance_prompt.user
    assert entry.solution in execution_prompt.user
    assert "<parent_code>" in execution_prompt.user


def test_one_step_links_only_guidance_receipts_and_skips_executor_on_bad_format(tmp_path: Path) -> None:
    library = prepare_library(
        seed_path=SEED,
        run_path=tmp_path / "library.json",
        groups_per_step=2,
        rollouts_per_group=2,
    )
    reef = _ReefClient()
    executor = _ExecutionClient()
    harness = ReefGuidanceTTTHarness(
        reef,
        executor,
        library,
        scenario="guidance-smoke",
        model="Qwen/Qwen3-8B",
        task=_task(),
        groups_per_step=2,
        rollouts_per_group=2,
        guidance_max_tokens=64,
        max_workers=4,
    )

    results = harness.run_step(0)

    assert len(results) == 4
    assert len(reef.inferences) == 4
    assert len(reef.reports) == 4
    assert len(executor.requests) == 3
    assert sum(result.guidance_format_ok for result in results) == 3
    assert {tuple(report["references"]) for report in reef.reports} == {
        ("receipt-0",),
        ("receipt-1",),
        ("receipt-2",),
        ("receipt-3",),
    }
    assert {report["metadata"]["comparison_set"] for report in reef.reports} == {
        "tttd-step-0-group-0",
        "tttd-step-0-group-1",
    }
    assert all(report["metadata"]["algorithm"] == "ttt-discover" for report in reef.reports)
    assert all(report["metadata"]["prompt_mode"] == "summary_only" for report in reef.reports)
    assert all("#include <iostream>" not in request["messages"][1]["content"] for request in reef.inferences)
    assert all("#include" in request.user for request in executor.requests)

    snapshot = library.snapshot()
    generated = [entry for entry in snapshot["entries"].values() if entry["timestep"] == 1]
    assert len(generated) == 4
    assert snapshot["puct_T"] == 4
    assert all(entry["metadata"]["guidance_generation_attempts"] == 1 for entry in generated)
    assert all(entry["metadata"]["prompt_mode"] == "summary_only" for entry in generated)


def test_frozen_policy_control_advances_archive_without_training_reports(tmp_path: Path) -> None:
    library = prepare_library(
        seed_path=SEED,
        run_path=tmp_path / "library.json",
        groups_per_step=1,
        rollouts_per_group=2,
    )
    reef = _ReefClient()
    harness = ReefGuidanceTTTHarness(
        reef,
        _ExecutionClient(),
        library,
        scenario="guidance-smoke",
        model="Qwen/Qwen3-8B",
        task=_task(),
        groups_per_step=1,
        rollouts_per_group=2,
        guidance_max_tokens=64,
        max_workers=2,
        sampling_seed=40,
        report_training=False,
    )

    results = harness.run_step(0)

    assert len(results) == 2
    assert len(reef.inferences) == 2
    assert {payload["seed"] for payload in reef.inferences} == {40, 41}
    assert reef.reports == []
    generated = [
        entry
        for entry in library.snapshot()["entries"].values()
        if entry["timestep"] == 1
    ]
    assert len(generated) == 2
    assert all(entry["metadata"]["training_reported"] is False for entry in generated)


def test_openrouter_secret_is_environment_only_and_not_serialized(monkeypatch) -> None:
    sentinel = "secret-that-must-not-be-serialized"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    backend = openrouter_glm_5_2_backend(concurrency=2)
    client = OpenAICompatibleExecutionClient(backend)

    assert client.backend.model == "z-ai/glm-5.2"
    assert sentinel not in json.dumps(backend.safe_dict())
    assert backend.safe_dict()["api_key_env"] == "OPENROUTER_API_KEY"
    assert backend.safe_dict()["max_retries"] == 6
    assert backend.safe_dict()["request_options"] == {"reasoning": {"effort": "high", "exclude": False}}


def test_gpt_oss_executor_matches_high_reasoning_reference_setting() -> None:
    backend = gpt_oss_120b_backend()

    assert backend.temperature == 0.0
    assert backend.max_tokens is None
    assert backend.max_retries == 0
    assert backend.request_options == {"reasoning_effort": "high"}


def test_gpt_oss_executor_reasoning_effort_is_configurable() -> None:
    assert gpt_oss_120b_backend(reasoning_effort="medium").request_options == {"reasoning_effort": "medium"}
    with pytest.raises(ValueError, match="reasoning_effort"):
        gpt_oss_120b_backend(reasoning_effort="invalid")


def _entry(parent_id: str, suffix: str, *, status: str = "valid", raw_score: float | None = 10.0) -> LibraryEntry:
    return LibraryEntry(
        id=f"entry-{suffix}",
        parent_id=parent_id,
        problem_id="polyomino_packing",
        timestep=1,
        guidance=f"idea {suffix}",
        execution_thinking="",
        solution=f"solution {suffix}",
        verifier_reward=0.0 if raw_score is None else raw_score,
        verifier_raw_score=raw_score,
        verifier_status=status,
        verifier_message=status,
        summary=f"summary {suffix}",
        reusable_idea=f"summary {suffix}",
        failure_mode=None if status == "valid" else status,
    )


def test_discover_archive_counts_failed_rollout_but_does_not_archive_it(tmp_path: Path) -> None:
    root = make_root_node(problem_id="polyomino_packing", raw_score=5.0, reward=5.0)
    library = GuidanceLibrary(
        tmp_path / "library.json",
        initial_nodes=[root],
        rollout_n=2,
        puct_q_mode="best_child",
        discover_compat=True,
        groups_per_batch=1,
        score_direction="max",
    )
    selected = library.acquire_group("1:0", visible_timestep_exclusive=1)
    valid = library.submit_child("1:0", _entry(selected.id, "valid", raw_score=7.0))
    failed = library.submit_child(
        "1:0",
        _entry(selected.id, "failed", status="guidance_format_error", raw_score=None),
    )
    snapshot = library.snapshot()

    assert valid is not None
    assert failed is None
    assert snapshot["puct_T"] == 2
    assert snapshot["puct_n"][selected.id] == 2
    assert snapshot["puct_m"][selected.id] == 7.0
    assert "entry-failed" in snapshot["entries"]
    assert all(node["entry_id"] != "entry-failed" for node in snapshot["nodes"].values())


def test_discover_same_step_uses_independent_root_lineages(tmp_path: Path) -> None:
    library = prepare_library(
        seed_path=SEED,
        run_path=tmp_path / "library.json",
        groups_per_step=2,
        rollouts_per_group=2,
    )

    first = library.acquire_group("1:0", visible_timestep_exclusive=1, require_solution=True)
    second = library.acquire_group("1:1", visible_timestep_exclusive=1, require_solution=True)

    assert first.id != second.id
    assert first.parent_id is None
    assert second.parent_id is None


def test_best_child_puct_mode_matches_guidance_reference() -> None:
    root = make_root_node(problem_id="polyomino_packing", raw_score=100.0, reward=100.0)

    score = archive_puct_score(
        node=root,
        visit_count=1,
        best_reachable_value=50.0,
        prior=0.0,
        scale=1.0,
        total_visits=1,
        puct_c=0.0,
        q_mode="best_child",
    )

    assert score == 50.0


def test_frontiercs_adapter_and_polyomino_verifier_surface_score(monkeypatch, tmp_path: Path) -> None:
    seen = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def urlopen(request, *, timeout):
        url = request.full_url if hasattr(request, "full_url") else request
        seen.append((url, timeout, request))
        if url.endswith("/submit"):
            body = request.data
            assert request.get_method() == "POST"
            assert b'name="pid"\r\n\r\n0\r\n' in body
            assert b'name="lang"\r\n\r\ncpp\r\n' in body
            assert b'name="code"; filename="solution.cpp"' in body
            assert b"int main() { return 0; }" in body
            # FrontierCS's SubmissionManager allocates numeric IDs even though
            # they are interpolated into the polling URL as strings.
            return Response({"sid": 1})
        assert url.endswith("/result/1")
        return Response({"status": "done", "score": 91.5, "scoreUnbounded": 93.0})

    monkeypatch.setattr("examples.guidance_ttt.verifier.frontiercs_adapter.urllib.request.urlopen", urlopen)
    result = evaluate_cpp_solution(
        "int main() { return 0; }",
        problem_id="0",
        timeout_s=7,
        config={
            "base_dir": str(tmp_path),
            "judge_url": "http://127.0.0.1:8081",
            "poll_interval_s": 0,
        },
    )

    assert result == FrontierCSResult(
        valid=True,
        score=91.5,
        message="accepted",
        artifacts={
            "problem_id": "0",
            "submission_id": "1",
            "frontiercs_config": {},
            "frontiercs_base_dir": str(tmp_path),
            "judge_status": "done",
            "score_unbounded": 93.0,
        },
    )
    assert [call[0] for call in seen] == [
        "http://127.0.0.1:8081/submit",
        "http://127.0.0.1:8081/result/1",
    ]
    assert all(call[1] > 0 for call in seen)


def test_guidance_deployment_defaults_are_small_but_not_hard_coded() -> None:
    args = build_parser().parse_args(["--executor", "gpt_oss_120b"])

    assert args.groups == 2
    assert args.rollouts == 4
    assert args.guidance_max_tokens == 8_192
    assert args.sequence_length == 12_288
    assert args.lora_rank == 32
    assert args.steps == 1
    assert args.training_horizon_steps is None
    assert args.verifier_timeout_s is None
    assert args.require_nonconstant_step is False
    assert args.frozen_policy_control is False


def test_guidance_deployment_exposes_matched_frozen_policy_control() -> None:
    args = build_parser().parse_args(
        ["--executor", "gpt_oss_120b", "--task", "trimul", "--frozen-policy-control"]
    )

    assert args.task == "trimul"
    assert args.frozen_policy_control is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_count": 0}, "gpu_count"),
        ({"rollouts": 1}, "rollouts"),
        ({"tensor_parallel_size": 3}, "divisible"),
        ({"guidance_max_tokens": 12_288}, "sequence_length"),
    ],
)
def test_guidance_deployment_rejects_invalid_settings(overrides: dict[str, int], message: str) -> None:
    settings = {
        "gpu_count": 4,
        "tensor_parallel_size": 2,
        "groups": 2,
        "rollouts": 4,
        "guidance_max_tokens": 8_192,
        "sequence_length": 12_288,
        "lora_rank": 32,
        "steps": 1,
        "verifier_timeout_s": 340,
    }
    settings.update(overrides)

    with pytest.raises(ValueError, match=message):
        _validate_settings(**settings)
