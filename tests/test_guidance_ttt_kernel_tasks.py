from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from examples.guidance_ttt import (
    TRIMUL_TASK,
    VLIW_TASK,
    create_verified_baseline_seed,
    get_task_spec,
    prepare_library,
    task_ids,
)
from examples.guidance_ttt.deployments.qwen3_8b_lora.runner import (
    _default_verifier_timeout_s,
    _require_frozen_step_success,
    _resolve_verifier_config,
    _safe_verifier_config,
)
from examples.guidance_ttt.prompts import build_execution_prompt, build_guidance_prompt
from examples.guidance_ttt.state import LibraryEntry, VerificationResult
from examples.guidance_ttt.tasks.trimul import TRIMUL_BASELINE_SOLUTION, TRIMUL_BASELINE_SUMMARY
from examples.guidance_ttt.tasks.trimul_prompt import TRIMUL_PROMPT
from examples.guidance_ttt.verifier.edgebench_adapter import (
    EdgeBenchResult,
    _result_from_payload as edgebench_result_from_payload,
    evaluate_vliw_solution,
    rescale_vliw_cycles,
    vliw_training_reward,
)
from examples.guidance_ttt.verifier.trimul import extract_trimul_solution_code, verify_trimul_solution_text
from examples.guidance_ttt.verifier.trimul_adapter import TriMulResult, evaluate_trimul_solution
from examples.guidance_ttt.verifier.trimul_official_runner import build_case_file, geometric_mean_runtime_us
from examples.guidance_ttt.verifier.service import VerifierServiceConfig, evaluate_request
from examples.guidance_ttt.verifier.vliw_kernel import extract_vliw_solution_code, verify_vliw_solution_text

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIMUL_SEED = REPO_ROOT / "examples/guidance_ttt/seeds/trimul/glm52_scratch_bootstrap_library.json"
TRIMUL_EVALUATOR = REPO_ROOT / "examples/guidance_ttt/tasks/assets/trimul_evaluator"
EVALUATOR_HASHES = {
    "task.yml": "77a36e2e74141dccb35341f54cc18731504687dcaf80e15f5344c0ba14b73be9",
    "task.py": "856c8f04c06fc4b0768fca5e267c34f1884355054f89c629a6aac07b3c099174",
    "utils.py": "a8a6a725c4fcc3d5e9cd50b9299a77d9f8a39e41830a994481171033f95b2681",
    "reference.py": "7fd9dfc1de86cd41063b50584ad0e214edd43285fc080e767d921ba86e1024ae",
    "eval.py": "94f5b0a0ff72e797c0c83c30478697a897dc2ac369f8cc8e07dbcd0272803f51",
}


def test_guidance_temperature_is_shared_by_sampling_and_logp_recomputation() -> None:
    deployment_config = (
        REPO_ROOT / "examples/tttd/deployments/qwen3_8b_lora/config.yaml"
    ).read_text()
    guidance_runner = (
        REPO_ROOT / "examples/guidance_ttt/deployments/qwen3_8b_lora/runner.py"
    ).read_text()
    tttd_runner = (
        REPO_ROOT / "examples/tttd/deployments/qwen3_8b_lora/runner.py"
    ).read_text()

    assert "--rollout-temperature=${TTTD_ROLLOUT_TEMPERATURE}" in deployment_config
    assert '"TTTD_ROLLOUT_TEMPERATURE": str(float(guidance_temperature))' in guidance_runner
    assert '"TTTD_ROLLOUT_TEMPERATURE": "1.0"' in tttd_runner


def _python_response(code: str, summary: str = "Preserve correctness while reducing runtime.") -> str:
    return f"""<solution>
```python
{code}
```
</solution>

<summary>
{summary}
</summary>"""


def test_task_registry_exposes_both_kernel_tasks() -> None:
    assert task_ids() == ("polyomino_packing", "trimul", "vliw_kernel_optimization")
    assert get_task_spec("vliw_kernel_optimization") is VLIW_TASK
    assert get_task_spec("trimul") is TRIMUL_TASK
    assert VLIW_TASK.score_direction == "min"
    assert TRIMUL_TASK.score_direction == "min"


def test_vliw_official_starter_and_contract_are_pinned() -> None:
    assert VLIW_TASK.bootstrap_solution is not None
    assert hashlib.sha256(VLIW_TASK.bootstrap_solution.encode()).hexdigest() == (
        "fb165d18bf4230b4b91ac889b3eff09f9931f559b151cf11c7890fb53e22c01e"
    )
    assert "class KernelBuilder" in VLIW_TASK.bootstrap_solution
    assert "dependency-aware bundle packing" in VLIW_TASK.guidance_mechanism_constraint
    assert "Lower raw cycles" in VLIW_TASK.guidance_objective(None)


def test_vliw_reward_preserves_minimize_raw_metric() -> None:
    assert rescale_vliw_cycles(None) == 0.0
    assert rescale_vliw_cycles(4475.526541978607) == 0.0
    assert rescale_vliw_cycles(1000.0) == pytest.approx(100.0)
    assert vliw_training_reward(100_000) == 10.0
    assert vliw_training_reward(50_000) == 20.0
    invalid = edgebench_result_from_payload(
        {"report": {"all_correct": True, "score_cycles": float("nan")}}
    )
    assert invalid.valid is False
    assert invalid.cycles is None


def test_vliw_parser_and_verifier_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    code = "class KernelBuilder:\n    def build_kernel(self, *args):\n        self.instrs = []"
    assert extract_vliw_solution_code(_python_response(code)) == code
    assert extract_vliw_solution_code("<solution>```cpp\nint main(){}\n```</solution>") is None

    def fake_evaluate(solution: str, *, timeout_s: int, config: dict):
        assert (solution, timeout_s, config) == (code, 700, {"provider": "http"})
        return EdgeBenchResult(True, 2000.0, rescale_vliw_cycles(2000.0), "accepted", {})

    monkeypatch.setattr("examples.guidance_ttt.verifier.vliw_kernel.evaluate_vliw_solution", fake_evaluate)
    result = verify_vliw_solution_text(_python_response(code), timeout_s=700, config={"provider": "http"})
    assert result.valid is True
    assert result.raw_score == 2000.0
    assert result.reward == 500.0
    assert result.artifacts["official_normalized_score"] == rescale_vliw_cycles(2000.0)


def test_vliw_http_adapter_is_provider_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({"report": {"all_correct": True, "score_cycles": 4321}}).encode()

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TEST_VLIW_URL", "https://judge.example.test/evaluate")
    monkeypatch.setenv("TEST_VLIW_TOKEN", "test-token")
    monkeypatch.setattr("examples.guidance_ttt.verifier.edgebench_adapter.urllib.request.urlopen", fake_urlopen)
    result = evaluate_vliw_solution(
        "class KernelBuilder: pass",
        timeout_s=120,
        config={
            "provider": "http",
            "endpoint_env": "TEST_VLIW_URL",
            "api_key_env": "TEST_VLIW_TOKEN",
            "runner_timeout_s": 37,
            "request_timeout_s": 45,
            "max_retries": 0,
        },
    )
    assert captured["url"] == "https://judge.example.test/evaluate"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["payload"] == {"solution": "class KernelBuilder: pass", "runner_timeout_s": 37}
    assert captured["timeout"] == 45
    assert result.cycles == 4321


def test_verified_vliw_baseline_builds_a_pristine_minimize_seed(tmp_path: Path) -> None:
    def verify(text: str, *, timeout_s: int, config: dict) -> VerificationResult:
        assert VLIW_TASK.bootstrap_solution in text
        assert timeout_s == 60
        assert config == {
            "provider": "test",
            "api_key": "must-not-persist",
            "headers": {"Authorization": "must-not-persist-either", "X-Trace": "kept"},
        }
        return VerificationResult(10.0, 100_000.0, True, "valid", "accepted", {"official": True})

    task = replace(VLIW_TASK, verifier=verify)
    seed_path = tmp_path / "vliw-seed.json"
    snapshot = create_verified_baseline_seed(
        seed_path,
        task=task,
        verifier_timeout_s=60,
        verifier_config={
            "provider": "test",
            "api_key": "must-not-persist",
            "headers": {"Authorization": "must-not-persist-either", "X-Trace": "kept"},
        },
    )
    root = next(iter(snapshot["nodes"].values()))
    entry = next(iter(snapshot["entries"].values()))
    assert root["problem_id"] == "vliw_kernel_optimization"
    assert root["raw_score"] == 100_000.0
    assert entry["solution"] == VLIW_TASK.bootstrap_solution
    assert entry["metadata"]["bootstrap_source"] == "task_baseline"
    serialized = json.dumps(snapshot)
    assert "must-not-persist" not in serialized
    assert entry["metadata"]["task"]["verifier"] == {
        "provider": "test",
        "headers": {"X-Trace": "kept"},
    }

    library = prepare_library(
        seed_path=seed_path,
        run_path=tmp_path / "run.json",
        groups_per_step=2,
        rollouts_per_group=2,
        task=task,
    )
    assert library.score_direction == "min"
    assert {node["value"] for node in library.snapshot()["nodes"].values()} == {-100_000.0}


def test_trimul_prompt_evaluator_and_seed_are_pinned() -> None:
    assert hashlib.sha256(TRIMUL_PROMPT.encode()).hexdigest() == (
        "b9add300e4bb518525701c9d2972b555675425bfe0d9da61a90e9d5a8b64c6f6"
    )
    for name, expected_hash in EVALUATOR_HASHES.items():
        assert hashlib.sha256((TRIMUL_EVALUATOR / name).read_bytes()).hexdigest() == expected_hash
    assert "@triton.jit" in TRIMUL_BASELINE_SOLUTION
    assert TRIMUL_BASELINE_SUMMARY
    seed = json.loads(TRIMUL_SEED.read_text())
    assert len(seed["nodes"]) == 1
    assert len(seed["entries"]) == 1
    root = next(iter(seed["nodes"].values()))
    assert root["problem_id"] == "trimul"
    assert root["raw_score"] == pytest.approx(10177.396849081848)


def test_trimul_summary_only_prompt_hides_parent_code() -> None:
    library_data = json.loads(TRIMUL_SEED.read_text())
    node_data = next(iter(library_data["nodes"].values()))
    entry_data = next(iter(library_data["entries"].values()))
    from examples.guidance_ttt.state import LibraryNode

    node = LibraryNode.from_dict(node_data)
    entry = LibraryEntry.from_dict(entry_data)
    guidance_prompt = build_guidance_prompt(
        problem_prompt=TRIMUL_TASK.problem_prompt,
        selected_node=node,
        selected_entry=entry,
        objective_text=TRIMUL_TASK.guidance_objective(node.raw_score),
        mechanism_constraint=TRIMUL_TASK.guidance_mechanism_constraint,
        raw_score_label=TRIMUL_TASK.raw_score_label,
    )
    execution_prompt = build_execution_prompt(
        problem_prompt=TRIMUL_TASK.problem_prompt,
        selected_entry=entry,
        guidance="Reduce launch overhead without changing the equations.",
        solution_language="python",
        solution_contract=TRIMUL_TASK.execution_solution_contract,
        score_direction="min",
        raw_score_label=TRIMUL_TASK.raw_score_label,
    )
    assert entry.solution not in guidance_prompt.user
    assert "<selected_summary>" in guidance_prompt.user
    assert entry.solution in execution_prompt.user
    assert "<parent_code>" in execution_prompt.user


def test_trimul_archive_uses_minimize_direction_and_rejects_cross_task_seed(tmp_path: Path) -> None:
    library = prepare_library(
        seed_path=TRIMUL_SEED,
        run_path=tmp_path / "trimul.json",
        groups_per_step=2,
        rollouts_per_group=2,
        task=TRIMUL_TASK,
    )
    assert library.score_direction == "min"
    assert all(node["value"] < 0 for node in library.snapshot()["nodes"].values())

    with pytest.raises(ValueError, match="task mismatch"):
        prepare_library(
            seed_path=TRIMUL_SEED,
            run_path=tmp_path / "wrong-task.json",
            groups_per_step=1,
            rollouts_per_group=2,
            task=VLIW_TASK,
        )


def test_trimul_parser_prechecks_and_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    code = """import triton
@triton.jit
def kernel(x):
    pass
def custom_kernel(data):
    return data[0]
"""
    assert extract_trimul_solution_code(_python_response(code)) == code.strip()

    def fake_evaluate(solution: str, *, timeout_s: int, config: dict):
        assert solution == code.strip()
        assert timeout_s == 1200
        return TriMulResult(True, 750.0, "accepted", {"benchmarks": []})

    monkeypatch.setattr("examples.guidance_ttt.verifier.trimul.evaluate_trimul_solution", fake_evaluate)
    result = verify_trimul_solution_text(
        _python_response(code),
        timeout_s=1200,
        config={"provider": "http", "reward_scale": 1500.0},
    )
    assert result.valid is True
    assert result.raw_score == 750.0
    assert result.reward == 2.0
    assert result.artifacts["reward_formula"] == "reward_scale / geometric_mean_runtime_us"


def test_trimul_http_adapter_retries_and_forwards_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({"report": {"all_correct": True, "score_us": 625.0}}).encode()

    def fake_urlopen(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError(request.full_url, 408, "cold start", {}, io.BytesIO(b""))
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TEST_TRIMUL_URL", "https://judge.example.test/evaluate")
    monkeypatch.setenv("TEST_TRIMUL_TOKEN", "secret-token")
    monkeypatch.setattr("examples.guidance_ttt.verifier.trimul_adapter.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("examples.guidance_ttt.verifier.trimul_adapter.time.sleep", lambda _: None)
    result = evaluate_trimul_solution(
        "def custom_kernel(data): pass",
        timeout_s=1200,
        config={
            "provider": "http",
            "endpoint_env": "TEST_TRIMUL_URL",
            "api_key_env": "TEST_TRIMUL_TOKEN",
            "runner_timeout_s": 520,
            "request_timeout_s": 1150,
            "max_retries": 2,
        },
    )
    assert attempts == 2
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["payload"] == {"solution": "def custom_kernel(data): pass", "runner_timeout_s": 520}
    assert captured["timeout"] == 1150
    assert result.score_us == 625.0
    assert result.artifacts["http_attempts"] == 2


def test_official_trimul_score_helpers_match_contract() -> None:
    assert build_case_file(
        [{"seqlen": 32, "bs": 1, "nomask": True, "distribution": "normal"}]
    ) == "seqlen: 32; bs: 1; nomask: True; distribution: normal\n"
    assert geometric_mean_runtime_us([{"mean_us": 100.0}, {"mean_us": 400.0}]) == pytest.approx(200.0)


def test_runner_defaults_are_task_specific_and_secret_safe(tmp_path: Path) -> None:
    vliw = _resolve_verifier_config(
        task_id="vliw_kernel_optimization",
        frontiercs_base_dir=tmp_path,
        judge_url="unused",
        overrides={"provider": "http", "endpoint_env": "VLIW_URL"},
    )
    assert vliw["provider"] == "http"
    assert vliw["reward_scale"] == 1_000_000.0
    assert vliw["runner_timeout_s"] == 60
    assert vliw["request_timeout_s"] == 90

    trimul = _resolve_verifier_config(
        task_id="trimul",
        frontiercs_base_dir=tmp_path,
        judge_url="unused",
        overrides={"api_key": "must-not-persist"},
    )
    assert trimul["provider"] == "http"
    assert trimul["reward_scale"] == 1500.0
    assert trimul["runner_timeout_s"] == 520
    assert trimul["request_timeout_s"] == 1150
    assert "api_key" not in _safe_verifier_config(trimul)
    assert _safe_verifier_config(trimul)["api_key_env"] == "TRIMUL_JUDGE_TOKEN"
    assert _default_verifier_timeout_s("polyomino_packing") == 340
    assert _default_verifier_timeout_s("vliw_kernel_optimization") == 120
    assert _default_verifier_timeout_s("trimul") == 1160


def test_frozen_control_requires_zero_training_and_stable_serving_weights() -> None:
    health = {"ok": True, "phase": "serving", "completed_train_steps": 0}
    _require_frozen_step_success(
        expected_rollouts=8,
        actual_rollouts=8,
        initial_weight_version="base",
        version_before="base",
        version_after="base",
        bridge_health=health,
    )

    with pytest.raises(RuntimeError, match="unexpectedly trained"):
        _require_frozen_step_success(
            expected_rollouts=8,
            actual_rollouts=8,
            initial_weight_version="base",
            version_before="base",
            version_after="base",
            bridge_health={**health, "completed_train_steps": 1},
        )

    with pytest.raises(RuntimeError, match="serving weight changed"):
        _require_frozen_step_success(
            expected_rollouts=8,
            actual_rollouts=8,
            initial_weight_version="base",
            version_before="base",
            version_after="updated",
            bridge_health=health,
        )


def test_provider_neutral_vliw_service_normalizes_official_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(code: str, *, timeout_s: int, config: dict):
        assert code == "class KernelBuilder: pass"
        assert timeout_s == 90
        assert config == {"provider": "docker", "judge_image": "pinned-image"}
        return EdgeBenchResult(
            True,
            4321.0,
            1.0,
            "accepted",
            {"judge_image": "pinned-image", "results": [{"correct": True}]},
        )

    monkeypatch.setattr("examples.guidance_ttt.verifier.service.evaluate_vliw_solution", fake_evaluate)
    result = evaluate_request(
        {"solution": "class KernelBuilder: pass", "runner_timeout_s": 90},
        VerifierServiceConfig(
            task_id="vliw_kernel_optimization",
            token="secret",
            vliw_config={"judge_image": "pinned-image"},
        ),
    )
    assert result["provider"] == "reef_http_verifier"
    assert result["report"]["all_correct"] is True
    assert result["report"]["score_cycles"] == 4321.0


def test_provider_neutral_trimul_service_caps_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRIMUL_JUDGE_TOKEN", "must-not-reach-candidate")

    def fake_evaluate(code: str, *, evaluator_dir: Path, timeout_s: int, subprocess_env: dict[str, str]):
        assert code == "def custom_kernel(data): pass"
        assert evaluator_dir == tmp_path
        assert timeout_s == 300
        assert "TRIMUL_JUDGE_TOKEN" not in subprocess_env
        assert subprocess_env["PYTHONNOUSERSITE"] == "1"
        return {"report": {"all_correct": True, "score_us": 500.0}}

    monkeypatch.setattr(
        "examples.guidance_ttt.verifier.service.run_official_trimul_evaluation",
        fake_evaluate,
    )
    result = evaluate_request(
        {"solution": "def custom_kernel(data): pass", "runner_timeout_s": 900},
        VerifierServiceConfig(
            task_id="trimul",
            token="secret",
            max_runner_timeout_s=300,
            evaluator_dir=tmp_path,
            evaluation_gpu_label="NVIDIA H100",
        ),
    )
    assert result["provider"] == "reef_http_verifier"
    assert result["evaluation_gpu"] == "NVIDIA H100"
    assert result["report"]["score_us"] == 500.0
