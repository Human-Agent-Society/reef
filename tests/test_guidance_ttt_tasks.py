"""Task contracts and real HTTP feedback boundaries for the four discovery examples."""

from __future__ import annotations

import importlib.util
import json
import math
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from recipes.tttd.examples.guidance_ttt.harness.agent import prepare_library
from recipes.tttd.examples.guidance_ttt.harness.bootstrap import prepare_seed
from recipes.tttd.examples.guidance_ttt.harness.config import EXAMPLE_DIR, TASKS, RunConfig
from recipes.tttd.examples.guidance_ttt.harness.scorer import JudgeScorer, JudgeUnavailableError


@pytest.fixture
def judge_endpoint():
    responses = {"result": {"status": "done", "score": 1, "scoreUnbounded": 1, "valid": True}}
    submissions = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            submissions.append(self.rfile.read(int(self.headers["Content-Length"])).decode())
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"sid":"sample"}')

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(responses["result"]).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", responses, submissions
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("task", TASKS)
def test_each_task_loads_and_verifies_a_bootstrap(task, monkeypatch, tmp_path, judge_endpoint):
    url, replies, submissions = judge_endpoint
    monkeypatch.setenv("GUIDANCE_TASK", task)
    config = replace(RunConfig.load(), state_dir=tmp_path, judge_url=url)
    contract = config.contract()
    suffix = ".cpp" if contract.solution_language == "cpp" else ".py"
    seed = tmp_path / f"bootstrap{suffix}"
    seed.write_text("int main() {}" if suffix == ".cpp" else 'CPP_CODE = "example"')
    seed.with_suffix(".md").write_text("An initial candidate summary.")
    monkeypatch.setenv("GUIDANCE_SEED", str(seed))
    replies["result"].update(score=1.5, scoreUnbounded=1000)
    scorer = JudgeScorer(url, problem_id=contract.judge_problem_id, language=contract.solution_language)
    verified = prepare_seed(config, scorer)
    library = prepare_library(
        seed_path=verified,
        run_path=tmp_path / "run.json",
        groups_per_step=2,
        rollouts_per_group=4,
        score_direction=contract.score_direction,
    )
    snapshot = library.snapshot()
    entry = next(iter(snapshot["entries"].values()))
    assert entry["verifier_reward"] == 1.5
    assert entry["verifier_raw_score"] == 1000
    assert len(snapshot["nodes"]) == 2
    assert library.score_direction == contract.score_direction
    assert f"\r\n\r\n{contract.judge_problem_id}\r\n" in submissions[0]
    assert f'filename="solution{suffix}"' in submissions[0]
    assert config.task_dir.joinpath("tests/grade.py").is_file()


def test_trimul_ranks_lower_raw_latency_first(monkeypatch, tmp_path, judge_endpoint):
    from recipes.tttd.examples.guidance_ttt.harness.state import LibraryEntry

    url, replies, _ = judge_endpoint
    monkeypatch.setenv("GUIDANCE_TASK", "trimul")
    config = replace(RunConfig.load(), state_dir=tmp_path)
    (tmp_path / "bootstrap.py").write_text("def custom_kernel(data): return data")
    (tmp_path / "bootstrap.md").write_text("The initial kernel.")
    replies["result"].update(score=1.0, scoreUnbounded=1500.0)
    scorer = JudgeScorer(url, problem_id="trimul", language="python")
    seed = prepare_seed(config, scorer)
    library = prepare_library(
        seed_path=seed, run_path=tmp_path / "run.json", groups_per_step=1, rollouts_per_group=2, score_direction="min"
    )
    parent = library.acquire_group("1:0", visible_timestep_exclusive=1, require_solution=True)
    for index, latency in enumerate((1200.0, 1800.0)):
        library.submit_child(
            "1:0",
            LibraryEntry(
                id=f"child-{index}",
                parent_id=parent.id,
                problem_id="trimul",
                timestep=1,
                guidance="improve",
                execution_thinking="",
                solution=f"kernel {index}",
                verifier_reward=1500 / latency,
                verifier_raw_score=latency,
                verifier_status="valid",
                verifier_message="accepted",
                summary="kernel",
                reusable_idea="kernel",
                failure_mode=None,
            ),
        )
    snapshot = library.snapshot()
    assert snapshot["nodes"][snapshot["best_node_id"]]["raw_score"] == 1200


def test_ahc_partial_reward_is_not_a_valid_archive_candidate(judge_endpoint):
    url, replies, _ = judge_endpoint
    replies["result"].update(score=1.2, scoreUnbounded=3600000, valid=False, trainingRewardOnInvalid=True)
    result = JudgeScorer(url, problem_id="ahc058")("int main() {}")
    assert result.reward == 1.2
    assert result.raw_score == 3600000
    assert result.valid is False
    assert result.status == "invalid"


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"scoreUnbounded": float("inf")},
        {"score": -1},
        {"valid": "false"},
        {"status": "environment_error"},
        {"failureDomain": "infrastructure"},
    ],
)
def test_judge_contract_errors_stop_the_step(judge_endpoint, change):
    url, replies, _ = judge_endpoint
    replies["result"].update(change)
    with pytest.raises(JudgeUnavailableError):
        JudgeScorer(url)("int main() {}")


def test_rejected_bootstrap_never_creates_an_archive(judge_endpoint, monkeypatch, tmp_path):
    url, replies, _ = judge_endpoint
    monkeypatch.setenv("GUIDANCE_TASK", "polyomino_packing")
    replies["result"].update(valid=False, status="error", message="wrong answer")
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        prepare_seed(replace(RunConfig.load(), state_dir=tmp_path), JudgeScorer(url))
    assert not (tmp_path / "verified-bootstrap-library.json").exists()


def test_configuration_rejects_grid_mismatch_before_startup(tmp_path, monkeypatch):
    config = yaml.safe_load((EXAMPLE_DIR / "serve.yaml").read_text())
    config["training"]["options"]["global-batch-size"] = 999
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("GUIDANCE_CONFIG", str(path))
    with pytest.raises(ValueError, match="global-batch-size"):
        RunConfig.load()


def test_task_and_executor_selection_reject_unknown_values(monkeypatch):
    monkeypatch.setenv("GUIDANCE_TASK", "../../other")
    with pytest.raises(ValueError, match="GUIDANCE_TASK"):
        RunConfig.load()
    monkeypatch.setenv("GUIDANCE_TASK", "lasso_path")
    monkeypatch.setenv("GUIDANCE_EXECUTOR", "unknown")
    with pytest.raises(ValueError, match="GUIDANCE_EXECUTOR"):
        RunConfig.load()


@pytest.mark.parametrize("task", TASKS)
def test_final_verifier_rejects_partial_or_nonfinite_results(task, monkeypatch):
    path = EXAMPLE_DIR / "harbor" / task / "tests/grade.py"
    spec = importlib.util.spec_from_file_location(f"grade_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SOLUTION_PATH", path)  # A readable candidate, never executed here.
    monkeypatch.setattr(module, "submit", lambda *args: "sample")
    for result in [{"status": "done", "score": 5, "valid": False}, {"status": "done", "score": math.nan}]:
        monkeypatch.setattr(module, "poll", lambda *args, result=result: result)
        assert module.grade()["reward"] == 0


def test_constant_reward_step_can_commit_without_a_parameter_change():
    from recipes.tttd.examples.guidance_ttt.harness.run_controller import require_step_success

    health = {
        "ok": True,
        "phase": "serving",
        "completed_train_steps": 1,
        "last_train_rollout_id": 0,
        "last_train_metrics": {
            "train/global_batch_size": 2,
            "train/lora_trainable_parameters": 1024,
            "train/lora_base_trainable_parameters": 0,
            "train/lora_b_nonzero": 0,
            "train/lora_b_l1": 0,
        },
    }
    args = {
        "expected_rollouts": 4,
        "actual_rollouts": 4,
        "retained_trajectories": 2,
        "bridge_health": health,
        "expected_completed_train_steps": 1,
        "expected_rollout_id": 0,
        "grad_norm": 0.0,
        "lora_rank": 32,
    }
    assert require_step_success(**args, allow_zero_signal=True) == 0.0
    with pytest.raises(RuntimeError, match="grad norm"):
        require_step_success(**args)
    health["last_train_metrics"]["train/lora_base_trainable_parameters"] = 1
    with pytest.raises(RuntimeError, match="base parameters"):
        require_step_success(**args, allow_zero_signal=True)


def test_resume_rejects_a_changed_task_or_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIDANCE_TASK", "polyomino_packing")
    config = replace(RunConfig.load(), state_dir=tmp_path)
    config.validate_state()
    config.validate_state()
    with pytest.raises(ValueError, match="new GUIDANCE_STATE_DIR"):
        replace(config, task="ahc058").validate_state()
    with pytest.raises(ValueError, match="new GUIDANCE_STATE_DIR"):
        replace(config, backend=replace(config.backend, model="another-model")).validate_state()


def test_trimul_evaluator_maps_latency_and_rejects_the_wrong_suite():
    from recipes.tttd.examples.guidance_ttt.judges.trimul_server import score_report

    report = {"all_correct": True, "score_us": 1200, "test_count": 18, "benchmark_count": 7, "benchmarks": []}
    result = score_report({"report": report})
    assert result["score"] == 1.25
    assert result["score_unbounded"] == 1200
    report["benchmark_count"] = 6
    with pytest.raises(ValueError, match="seven benchmarks"):
        score_report({"report": report})
    report["all_correct"] = False
    assert score_report({"report": report})["valid"] is False


def test_official_trimul_timing_parser_uses_microseconds():
    from recipes.tttd.examples.guidance_ttt.judges.trimul_runner import benchmark_records, geometric_mean_runtime_us

    records = benchmark_records({"benchmark-count": "2", "benchmark.0.mean": "1000", "benchmark.1.mean": "4000"})
    assert geometric_mean_runtime_us(records) == 2
    with pytest.raises(ValueError, match="invalid"):
        benchmark_records({"benchmark-count": "1", "benchmark.0.mean": "-1"})


def test_trimul_modal_container_import_does_not_require_deployer_files(monkeypatch, tmp_path):
    import runpy
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    image = MagicMock()
    image.pip_install.return_value = image
    app = MagicMock()
    app.function.return_value = lambda function: function
    modal = SimpleNamespace(
        is_local=lambda: False,
        App=lambda name: app,
        Image=SimpleNamespace(debian_slim=lambda **kwargs: image),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setenv("GUIDANCE_DISCOVER_ROOT", str(tmp_path / "absent-on-worker"))
    module = runpy.run_path(str(EXAMPLE_DIR / "judges/trimul_modal.py"))
    assert callable(module["evaluate"])
    image.add_local_dir.assert_not_called()
    image.add_local_file.assert_not_called()


def test_executor_rejection_is_infrastructure_and_preserves_output_budget(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from recipes.tttd.examples.guidance_ttt.harness.execution import (
        ExecutionBackend,
        ExecutorUnavailableError,
        OpenAICompatibleExecutionClient,
    )
    from recipes.tttd.examples.guidance_ttt.harness.state import LLMRequest

    requests = []

    def reject(request, *, timeout):
        requests.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 402, "budget exceeded", {}, io.BytesIO(b"budget exceeded"))

    monkeypatch.setattr(urllib.request, "urlopen", reject)
    client = OpenAICompatibleExecutionClient(ExecutionBackend("test", "test", "http://executor.invalid/v1"))
    with pytest.raises(ExecutorUnavailableError, match="HTTP 402"):
        client.complete(LLMRequest(system="system", user="task", model="test", temperature=1, max_tokens=16384))
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 16384


def test_executor_output_budget_configuration(monkeypatch):
    monkeypatch.setenv("GUIDANCE_EXECUTOR_MAX_TOKENS", "16384")
    assert RunConfig.load().backend.max_tokens == 16384
    monkeypatch.setenv("GUIDANCE_EXECUTOR_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="max_tokens"):
        RunConfig.load()
