from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.tttd.examples.guidance_ttt.harness import (
    ExecutionBackend,
    OpenAICompatibleExecutionClient,
    ReefGuidanceTTTHarness,
    TaskContract,
    gpt_oss_120b_backend,
    openrouter_glm_5_2_backend,
    prepare_library,
)
from recipes.tttd.examples.guidance_ttt.harness.library import GuidanceLibrary
from recipes.tttd.examples.guidance_ttt.harness.prompts import (
    build_execution_prompt,
    build_guidance_prompt,
    extract_strict_guidance,
)
from recipes.tttd.examples.guidance_ttt.harness.puct import archive_puct_score
from recipes.tttd.examples.guidance_ttt.harness.run_controller import (
    GuidanceRunIdentity,
    GuidanceRunStateError,
    GuidanceRunStateStore,
    RayTrainingBridge,
    read_json,
    require_step_success,
    scenario_status,
    wait_for_training_step,
    write_json,
)
from recipes.tttd.examples.guidance_ttt.harness.scorer import JudgeResult, JudgeScorer, extract_solution_code
from recipes.tttd.examples.guidance_ttt.harness.state import (
    LibraryEntry,
    LLMRequest,
    LLMResponse,
    VerificationResult,
    make_root_node,
)

TASK_DIR = REPO_ROOT / "recipes" / "tttd" / "examples" / "guidance_ttt" / "harbor" / "polyomino_packing"
SEED = TASK_DIR / "solution" / "gpt_oss_120b_bootstrap_library.json"
INSTRUCTION = (TASK_DIR / "instruction.md").read_text()
CONTRACT_PATH = TASK_DIR / "contract.json"


def _contract() -> TaskContract:
    return TaskContract.load(CONTRACT_PATH, problem_prompt=INSTRUCTION)


def _length_scorer(code: str) -> VerificationResult:
    """Stand in for the external judge: a deterministic, non-constant score."""
    score = float(len(code))
    return VerificationResult(score, score, True, "valid", "accepted", {"code": code})


class _ReefClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inferences: list[dict] = []
        self.reports: list[dict] = []

    def inference_with_record(self, scenario, path, payload, *, recipe=None, release_id=None):
        assert (scenario, path, recipe, release_id) == (
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

    def report(self, scenario, payload, *, recipe=None, release_id=None):
        assert (scenario, recipe, release_id) == ("guidance-smoke", "tttd", None)
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
        contract=_contract(),
        scorer=_length_scorer,
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


def test_judge_scorer_speaks_the_judge_protocol_and_surfaces_its_score(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("recipes.tttd.examples.guidance_ttt.harness.scorer.urllib.request.urlopen", urlopen)
    scorer = JudgeScorer(
        "http://127.0.0.1:8081",
        problem_id="0",
        timeout_s=7,
        poll_interval_s=0,
        base_dir=str(tmp_path),
    )
    result = scorer.evaluate("int main() { return 0; }")

    assert result == JudgeResult(
        valid=True,
        score=91.5,
        message="accepted",
        artifacts={
            "problem_id": "0",
            "submission_id": "1",
            "judge_base_dir": str(tmp_path),
            "judge_status": "done",
            "score_unbounded": 93.0,
        },
    )
    assert [call[0] for call in seen] == [
        "http://127.0.0.1:8081/submit",
        "http://127.0.0.1:8081/result/1",
    ]
    assert all(call[1] > 0 for call in seen)
    # As a Scorer, the same verdict becomes the harness's VerificationResult.
    verification = scorer("int main() { return 0; }")
    assert (verification.reward, verification.valid, verification.status) == (91.5, True, "valid")


def _bridge_health(**overrides) -> dict:
    health = {
        "ok": True,
        "completed_train_steps": 1,
        "last_train_rollout_id": 0,
        "phase": "awaiting_commit",
    }
    health.update(overrides)
    return health


def test_training_wait_keeps_polling_until_the_bridge_returns_to_serving() -> None:
    health = iter([_bridge_health(), _bridge_health(phase="serving")])

    result = wait_for_training_step(
        health=lambda: next(health),
        status=lambda: scenario_status(_reef_status(), "guidance-run"),
        expected_completed_steps=1,
        expected_rollout_id=0,
        timeout_s=1,
        poll_interval_s=0,
    )

    assert result["phase"] == "serving"


def test_training_wait_fails_when_tttd_reports_the_expected_step_failure() -> None:
    failed = _reef_status(
        failed_steps=[
            {
                "step": 1,
                "reason": "mixed_release_ids",
                "release_ids": ["artifact-old", "artifact-new"],
            }
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"step 1 failed \(mixed_release_ids\).*artifact-old.*artifact-new",
    ):
        wait_for_training_step(
            health=lambda: _bridge_health(completed_train_steps=0, last_train_rollout_id=None, phase="serving"),
            status=lambda: scenario_status(failed, "guidance-run"),
            expected_completed_steps=1,
            expected_rollout_id=1,
            timeout_s=1,
            poll_interval_s=0,
        )


def test_training_wait_fails_on_reef_training_error() -> None:
    """Issue #344: a pre-bridge failure must not look like healthy serving."""
    body = _reef_status(error="guidance-run: RuntimeError: failed before RolloutManager submission")

    with pytest.raises(RuntimeError, match="failed before RolloutManager submission"):
        wait_for_training_step(
            health=lambda: _bridge_health(last_train_rollout_id=11, phase="serving"),
            status=lambda: scenario_status(body, "guidance-run"),
            expected_completed_steps=2,
            expected_rollout_id=12,
            timeout_s=0.01,
            poll_interval_s=0,
        )


def _reef_status(
    *,
    error: str | None = None,
    failed_steps: list[dict] | None = None,
) -> dict:
    return {
        "error": error,
        "scenarios": {
            "guidance-run": {
                "processor": {"failed_steps": failed_steps or []},
            }
        },
    }


def test_the_harness_takes_its_task_vocabulary_from_the_harbor_task(tmp_path: Path) -> None:
    contract = TaskContract.load(CONTRACT_PATH, problem_prompt=INSTRUCTION)

    assert contract.problem_prompt == INSTRUCTION
    assert "Polyomino Packing" in INSTRUCTION
    assert (contract.solution_language, contract.score_direction) == ("cpp", "max")
    assert contract.judge_problem_id == "0"

    with pytest.raises(ValueError, match="problem_prompt"):
        TaskContract.load(CONTRACT_PATH, problem_prompt="  ")

    unknown = tmp_path / "contract.json"
    unknown.write_text(json.dumps({**json.loads(CONTRACT_PATH.read_text()), "verifier": "polyomino"}))
    with pytest.raises(ValueError, match="unknown task contract fields"):
        TaskContract.load(unknown, problem_prompt=INSTRUCTION)


def test_solution_extraction_prefers_the_solution_block_then_the_last_fence() -> None:
    text = "```cpp\nint old();\n```\n<solution>\n```cpp\nint main() { return 0; }\n```\n</solution>"

    assert extract_solution_code(text) == "int main() { return 0; }"
    assert extract_solution_code("```C++\nint x;\n```") == "int x;"
    assert extract_solution_code("no fenced program") is None


def test_execution_client_sends_and_decodes_openai_compatible_request(monkeypatch) -> None:
    from recipes.tttd.examples.guidance_ttt.harness import execution

    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return json.dumps(
                {
                    "model": "executor-revision",
                    "choices": [
                        {
                            "message": {"content": "final program", "reasoning_content": "checked it"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"completion_tokens": 4},
                }
            ).encode()

    def urlopen(request, timeout):
        seen.append((request, timeout))
        return Response()

    monkeypatch.setattr(execution.urllib.request, "urlopen", urlopen)
    backend = ExecutionBackend(
        name="host-test",
        model="executor",
        base_url="http://executor.invalid/v1",
        temperature=0.25,
        max_tokens=64,
        timeout_s=9,
        max_retries=0,
        request_options={"reasoning_effort": "high"},
    )
    result = OpenAICompatibleExecutionClient(backend).complete(
        LLMRequest("system", "user", "executor", 0.25, 64, {"step": 1})
    )

    request, timeout = seen[0]
    assert (request.full_url, timeout) == ("http://executor.invalid/v1/chat/completions", 9)
    assert request.headers["Authorization"] == "Bearer local-executor"
    payload = json.loads(request.data)
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert payload["reasoning_effort"] == "high"
    assert result.text == "final program"
    assert result.reasoning == "checked it"
    assert result.metadata["step"] == 1
    assert result.metadata["api_attempts"] == 1


def test_guidance_training_success_gate_accepts_complete_lora_update() -> None:
    health = {
        "ok": True,
        "phase": "serving",
        "completed_train_steps": 1,
        "last_train_rollout_id": 0,
        "last_train_metrics": {
            "train/global_batch_size": 8,
            "train/lora_trainable_parameters": 2048,
            "train/lora_base_trainable_parameters": 0,
            "train/lora_b_nonzero": 128,
            "train/lora_b_l1": 1.5,
        },
    }

    assert (
        require_step_success(
            expected_rollouts=8,
            actual_rollouts=8,
            retained_trajectories=8,
            bridge_health=health,
            expected_completed_train_steps=1,
            expected_rollout_id=0,
            grad_norm=0.75,
            lora_rank=32,
        )
        == 0.75
    )


def _identity(**overrides) -> GuidanceRunIdentity:
    settings = {
        "model": "Qwen/Qwen3-8B",
        "executor": "gpt_oss_120b",
        "gpt_oss_reasoning_effort": "high",
        "groups_per_step": 2,
        "rollouts_per_group": 4,
        "guidance_max_tokens": 8_192,
        "sequence_length": 12_288,
        "lora_rank": 32,
        "tensor_parallel_size": 2,
    }
    settings.update(overrides)
    return GuidanceRunIdentity(**settings)


def test_guidance_resume_state_round_trips_and_fails_closed_on_a_changed_run(tmp_path: Path) -> None:
    store = GuidanceRunStateStore(tmp_path / "state", _identity())
    store.save(next_step=3, step_summaries=[{"step": 2}], extra={"prompt_mode": "summary_only"})

    saved = store.load()
    assert saved is not None
    assert saved["next_step"] == 3
    assert saved["step_summaries"] == [{"step": 2}]
    assert saved["prompt_mode"] == "summary_only"
    assert read_json(store.resume_path) == saved

    changed = GuidanceRunStateStore(tmp_path / "state", _identity(rollouts_per_group=8))
    with pytest.raises(GuidanceRunStateError, match="rollouts_per_group"):
        changed.load()


def test_guidance_committed_archive_is_the_only_resume_source(tmp_path: Path) -> None:
    store = GuidanceRunStateStore(tmp_path / "state", _identity())
    with pytest.raises(GuidanceRunStateError, match="committed Guidance archive is missing"):
        store.restore_working_library()

    write_json(store.working_library_path, {"nodes": {}, "step": 1})
    store.commit_library()
    write_json(store.working_library_path, {"nodes": {}, "step": 2})
    store.restore_working_library()

    assert read_json(store.working_library_path)["step"] == 1


def test_ray_bridge_rejects_an_invalid_training_timeout() -> None:
    with pytest.raises(ValueError, match="training timeout"):
        RayTrainingBridge("http://127.0.0.1:8900", "guidance-run", timeout_s=0)


def _harbor_agent_module(monkeypatch):
    """Load the Harbor agent against the minimal Harbor protocol it uses."""
    modules = {
        "harbor": ModuleType("harbor"),
        "harbor.agents": ModuleType("harbor.agents"),
        "harbor.agents.base": ModuleType("harbor.agents.base"),
        "harbor.environments": ModuleType("harbor.environments"),
        "harbor.environments.base": ModuleType("harbor.environments.base"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.agent": ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.base"].BaseAgent = type("BaseAgent", (), {})
    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.models.agent.context"].AgentContext = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module("recipes.tttd.examples.guidance_ttt.harness.harbor_agent")


class _Environment:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def test_harbor_agent_runs_one_committed_step_and_submits_the_best_candidate(tmp_path, monkeypatch) -> None:
    agent_module = _harbor_agent_module(monkeypatch)

    scores = iter([1_000_000.0, 2_000_000.0])

    class _Scorer:
        def __init__(self, *args, **kwargs) -> None:
            self.args = (args, kwargs)

        def __call__(self, code: str) -> VerificationResult:
            score = next(scores)
            return VerificationResult(score, score, True, "valid", "accepted", {"code": code})

    monkeypatch.setattr(agent_module, "JudgeScorer", _Scorer)

    checkpoint_root = tmp_path / "checkpoints" / "megatron"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("1")

    class _Bridge:
        def __init__(self, *args, **kwargs) -> None:
            self.args = (args, kwargs)

        def start_step(self) -> int:
            return 0

        def wait_for_step(self, *, expected_completed_steps, expected_rollout_id):
            assert (expected_completed_steps, expected_rollout_id) == (1, 0)
            return {
                "ok": True,
                "phase": "serving",
                "completed_train_steps": 1,
                "last_train_rollout_id": 0,
                "runtime_load_id": "guidance:1",
                "last_train_metrics": {
                    "train/grad_norm": 0.5,
                    "train/global_batch_size": 2,
                    "train/lora_trainable_parameters": 1024,
                    "train/lora_base_trainable_parameters": 0,
                    "train/lora_b_nonzero": 8,
                    "train/lora_b_l1": 0.25,
                },
            }

    monkeypatch.setattr(agent_module, "RayTrainingBridge", _Bridge)
    monkeypatch.setattr(agent_module, "OpenAICompatibleExecutionClient", lambda _backend: _ExecutionClient())
    for name, value in {
        "SCENARIO": "guidance-smoke",
        "GROUPS_PER_STEP": 1,
        "ROLLOUTS_PER_GROUP": 2,
        "STEPS": 1,
        "MAX_TOKENS": 64,
        "MAX_WORKERS": 2,
        "SEED_LIBRARY": SEED,
        "TASK_CONTRACT": CONTRACT_PATH,
        "RUN_DIR": tmp_path / "guidance-run",
        "STATE_DIR": tmp_path,
    }.items():
        monkeypatch.setattr(agent_module, name, value)

    agent = object.__new__(agent_module.HarborAgent)
    agent._client = _ReefClient()
    agent.model_name = "Qwen/Qwen3-8B"
    agent.logger = logging.getLogger("guidance-test")
    environment = _Environment()
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run(INSTRUCTION, environment, context))

    assert environment.commands and "#include <iostream>" in environment.commands[0]
    assert "/workspace/solution.cpp" in environment.commands[0]
    reef = context.metadata["reef"]
    assert reef["agent_record_ids"] == ["receipt-1"]
    assert (reef["start_step"], reef["next_step"]) == (0, 1)
    assert reef["step_summaries"][0]["sampled_rollouts"] == 2
    assert reef["step_summaries"][0]["well_formed_guidance"] == 1
    resumed = GuidanceRunStateStore(
        tmp_path / "guidance-run", _identity(groups_per_step=1, rollouts_per_group=2, guidance_max_tokens=64)
    ).load()
    assert resumed is not None and resumed["next_step"] == 1
