"""Executable host-side contracts for the shipped REEF Eval/Harbor examples."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
# An example lives beside its method under recipes/<method>/examples/; the
# record-only ``basic`` example has no method package.
EXAMPLE_DIRS = {
    "basic": ROOT / "recipes" / "basic",
    "sao": ROOT / "recipes" / "sao" / "examples" / "sao",
    "tttd": ROOT / "recipes" / "tttd" / "examples" / "tttd",
    "guidance_ttt": ROOT / "recipes" / "tttd" / "examples" / "guidance_ttt",
}
REEF_EVAL_EXAMPLE_DIRS = (
    *EXAMPLE_DIRS.values(),
    ROOT / "recipes" / "harness_evolve" / "examples" / "skillclaw",
    ROOT / "recipes" / "openclawrl" / "examples" / "openclawrl",
)


@pytest.mark.unit
@pytest.mark.parametrize("example_dir", REEF_EVAL_EXAMPLE_DIRS, ids=lambda path: path.name)
def test_reef_eval_examples_declare_their_runtime_dependency(example_dir: Path) -> None:
    config = tomllib.loads((example_dir / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    source = config["tool"]["uv"]["sources"]["reef-eval"]

    assert project["requires-python"] == ">=3.12"
    assert "reef-eval[harbor]" in project["dependencies"]
    assert source["editable"] is True
    assert (example_dir / source["path"]).resolve() == ROOT / "reef-eval"


def _install_harbor_protocol(monkeypatch) -> None:
    """Supply only the Harbor types used for annotations and inheritance."""

    class BaseAgent:
        pass

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
    modules["harbor.agents.base"].BaseAgent = BaseAgent
    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.models.agent.context"].AgentContext = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_harness(monkeypatch, example: str):
    _install_harbor_protocol(monkeypatch)
    package_name = f"_reef_{example}_example_harness"
    package_dir = EXAMPLE_DIRS[example] / "harness"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.agent"), importlib.import_module(f"{package_name}.report")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("example", "recipe", "task_name", "expected_tasks", "expected_tags", "expected_lab", "expected_model"),
    [
        # Every example names its model as a literal now, so each row pins its own.
        ("basic", "basic", None, ("harbor",), (None,), "work/lab", "gpt-4o"),
        (
            "sao",
            "sao",
            None,
            ("harbor/imo-4", "harbor/imo-8", "harbor/imo-12"),
            ({"position": 0}, {"position": 1}, {"position": 2}),
            "work/lab",
            "reef",
        ),
        (
            "guidance_ttt",
            "tttd",
            None,
            ("harbor/polyomino_packing",),
            (None,),
            "work/polyomino_packing/lab",
            "Qwen/Qwen3-8B",
        ),
        (
            "tttd",
            "tttd",
            "erdos_min_overlap",
            ("harbor/erdos_min_overlap",),
            (None,),
            "work/erdos_min_overlap/lab",
            "Qwen/Qwen3-8B",
        ),
        (
            "tttd",
            "tttd",
            "circle_packing_26",
            ("harbor/circle_packing_26",),
            (None,),
            "work/circle_packing_26/lab",
            "Qwen/Qwen3-8B",
        ),
        (
            "tttd",
            "tttd",
            "circle_packing_32",
            ("harbor/circle_packing_32",),
            (None,),
            "work/circle_packing_32/lab",
            "Qwen/Qwen3-8B",
        ),
    ],
)
def test_reef_eval_entrypoint_dispatches_the_documented_workload(
    monkeypatch,
    capsys,
    example,
    recipe,
    task_name,
    expected_tasks,
    expected_tags,
    expected_lab,
    expected_model,
) -> None:
    calls = []

    class Lab:
        def __init__(self, path):
            self.path = Path(path)

        async def run(self, task, agent, tags=None):
            calls.append((self.path, Path(task), agent, tags))
            return SimpleNamespace(rewards={"reward": 1.0}, tags={}, uri="file:///trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    for key, value in {
        "REEF_SERVICE_URL": "http://127.0.0.1:8900",
        "REEF_SCENARIO": f"{example}-host-test",
        "REEF_RECIPE": recipe,
        "REEF_TOKEN": "reef-local",
        "REEF_MODEL": "reef-test-model",
        "REEF_TIMEOUT_S": "30",
        "SAO_ROLLOUTS": "2",
        "SAO_MAX_TOKENS": "32",
        "JUDGE_URL": "http://127.0.0.1:8082",
    }.items():
        monkeypatch.setenv(key, value)

    if task_name is not None:
        monkeypatch.setenv("TTTD_TASK", task_name)

    runpy.run_path(str(EXAMPLE_DIRS[example] / "run.py"))

    example_root = EXAMPLE_DIRS[example]
    assert [str(task.relative_to(example_root)) for _, task, _, _ in calls] == list(expected_tasks)
    assert [tags for _, _, _, tags in calls] == list(expected_tags)
    assert all(agent == {"name": "harness:HarborAgent", "model_name": expected_model} for _, _, agent, _ in calls)
    assert all(lab == example_root / expected_lab for lab, _, _, _ in calls)
    assert "reward" in capsys.readouterr().out


@pytest.mark.unit
def test_tttd_entrypoint_rejects_an_unknown_task(monkeypatch) -> None:
    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = object
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    monkeypatch.setenv("TTTD_TASK", "not-a-task")

    with pytest.raises(SystemExit, match=r"unknown TTTD_TASK.*circle_packing_32"):
        runpy.run_path(str(EXAMPLE_DIRS["tttd"] / "run.py"))


@pytest.mark.unit
def test_tttd_entrypoint_fails_when_harbor_reports_an_error(monkeypatch) -> None:
    class Lab:
        def __init__(self, _path):
            pass

        async def run(self, _task, _agent):
            return SimpleNamespace(rewards={}, tags={"error": "environment failed"}, uri="file:///failed-trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)

    with pytest.raises(RuntimeError, match="Harbor trial failed: environment failed"):
        runpy.run_path(str(EXAMPLE_DIRS["tttd"] / "run.py"))


class _Environment:
    def __init__(self) -> None:
        self.commands = []

    async def exec(self, command):
        self.commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


@pytest.mark.unit
def test_basic_harness_runs_inference_and_preserves_its_receipt(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "basic")

    async def call_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(agent_module.asyncio, "to_thread", call_inline)
    agent = object.__new__(agent_module.HarborAgent)
    agent._ask_reef = lambda _instruction: (
        {
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            "metadata": {"weight_version": "basic:0"},
        },
        "receipt-basic",
    )
    environment = _Environment()
    context = SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)

    asyncio.run(agent.run("answer the task", environment, context))

    assert environment.commands and "42" in environment.commands[0]
    assert context.metadata["reef"] == {
        "agent_record_ids": ["receipt-basic"],
        "weight_versions": ["basic:0"],
    }
    assert (context.n_input_tokens, context.n_output_tokens) == (3, 1)


@pytest.mark.unit
def test_sao_harness_scores_and_reports_every_rollout(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "sao")
    monkeypatch.setattr(agent_module, "ROLLOUTS", 2)

    async def call_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(agent_module.asyncio, "to_thread", call_inline)

    class Client:
        def __init__(self):
            self.reports = []

        def report(self, scenario, payload, *, recipe=None):
            self.reports.append((scenario, payload, recipe))

    responses = iter(
        [
            ({"choices": [{"message": {"content": r"\boxed{2^{u-2}}"}}]}, "receipt-correct"),
            (
                {
                    "choices": [{"message": {"content": r"\boxed{0}"}}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
                },
                "receipt-wrong",
            ),
        ]
    )
    agent = object.__new__(agent_module.HarborAgent)
    agent._rollouts = 2
    agent._client = Client()
    agent._scenario = "sao-host-test"
    agent._recipe = "sao"
    agent._ask_reef = lambda _instruction: next(responses)
    environment = _Environment()
    context = SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)

    asyncio.run(agent.run(r"Let $u \ge 2$ be the problem prefix", environment, context))

    assert [report[1]["score"] for report in agent._client.reports] == [1.0, 0.0]
    assert [report[1]["references"] for report in agent._client.reports] == [
        ["receipt-correct"],
        ["receipt-wrong"],
    ]
    assert context.metadata["reef"]["agent_record_ids"] == ["receipt-correct", "receipt-wrong"]
    assert (context.n_input_tokens, context.n_output_tokens) == (9, 2)


@pytest.mark.unit
def test_sao_harness_accepts_each_bundled_symbolic_gold_answer(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "sao")

    for gold in agent_module.GOLD_ANSWERS.values():
        assert agent_module.answers_equal(gold, gold.strip("$"))


@pytest.mark.unit
@pytest.mark.parametrize("example", ["basic", "sao"])
def test_harbor_result_posts_verifier_reward_to_its_exact_receipts(monkeypatch, example) -> None:
    _, report_module = _load_harness(monkeypatch, example)

    class Client:
        def __init__(self):
            self.call = None

        def report(self, scenario, payload, *, recipe=None):
            self.call = (scenario, payload, recipe)
            return {"accepted": True}

    result = {
        "id": "trial-7",
        "task_name": "smoke",
        "agent_result": {"metadata": {"reef": {"agent_record_ids": ["receipt-7"]}}},
        "verifier_result": {"rewards": {"reward": 0.75}},
    }
    client = Client()

    assert report_module.post_report(result, client=client, scenario="example-host-test", recipe=example) == {
        "accepted": True
    }
    scenario, payload, recipe = client.call
    assert (scenario, recipe) == ("example-host-test", example)
    assert payload["score"] == 0.75
    assert payload["references"] == ["receipt-7"]
