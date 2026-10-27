"""Guarantees of the tutorials/evolve-your-harness example, hermetic: the served model is stubbed, episodes run a
fake pi binary, and the boot tests read the serve files the way run.sh and ``reef serve`` do.

This file covers what the tutorial owns: its grader, its two methods (``RulesMethod`` on pi, ``NativeMethod`` on the
native loop), the serve files and the replay page.
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from tests.support._trajectories import recorded_trajectory
from tests.support.config_helpers import load_harness_deployment as load_config
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_backend

from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import NoEvaluationPolicy
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.episodes.trajectory import final_assistant_text
from reef.harness.tree import HarnessTree, validate_patch
from reef.inference.http import InferenceProxyRuntime
from reef.recipe import load_recipe_config
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.service.deploy.service_config import service_config_from_mapping
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.backend import ComponentReleases
from reef.train.trainer import Trainer
from reef.train.types import JobData

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "evolve-your-harness"
REPO_ROOT = EXAMPLE_DIR.parents[1]
PI_SEED = HarnessTree.read(EXAMPLE_DIR / "seeds" / "pi")
NATIVE_SEED = HarnessTree.read(EXAMPLE_DIR / "seeds" / "native")
#: A failing trace with its reply and its report's feedback, and a passing one.
FAILING = recorded_trajectory(
    "a1",
    {
        "messages": [{"role": "user", "content": "[fib] compute fib(90)"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "about 2.88e18"}}]},
    },
    0.0,
    "not exact",
)
PASSING = recorded_trajectory("a2", {"messages": [{"role": "user", "content": "[csv] compute the median"}]}, 1.0)


def _method(monkeypatch: pytest.MonkeyPatch, module: str) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    # Every example names its package ``harness``: drop any sibling example's cached import so this file's
    # syspath entry wins.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        monkeypatch.delitem(sys.modules, name)
    return importlib.import_module(f"harness.{module}")


@pytest.fixture
def evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "evolution")


@pytest.fixture
def native_evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "native_evolution")


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "plugins")


class Served(ModelBinding):
    """The served model: it answers ``reply``, or raises it, and keeps every prompt it was asked."""

    def __init__(self, reply: str | Exception) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="served", api_key="dummy")
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params) -> str:
        self.prompts.append(messages[-1]["content"])
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def method_backend(tmp_path: Path, method, served: Served, **fields):
    """A backend that runs ``method`` with ``served`` as its served model; ``fields`` set the recipe."""
    backend, _ = harness_backend(tmp_path, method, **fields)
    backend.models = ModelBindings(served=served)
    return backend


def episode(trajectory: tuple[dict, ...]) -> EpisodeResult:
    return EpisodeResult(exit_code=0, stdout="", stderr="", trajectory=trajectory, residue=())


def pi_message(text: str) -> dict:
    return {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def releases_for(recipe: HarnessEvolutionRecipe) -> ComponentReleases:
    return ComponentReleases(TreeReleases(recipe.seed), COMPONENT)


# -- the grader: exact last-line answers ---------------------------------------------------------


def test_evaluate_grades_exact_final_lines(evolution) -> None:
    prefix, answer = next(iter(evolution.ANSWERS.items()))
    task = f"{prefix} the task"
    assert evolution.evaluate(task, episode((pi_message(f"Working...\n\n{answer}"),))) == 1.0
    assert evolution.evaluate(task, episode(({"role": "assistant", "content": answer},))) == 1.0


def test_evaluate_grades_non_exact_as_zero(evolution) -> None:
    prefix, answer = next(iter(evolution.ANSWERS.items()))
    task = f"{prefix} the task"
    assert evolution.evaluate(task, episode((pi_message(f"{answer}!"),))) == 0.0
    assert evolution.evaluate(task, episode((pi_message(f"The answer is {answer}"),))) == 0.0
    assert evolution.evaluate(task, episode(())) == 0.0
    assert evolution.evaluate("no such prefix", episode((pi_message(answer),))) == 0.0


def test_the_tutorial_grader_keeps_its_own_answers(evolution) -> None:
    """The tutorial's grader knows the three arithmetic answers its deployments run, and no other task; it reads
    the trajectory through the built-in reader."""
    assert set(evolution.ANSWERS) == {"[sieve]", "[fib]", "[csv]"}
    assert evolution.evaluate("[health] Run `echo reef-ok`.", episode((pi_message("reef-ok"),))) == 0.0
    assert evolution.evaluate("[sieve] count the primes", episode((pi_message("Sieving...\n\n9592"),))) == 1.0
    assert final_assistant_text([pi_message("a"), {"role": "user", "content": "b"}]) == "a"


# -- the pi variant's method: one call rewrites rules.md -------------------------------------------


def test_the_rules_method_rewrites_rules_md_from_the_failing_traces(tmp_path, plugins) -> None:
    """The model sees each failing trace with its reply and feedback, and the current rules; the passing trace
    stays out. Its fenced answer becomes rules.md, which wins the evaluation (the scorer prefers the marker)."""
    served = Served("The new rules:\n```markdown\nEnd every reply with the answer alone. marker\n```\n")
    backend = method_backend(tmp_path, plugins.RulesMethod(), served, seed=PI_SEED)
    result = backend_step(backend, JobData("demo:1", (FAILING, PASSING)), backend.initial_state())
    [prompt] = served.prompts
    assert "[fib] compute fib(90)" in prompt and "about 2.88e18" in prompt and "not exact" in prompt
    assert "[csv] compute the median" not in prompt and PI_SEED.files["rules.md"] in prompt
    assert result.metrics["patch"] == {"rules.md": "End every reply with the answer alone. marker\n"}
    assert result.metrics["method_summary"] == "rewrote rules.md from 1 failing trace(s)"
    assert result.metrics["method_notes"] == {"failing_traces": 1} and result.metrics["selected"] is True


def test_the_rules_method_answers_a_request_and_skips_when_it_has_nothing_to_learn(tmp_path, plugins) -> None:
    served = Served("```\nAnswer in English.\n```")
    backend = method_backend(tmp_path, plugins.RulesMethod(), served, seed=PI_SEED)
    request = TrainingRequest("Always answer in English.", "s", "base", "ask-1")
    answered = backend_step(backend, JobData("demo:1", (), request=request), backend.initial_state())
    assert "Always answer in English." in served.prompts[-1]
    assert answered.metrics["method_summary"] == "rewrote rules.md from 0 failing trace(s) and the request"
    # A passing trace alone teaches nothing: no model call, and the step skips with the method's reason.
    passing = backend_step(backend, JobData("demo:2", (PASSING,)), answered.state)
    assert len(served.prompts) == 1 and passing.metrics["skipped"] == "the method left the head tree unchanged"
    assert passing.metrics["method_summary"] == "no failing trace and no request to learn from"
    backend.models = ModelBindings(served=Served(ModelBindingError("model endpoint unreachable: refused")))
    failed = backend_step(backend, JobData("demo:3", (FAILING,)), passing.state)
    assert failed.metrics["method_summary"] == "the model call failed: model endpoint unreachable: refused"
    assert failed.metrics["skipped"] == "the method left the head tree unchanged"


# -- the serve files boot the recipe --------------------------------------------------------------


def serve_file_base(filename: str) -> Path:
    """The directory a serve file's relative paths resolve from: run.sh starts Reef in the tutorial, and the
    long-running deployment runs from the repository root."""
    return REPO_ROOT if filename == "deployment.yaml" else EXAMPLE_DIR


@pytest.mark.parametrize("filename", ["serve.yaml", "serve-native.yaml", "deployment.yaml"])
def test_materializer_preserves_executor_profiles_and_recipe_selection(monkeypatch, tmp_path, filename):
    materializer = _method(monkeypatch, "materialize_recipe")
    config = yaml.safe_load((EXAMPLE_DIR / "configs" / filename).read_text())
    config["executors"] = {"cpu-pool": {"backend": "mp", "workers": 2, "resources": {"cpus_per_worker": 2}}}
    config["execution"] = {"evolution": "cpu-pool"}
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    settings = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert settings["execution"] == config["execution"]
    assert settings["executors"] == config["executors"]
    assert "reef" not in settings and "services" not in settings
    tasks = config["recipe"]["config"]["evolution"]["evaluation"]["tasks"]
    assert json.loads((tmp_path / "work/tasks.json").read_text()) == tasks
    # Boot the real recipe; a retained selector without its profile would fail here.
    monkeypatch.chdir(serve_file_base(filename))
    recipe = HarnessEvolutionRecipe.from_environment(
        {},
        config=recipe_config_from_file_layout(settings),
        runtime=InferenceProxyRuntime(model_path="test", base_url="http://unused", api_key="dummy"),
    )
    assert recipe.worker_executor.backend == "mp"
    assert recipe.worker_executor.workers == 2
    assert recipe.worker_executor.resources.cpus_per_worker == 2


def test_materializer_accepts_config_without_execution_sections(monkeypatch, tmp_path):
    materializer = _method(monkeypatch, "materialize_recipe")
    config = yaml.safe_load((EXAMPLE_DIR / "configs/serve.yaml").read_text())
    config.pop("execution")
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    result = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert set(result) == {"recipe", "inference"}
    assert result["recipe"]["config"]["evolution"]["batch_size"] == 1


def write_recipe_file(path: Path, config: dict, evolution: dict) -> None:
    """Write a loaded harness deployment's recipe sections as a recipe file, as run.sh does."""
    path.write_text(
        yaml.safe_dump(
            {
                "recipe": {
                    "implementation": config["implementation"],
                    "config": {**config["data"], "evolution": evolution},
                },
                "execution": config["execution"],
            }
        )
    )


def boot_serve_file(tmp_path: Path, monkeypatch, filename: str, **evolution) -> HarnessEvolutionRecipe:
    """The recipe a serve file boots, through reef's config loader and the service's upstream runtime.

    Its state goes under ``tmp_path``; ``evolution`` sets further ``evolution`` keys.
    """
    from reef.service.assembly import _upstream_runtime

    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / filename)
    materialized = tmp_path / "harness_evolve.yaml"
    state = {"state_dir": str(tmp_path / "harness")}
    write_recipe_file(materialized, config, {**config["evolution"], **state, **evolution})
    monkeypatch.chdir(serve_file_base(filename))
    service = service_config_from_mapping(config)
    return HarnessEvolutionRecipe.from_environment(
        {}, config=load_recipe_config(materialized), runtime=_upstream_runtime(service)
    )


def test_the_serve_file_boots_the_recipe_through_from_environment(tmp_path, monkeypatch, evolution) -> None:
    """The run.sh contract, hermetic: interpolate serve.yaml through reef's config loader, materialize the recipe
    sections as a named config, and boot the recipe (seed validation included) with a fake binary."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve.yaml")
    # The deployment names the upstream once; the service builds the recipe's runtime from it.
    service = service_config_from_mapping(config)
    assert (service.upstream_url, service.upstream_api_key, service.upstream_model) == (
        "http://127.0.0.1:8000",
        "dummy",
        "qwen3-8b",
    )
    # serve.yaml names no binary; this run has no pi on PATH.
    built = boot_serve_file(tmp_path, monkeypatch, "serve.yaml", binary=str(tmp_path / "fake-pi"))
    assert built.evolution.adapter == "pi" and built.evolution.binary == str(tmp_path / "fake-pi")
    tasks = built.configured_tasks()
    assert len(tasks) == 3
    assert all(any(task.startswith(prefix) for prefix in evolution.ANSWERS) for task in tasks)
    assert (built.batch_size, built.training_mode, built.evaluation.policy) == (1, "auto", "compare")
    assert type(built.method).__name__ == "RulesMethod" and type(built.scorer).__name__ == "AnswerScorer"
    # The seed is the starter rules and carries no provider; the binding comes from the runtime.
    assert built.seed == PI_SEED and set(PI_SEED.files) == {"rules.md"}
    assert not any("upstream" in text or "8000" in text for text in PI_SEED.files.values())
    binding = built.model_binding()
    assert (binding.base_url, binding.model, binding.api_key, binding.api) == (
        "http://127.0.0.1:8000",
        "qwen3-8b",
        "dummy",
        "openai",
    )
    assert list(built.model_bindings()) == ["served"]
    trainer = built.build("demo", SQLiteRecordStore(), releases=releases_for(built))
    assert isinstance(trainer, Trainer)
    trainer.close()


def test_the_native_serve_file_boots_the_recipe_with_the_shipped_seed(tmp_path, monkeypatch) -> None:
    """The run.sh native contract, hermetic: the native serve file materializes like serve.yaml and boots with
    the loop's own tools, hook and graph as the seed directory holds them."""
    built = boot_serve_file(tmp_path, monkeypatch, "serve-native.yaml")
    assert built.evolution.adapter == "native" and built.evolution.binary is None
    assert type(built.method).__name__ == "NativeMethod" and type(built.scorer).__name__ == "NativeAnswerScorer"
    # The same three tasks as the pi variant, so the two runs are comparable.
    pi_tasks = load_config(EXAMPLE_DIR / "configs" / "serve.yaml")["evolution"]["evaluation"]["tasks"]
    assert built.configured_tasks() == tuple(pi_tasks)
    assert (built.batch_size, built.training_mode) == (1, "auto")
    assert built.seed == NATIVE_SEED
    assert {
        "native/tools/read_file.py",
        "native/tools/write_file.py",
        "native/tools/run_bash.py",
        "native/tools/execute.py",
        "native/hooks/loop_guard.py",
        "native/graphs/main.json",
        "skills/answer-style/SKILL.md",
    } == set(NATIVE_SEED.files)
    # A fresh scenario's base release is the seed's source tree, which names no endpoint.
    assert built.base_artifact_files() == dict(NATIVE_SEED.files)
    assert not any("upstream" in text for text in NATIVE_SEED.files.values())
    assert built.build_surface("demo").harness.served_model == "qwen3-8b"
    empty = tmp_path / "empty-seed"
    empty.mkdir()
    assert boot_serve_file(tmp_path, monkeypatch, "serve-native.yaml", seed=str(empty)).base_artifact_files() is None


@pytest.mark.parametrize("model_id", ["provider/model-a", "provider/model-b"])
def test_deployment_yaml_names_directories_that_exist_and_boots_its_named_recipe(
    monkeypatch, tmp_path, model_id
) -> None:
    """The README deployment: ``reef.recipe: deployment`` is read back from the directory the service's own env
    names; a stale directory name here fails at boot, so the file's own paths are checked against the checkout."""
    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", model_id)
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(REPO_ROOT))
    monkeypatch.chdir(REPO_ROOT)
    path = EXAMPLE_DIR / "configs" / "deployment.yaml"
    config = load_config(path)
    for setting in (config["evolution"]["method"], config["evolution"]["evaluation"]["scorer"]):
        module, _, name = setting.partition(":")
        assert callable(getattr(importlib.import_module(module), name))
    assert (REPO_ROOT / config["evolution"]["seed"]).is_dir()
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith("tutorials/evolve-your-harness/")
    assert config["run_dir"].startswith("tutorials/evolve-your-harness/")
    service = service_config_from_mapping(config)
    monkeypatch.delenv("REEF_UPSTREAM_MODEL")  # Recipe construction uses the resolved runtime, not the environment.
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(EXAMPLE_DIR / "configs")},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, HarnessEvolutionRecipe) and built.evolution.adapter == "pi"
    assert service.upstream_model == model_id
    assert built.model_binding().model == model_id
    assert built.build_surface("demo").harness.served_model == model_id
    # The person asks while it keeps learning from failures; the method reads the request beside the traces.
    assert built.training_mode == "hybrid" and type(built.method).__name__ == "RulesMethod"
    records = SQLiteRecordStore()
    local = replace(built.evolution, binary=str(tmp_path / "fake-pi"), state_dir=str(tmp_path / "harness"))
    trainer = replace(built, evolution=local).build("demo", records, releases=releases_for(built))
    assert trainer.pipeline.training_mode == "hybrid"
    # Each report is one job's data: the deployment sets evolution.batch_size 1, the default job_samples.
    assert trainer.job_samples == 1
    trainer.close()
    records.close()


def test_deployment_yaml_holds_evolved_extensions_for_review(monkeypatch) -> None:
    """The long-running deployment takes requests and holds a win that touches a code extension for a promote;
    the two demo files run in auto, where a request is refused, and hold nothing back."""
    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "provider/model-a")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(REPO_ROOT))
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(EXAMPLE_DIR / "configs" / "deployment.yaml")
    assert config["data"]["training_mode"] == "hybrid"
    assert config["evolution"]["review_kinds"] == ["code_extension"]
    for name in ("serve", "serve-native"):
        demo = load_config(EXAMPLE_DIR / "configs" / f"{name}.yaml")
        assert demo["data"]["training_mode"] == "auto" and "review_kinds" not in demo["evolution"]
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(EXAMPLE_DIR / "configs")},
        default_runtime=_upstream_runtime(service_config_from_mapping(config)),
    )
    assert built.evolution.review_kinds == ("code_extension",)
    # The seed is the starter tree alone: Reef adds no entries of its own to it.
    assert built.seed == PI_SEED


# -- the native variant: skills, tools, hooks, agents and the graph -------------------------------

TOOL = {
    "description": "Run python.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    "code": "def run(args, workdir):\n    return 'ok'\n",
}
HOOK = {"event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"}


def native_proposal(kind: str, name: str, **config) -> str:
    return json.dumps({"id": name, "name": kind, "config": {"name": name, **config}})


def native_message(text: str) -> dict:
    return {
        "type": "assistant/message",
        "seq": 4,
        "time": 0,
        "data": {"content": text, "tool_calls": [], "finish": "stop"},
    }


def test_the_native_method_writes_one_change_of_the_loop(tmp_path, plugins) -> None:
    """One failing trace, one model call, one native file; a reply that holds no change leaves the tree."""
    served = Served(native_proposal("native_tool", "run_python", **TOOL))
    backend = method_backend(
        tmp_path, plugins.NativeMethod(), served, seed=NATIVE_SEED, adapter="native", evaluation={"policy": "none"}
    )
    result = backend_step(
        backend, JobData("demo:1", (FAILING, PASSING)), backend.initial_state(), NoEvaluationPolicy()
    )
    [prompt] = served.prompts
    assert "about 2.88e18" in prompt and "[csv] compute the median" not in prompt
    assert '"native/graphs/main.json"' in prompt
    assert list(result.metrics["patch"]) == ["native/tools/run_python.py"] and result.metrics["selected"] is True
    assert result.metrics["method_summary"] == "wrote native/tools/run_python.py from 1 failing trace(s)"
    backend.models = ModelBindings(served=Served("no json here"))
    unusable = backend_step(backend, JobData("demo:2", (FAILING,)), result.state, NoEvaluationPolicy())
    assert unusable.metrics["method_summary"] == "the reply held no usable change"
    assert unusable.metrics["skipped"] == "the method left the head tree unchanged"


def test_native_changes_are_files_the_native_adapter_takes(native_evolution) -> None:
    native = get_adapter("native")
    tool = native_evolution.patch_of(NATIVE_SEED, native_proposal("native_tool", "run_python", **TOOL))
    assert list(tool.changes) == ["native/tools/run_python.py"]
    module = tool.changes["native/tools/run_python.py"]
    assert module.startswith(TOOL["code"].rstrip()) and "NAME = 'run_python'" in module
    assert "DESCRIPTION = 'Run python.'" in module
    hook = native_evolution.patch_of(NATIVE_SEED, native_proposal("native_hook", "answer_last", **HOOK))
    assert list(hook.changes) == ["native/hooks/answer_last.py"]
    assert "EVENT = 'pre_step'" in hook.changes["native/hooks/answer_last.py"]
    # A name a file of the same kind holds rewrites that file: the shipped seed tools are addressable by name.
    same = native_evolution.patch_of(NATIVE_SEED, native_proposal("native_tool", "read_file", **TOOL))
    assert list(same.changes) == ["native/tools/read_file.py"] and "native/tools/read_file.py" in NATIVE_SEED.files
    skill = native_evolution.patch_of(
        NATIVE_SEED, native_proposal("skill", "answer-style", text="# answer-style\n\nBe exact.\n")
    )
    assert skill.changes == {"skills/answer-style/SKILL.md": "# answer-style\n\nBe exact.\n"}
    # qwen2.5:7b was measured swapping the kind and the id; the config name settles it.
    swapped = json.dumps({"id": "native_tool", "name": "median", "config": {"name": "median", **TOOL}})
    assert list(native_evolution.patch_of(NATIVE_SEED, swapped).changes) == ["native/tools/median.py"]
    for patch in (tool, hook, same, skill):
        assert validate_patch(NATIVE_SEED, patch, native).passed, validate_patch(NATIVE_SEED, patch, native).problems


def test_native_replies_of_a_malformed_shape_hold_no_change(native_evolution) -> None:
    for reply in (
        native_proposal("rules", "answer-style", text="x"),
        native_proposal("native_hook", "h", event="on_exit", code="x = 1"),
        native_proposal("native_hook", "h", event="pre_step", code="  "),
        native_proposal("native_tool", "t", description="d", parameters="not a schema", code="x = 1"),
        native_proposal("native_tool", "t", description="", parameters={}, code="x = 1"),
        json.dumps({"id": "t", "name": "native_tool", "config": {"name": "other", **TOOL}}),
        "no json here",
    ):
        assert native_evolution.patch_of(NATIVE_SEED, reply) is None


def test_native_hooks_take_every_event_the_shape_offers(native_evolution) -> None:
    for event in native_evolution.EVENTS:
        reply = native_proposal(
            "native_hook", "guard", event=event, code="def listen(payload, next):\n    return next()\n"
        )
        patch = native_evolution.patch_of(NATIVE_SEED, reply)
        assert f"EVENT = {event!r}" in patch.changes["native/hooks/guard.py"]


def test_a_native_agent_change_routes_the_main_graph_through_the_agent(native_evolution) -> None:
    """An agent alone can never win: only a subagent stage runs one, and its text comes back as a user message the
    grader never reads. The change carries the graph edit: ask the agent, then a model stage answers."""
    native = get_adapter("native")
    reply = native_proposal("native_agent", "helper", prompt="Solve the task alone.", tools=["read_file"])
    patch = native_evolution.patch_of(NATIVE_SEED, reply)
    assert sorted(patch.changes) == ["native/agents/helper.json", "native/graphs/main.json"]
    assert json.loads(patch.changes["native/agents/helper.json"])["prompt"] == "Solve the task alone."
    graph = json.loads(patch.changes["native/graphs/main.json"])
    assert graph["stages"]["ask-helper"] == {"kind": "subagent", "agent": "helper"}
    assert graph["stages"]["answer-helper"] == {"kind": "model"}
    assert [(edge["when"], edge["to"]) for edge in graph["edges"] if edge["from"] == "ask-helper"] == [
        ("completed", "answer-helper"),
        ("gave_up", "answer-helper"),
        ("budget", "answer-helper"),
        ("ask", "answer-helper"),
    ]
    assert validate_patch(NATIVE_SEED, patch, native).passed, validate_patch(NATIVE_SEED, patch, native).problems
    # The agent stands alone when the graph already runs it, and when the stage names it would add are taken.
    routed = NATIVE_SEED.apply(patch)
    assert list(native_evolution.patch_of(routed, reply).changes) == ["native/agents/helper.json"]
    taken = {**graph, "stages": {**json.loads(NATIVE_SEED.files["native/graphs/main.json"])["stages"]}}
    taken["stages"]["ask-helper"] = {"kind": "message", "text": "hi"}
    occupied = HarnessTree({**NATIVE_SEED.files, "native/graphs/main.json": json.dumps(taken)})
    assert list(native_evolution.patch_of(occupied, reply).changes) == ["native/agents/helper.json"]


def test_native_evaluate_reads_the_last_assistant_message_with_content(native_evolution) -> None:
    task = "[sieve] count the primes below 100000"
    call = {
        "type": "assistant/message",
        "seq": 2,
        "time": 0,
        "data": {"content": "", "tool_calls": [{}], "finish": "x"},
    }
    assert native_evolution.evaluate(task, episode((call, native_message("Sieving...\n\n9592")))) == 1.0
    # A tool-call message carries no text; the last text answer is the one graded.
    assert native_evolution.evaluate(task, episode((native_message("9592"), call))) == 1.0
    assert native_evolution.evaluate(task, episode((native_message("The count is 9592"),))) == 0.0
    assert native_evolution.evaluate(task, episode(())) == 0.0


# -- the replay page ------------------------------------------------------------------------------


def session_events(session: str, release: str, stages, tool: str | None = None) -> list[dict]:
    events = [
        {
            "type": "session",
            "seq": 0,
            "time": 1000,
            "data": {"session": session, "release_id": release, "model": "m", "tools": ["read_file"], "agent": "root"},
        },
        {"type": "turn/start", "seq": 1, "time": 1001, "data": {"turn": 1, "prompt": "go"}},
    ]
    seq = 2
    for index, stage in enumerate(stages):
        events.append(
            {
                "type": "stage/enter",
                "seq": seq,
                "time": 1002 + index,
                "data": {"step": index, "stage": stage, "kind": "model"},
            }
        )
        seq += 1
        if tool and index == 0:
            events.append(
                {
                    "type": "tool/call",
                    "seq": seq,
                    "time": 1002 + index,
                    "data": {"step": index, "name": tool, "call_id": "c", "arguments": "{}"},
                }
            )
            seq += 1
        events.append(
            {
                "type": "stage/exit",
                "seq": seq,
                "time": 1003 + index,
                "data": {"step": index, "stage": stage, "outcome": "text", "to": "done"},
            }
        )
        seq += 1
    events.append({"type": "turn/end", "seq": seq, "time": 1100, "data": {"turn": 1, "reason": {"kind": "completed"}}})
    return events


def write_commits(directory: Path, *commits: dict) -> None:
    """Replace the commits of one scenario database with the given JSON objects, at steps 1, 2, ..."""
    database = directory / "x.sqlite3"
    SQLiteRecordStore(database).close()
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("DELETE FROM scenario_commit")
        connection.executemany(
            "INSERT INTO scenario_commit (step, commit_json) VALUES (?, ?)",
            [(step, json.dumps(commit)) for step, commit in enumerate(commits, start=1)],
        )


def test_replay_collects_a_run_and_renders_one_self_contained_page(tmp_path: Path, monkeypatch) -> None:
    replay = _method(monkeypatch, "replay")
    work = tmp_path / "work"
    native = work / "tree" / "native"
    (native / "sessions" / "s1").mkdir(parents=True)
    (native / "sessions" / "s2").mkdir(parents=True)
    (work / "agent-record").mkdir()
    commit = {
        "recorded_at": 1050.0,
        "operation": "training",
        "artifact_ref": {"release_id": "r2", "parent_release_id": "r1"},
        "metrics": {
            "steps": 1,
            "selected": True,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "candidate_score": 2.0,
            "current_score": 0.0,
            "patch": {"rules.md": "one line\n"},
            "proposal": {"id": "p1", "session": "s1", "reason": "because"},
            "selection": {"reason": "candidate won 2 task pairings and lost 0"},
            "method_summary": "rewrote rules.md from 1 failing trace(s)",
            "model_seconds": 1.5,
        },
    }
    write_commits(work / "agent-record", commit)
    for name, release, tool in (("s1", "r1", "harness_propose"), ("s2", "r2", None)):
        events = session_events(name, release, ["think"], tool)
        (native / "sessions" / name / "session.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    mounts = (
        {"type": "harness/mount", "seq": 0, "time": 999, "data": {"release_id": "r1", "source": "boot"}},
        {"type": "harness/mount", "seq": 1, "time": 1051, "data": {"release_id": "r2", "source": "release"}},
    )
    (native / "sessions" / "serve.jsonl").write_text("".join(json.dumps(event) + "\n" for event in mounts))

    data = replay.collect(work)
    assert [(release["step"], release["kind"], release["release_id"]) for release in data["releases"]] == [
        (0, "seed", "seed"),
        (1, "published", "r2"),
    ]
    seed_release, published = data["releases"]
    # The native run's seed is the native seed directory; a published release is it with the step's patch applied.
    assert seed_release["files"] == dict(NATIVE_SEED.files) == data["seed_files"]
    assert published["files"] == {**NATIVE_SEED.files, "rules.md": "one line\n"}
    assert published["diff"] == {"added": ["rules.md"], "updated": [], "removed": []}
    assert published["proposal"] == {"id": "p1", "session": "s1", "reason": "because"}
    assert published["summary"] == "rewrote rules.md from 1 failing trace(s)"
    assert published["result"]["wins"] == 2 and published["result"]["reason"].startswith("candidate won")
    assert published["result"]["model_seconds"] == 1.5
    assert [(s["session"], s["release_id"], len(s["events"])) for s in data["sessions"]] == [
        ("s1", "r1", 6),
        ("s2", "r2", 5),
    ]
    assert [event["type"] for event in data["process"]] == ["harness/mount", "harness/mount"]

    hostile = {"type": "user/message", "seq": 9, "time": 1200, "data": {"content": "<!--<script>x</script>"}}
    page = replay.render(
        {**data, "sessions": [{**data["sessions"][0], "events": [*data["sessions"][0]["events"], hostile]}]}
    )
    assert page.startswith("<title>Harness Evolution Replay</title>")
    assert '<script id="data" type="application/json">' in page and "harness_propose" in page
    # No "<" survives inside the data element: neither a closing tag nor the comment opener that would keep the
    # element from closing and swallow the page's own script.
    payload = page.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in payload and "\\u003c!--\\u003cscript" in payload
    assert page.count("</script>") == 2
    assert "http://" not in page.split("</style>")[0] and "cdn" not in page
    out = tmp_path / "replay.html"
    assert replay.main([str(work), str(out)]) == 0
    assert out.read_text(encoding="utf-8") == replay.render(replay.collect(work))

    # A step that rewrote a seed file names it as updated.
    graph = {**commit, "metrics": {**commit["metrics"], "patch": {"native/graphs/main.json": "{}\n"}}}
    write_commits(work / "agent-record", graph)
    assert replay.collect(work)["releases"][1]["diff"] == {
        "added": [],
        "updated": ["native/graphs/main.json"],
        "removed": [],
    }

    # A rollback row is not a step: it is shown as its own kind, with no result, over the files it followed.
    rollback = {
        "recorded_at": 1060.0,
        "operation": "rollback",
        "step": 2,
        "artifact_ref": {"release_id": "r1", "parent_release_id": "r2"},
        "metrics": None,
        "algorithm_state": None,
    }
    write_commits(work / "agent-record", commit, rollback)
    with_rollback = replay.collect(work)
    assert [(release["kind"], release["step"], release["release_id"]) for release in with_rollback["releases"]] == [
        ("seed", 0, "seed"),
        ("published", 1, "r2"),
        ("rollback", 2, "r1"),
    ]
    assert with_rollback["releases"][2]["result"] is None
    assert with_rollback["releases"][2]["files"] == with_rollback["releases"][1]["files"]

    # An empty work directory still renders a page: the pi variant's seed alone.
    empty = replay.collect(tmp_path / "nothing")
    assert [release["kind"] for release in empty["releases"]] == ["seed"]
    assert empty["releases"][0]["files"] == dict(PI_SEED.files) == empty["seed_files"]
    assert empty["sessions"] == [] and empty["process"] == []
    assert '<script id="data" type="application/json">' in replay.render(empty)
