"""Guarantees of the tutorials/evolve-your-harness cookbook example, hermetic: the
model binding is stubbed, episodes never run, and the boot test drives the
same serve.yaml materialization run.sh performs."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from reef.harness.episodes.model_binding import ModelBindingError
from reef.harness.episodes.run import EpisodeResult
from reef.recipe import load_recipe_config
from reef.records import RecordStore
from reef.service.deploy.config import load_config
from reef.service.deploy.settings import service_settings_from_config
from reef.train.cordis_backend import CordisRecipe, Mutation
from reef.train.trainer import Trainer
from reef.train.types import TraceSample

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "evolve-your-harness"

#: The composition the proposer sees: the starter skill, as (kind, config)
#: pairs exactly like the backend passes. No provider node: the model binding
#: reef hands the proposer is the only endpoint in play.
NODES = (("skill", {"name": "answer-style", "text": "# answer-style\n\nStarter skill."}),)

SAMPLES = (TraceSample("a1", {"messages": [{"role": "user", "content": "[fib] compute fib(90)"}]}, 0.0),)


def _method(monkeypatch: pytest.MonkeyPatch, module: str) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    # Every example names its package ``harness``: drop any sibling
    # example's cached import so this file's syspath entry wins.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module(f"harness.{module}")


@pytest.fixture
def evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "evolution")


@pytest.fixture
def native_evolution(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _method(monkeypatch, "native_evolution")


class Model:
    """A ModelBindings stand-in: ``served`` answers one canned reply, or raises."""

    def __init__(self, reply: str | None = None, failure: Exception | None = None) -> None:
        self.reply, self.failure, self.calls = reply, failure, 0
        self.served = self

    def chat(self, messages, **params):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.reply


def canned(reply: str) -> Model:
    return Model(reply)


def proposal(entry_id: str, name: str = "skill", *, config_name: str | None = None) -> str:
    return json.dumps(
        {"id": entry_id, "name": name, "config": {"name": config_name or entry_id, "text": "# improved\n\ntext"}}
    )


# -- propose: parsing, routing, refusal ----------------------------------


def test_propose_updates_an_existing_skill_from_a_fenced_reply(evolution, monkeypatch) -> None:
    model = canned(f"```json\n{proposal('answer-style')}\n```")
    mutation = evolution.propose(NODES, SAMPLES, model)
    assert isinstance(mutation, Mutation)
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    assert mutation.options == {"name": "skill", "config": {"name": "answer-style", "text": "# improved\n\ntext"}}


def test_propose_creates_a_new_skill_for_an_unknown_id(evolution, monkeypatch) -> None:
    model = canned(proposal("csv-median"))
    mutation = evolution.propose(NODES, SAMPLES, model)
    assert (mutation.op, mutation.id) == ("create", "csv-median")


def test_propose_refuses_non_skill_kinds(evolution) -> None:
    model = canned(proposal("answer-style", name="rules"))
    assert evolution.propose(NODES, SAMPLES, model) is None


def test_propose_returns_none_on_garbage(evolution, monkeypatch) -> None:
    for reply in (
        "no json here",
        '{"id": 1, "name": "skill", "config": {}}',
        proposal("bad id with spaces"),
        proposal("answer-style", config_name="other-name"),
        json.dumps({"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "  "}}),
    ):
        model = canned(reply)
        assert evolution.propose(NODES, SAMPLES, model) is None


def test_propose_skips_when_the_endpoint_is_down(evolution) -> None:
    down = Model(failure=ModelBindingError("model endpoint unreachable: connection refused"))
    assert evolution.propose(NODES, SAMPLES, down) is None


def test_propose_without_failures_skips_without_calling_the_model(evolution) -> None:
    never = Model(failure=AssertionError("no samples, no model call"))
    assert evolution.propose(NODES, (), never) is None
    assert never.calls == 0


# -- evaluate: exact last-line grading ------------------------------------


def episode(trajectory: tuple[dict, ...]) -> EpisodeResult:
    return EpisodeResult(exit_code=0, stdout="", stderr="", trajectory=trajectory, residue=())


def pi_message(text: str) -> dict:
    return {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def test_evaluate_grades_exact_final_lines(evolution) -> None:
    task = "[sieve] count the primes below 100000"
    assert evolution.evaluate(task, episode((pi_message("Sieving...\n\n9592"),))) == 1.0
    assert evolution.evaluate(task, episode(({"role": "assistant", "content": "9592"},))) == 1.0


def test_evaluate_grades_non_exact_as_zero(evolution) -> None:
    task = "[sieve] count the primes below 100000"
    assert evolution.evaluate(task, episode((pi_message("9,592"),))) == 0.0
    assert evolution.evaluate(task, episode((pi_message("The count is 9592"),))) == 0.0
    assert evolution.evaluate(task, episode(())) == 0.0
    assert evolution.evaluate("no such prefix", episode((pi_message("9592"),))) == 0.0


# -- serve.yaml boots the recipe ------------------------------------------


@pytest.mark.parametrize("filename", ["serve.yaml", "serve-native.yaml", "deployment.yaml"])
@pytest.mark.parametrize("selector", ["role", "worker"])
def test_materializer_preserves_executor_profiles_and_recipe_selection(monkeypatch, tmp_path, filename, selector):
    materializer = _method(monkeypatch, "materialize_recipe")
    config = yaml.safe_load((EXAMPLE_DIR / "configs" / filename).read_text())
    config["executors"] = {"cpu-pool": {"backend": "mp", "workers": 2, "resources": {"cpus_per_worker": 2}}}
    config["execution"] = {"services": "local", "evolution": "cpu-pool"}
    if selector == "worker":
        config["evolution"]["worker_executor"] = "cpu-pool"
        config["execution"]["evolution"] = "uni"  # The explicit worker profile must win.
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    settings = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert settings["execution"] == config["execution"]
    assert settings["executors"] == config["executors"]
    assert "reef" not in settings and "services" not in settings
    assert json.loads((tmp_path / "work/tasks.json").read_text()) == config["evolution"]["tasks"]
    # Boot the real recipe; a retained selector without its profile would fail here.
    from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime

    recipe = CordisRecipe.from_environment(
        {},
        config=settings,
        runtime=InferenceProxyRuntime(model_path="test", base_url="http://unused", api_key="dummy"),
    )
    assert recipe.worker_executor.backend == "mp"
    assert recipe.episode_workers == 2
    assert recipe.worker_executor.workers == 2
    assert recipe.worker_executor.resources.cpus_per_worker == 2


def test_materializer_accepts_legacy_config_without_execution_sections(monkeypatch, tmp_path):
    materializer = _method(monkeypatch, "materialize_recipe")
    config = yaml.safe_load((EXAMPLE_DIR / "configs/serve.yaml").read_text())
    config.pop("execution")
    serve = tmp_path / "serve.yaml"
    serve.write_text(yaml.safe_dump(config))
    materializer.materialize(serve, tmp_path / "work")
    result = yaml.safe_load((tmp_path / "work/recipes/harness_evolve.yaml").read_text())
    assert set(result) == {"implementation", "model", "evolution", "data"}


def test_example_yaml_boots_the_recipe_through_from_environment(evolution, tmp_path, monkeypatch) -> None:
    """The run.sh contract, hermetic: interpolate serve.yaml through reef's
    config loader, materialize the recipe sections as a named config, and
    boot the recipe (seed validation included) with a fake binary."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    # serve.yaml names the real binary; this run has no pi on PATH.
    recipe_sections["evolution"] = {**recipe_sections["evolution"], "binary": str(tmp_path / "fake-pi")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))

    # The deployment names the upstream once, on the reef section; the
    # service builds the recipe's runtime from it.
    service = service_settings_from_config(config)
    assert (service.upstream_url, service.upstream_api_key, service.upstream_model) == (
        "http://127.0.0.1:8000",
        "dummy",
        "qwen3-8b",
    )
    from reef.service.assembly import _upstream_runtime

    settings = load_recipe_config(materialized)
    built = CordisRecipe.from_environment({}, config=settings, runtime=_upstream_runtime(service))
    assert built.adapter == "pi"
    assert built.binary == str(tmp_path / "fake-pi")
    assert len(built.tasks) == 3
    assert all(any(task.startswith(prefix) for prefix in evolution.ANSWERS) for task in built.tasks)
    assert (built.batch_size, built.max_score) == (1, 0.0)

    # The seed carries no provider node and the binding comes from the runtime.
    assert [entry["id"] for entry in built.seed] == ["answer-style"]
    assert "upstream" not in yaml.safe_dump(list(built.seed))
    binding = built.model_binding()
    assert (binding.base_url, binding.model, binding.api_key, binding.api) == (
        "http://127.0.0.1:8000",
        "qwen3-8b",
        "dummy",
        "openai",
    )
    assert list(built.model_bindings()) == ["served"]
    assert isinstance(built.build("demo", RecordStore()), Trainer)  # loads the seed; no episodes


# -- the native variant: skills, tools, and hooks --------------------------

NATIVE_NODES = (
    (
        "native_tool",
        {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {"type": "object"},
            "code": "def run(args, workdir):\n    return ''\n",
        },
    ),
    (
        "native_hook",
        {"name": "loop_guard", "event": "post_execute", "code": "def listen(payload, next):\n    return next()\n"},
    ),
    *NODES,
)

TOOL = {
    "description": "Run python.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    "code": "def run(args, workdir):\n    return 'ok'\n",
}
HOOK = {"event": "pre_step", "code": "def listen(payload, next):\n    return next()\n"}


def native_proposal(kind: str, entry_id: str, **config) -> str:
    return json.dumps({"id": entry_id, "name": kind, "config": {"name": entry_id, **config}})


def native_message(text: str) -> dict:
    return {
        "type": "assistant/message",
        "seq": 4,
        "time": 0,
        "data": {"content": text, "tool_calls": [], "finish": "stop"},
    }


def test_native_propose_routes_skills_tools_and_hooks(native_evolution) -> None:
    tool = native_evolution.propose(
        NATIVE_NODES, SAMPLES, canned(native_proposal("native_tool", "run_python", **TOOL))
    )
    assert (tool.op, tool.id) == ("create", "run_python")
    assert tool.options == {"name": "native_tool", "config": {"name": "run_python", **TOOL}}
    hook = native_evolution.propose(
        NATIVE_NODES, SAMPLES, canned(native_proposal("native_hook", "answer_last", **HOOK))
    )
    assert (hook.op, hook.id, hook.options["name"]) == ("create", "answer_last", "native_hook")
    # A name held by a node of the same kind updates it: the shipped seed tools are addressable by name.
    same = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(native_proposal("native_tool", "read_file", **TOOL)))
    assert (same.op, same.id) == ("update", "read_file")
    skill = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(proposal("answer-style")))
    assert (skill.op, skill.id, skill.options["name"]) == ("update", "answer-style", "skill")
    # qwen2.5:7b was measured swapping the kind and the id; the config name settles it.
    swapped = json.dumps({"id": "native_tool", "name": "median", "config": {"name": "median", **TOOL}})
    mutation = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(swapped))
    assert (mutation.op, mutation.id, mutation.options["name"]) == ("create", "median", "native_tool")


def test_native_propose_refuses_malformed_shapes(native_evolution) -> None:
    for reply in (
        proposal("answer-style", name="rules"),
        native_proposal("native_hook", "h", event="on_exit", code="x = 1"),
        native_proposal("native_hook", "h", event="pre_step", code="  "),
        native_proposal("native_tool", "t", description="d", parameters="not a schema", code="x = 1"),
        native_proposal("native_tool", "t", description="", parameters={}, code="x = 1"),
        json.dumps({"id": "t", "name": "native_tool", "config": {"name": "other", **TOOL}}),
        "no json here",
    ):
        assert native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply)) is None


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


def test_native_example_yaml_boots_the_recipe_with_the_shipped_seed(native_evolution, tmp_path, monkeypatch) -> None:
    """The run.sh native contract, hermetic: the native serve file materializes
    like serve.yaml and boots with the loop's own tools and hook seeded by reference."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve-native.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))
    from reef.service.assembly import _upstream_runtime

    service = service_settings_from_config(config)
    built = CordisRecipe.from_environment(
        {}, config=load_recipe_config(materialized), runtime=_upstream_runtime(service)
    )
    assert built.adapter == "native" and built.binary is None
    # The same three tasks as the pi variant, so the two runs are comparable.
    assert built.tasks == tuple(load_config(EXAMPLE_DIR / "configs" / "serve.yaml")["evolution"]["tasks"])
    assert [entry["id"] for entry in built.seed] == [
        "read_file",
        "write_file",
        "run_bash",
        "execute",
        "loop_guard",
        "main",
        "answer-style",
    ]
    assert "upstream" not in yaml.safe_dump(list(built.seed))
    assert isinstance(built.build("demo", RecordStore()), Trainer)  # loads the seed; no episodes


def test_deployment_yaml_names_directories_that_exist_and_boots_its_named_recipe(monkeypatch) -> None:
    """The README deployment: ``reef.recipe: deployment`` is read back from the
    directory the service's own env names, and the harness package is on the
    PYTHONPATH the same env sets; a stale directory name here fails at boot, so
    the file's own paths are checked against the checkout."""
    import os

    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    repo_root = EXAMPLE_DIR.parents[1]
    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_PYTHON", sys.executable)
    monkeypatch.setenv("PWD", str(repo_root))
    path = EXAMPLE_DIR / "configs" / "deployment.yaml"
    config = load_config(path)
    env = next(service for service in config["services"] if service["name"] == "reef")["env"]
    recipe_dir = repo_root / env["REEF_RECIPE_CONFIG_DIR"]
    assert (recipe_dir / "deployment.yaml").resolve() == path.resolve()
    method_root = Path(env["PYTHONPATH"].split(":")[0])
    assert (method_root / "harness" / "evolution.py").is_file()
    monkeypatch.syspath_prepend(str(method_root))  # what the service env PYTHONPATH gives the recipe
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith("tutorials/evolve-your-harness/")
    assert config["run_dir"].startswith("tutorials/evolve-your-harness/")
    service = service_settings_from_config(config)
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(recipe_dir)},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, CordisRecipe) and built.adapter == "pi"


def test_native_example_recipe_renders_its_seed_as_the_base_files(native_evolution, tmp_path, monkeypatch) -> None:
    """The seed a deployment ships is what a fresh scenario serves, rendered once by the recipe."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve-native.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data", "execution")}
    materialized = tmp_path / "harness_evolve.yaml"
    materialized.write_text(yaml.safe_dump(recipe_sections))
    from reef.service.assembly import _upstream_runtime

    service = service_settings_from_config(config)
    built = CordisRecipe.from_environment(
        {}, config=load_recipe_config(materialized), runtime=_upstream_runtime(service)
    )
    files = built.base_artifact_files()
    assert files is not None
    assert {"native/tools/read_file.py", "native/graphs/main.json", "native/skills/answer-style/SKILL.md"} <= set(
        files
    )
    assert "upstream" not in files["native/models.json"]  # the seed carries no provider
    info = built.build_surface("demo").harness
    assert info is not None
    assert [entry["id"] for entry in info.seed_entries][:3] == ["read_file", "write_file", "run_bash"]
    assert info.served_model == "qwen3-8b"
    assert (
        CordisRecipe.from_environment(
            {},
            config={**load_recipe_config(materialized), "evolution": {**recipe_sections["evolution"], "seed": []}},
            runtime=_upstream_runtime(service),
        ).base_artifact_files()
        is None
    )
