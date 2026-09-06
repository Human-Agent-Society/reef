"""Guarantees of the tutorials/evolve-your-harness cookbook example, hermetic: the
model binding is stubbed, episodes never run, and the boot test drives the
same serve.yaml materialization run.sh performs."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
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

#: One queued instruction, as the backend forwards it to a proposer that names ``requests``.
REQUEST = {
    "id": "ask-1",
    "text": "Add a skill that runs the tests before answering",
    "session": "session-1",
    "release_id": "release-1",
    "untrusted": True,
}


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
        self.prompt: str | None = None
        self.served = self

    def chat(self, messages, **params):
        self.calls += 1
        self.prompt = messages[-1]["content"]
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


def test_propose_answers_a_queued_request_with_the_failures_as_context(evolution) -> None:
    model = canned(proposal("run-tests"))
    (mutation,) = evolution.propose(NODES, SAMPLES, model, requests=(REQUEST,))
    assert (mutation.op, mutation.id) == ("create", "run-tests")
    assert model.prompt.index(REQUEST["text"]) < model.prompt.index("[fib] compute fib(90)")
    assert "gives the user what the request names" in model.prompt


def test_propose_answers_a_request_alone_without_failures(evolution) -> None:
    model = canned(proposal("answer-style"))
    (mutation,) = evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    assert model.calls == 1
    assert REQUEST["text"] in model.prompt and "Recent failing requests" not in model.prompt


API_SKILL = (
    "skill",
    {"name": "reef-pi-extension-api", "text": "---\nname: reef-pi-extension-api\n---\n# pi API\npi.registerTool"},
)


def request_reply(*entries: dict) -> str:
    return json.dumps(list(entries))


def test_propose_answers_a_request_with_the_entries_the_request_and_the_api_skill_in_its_prompt(evolution) -> None:
    skill = {"name": "test-first", "text": "---\nname: test-first\ndescription: run tests first\n---\n# test-first\n"}
    model = canned(request_reply({"id": "test-first", "name": "skill", "config": skill}))
    mutations = evolution.propose(
        (*NODES, ("rules", {"text": "Be brief."}), API_SKILL), (), model, requests=(REQUEST,)
    )
    assert model.calls == 1
    prompt = model.prompt
    assert REQUEST["text"] in prompt and "[BEGIN user request" in prompt
    assert '"id": "answer-style"' in prompt and '"body": "# answer-style' in prompt
    assert '"kind": "rules"' in prompt and '"id": null' in prompt
    assert "pi.registerTool" in prompt and "reef-pi-extension-api" in prompt
    for reserved in ("reef-version-check", "reef-requests"):
        assert reserved in prompt
    for kind in ("skill", "rules", "agent_command", "code_extension"):
        assert f"- {kind}:" in prompt
    assert [(m.op, m.id, m.options) for m in mutations] == [
        ("create", "test-first", {"name": "skill", "config": skill})
    ]
    # Without the API skill in the tree the prompt carries no reference section.
    model = canned(request_reply({"id": "test-first", "name": "skill", "config": skill}))
    evolution.propose(NODES, (), model, requests=(REQUEST,))
    assert "pi.registerTool" not in model.prompt and "code_extension" in model.prompt


def test_propose_parses_every_request_kind_from_one_reply(evolution) -> None:
    code = "export default function (pi) {}\n"
    reply = request_reply(
        {"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": "# updated"}},
        {"id": "test-first", "name": "rules", "config": {"text": "Run the tests first."}},
        {"id": "notify", "name": "code_extension", "config": {"name": "notify", "code": code, "extra": 1}},
        {"id": "review", "name": "agent_command", "config": {"name": "review", "text": "Review the diff."}},
        {
            "id": "test-first-skill",
            "name": "skill",
            "config": {"text": "--- name: test-first-skill ---\nRun the tests."},
        },
    )
    mutations = evolution.propose(NODES, (), canned(reply), requests=(REQUEST,))
    assert [(m.op, m.id, m.options) for m in mutations] == [
        ("update", "answer-style", {"name": "skill", "config": {"name": "answer-style", "text": "# updated"}}),
        ("create", "test-first", {"name": "rules", "config": {"text": "Run the tests first."}}),
        ("create", "notify", {"name": "code_extension", "config": {"name": "notify", "code": code}}),
        ("create", "review", {"name": "agent_command", "config": {"name": "review", "text": "Review the diff."}}),
        # A config that omits the name takes the entry id; that is how the served model writes a skill.
        (
            "create",
            "test-first-skill",
            {
                "name": "skill",
                "config": {"name": "test-first-skill", "text": "--- name: test-first-skill ---\nRun the tests."},
            },
        ),
    ]
    # One fenced object is a proposal too, and a named kind already in the tree updates by its name.
    fenced = f"```json\n{json.dumps({'id': 'notify', 'name': 'code_extension', 'config': {'name': 'notify', 'code': code}})}\n```"
    tree = (*NODES, ("code_extension", {"name": "notify", "code": "old"}))
    (mutation,) = evolution.propose(tree, (), canned(fenced), requests=(REQUEST,))
    assert (mutation.op, mutation.id) == ("update", "notify")


def test_propose_drops_a_reserved_id_and_a_malformed_object_from_a_request_reply(evolution) -> None:
    reply = request_reply(
        {"id": "reef-requests", "name": "code_extension", "config": {"name": "reef-requests", "code": "x"}},
        {"id": "reef-version-check", "name": "rules", "config": {"text": "x"}},
        {"id": "notify", "name": "code_extension", "config": {"name": "other", "code": "x"}},
        {"id": "shout", "name": "native_tool", "config": {"name": "shout", "code": "x"}},
        {"id": "bad id", "name": "rules", "config": {"text": "x"}},
        {"id": "empty", "name": "rules", "config": {"text": "  "}},
        {"id": "ok", "name": "rules", "config": {"text": "ok"}},
    )
    mutations = evolution.propose(NODES, (), canned(reply), requests=(REQUEST,))
    assert [(m.op, m.id) for m in mutations] == [("create", "ok")]
    only_reserved = json.dumps(
        {"id": "reef-pi-extension-api", "name": "skill", "config": {"name": "reef-pi-extension-api", "text": "x"}}
    )
    assert evolution.propose(NODES, (), canned(only_reserved), requests=(REQUEST,)) is None
    assert evolution.propose(NODES, (), canned("no json here"), requests=(REQUEST,)) is None


def test_propose_without_a_request_keeps_the_failure_path(evolution) -> None:
    never = Model(failure=AssertionError("no samples, no request, no model call"))
    assert evolution.propose(NODES, (), never, requests=()) is None and never.calls == 0
    # The failure path still writes skills only, whatever kinds a request may name.
    assert evolution.propose(NODES, SAMPLES, canned(proposal("answer-style", name="rules"))) is None
    down = Model(failure=ModelBindingError("model endpoint unreachable: connection refused"))
    assert evolution.propose(NODES, (), down, requests=(REQUEST,)) is None


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
    assert (built.batch_size, built.max_score, built.training_mode) == (1, 0.0, "auto")

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


def test_native_propose_accepts_every_event_the_shape_offers(native_evolution) -> None:
    for event in ("pre_step", "pre_execute", "request_error", "post_execute"):
        reply = native_proposal(
            "native_hook", "guard", event=event, code="def listen(payload, next):\n    return next()\n"
        )
        mutation = native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply))
        assert isinstance(mutation, Mutation) and mutation.options["config"]["event"] == event


MAIN_GRAPH = {
    "name": "main",
    "start": "think",
    "max_steps": 12,
    "stages": {
        "think": {"kind": "model"},
        "act": {"kind": "tools"},
        "check": {"kind": "verify", "check": "last_line_integer"},
        "done": {"kind": "end", "reason": "completed"},
    },
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "check"},
        {"from": "act", "when": "done", "to": "think"},
        {"from": "check", "when": "pass", "to": "done"},
        {"from": "check", "when": "fail", "to": "think"},
    ],
}


def test_native_propose_routes_the_main_graph_through_a_proposed_agent(native_evolution) -> None:
    """An agent alone can never win: only a subagent stage runs one, and its text comes back as a user message
    the grader never reads. The proposal carries the graph edit: ask the agent, then a model stage answers."""
    from reef.harness.tree.nodes import NODE_KINDS

    reply = native_proposal("native_agent", "helper", prompt="Solve the task alone.", tools=["read_file"])
    nodes = (*NATIVE_NODES, ("native_graph", MAIN_GRAPH))
    proposal = native_evolution.propose(nodes, SAMPLES, canned(reply))
    assert isinstance(proposal, list) and [m.op for m in proposal] == ["create", "update"]
    agent, route = proposal
    assert agent.id == "helper" and agent.options["config"]["prompt"] == "Solve the task alone."
    graph = route.options["config"]
    assert route.id == "main"
    assert graph["stages"]["ask-helper"] == {"kind": "subagent", "agent": "helper"}
    assert graph["stages"]["answer-helper"] == {"kind": "model"}
    assert {"from": "think", "when": "text", "to": "ask-helper"} in graph["edges"]
    assert [(e["when"], e["to"]) for e in graph["edges"] if e["from"] == "ask-helper"] == [
        ("completed", "answer-helper"),
        ("gave_up", "answer-helper"),
        ("budget", "answer-helper"),
        ("ask", "answer-helper"),
    ]
    assert sorted((e["when"], e["to"]) for e in graph["edges"] if e["from"] == "answer-helper") == [
        ("text", "check"),
        ("tool_calls", "act"),
    ]
    NODE_KINDS["native_graph"](None, graph)
    # The agent stands alone without a main graph to route through, when the graph already runs it under any
    # stage name, and when the stage names it would add are taken.
    assert isinstance(native_evolution.propose(NATIVE_NODES, SAMPLES, canned(reply)), Mutation)
    routed = (*NATIVE_NODES, ("native_graph", graph))
    assert isinstance(native_evolution.propose(routed, SAMPLES, canned(reply)), Mutation)
    consult = {
        **MAIN_GRAPH,
        "stages": {**MAIN_GRAPH["stages"], "consult": {"kind": "subagent", "agent": "helper"}},
        "edges": [
            {**e, "to": "consult"} if e["from"] == "think" and e["when"] == "text" else e for e in MAIN_GRAPH["edges"]
        ]
        + [{"from": "consult", "when": o, "to": "check"} for o in ("completed", "gave_up", "budget", "ask")],
    }
    NODE_KINDS["native_graph"](None, consult)
    assert isinstance(
        native_evolution.propose((*NATIVE_NODES, ("native_graph", consult)), SAMPLES, canned(reply)), Mutation
    )
    taken = {**MAIN_GRAPH, "stages": {**MAIN_GRAPH["stages"], "ask-helper": {"kind": "message", "text": "hi"}}}
    assert isinstance(
        native_evolution.propose((*NATIVE_NODES, ("native_graph", taken)), SAMPLES, canned(reply)), Mutation
    )


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
    assert built.training_mode == "auto"
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


@pytest.mark.parametrize("model_id", ["provider/model-a", "provider/model-b"])
def test_deployment_yaml_names_directories_that_exist_and_boots_its_named_recipe(
    monkeypatch, tmp_path, model_id
) -> None:
    """The README deployment: ``reef.recipe: deployment`` is read back from the
    directory the service's own env names, and the harness package is on the
    PYTHONPATH the same env sets; a stale directory name here fails at boot, so
    the file's own paths are checked against the checkout."""
    import os

    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    repo_root = EXAMPLE_DIR.parents[1]
    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", model_id)
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
    monkeypatch.delenv("REEF_UPSTREAM_MODEL")  # Recipe construction uses the resolved runtime, not the environment.
    built = build_named_recipe(
        "deployment",
        {**os.environ, "REEF_RECIPE_CONFIG_DIR": str(recipe_dir)},
        default_runtime=_upstream_runtime(service),
    )
    assert isinstance(built, CordisRecipe) and built.adapter == "pi"
    assert service.upstream_model == model_id
    assert built.model_binding().model == model_id
    assert built.build_surface("demo").harness.served_model == model_id
    # The person asks while it keeps learning from failures, so the tutorial proposer must take requests.
    assert built.training_mode == "hybrid"
    records = RecordStore()
    trainer = replace(built, binary=str(tmp_path / "fake-pi")).build("demo", records)
    assert trainer.training_mode == "hybrid"
    trainer.close()
    records.close()


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


# -- the replay page --------------------------------------------------------


def _session_events(session: str, release: str, stages, tool: str | None = None) -> list[dict]:
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


def test_replay_collects_a_run_and_renders_one_self_contained_page(tmp_path: Path, monkeypatch) -> None:
    replay = _method(monkeypatch, "replay")
    work = tmp_path / "work"
    native = work / "tree" / "native"
    (native / "sessions" / "s1").mkdir(parents=True)
    (native / "sessions" / "s2").mkdir(parents=True)
    (work / "agent-record").mkdir()
    graph = {
        "name": "main",
        "start": "think",
        "stages": {"think": {"kind": "model"}, "done": {"kind": "end"}},
        "edges": [{"from": "think", "when": "text", "to": "done"}],
    }
    seed = [
        {"id": "read_file", "name": "native_tool", "config": {"name": "read_file"}},
        {"id": "main", "name": "native_graph", "config": graph},
    ]
    rule = {"id": "answer-format", "name": "rules", "config": {"text": "one line"}}
    (native / "tree.json").write_text(json.dumps([*seed, rule]))
    commit = {
        "recorded_at": 1050.0,
        "artifact_ref": {"release_id": "r2", "parent_release_id": "r1"},
        "algorithm_state": {"entries": [*seed, rule], "steps": 1},
        "metrics": {
            "steps": 1,
            "published": True,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "candidate_score": 2.0,
            "current_score": 0.0,
            "mutations": [
                {"op": "create", "id": "answer-format", "options": {"name": "rules", "config": {"text": "one line"}}}
            ],
            "proposal": {"id": "p1", "session": "s1", "release_id": "r1"},
            "selection": {"reason": "candidate won 2 task pairings and lost 0"},
        },
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(commit) + "\n")
    for name, release, tool in (("s1", "r1", "harness_propose"), ("s2", "r2", None)):
        events = _session_events(name, release, ["think"], tool)
        (native / "sessions" / name / "session.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (native / "sessions" / "serve.jsonl").write_text(
        json.dumps(
            {
                "type": "harness/mount",
                "seq": 0,
                "time": 999,
                "data": {"release_id": "r1", "source": "boot", "entries": 2},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "harness/mount",
                "seq": 1,
                "time": 1051,
                "data": {"release_id": "r2", "parent_release_id": "r1", "source": "release", "entries": 3},
            }
        )
        + "\n"
    )

    data = replay.collect(work)
    assert [(r["step"], r["kind"], r["release_id"]) for r in data["releases"]] == [
        (0, "seed", "r1"),
        (1, "published", "r2"),
    ]
    seed_release, published = data["releases"]
    # The seed is the published state with the step's creates undone; the diff names what the step added.
    assert [e["id"] for e in seed_release["entries"]] == ["read_file", "main"]
    assert published["diff"] == {"added": [{"id": "answer-format", "kind": "rules"}], "updated": [], "removed": []}
    assert published["proposal"] == {"id": "p1", "session": "s1", "release_id": "r1"}
    assert published["verdict"]["wins"] == 2 and published["verdict"]["reason"].startswith("candidate won")
    assert [(s["session"], s["release_id"], len(s["events"])) for s in data["sessions"]] == [
        ("s1", "r1", 6),
        ("s2", "r2", 5),
    ]
    assert [e["type"] for e in data["process"]] == ["harness/mount", "harness/mount"]

    page = replay.render(
        {
            **data,
            "sessions": [
                {
                    **data["sessions"][0],
                    "events": [
                        *data["sessions"][0]["events"],
                        {
                            "type": "user/message",
                            "seq": 9,
                            "time": 1200,
                            "data": {"content": "<!--<script>x</script>"},
                        },
                    ],
                }
            ],
        }
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
    assert replay.main([str(work), str(out)]) == 0 and out.read_text(encoding="utf-8") == replay.render(
        replay.collect(work)
    )

    # A first step that updated an entry: its previous options are not on the record, so the seed keeps the
    # published entry and the step's diff names it from the mutation.
    updated = {
        **commit,
        "metrics": {
            **commit["metrics"],
            "mutations": [{"op": "update", "id": "main", "options": {"name": "native_graph", "config": graph}}],
        },
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(updated) + "\n")
    first_update = replay.collect(work)
    assert first_update["releases"][1]["diff"] == {
        "added": [],
        "updated": [{"id": "main", "kind": "native_graph"}],
        "removed": [],
    }
    assert [e["id"] for e in first_update["releases"][0]["entries"]] == ["read_file", "main", "answer-format"]

    # A rollback row is not a step: it is shown as its own kind, with no verdict.
    rollback = {
        "recorded_at": 1060.0,
        "operation": "rollback",
        "step": 2,
        "artifact_ref": {"release_id": "r1", "parent_release_id": "r2"},
        "metrics": None,
        "algorithm_state": None,
    }
    (work / "agent-record" / "x.commits.jsonl").write_text(json.dumps(commit) + "\n" + json.dumps(rollback) + "\n")
    with_rollback = replay.collect(work)
    assert [(r["kind"], r["step"], r["release_id"]) for r in with_rollback["releases"]] == [
        ("seed", 0, "r1"),
        ("published", 1, "r2"),
        ("rollback", 2, "r1"),
    ]
    assert with_rollback["releases"][2]["verdict"] is None
    assert with_rollback["releases"][2]["entries"] == with_rollback["releases"][1]["entries"]

    # An empty work directory still renders a page.
    empty = replay.collect(tmp_path / "nothing")
    assert empty == {"releases": [], "sessions": [], "process": [], "seed_entries": []}
    assert '<script id="data" type="application/json">' in replay.render(empty)
