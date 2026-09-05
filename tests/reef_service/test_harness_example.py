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


def test_example_yaml_boots_the_recipe_through_from_environment(evolution, tmp_path, monkeypatch) -> None:
    """The run.sh contract, hermetic: interpolate serve.yaml through reef's
    config loader, materialize the recipe sections as a named config, and
    boot the recipe (seed validation included) with a fake binary."""
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    config = load_config(EXAMPLE_DIR / "configs" / "serve.yaml")
    recipe_sections = {key: config[key] for key in ("implementation", "model", "evolution", "data")}
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
                        {"type": "user/message", "seq": 9, "time": 1200, "data": {"content": "</script><b>x</b>"}},
                    ],
                }
            ],
        }
    )
    assert page.startswith("<title>Harness Evolution Replay</title>")
    assert '<script id="data" type="application/json">' in page and "harness_propose" in page
    # The inline JSON never closes its own script tag early.
    assert page.count("</script>") == 2 and "<\\/script>" in page
    assert "http://" not in page.split("</style>")[0] and "cdn" not in page
    out = tmp_path / "replay.html"
    assert replay.main([str(work), str(out)]) == 0 and out.read_text(encoding="utf-8") == replay.render(
        replay.collect(work)
    )
