"""Guarantees of the recipes/skillclaw method package, hermetic: the
night's LLM is stubbed, probe episodes run a fake pi binary, docker tasks are
faked, and the driver's dry run replays traffic through the real embedded
service so the recorded requests carry the served pool's catalog."""

from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from reef.core import AgentRecord, RequestType
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.recipe import RecipeConfigError
from reef.recipe.registry import build_recipe
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.processor import CordisProcessor
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.evaluation import AlwaysSelect
from reef.train.types import ProcessorContext, TraceSample

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "recipes" / "skillclaw"

# A pi stand-in for the probe episodes: it writes the session file the
# runner reads back as the trajectory and answers nothing, so probes dead
# tie at 0.0 and publishing is the selection policy's call alone.
PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = agent_dir / "AGENTS.md"
event = {"type": "agent_end", "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

_STARTER_TEXT = "---\nname: answer-style\ndescription: How to format answers\n---\n\nAnswer briefly.\n"

#: The composition the night reads: one skill, as (kind, config) pairs
#: exactly like the backend passes. No provider node: the model binding the
#: driver hands the recipe is the only endpoint in play.
NODES = (("skill", {"name": "answer-style", "text": _STARTER_TEXT}),)

SEED = ({"id": "answer-style", "name": "skill", "config": {"name": "answer-style", "text": _STARTER_TEXT}},)

#: The deployment's model binding, as reef passes it to propose and renders
#: it into probe episodes. Tests stub the evolver's chat loop, so no call
#: ever reaches this endpoint.
MODEL = ModelBindings(served=ModelBinding(base_url="http://127.0.0.1:9", model="demo-model", api_key="dummy"))


def runtime() -> InferenceProxyRuntime:
    served = MODEL.served
    return InferenceProxyRuntime(model_path=served.model, base_url=served.base_url, api_key=served.api_key)


#: A recorded exchange whose agent read the starter skill: parse_recorded
#: turns the read tool call into the session's referenced skill.
PAYLOAD_WITH_READ = {
    "messages": [
        {"role": "user", "content": "[fib] compute fib(90)"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "/root/skills/answer-style/SKILL.md"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "skill body"},
    ],
    "response": {"choices": [{"message": {"role": "assistant", "content": "2880067194370816120"}}]},
}

PAYLOAD_PLAIN = {
    "messages": [{"role": "user", "content": "[csv] compute the median"}],
    "response": {"choices": [{"message": {"role": "assistant", "content": "wrong"}}]},
}


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    # Every example names its package ``harness``: drop any sibling
    # example's cached import so this file's syspath entry wins.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    names = ("skillclaw", "evolver", "prompts", "night")
    modules = {name: importlib.import_module(f"harness.{name}") for name in names}
    modules["skillclaw_recipe"] = importlib.import_module("recipes.skillclaw.recipe")
    return modules


@pytest.fixture
def skillclaw(example: dict[str, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = example["skillclaw"]
    monkeypatch.setattr(module, "WORKDIR", tmp_path / "campaign")
    monkeypatch.setattr(module, "RUN", "dry")
    return module


@pytest.fixture
def driver(example: dict[str, ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = importlib.import_module("run")
    monkeypatch.setattr(module, "RUN", "dry")
    return module


def night_llm(
    example: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: dict[str, dict[str, Any]] | None = None,
    create: dict[str, Any] | None = None,
) -> None:
    """Stub the evolve LLM: canned summarize/judge, per-group decisions."""
    prompts = example["prompts"]

    def chat(system: str, user: str, temperature: float, max_tokens: int = 0) -> str:
        del user, temperature, max_tokens
        if system == prompts.SUMMARIZE_SYSTEM:
            return "summary"
        if system == prompts.JUDGE_SYSTEM:
            return json.dumps({"task_completion": 0.2, "response_quality": 0.2, "efficiency": 0.5, "tool_usage": 0.5})
        if system == prompts.CREATE_SYSTEM:
            return json.dumps(create or {"action": "skip", "rationale": "nothing new"})
        for name, decision in (decisions or {}).items():
            if f"``{name}``" in system:
                return json.dumps(decision)
        return json.dumps({"action": "skip", "rationale": "no evidence"})

    monkeypatch.setattr(example["evolver"], "chat_client", lambda model: chat)


# -- the night flow as the proposer ----------------------------------------


def test_propose_maps_the_no_skill_bucket_to_a_create_mutation(skillclaw, example, monkeypatch) -> None:
    night_llm(
        example,
        monkeypatch,
        create={
            "action": "create_skill",
            "rationale": "gap",
            "skill": {"name": "csv-median", "description": "Median of a csv", "content": "Sort, take the middle."},
        },
    )
    samples = (TraceSample("a1", PAYLOAD_PLAIN, -1.0),)
    mutations = skillclaw.propose(NODES, samples, MODEL)
    assert mutations is not None
    (mutation,) = mutations
    assert (mutation.op, mutation.id) == ("create", "csv-median")
    text = mutation.options["config"]["text"]
    assert text.startswith("---\nname: csv-median\ndescription: Median of a csv\n")
    assert "Sort, take the middle." in text


def test_propose_maps_a_group_decision_to_an_update_mutation(skillclaw, example, monkeypatch) -> None:
    """A session that read a skill lands in that skill's group, and the
    group's improve_skill decision becomes an update of that node."""
    night_llm(
        example,
        monkeypatch,
        decisions={
            "answer-style": {
                "action": "improve_skill",
                "rationale": "missing guidance",
                "skill": {
                    "name": "answer-style",
                    "description": "How to format answers",
                    "content": "Answer with the bare number on the last line.",
                },
            }
        },
    )
    samples = (TraceSample("a1", PAYLOAD_WITH_READ, 0.0),)
    mutations = skillclaw.propose(NODES, samples, MODEL)
    assert mutations is not None
    (mutation,) = mutations
    assert (mutation.op, mutation.id) == ("update", "answer-style")
    assert "bare number on the last line" in mutation.options["config"]["text"]


def test_propose_optimize_description_keeps_the_body(skillclaw, example, monkeypatch) -> None:
    night_llm(
        example,
        monkeypatch,
        decisions={
            "answer-style": {
                "action": "optimize_description",
                "rationale": "wrong trigger",
                "skill": {"name": "answer-style", "description": "Only for numeric final answers"},
            }
        },
    )
    mutations = skillclaw.propose(NODES, (TraceSample("a1", PAYLOAD_WITH_READ, 0.0),), MODEL)
    assert mutations is not None
    (mutation,) = mutations
    text = mutation.options["config"]["text"]
    assert "description: Only for numeric final answers" in text
    assert "Answer briefly." in text  # the incumbent body survives


def test_propose_returns_none_when_every_decision_skips(skillclaw, example, monkeypatch) -> None:
    night_llm(example, monkeypatch)
    assert skillclaw.propose(NODES, (TraceSample("a1", PAYLOAD_WITH_READ, 0.0),), MODEL) is None
    audit = json.loads((skillclaw.WORKDIR / "dry" / "round-0" / "night" / "audit.json").read_text())
    assert audit["advanced"] is False


def test_propose_returns_none_on_an_empty_day(skillclaw, example, monkeypatch) -> None:
    def never(system, user, temperature, max_tokens=0):
        raise AssertionError("no samples, no model call")

    monkeypatch.setattr(example["evolver"], "chat_client", lambda model: never)
    assert skillclaw.propose(NODES, (), MODEL) is None


def test_propose_never_proposes_remove(skillclaw, example, monkeypatch) -> None:
    """The sealed night only adds or rewrites skills; whatever the decisions
    are, the mapped mutations are creates and updates only."""
    night_llm(
        example,
        monkeypatch,
        decisions={
            "answer-style": {
                "action": "improve_skill",
                "rationale": "r",
                "skill": {"name": "answer-style", "description": "d", "content": "new body"},
            }
        },
        create={
            "action": "create_skill",
            "rationale": "r",
            "skill": {"name": "fresh", "description": "d", "content": "body"},
        },
    )
    samples = (TraceSample("a1", PAYLOAD_WITH_READ, 0.0), TraceSample("a2", PAYLOAD_PLAIN, -1.0))
    mutations = skillclaw.propose(NODES, samples, MODEL)
    assert mutations is not None
    assert {mutation.op for mutation in mutations} <= {"create", "update"}
    assert sorted(mutation.id for mutation in mutations) == ["answer-style", "fresh"]


def test_propose_maps_a_remove_requesting_decision_to_no_remove(skillclaw, example, monkeypatch) -> None:
    """An LLM decision naming an unknown action (here a removal request) flows
    through the sealed parser; the mapped mutations still never remove."""
    night_llm(
        example,
        monkeypatch,
        decisions={
            "answer-style": {
                "action": "remove_skill",
                "rationale": "r",
                "skill": {"name": "answer-style", "description": "d", "content": ""},
            }
        },
    )
    samples = (TraceSample("a1", PAYLOAD_WITH_READ, 0.0),)
    mutations = skillclaw.propose(NODES, samples, MODEL)
    assert mutations is not None  # the request still materializes, as an edit
    assert all(mutation.op in ("create", "update") for mutation in mutations)


def test_the_day_ledger_feeds_the_digest_and_the_sentinel_means_unscored(skillclaw) -> None:
    sample = TraceSample("ref-1", PAYLOAD_PLAIN, -1.0)
    fallback = skillclaw._fallback_meta(sample)
    assert fallback["score"] is None  # -1.0 is the unscored sentinel, not a grade
    assert fallback["success"] is False

    reports = skillclaw.WORKDIR / "dry" / "round-2" / "reports"
    reports.mkdir(parents=True)
    meta = {
        "reference": "ref-1",
        "task_id": "t-9",
        "prompt": "solve it",
        "category": "01_Demo",
        "round": 2,
        "score": 0.75,
        "success": True,
        "error": "",
        "breakdown": {"overall_score": 0.75},
    }
    (reports / "t-9.json").write_text(json.dumps(meta))
    assert skillclaw._report_index()["ref-1"]["task_id"] == "t-9"

    session = skillclaw.digest(sample, meta)
    assert session["task_id"] == "t-9"
    assert session["aggregate"]["mean_score"] == 0.75
    assert session["round"] == 2


# -- the recipe: yaml boot, delivery surface --------------------------------


def test_example_yaml_boots_the_recipe_with_the_paper_wiring(example, tmp_path, monkeypatch) -> None:
    """The driver's load_recipe contract, hermetic: interpolate skillclaw.yaml
    through reef's config loader and build the explicit implementation - selection
    always, batch_size 60, the seed composition plus the seed_skills pool."""
    from reef.records import RecordStore
    from reef.service.deploy.config import load_config
    from reef.surface import Surface
    from reef.surface.skills import SkillInferenceHooks
    from reef.train.trainer import Trainer

    skills = tmp_path / "skills"
    (skills / "alpha").mkdir(parents=True)
    (skills / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n\nBody.\n")
    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REEF_MODEL", "demo-model")
    monkeypatch.setenv("REEF_PI_BINARY", str(tmp_path / "fake-pi"))
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.setenv("REEF_SC_SKILLS", str(skills))
    config = load_config(EXAMPLE_DIR / "skillclaw.yaml")
    sections = {key: config[key] for key in ("implementation", "model", "evolution", "data")}
    assert sections["implementation"] == "recipes.skillclaw.recipe:SkillClawRecipe"

    built = build_recipe(str(sections["implementation"]), {}, config=sections, runtime=runtime())
    assert type(built).__name__ == "SkillClawRecipe"
    assert built.name == "skillclaw"
    assert isinstance(built.candidate_selector, AlwaysSelect)
    assert built.batch_size == 60
    assert built.max_score == float("inf")  # the whole day batches, passes included
    assert [entry["id"] for entry in built.seed] == ["alpha"]
    assert built.seed[0]["config"]["name"] == "alpha"
    assert len(built.tasks) == 3
    assert all(any(task.startswith(prefix) for prefix in example["skillclaw"].ANSWERS) for task in built.tasks)

    surface = built.build_surface("s")
    assert type(surface) is Surface
    assert isinstance(surface.inference, SkillInferenceHooks)
    assert [layer.layer for layer in surface.inference.layers] == ["pi-agent"]
    assert [layer.layer for layer in built.build_artifact_validator().layers] == ["pi-agent"]
    assert isinstance(built.build("demo", RecordStore()), Trainer)  # loads the seed; no episodes


def test_seed_skills_must_name_an_existing_directory(example, tmp_path) -> None:
    recipe_module = example["skillclaw_recipe"]
    with pytest.raises(RecipeConfigError, match="seed_skills"):
        recipe_module.seed_skill_entries(tmp_path / "missing")


def test_seed_skill_directory_names_land_verbatim(example, tmp_path) -> None:
    """The benchmark's bootstrap copy keeps directory names as shipped, dots
    included; the seeded composition renders the same tree."""
    skill = tmp_path / "pool" / "self-improving-agent-3.0.5"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: self-improvement\n---\nbody\n", encoding="utf-8")
    (entry,) = example["skillclaw_recipe"].seed_skill_entries(tmp_path / "pool")
    assert entry["id"] == "self-improving-agent-3.0.5"
    assert entry["config"]["name"] == "self-improving-agent-3.0.5"
    from reef.harness.adapters import get_adapter
    from reef.harness.render import render_composition

    files = render_composition((("skill", entry["config"]),), get_adapter("pi"))
    assert "pi-agent/skills/self-improving-agent-3.0.5/SKILL.md" in files


def test_the_frozen_task_list_is_the_paper_day() -> None:
    data = json.loads((EXAMPLE_DIR / "harness" / "tasks.json").read_text())
    assert sum(len(entries) for entries in data["tasks"].values()) == 60
    assert len(data["tasks"]) == 6  # the six benchmark categories


def test_the_days_last_report_completes_the_batch_passes_included() -> None:
    """SkillClaw's night reads the whole day: no score window, so passing
    reports batch with the failures, and the batch closes exactly at
    batch_size - the day's task count."""

    def _inference(agent_record_id: str) -> AgentRecord:
        return AgentRecord.create(
            scenario="s",
            request_type=RequestType.INFERENCE,
            payload={"messages": [{"role": "user", "content": agent_record_id}]},
            agent_record_id=agent_record_id,
        )

    def _report(agent_record_id: str, score: float, reference: str) -> AgentRecord:
        return AgentRecord.create(
            scenario="s",
            request_type=RequestType.REPORT,
            payload={"score": score, "references": [reference]},
            agent_record_id=agent_record_id,
        )

    processor = CordisProcessor(ProcessorContext("s", {"batch_size": 3}))
    for index, score in enumerate((1.0, 0.0), start=1):
        processor.ingest(_inference(f"inf-{index}"))
        processor.ingest(_report(f"rep-{index}", score, f"inf-{index}"))
    assert not processor.ready()  # two of three reports: the day is not done
    processor.ingest(_inference("inf-3"))
    processor.ingest(_report("rep-3", 1.0, "inf-3"))
    assert processor.ready()
    batch = processor.build_batch()
    assert sorted(sample.score for sample in batch.samples) == [0.0, 1.0, 1.0]


# -- the campaign driver, dry: embedded service, faked docker day -----------


def _stub_backend() -> Any:
    from reef.runtime.inference import InferenceBackend

    class StubModel(InferenceBackend):
        async def inference(self, artifact, path, payload):
            del artifact, path, payload
            return {"choices": [{"message": {"role": "assistant", "content": "42"}}]}

    return StubModel()


def _dry_recipe(example: dict[str, ModuleType], tmp_path: Path, batch_size: int) -> Any:
    skillclaw = example["skillclaw"]
    return example["skillclaw_recipe"].SkillClawRecipe(
        resolve_proposer(skillclaw.propose),
        resolve_episode_scorer(skillclaw.evaluate),
        ("[sieve] probe",),
        binary=str(make_binary(tmp_path)),
        seed=SEED,
        candidate_selector=AlwaysSelect(),
        batch_size=batch_size,
        max_score=float("inf"),
        runtime=runtime(),
    )


def test_replay_driver_dry_run(driver, skillclaw, example, tmp_path, monkeypatch) -> None:
    """One day end to end on the embedded service: the day's traffic records
    with the served pool's catalog injected, the last report schedules the
    background night (stubbed LLM, fake pi probes), selection always publishes
    the composite night, the driver pulls the evolved pool from GET /reef/harness,
    and a replayed report lands on its 409 without failing the day."""
    from reef_client import ReefClient

    night_llm(
        example,
        monkeypatch,
        create={
            "action": "create_skill",
            "rationale": "gap",
            "skill": {"name": "csv-median", "description": "Median of a csv", "content": "Sort, take the middle."},
        },
    )
    run_dir = skillclaw.WORKDIR / "dry"
    round_dir = run_dir / "round-1"
    # Spy on the night's input: the batch samples are the recorded request
    # payloads, and the night's commit compacts the consumed records away,
    # so this seam is where the recorded traffic is observable.
    night_input: dict[str, Any] = {}
    real_propose = skillclaw.propose

    def spying_propose(nodes, samples, model):
        night_input["samples"] = samples
        return real_propose(nodes, samples, model)

    monkeypatch.setattr(skillclaw, "propose", spying_propose)
    recipe = _dry_recipe(example, tmp_path, batch_size=2)
    service = driver.RunService(
        scenario="dry",
        recipe_name="skillclaw",
        recipe=recipe,
        bootstrap_pool=driver._bootstrap_pool(run_dir, "dry", recipe),
        run_dir=run_dir,
        upstream_url="http://127.0.0.1:9",
        upstream_key="dummy",
        port=0,
        inference_backend=_stub_backend(),
    )
    service.start()
    try:
        client = ReefClient(service.base_url, timeout_s=60.0)
        base_url = service.base_url

        def fake_run_task(*, task_file, pool, image, brave_key, output_dir, agent_base_url):
            del image, brave_key, agent_base_url
            # The day materialized the pulled pool for the containers.
            assert (pool / "skills" / "answer-style" / "SKILL.md").is_file()
            prompt = f"solve {task_file.stem}"
            # The container's agent knows only an OpenAI-compatible base url:
            # no reef headers; the service stamps the run's scenario.
            request = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                assert json.load(response)["choices"][0]["message"]["content"] == "42"
            output_dir.mkdir(parents=True, exist_ok=True)
            score = 1.0 if task_file.stem == "one" else None
            return {
                "task_id": task_file.stem,
                "prompt": prompt,
                "score": score,
                "breakdown": {},
                "error": "",
                "chat": output_dir / "chat.jsonl",
            }

        monkeypatch.setattr(driver.day, "run_task", fake_run_task)
        monkeypatch.setattr(driver.day, "image_tag", lambda: "img")
        monkeypatch.setattr(
            driver,
            "_tasks",
            lambda: [
                ("01_Demo", {"id": "one", "file": "tasks/one.md"}),
                ("01_Demo", {"id": "two", "file": "tasks/two.md"}),
            ],
        )
        monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")

        assert service.training_versions() == 0
        step_before = service.training_step()
        results = driver.run_day(service, client, round_dir, 1, driver.Ledger(run_dir / "ledger.jsonl"))
        assert driver.category_scores(results) == {"01_Demo": 100.0}  # unscored stays out of the mean
        assert sorted(result["task_id"] for result in results) == ["one", "two"]

        service.wait_for_training_step(step_before)
        assert service.training_versions() == 1
        manifest = driver.pull_pool(client, service, tmp_path / "pulled")
        assert manifest["gate"]["published"] is True
        assert manifest["gate"]["mutation"] == {"op": "create", "id": "csv-median"}
        assert "Sort, take the middle." in manifest["files"]["pi-agent/skills/csv-median/SKILL.md"]

        # The served pool's catalog section appears in the recorded request
        # payload - the batch the night received is those recorded payloads,
        # post-transform.
        catalog_cls = importlib.import_module("harness.catalog").SkillCatalogModule
        recorded = [sample for sample in night_input["samples"] if "solve one" in json.dumps(dict(sample.payload))]
        assert recorded, "the day's traffic did not reach the night"
        payload = dict(recorded[-1].payload)
        system = payload["messages"][0]
        assert system["role"] == "system"
        assert "## Skills (mandatory)" in system["content"]
        assert catalog_cls.catalog_names(payload) == ("answer-style",)

        # The day ledger the night read is on disk, keyed by the reference.
        reports = sorted((round_dir / "reports").glob("*.json"))
        assert [json.loads(path.read_text())["task_id"] for path in reports] == ["one", "two"]

        # A replayed day re-reports with the same id and different content:
        # the 409 means already reported, and the driver moves on.
        replayed = {
            "agent_record_id": "sc-dry-report-r1-one",
            "score": 0.123,
            "references": [json.loads(reports[0].read_text())["reference"]],
        }
        driver.report(service, client, replayed)  # must not raise
    finally:
        service.stop()


def test_poke_night_recovers_a_pending_batch(driver, skillclaw, example, tmp_path, monkeypatch) -> None:
    night_llm(example, monkeypatch)  # every decision skips: the night commits without an artifact
    run_dir = tmp_path / "run"
    recipe = _dry_recipe(example, tmp_path, batch_size=1)
    service = driver.RunService(
        scenario="dry2",
        recipe_name="skillclaw",
        recipe=recipe,
        bootstrap_pool=driver._bootstrap_pool(run_dir, "dry2", recipe),
        run_dir=run_dir,
        upstream_url="http://127.0.0.1:9",
        upstream_key="dummy",
        port=0,
        inference_backend=_stub_backend(),
    )
    try:
        scenario = service.dispatcher.get_or_create_scenario("dry2", "skillclaw")
        assert scenario is not None
        # Append straight to the store to simulate a crash before the trigger.
        scenario.records.append_result(
            AgentRecord.create(
                scenario="dry2",
                request_type=RequestType.INFERENCE,
                payload=dict(PAYLOAD_PLAIN),
                agent_record_id="i1",
            )
        )
        scenario.records.append_result(
            AgentRecord.create(
                scenario="dry2",
                request_type=RequestType.REPORT,
                payload={"score": 0.0, "references": ["i1"]},
                agent_record_id="r1",
            )
        )
        assert service.poke_night() is True  # the pending batch settles
        assert service.poke_night() is False  # nothing left to recover
    finally:
        service.dispatcher.close()


def test_persist_and_committed_pool_round_trip(driver, tmp_path) -> None:
    """pool-current survives via the double rename, and the commit log's last
    version wins as the restart bootstrap when its tree is still on disk."""
    import hashlib

    run_dir = tmp_path / "run"
    version_dir = run_dir / "artifacts" / "v-night-1" / "pi-agent" / "skills" / "alpha"
    version_dir.mkdir(parents=True)
    (version_dir / "SKILL.md").write_text("body")

    class FakeService:
        def current_pool(self) -> Path:
            return run_dir / "artifacts" / "v-night-1"

    driver._persist_pool(FakeService(), run_dir)
    driver._persist_pool(FakeService(), run_dir)  # idempotent under the double rename
    assert (run_dir / "pool-current" / "pi-agent" / "skills" / "alpha" / "SKILL.md").read_text() == "body"

    assert driver._committed_pool(run_dir, "dry") is None  # no commit log yet
    key = hashlib.sha256(b"dry").hexdigest()
    log = run_dir / "reef-data" / f"{key}.commits.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"artifact_ref": {"version": "v-gone"}})
        + "\n"
        + json.dumps({"artifact_ref": {"version": "v-night-1"}})
        + "\n"
    )
    committed = driver._committed_pool(run_dir, "dry")
    assert committed == run_dir / "artifacts" / "v-night-1"


def test_the_recipe_yaml_is_valid_yaml_after_interpolation(monkeypatch) -> None:
    from reef.service.deploy.config import load_config

    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("REEF_MODEL", "demo-model")
    monkeypatch.setenv("REEF_PI_BINARY", "pi")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "dummy")
    monkeypatch.delenv("REEF_SC_SKILLS", raising=False)
    config = load_config(EXAMPLE_DIR / "skillclaw.yaml")
    assert yaml.safe_load(yaml.safe_dump(config)) == config
    assert config["evolution"]["seed_skills"] == ""  # unset env seeds no skills


def test_mutation_type_is_the_mechanisms(skillclaw, example, monkeypatch) -> None:
    night_llm(
        example,
        monkeypatch,
        create={
            "action": "create_skill",
            "rationale": "r",
            "skill": {"name": "fresh", "description": "d", "content": "body"},
        },
    )
    mutations = skillclaw.propose(NODES, (TraceSample("a1", PAYLOAD_PLAIN, -1.0),), MODEL)
    assert mutations is not None
    assert all(isinstance(mutation, Mutation) for mutation in mutations)


def test_evolver_chat_client_runs_their_loop_over_the_model_binding(example) -> None:
    """The ported retry loop now drives reef's binding: temperature dropped
    on the provider's 400, stream-only fallback folded, model never named."""
    from reef.harness.model_binding import ModelBindingError

    evolver = example["evolver"]
    seen: list[dict[str, Any]] = []

    class Binding:
        model = "demo-model"

        def chat(self, messages, *, timeout_s=None, **params):
            seen.append({"messages": messages, **params})
            if len(seen) == 1:
                raise ModelBindingError("400", status=400, detail="'temperature' is not supported")
            if len(seen) == 2:
                raise ModelBindingError("400", status=400, detail="Stream must be set to true")
            return "reply"

    chat = evolver.chat_client(Binding())
    assert chat("system", "user", 0.4, 16) == "reply"
    assert "temperature" in seen[0] and "temperature" not in seen[1]
    assert seen[2]["stream"] is True
    assert seen[0]["messages"][0] == {"role": "system", "content": "system"}
    assert all("model" not in body for body in seen)  # the binding supplies it
