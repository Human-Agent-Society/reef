"""The page per catalog step: why a version exists, what it changed, the evaluation's result, its setup and its chain.

``GET /reef/harness/releases/{step}/page`` renders it from the releases row,
the step being the row's position oldest first with the creation row as 0. The
chain driven here runs in ``training_mode: hybrid``: one published win and one
rejected candidate answering requests posted to ``POST /reef/train``, one
rejected automatic step from a scored report with no request, and one pending
extension update, again from a request, whose page diffs the file against the
head.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_recipe import SEED_MODELS, SEED_SETTINGS, make_binary, runtime
from reef_service.test_harness_requests import _post, _request

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness.episodes.run import EpisodeResult
from reef.recipe.cordis import CordisRecipe
from reef.service.app import create_app
from reef.service.release_page import before_release_id, build_release_page, result_of, served_step
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

MODULE = Path(__file__).parents[2] / "reef" / "service" / "release_page.py"

OLD_CODE = "export default function hello(pi) {\n  if (1 < 2) return;\n}\n"
NEW_CODE = 'export default function hello(pi) {\n  if (1 < 2) return;\n  pi.on("session_start", () => {});\n}\n'
SEED_RULES = {"id": "r1", "name": "rules", "config": {"text": "Answer briefly."}}
SEED_EXTENSION = {"id": "ext", "name": "code_extension", "config": {"name": "hello", "code": OLD_CODE}}
SEED = (SEED_MODELS, SEED_SETTINGS, SEED_RULES, SEED_EXTENSION)

MARKER = Mutation("update", "r1", {"config": {"text": "marker rules"}})
PLAIN = Mutation("update", "r1", {"config": {"text": "Answer briefly, with care."}})
TWO_MARKERS = Mutation("update", "r1", {"config": {"text": "marker marker rules"}})
EXTENSION = Mutation("update", "ext", {"config": {"name": "hello", "code": NEW_CODE}})
NOTES = Mutation("create", "s1", {"name": "skill", "config": {"name": "notes", "text": "# notes"}})

FIRST = "run the tests before you answer"
SECOND = "answer with more care"
FOURTH = "log a note when a < b, before the turn ends"
# What the proposer answers each request with; an automatic step, with no request, proposes NOTES from the failure.
ANSWERS = {FIRST: MARKER, SECOND: PLAIN, FOURTH: (TWO_MARKERS, EXTENSION)}
SCENARIO = "agents"


def evaluate(task: str, result: EpisodeResult) -> float:
    # Marker count, so a second marker still beats a head that carries one.
    return float(result.trajectory[-1]["rules"].count("marker"))


def _propose(nodes, samples, models, *, requests=()):
    return ANSWERS[requests[0]["text"]] if requests else NOTES


def _dispatcher(tmp_path: Path, *, keep_records: bool = False) -> Dispatcher:
    recipe = CordisRecipe(
        resolve_proposer(_propose),
        resolve_episode_scorer(evaluate),
        ("task one",),
        binary=str(make_binary(tmp_path)),
        seed=SEED,
        runtime=runtime(),
        proposals_dir=str(tmp_path / "inbox"),
        step_record_dir=str(tmp_path / "steps") if keep_records else None,
        review_kinds=("code_extension",),
        training_mode="hybrid",
    )
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    # What the service assembly does: the recipe's seed is the base artifact every scenario forks from.
    for relative, text in (recipe.base_artifact_files() or {}).items():
        target = bootstrap / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    factory = InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository")
    return Dispatcher(
        recipe,
        factory,
        local_artifact_dir=tmp_path / "local",
        agent_record_dir=tmp_path / "agent-record",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )


def _report(dispatcher: Dispatcher, suffix: str) -> None:
    """One scored exchange through the dispatcher: what wakes an automatic step in ``hybrid`` mode."""
    inference = AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": "q"}]},
        agent_record_id=f"i{suffix}",
    )
    report = AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.REPORT,
        payload={"score": 0.0, "references": [f"i{suffix}"]},
        agent_record_id=f"r{suffix}",
    )
    dispatcher.accept_record(inference)
    dispatcher.accept_record(report)


async def _committed(scenario, count: int, seconds: float = 30.0) -> None:
    """Returns once the catalog holds ``count`` rows; the dispatcher's worker runs the steps on its own thread."""
    for _ in range(int(seconds / 0.05)):
        if len(scenario.releases()) >= count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the catalog did not reach {count} rows in {seconds}s: {scenario.releases()}")


def _chain(tmp_path: Path) -> Dispatcher:
    """Steps 1 to 4 on one scenario: published, rejected, rejected without a request, pending extension."""
    dispatcher = _dispatcher(tmp_path)
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    assert scenario is not None

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            # One step at a time, so the rows land in the order the pages are counted by.
            for count, body in ((2, _request(FIRST)), (3, _request(SECOND, session="s2"))):
                response = await _post(client, body, SCENARIO)
                assert response.status == 200, await response.text()
                await _committed(scenario, count)
            _report(dispatcher, "3")
            await _committed(scenario, 4)
            response = await _post(client, _request(FOURTH, session="s4", release_id="rel-1"), SCENARIO)
            assert response.status == 200, await response.text()
            await _committed(scenario, 5)
        finally:
            await client.close()

    asyncio.run(run())
    rows = list(reversed(scenario.releases()))
    assert [result_of(row) for row in rows] == ["creation", "selected", "rejected", "rejected", "pending"]
    assert ["training_request" in (row.get("metrics") or {}) for row in rows] == [False, True, True, False, True]
    return dispatcher


def _pages(dispatcher: Dispatcher, *paths: str) -> list[tuple[int, str, str]]:
    async def run() -> list[tuple[int, str, str]]:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            answers = []
            for path in paths:
                response = await client.get(path, headers={"x-reef-scenario": SCENARIO})
                answers.append((response.status, response.headers.get("content-type", ""), await response.text()))
            return answers
        finally:
            await client.close()

    return asyncio.run(run())


def _page(dispatcher: Dispatcher, step: int) -> str:
    ((status, content_type, page),) = _pages(dispatcher, f"/reef/harness/releases/{step}/page")
    assert status == 200 and content_type.startswith("text/html")
    page.encode("ascii")
    return page


def _sections(page: str) -> list[str]:
    return [line[4:-5] for line in page.splitlines() if line.startswith("<h2>")]


def _data(page: str) -> dict:
    _, _, tail = page.partition('<script id="data" type="application/json">')
    block, _, _ = tail.partition("</script>")
    assert "<" not in block
    return json.loads(block)


def _section(page: str, name: str) -> str:
    _, _, tail = page.partition(f"<h2>{name}</h2>")
    body, _, _ = tail.partition("<h2>")
    return body


def _sub(page: str) -> str:
    """The line under the title: the release id, the result, current on the served head, the commit time."""
    return next(line for line in page.splitlines() if line.startswith('<p class="sub">'))


def test_the_page_for_a_published_step_carries_the_request_the_result_and_the_chain(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 1)
        assert page.startswith("<title>Harness step 1</title>")
        assert "<h1>Harness step 1</h1>" in page
        assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
        assert FIRST in _section(page, "Why") and "3f1c2a9d0b7e" in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag">update</span>r1 <span class="tag">rules</span>' in changed
        assert "<pre>marker rules</pre>" in changed
        selection_result = _section(page, "Result")
        assert '<td class="selected">selected</td>' in selection_result
        assert "<th>wins</th><td>1</td>" in selection_result and "<th>losses</th><td>0</td>" in selection_result
        assert (
            "<th>candidate score</th><td>1.0</td>" in selection_result
            and "<th>current score</th><td>0.0</td>" in selection_result
        )
        assert "<th>episode failures</th><td>0</td>" in selection_result
        assert "nothing to set up" in _section(page, "Setup")
        chain = _section(page, "Chain")
        assert rows[0]["release_id"] in chain and rows[1]["release_id"] in chain
        # The pending extension update was evaluated against this head, so it is this release's child.
        assert f"<li>step 4 <span class=\"id\">{rows[4]['release_id']}</span>" in chain
        data = _data(page)
        assert data["release_id"] == rows[1]["release_id"] and data["metrics"]["training_request"]["text"] == FIRST
        assert data["metrics"]["training_request"]["requires"] == []
    finally:
        dispatcher.close()


def test_the_page_for_a_pending_extension_step_diffs_the_file_against_the_head(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 4)
        assert "<title>Harness step 4</title>" in page
        assert FOURTH.replace("<", "&lt;") in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag">update</span>ext <span class="tag">code_extension</span>' in changed
        # A unified diff of the rendered file against the head's copy, the added line marked.
        assert f"--- pi-agent/extensions/hello.ts ({rows[1]['release_id'][:8]})" in changed
        assert f"+++ pi-agent/extensions/hello.ts ({rows[4]['release_id'][:8]})" in changed
        assert '<span class="add">+  pi.on(&quot;session_start&quot;, () =&gt; {});</span>' in changed
        assert "if (1 &lt; 2) return;" in changed
        # The rules bump that made the composite win rides beside it.
        assert "<pre>marker marker rules</pre>" in changed
        selection_result = _section(page, "Result")
        assert '<td class="pending">pending</td>' in selection_result and "waits for a promote" in selection_result
        chain = _section(page, "Chain")
        assert f'<tr><th>parent</th><td class="id">{rows[1]["release_id"]}</td></tr>' in chain
        assert rows[4]["release_id"] in chain and "<li>" not in chain
        data = _data(page)
        assert data["pending"] is True and data["metrics"]["mutations"][1]["options"]["config"]["code"] == NEW_CODE
    finally:
        dispatcher.close()


def test_the_page_for_a_rejected_step_names_the_head_it_ran_on_and_the_candidate(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 2)
        assert SECOND in _section(page, "Why") and "s2" in _section(page, "Why")
        assert "<pre>Answer briefly, with care.</pre>" in _section(page, "What changed")
        selection_result = _section(page, "Result")
        assert '<td class="rejected">rejected</td>' in selection_result and "the head stayed" in selection_result
        assert "<th>wins</th><td>0</td>" in selection_result and "<th>losses</th><td>1</td>" in selection_result
        chain = _section(page, "Chain")
        assert rows[2]["release_id"] == rows[1]["release_id"]
        assert f'{rows[1]["release_id"]} (the head at this step; the candidate published nothing)' in chain
        assert before_release_id(rows[2]) == rows[1]["release_id"]
    finally:
        dispatcher.close()


def test_the_page_for_an_automatic_step_without_a_request_says_a_failure_in_the_batch(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        page = _page(dispatcher, 3)
        assert "<p>a failure in the batch</p>" in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag">create</span>s1 <span class="tag">skill</span>' in changed
        assert "<pre># notes</pre>" in changed and "notes" in changed
        assert '<td class="rejected">rejected</td>' in _section(page, "Result")
        # The trainer writes training_request on every request step's row; an automatic step has none.
        assert "training_request" not in _data(page)["metrics"]
    finally:
        dispatcher.close()


def test_the_chain_lists_the_steps_evaluated_against_a_release_as_its_children(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 0)
        assert "<title>Harness step 0</title>" in page
        assert "no step made it" in _section(page, "Why")
        assert "the seed as the recipe rendered it" in _section(page, "What changed")
        assert '<td class="creation">creation</td>' in _section(page, "Result")
        chain = _section(page, "Chain")
        # Only the win ran on the seed; the two rejected candidates ran on the win, whose id their rows carry.
        assert chain.count("<li>step ") == 1
        assert (
            f'<li>step 1 <span class="id">{rows[1]["release_id"]}</span> <span class="selected">selected</span>'
            in chain
        )
        assert '<span class="rejected">' not in chain
        head = _section(_page(dispatcher, 1), "Chain")
        assert head.count("<li>step ") == 3 and "<li>step 1 " not in head
        for step in (2, 3):
            assert f'<li>step {step} <span class="id">{rows[1]["release_id"]}</span> <span class="rejected">' in head
        assert (
            f'<li>step 4 <span class="id">{rows[4]["release_id"]}</span> <span class="pending">pending</span>' in head
        )
    finally:
        dispatcher.close()


def test_an_unknown_step_is_404_naming_the_range_and_a_non_number_is_404(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        (missing, letters, long, longer) = _pages(
            dispatcher,
            "/reef/harness/releases/9/page",
            "/reef/harness/releases/x/page",
            "/reef/harness/releases/9999999999/page",
            f"/reef/harness/releases/{'9' * 5000}/page",
        )
        assert missing[0] == 404 and "has no step 9" in missing[2] and "steps 0 to 4" in missing[2]
        assert letters[0] == 404
        # Ten digits and more never match the route: no catalog is that long, and int() would balk past 4300.
        assert long[0] == 404 and longer[0] == 404
    finally:
        dispatcher.close()


def test_the_page_module_is_ascii_and_the_builder_escapes_every_angle_bracket() -> None:
    MODULE.read_text(encoding="utf-8").encode("ascii")
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation", "current": False}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "pending": False,
        "current": True,
        "recorded_at": 1756400000.0,
        "metrics": {
            "steps": 1,
            "selected": True,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "current_score": 1.0,
            "candidate_score": 3.0,
            "episode_failures": 1,
            "proposer_input_tokens": 1200,
            "proposer_output_tokens": 80,
            "candidate_agents": {
                "root": {
                    "turns": 3,
                    "steps": 5,
                    "tool_calls": 2,
                    "tool_errors": 0,
                    "input_tokens": 500,
                    "output_tokens": 40,
                }
            },
            "current_agents": {
                "root": {
                    "turns": 3,
                    "steps": 4,
                    "tool_calls": 1,
                    "tool_errors": 0,
                    "input_tokens": 450,
                    "output_tokens": 35,
                }
            },
            "step_record": "/srv/reef/steps/agents/1",
            "selection": {"reason": "candidate won 2 of 3"},
            "mutation": {"op": "create", "id": "n1", "options": {"name": "rules", "config": {"text": "<b>bold</b>"}}},
            "training_request": {
                "id": "q-1",
                "session": "s1",
                "release_id": "rel-0",
                "text": "caf\u00e9 <script>alert(1)</script>",
                "requires": [
                    {"name": "SLACK_WEBHOOK", "kind": "env", "check": "SLACK_WEBHOOK"},
                    {
                        "name": "notifications",
                        "kind": "permission",
                        "check": "osascript -e '1 < 2'",
                        "prompt": 'Allow <notifications> & "more"',
                    },
                    {"name": "calendar", "kind": "service"},
                ],
            },
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "caf&#233;" in page
    assert "<pre>&lt;b&gt;bold&lt;/b&gt;</pre>" in page
    selection_result = _section(page, "Result")
    assert "<th>ties</th><td>1</td>" in selection_result and "<th>episode failures</th><td>1</td>" in selection_result
    assert (
        "<th>proposer input tokens</th><td>1200</td>" in selection_result
        and "<th>proposer output tokens</th><td>80</td>" in selection_result
    )
    assert "<th>evaluation tokens</th><td>950 in, 75 out</td>" in selection_result
    assert (
        '<td class="id">/srv/reef/steps/agents/1</td>' in selection_result
        and "candidate won 2 of 3" in selection_result
    )
    setup = _section(page, "Setup")
    assert "<thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th></tr></thead>" in setup
    assert '<tr><td>SLACK_WEBHOOK</td><td>env</td><td class="id">SLACK_WEBHOOK</td><td></td></tr>' in setup
    assert (
        '<tr><td>notifications</td><td>permission</td><td class="id">osascript -e &#x27;1 &lt; 2&#x27;</td>'
        "<td>Allow &lt;notifications&gt; &amp; &quot;more&quot;</td></tr>" in setup
    )
    assert '<tr><td>calendar</td><td>service</td><td class="id"></td><td></td></tr>' in setup
    assert "reef-pi setup" in setup and "carried from earlier steps" not in setup
    assert "refused by the step" not in setup
    # No proposal notes on the row: no Design and no Review section.
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    data = _data(page)
    assert data["metrics"]["training_request"]["text"] == "caf\u00e9 <script>alert(1)</script>"
    assert data["metrics"]["training_request"]["requires"][1]["check"] == "osascript -e '1 < 2'"
    assert "recorded at 1756400000" in page


def test_the_setup_section_splits_the_steps_own_items_from_what_the_chain_carries() -> None:
    """The same union the install script reads (required_by), shown by the step that named each item."""
    twilio = {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}
    notify = {"name": "notify", "kind": "permission", "check": "test -d /", "prompt": "Allow notifications"}
    never = {"name": "never", "kind": "service"}
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    first = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True, "training_request": {"id": "q-1", "text": "text me", "requires": [twilio]}},
    }
    # The rejected candidate's row carries the head's id; its own item never installed anywhere.
    rejected = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": False, "training_request": {"id": "q-2", "text": "more", "requires": [never]}},
    }
    second = {
        "release_id": "rel-2",
        "parent_release_id": "rel-1",
        "operation": "training",
        "pending": True,
        "metrics": {"selected": True, "training_request": {"id": "q-3", "text": "notify", "requires": [notify]}},
    }
    promote = {
        "release_id": "rel-3",
        "parent_release_id": "rel-1",
        "operation": "promote",
        "rollback_target_release_id": "rel-2",
    }
    rows = [creation, first, rejected, second, promote]
    twilio_row = '<tr><td>TWILIO_SID</td><td>env</td><td class="id">TWILIO_SID</td><td></td></tr>'
    notify_row = '<tr><td>notify</td><td>permission</td><td class="id">test -d /</td><td>Allow notifications</td></tr>'

    assert "nothing to set up" in _section(build_release_page(0, rows), "Setup")
    own_only = _section(build_release_page(1, rows), "Setup")
    assert twilio_row in own_only and "carried from earlier steps" not in own_only
    candidate = _section(build_release_page(2, rows), "Setup")
    assert '<tr><td>never</td><td>service</td><td class="id"></td><td></td></tr>' in candidate
    assert "TWILIO_SID" not in candidate and "carried from earlier steps" not in candidate
    pending = _section(build_release_page(3, rows), "Setup")
    own, _, carried = pending.partition("<h3>carried from earlier steps</h3>")
    assert notify_row in own and twilio_row not in own
    assert twilio_row in carried and notify_row not in carried
    assert "reef-pi setup" in carried
    # The promote row names nothing itself; through its target it carries the whole chain.
    promoted = _section(build_release_page(4, rows), "Setup")
    own, _, carried = promoted.partition("<h3>carried from earlier steps</h3>")
    assert "nothing of its own" in own and twilio_row in carried and notify_row in carried
    assert carried.index(twilio_row) < carried.index(notify_row)


def test_an_extension_update_without_the_parents_file_shows_its_new_text() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "pending": True,
        "metrics": {
            "selected": True,
            "mutation": {"op": "update", "id": "ext", "options": {"config": {"code": NEW_CODE}}},
        },
    }
    without = build_release_page(1, [creation, row], before_entries=[SEED_EXTENSION])
    assert "<pre>export default function hello(pi) {" in without and "+++ " not in without
    assert '<span class="tag">update</span>ext <span class="tag">code_extension</span>' in without
    files = {"pi-agent/extensions/hello.ts": OLD_CODE}
    paths = {"code_extension": "pi-agent/extensions/{name}.ts"}
    with_diff = build_release_page(
        1, [creation, row], before_entries=[SEED_EXTENSION], before_files=files, node_paths=paths
    )
    assert "+++ pi-agent/extensions/hello.ts (rel-1)" in with_diff and '<span class="add">+  pi.on(' in with_diff
    unchanged = {"pi-agent/extensions/hello.ts": NEW_CODE}
    same = build_release_page(
        1, [creation, row], before_entries=[SEED_EXTENSION], before_files=unchanged, node_paths=paths
    )
    assert "pi-agent/extensions/hello.ts is unchanged" in same


def test_current_marks_the_served_head_and_not_the_pending_row(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        # The catalog's own flag sits on the newest row, the pending one, which serves nothing.
        assert rows[4]["current"] is True and rows[1]["current"] is False
        assert served_step(rows) == 1
        assert "| current" in _sub(_page(dispatcher, 1))
        for step in (0, 2, 3, 4):
            assert "current" not in _sub(_page(dispatcher, step))
    finally:
        dispatcher.close()


def test_a_promoted_pending_step_reads_promoted_at_the_promote_step(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        pending_id = list(reversed(scenario.releases()))[4]["release_id"]
        dispatcher.promote(SCENARIO, pending_id)
        rows = list(reversed(scenario.releases()))
        assert rows[5]["operation"] == "promote" and rows[5]["rollback_target_release_id"] == pending_id
        assert rows[4]["pending"] is True and result_of(rows[4]) == "pending"
        assert result_of(rows[4], rows) == "promoted at step 5"
        assert served_step(rows) == 5
        page = _page(dispatcher, 4)
        sub = _sub(page)
        assert '<span class="promoted">promoted at step 5</span>' in sub and "current" not in sub
        selection_result = _section(page, "Result")
        assert '<td class="promoted">promoted at step 5</td>' in selection_result
        assert (
            "passed the checks and was promoted at step 5; the release that step published serves it"
            in selection_result
        )
        assert "waits for a promote" not in selection_result
        # The step still diffs against the head it was evaluated on; the promote's own page names it as the target.
        assert f"+++ pi-agent/extensions/hello.ts ({pending_id[:8]})" in _section(page, "What changed")
        promoted = _page(dispatcher, 5)
        assert "| current" in _sub(promoted)
        assert f"a person promoted release {pending_id} after reading it" in _section(promoted, "Why")
        head = _section(_page(dispatcher, 1), "Chain")
        assert '<span class="promoted">promoted at step 5</span>' in head
        # The promote was made on the head, so it is the head's child too.
        assert (
            f'<li>step 5 <span class="id">{rows[5]["release_id"]}</span> <span class="promote">promote</span>' in head
        )
    finally:
        dispatcher.close()


def test_a_step_that_published_nothing_chains_to_the_head_it_ran_on() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    head = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True},
    }
    skipped = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"skipped": "no proposal"},
    }
    rows = [creation, head, skipped]
    assert before_release_id(skipped) == "rel-1" and served_step(rows) == 1
    # Chain is the last section, so cut it before the data block, which carries the row's own parent id.
    chain = _section(build_release_page(2, rows), "Chain").partition("</main>")[0]
    assert '<tr><th>ran on</th><td class="id">rel-1 (the head at this step; nothing was evaluated)</td></tr>' in chain
    assert "none (the candidate published nothing)" in chain
    assert "<th>parent</th>" not in chain and "<th>this release</th>" not in chain and "rel-0" not in chain
    published = _section(build_release_page(1, rows), "Chain").partition("</main>")[0]
    assert '<tr><th>parent</th><td class="id">rel-0</td></tr>' in published
    assert '<tr><th>this release</th><td class="id">rel-1</td></tr>' in published
    # The skipped step ran on rel-1, so it is rel-1's child and not rel-0's, though its row names rel-0 as parent.
    assert '<li>step 2 <span class="id">rel-1</span> <span class="skipped">skipped</span></li>' in published
    seed = _section(build_release_page(0, rows), "Chain").partition("</main>")[0]
    assert seed.count("<li>step ") == 1 and "<li>step 1 " in seed


def test_the_diff_colours_lines_by_position_so_a_plus_plus_line_is_an_addition() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {
            "selected": True,
            "mutation": {"op": "update", "id": "ext", "options": {"config": {"code": "let n = 0;\n++n;\n"}}},
        },
    }
    files = {"pi-agent/extensions/hello.ts": "let n = 0;\n--n;\n"}
    paths = {"code_extension": "pi-agent/extensions/{name}.ts"}
    page = build_release_page(
        1, [creation, row], before_entries=[SEED_EXTENSION], before_files=files, node_paths=paths
    )
    changed = _section(page, "What changed")
    assert '<span class="hunk">--- pi-agent/extensions/hello.ts (rel-0)</span>' in changed
    assert '<span class="hunk">+++ pi-agent/extensions/hello.ts (rel-1)</span>' in changed
    assert '<span class="del">---n;</span>' in changed and '<span class="add">+++n;</span>' in changed


@pytest.mark.parametrize("review_key", ["result", "verdict"])
def test_the_page_shows_the_proposers_design_and_review_and_escapes_them(review_key) -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    notes = {
        "design": "Add an <away> command.\n\nA rule alone would assume the state holds.",
        "review": {
            review_key: "partial",
            "covered": ["the /away command toggles the state", "the rule reads a < b"],
            "uncovered": ["no <b>notice</b> when the state clears"],
        },
        "undeclared_env": ["SLACK_WEBHOOK", "X<Y"],
    }
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True, "training_request": {"id": "q-1", "text": "away mode"}, "proposal_notes": notes},
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    assert _sections(page) == ["Why", "Design", "What changed", "Review", "Result", "Setup", "Chain"]
    design = _section(page, "Design")
    assert '<p class="text">Add an &lt;away&gt; command.\n\nA rule alone would assume the state holds.</p>' in design
    review = _section(page, "Review")
    assert 'the proposer\'s review of its entries against the request: <span class="partial">partial</span>' in review
    assert (
        "<h3>covered</h3><ul><li>the /away command toggles the state</li><li>the rule reads a &lt; b</li></ul>"
        in review
    )
    assert "<h3>uncovered</h3><ul><li>no &lt;b&gt;notice&lt;/b&gt; when the state clears</li></ul>" in review
    assert (
        "the extension reads these variables and no requires item names them: "
        '<span class="id">SLACK_WEBHOOK, X&lt;Y</span>' in review
    )
    assert "<away>" not in page and "<b>notice" not in page and "X<Y" not in page
    assert _data(page)["metrics"]["proposal_notes"] == notes
    # A complete review with nothing left uncovered says so.
    complete = {"review": {"result": "complete", "covered": ["the command"], "uncovered": []}}
    review = _section(
        build_release_page(1, [creation, {**row, "metrics": {**row["metrics"], "proposal_notes": complete}}]), "Review"
    )
    assert '<span class="complete">complete</span>' in review
    assert '<h3>uncovered</h3><p class="empty">nothing left uncovered</p>' in review
    assert "no requires item names them" not in review
    # The review call failed: no result on record, but the undeclared variables still show.
    failed = {"design": "One command.", "undeclared_env": ["TWILIO_SID"]}
    page = build_release_page(1, [creation, {**row, "metrics": {**row["metrics"], "proposal_notes": failed}}])
    assert _sections(page) == ["Why", "Design", "What changed", "Review", "Result", "Setup", "Chain"]
    review = _section(page, "Review")
    assert '<p class="empty">no review on record</p>' in review and "<h3>covered</h3>" not in review
    assert '<span class="id">TWILIO_SID</span>' in review


def test_the_result_names_why_the_proposer_produced_nothing_when_the_step_recorded_it() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    failure = "model call failed after 58.2 s (max_tokens=16384): model endpoint returned <non-text> content"
    row = {
        "release_id": "rel-0",
        "parent_release_id": None,
        "operation": "training",
        "metrics": {
            "skipped": "no proposal",
            "training_request": {"id": "q-1", "text": "text me"},
            "proposal_notes": {"failure": failure},
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    # A failure alone adds no Design or Review section; the Result table names it after the skip row, escaped.
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    assert (
        "<tr><th>skipped</th><td>no proposal</td></tr><tr><th>proposer failure</th><td>model call failed after "
        "58.2 s (max_tokens=16384): model endpoint returned &lt;non-text&gt; content</td></tr>"
    ) in _section(page, "Result")
    assert "<non-text>" not in page.partition("<script")[0]
    # Without the note, or with one that is not text, there is no such row.
    for notes in ({}, {"failure": "  "}, {"failure": 3}):
        without = {**row, "metrics": {**row["metrics"], "proposal_notes": notes}}
        assert "proposer failure" not in build_release_page(1, [creation, without])
    # A design written before the reply came to nothing keeps its section beside the row.
    designed = {**row, "metrics": {**row["metrics"], "proposal_notes": {"design": "A tool.", "failure": failure}}}
    page = build_release_page(1, [creation, designed])
    assert _sections(page) == ["Why", "Design", "What changed", "Result", "Setup", "Chain"]
    assert "<th>proposer failure</th>" in _section(page, "Result")


def test_a_row_without_a_design_or_a_review_has_no_such_section() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    plain = ["Why", "What changed", "Result", "Setup", "Chain"]
    for metrics in (
        {"selected": True},
        {"selected": True, "proposal_notes": {}},
        # Other methods write other keys; an empty design and a review that is not a mapping count as absent.
        {"selected": True, "proposal_notes": {"design": "  ", "review": "complete", "plan": "other keys"}},
        {"skipped": "no proposal"},
    ):
        row = {"release_id": "rel-1", "parent_release_id": "rel-0", "operation": "training", "metrics": metrics}
        page = build_release_page(1, [creation, row])
        assert _sections(page) == plain and "Design" not in page.partition("<script")[0]


@pytest.mark.parametrize("sides_key", ["evaluation_sides", "gate_sides"])
def test_the_result_lists_the_floor_metrics_and_omits_the_current_score_when_no_current_side_ran(sides_key) -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {
            "selected": True,
            "passed": 1,
            "failed": 0,
            "floor_score": 1.0,
            sides_key: ["candidate"],
            "candidate_score": 1.0,
            "episode_failures": 0,
            "candidate_agents": {"root": {"turns": 1, "input_tokens": 300, "output_tokens": 20}},
            "selection": {
                "policy": "floor",
                "policy_version": "1",
                "reason": "candidate met the floor on all 1 tasks",
            },
        },
    }
    selection_result = _section(build_release_page(1, [creation, row]), "Result")
    assert '<td class="selected">selected</td>' in selection_result
    assert "<th>passed</th><td>1</td>" in selection_result and "<th>failed</th><td>0</td>" in selection_result
    assert (
        "<th>floor score</th><td>1.0</td>" in selection_result
        and "<th>evaluation sides</th><td>candidate</td>" in selection_result
    )
    assert (
        "<th>candidate score</th><td>1.0</td>" in selection_result
        and "<th>episode failures</th><td>0</td>" in selection_result
    )
    for absent in ("current score", "wins", "losses", "ties"):
        assert f"<th>{absent}</th>" not in selection_result
    # The floor fields sit between the comparison's and the scores.
    assert selection_result.index("<th>passed</th>") < selection_result.index("<th>candidate score</th>")
    assert "<th>evaluation tokens</th><td>300 in, 20 out</td>" in selection_result
    assert "candidate met the floor on all 1 tasks" in selection_result
    both = {**row, "metrics": {**row["metrics"], "evaluation_sides": ["candidate", "current"], "ties": 0}}
    selection_result = _section(build_release_page(1, [creation, both]), "Result")
    assert "<th>evaluation sides</th><td>candidate, current</td>" in selection_result
    assert selection_result.index("<th>ties</th>") < selection_result.index("<th>passed</th>")


def test_the_setup_section_lists_the_requires_the_step_refused_with_their_reasons() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    by_backend = [
        {
            "item": {"name": "bad name", "kind": "env", "check": 'test -n "$X" && echo 1 < 2'},
            "reason": "requires[0].name must be a non-empty string matching ^[A-Za-z0-9][A-Za-z0-9._-]*$",
        },
        # A malformed item need not be an object; its JSON stands where the name would.
        {"item": "SLACK_WEBHOOK", "reason": "requires[0] must be an object with a name and a kind"},
    ]
    by_method = [
        {
            "item": {"name": "phone", "kind": "sms", "prompt": "The number to text, with the <country> code"},
            "reason": "kind must be one of ('permission', 'env', 'service')",
        }
    ]
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {
            "selected": True,
            "training_request": {
                "id": "q-1",
                "text": "text me",
                "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}],
                "refused_requires": by_backend,
            },
            "proposal_notes": {"refused_requires": by_method},
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    setup = _section(page, "Setup")
    own, _, refused = setup.partition("<h3>refused by the step</h3>")
    assert '<tr><td>TWILIO_SID</td><td>env</td><td class="id">TWILIO_SID</td><td></td></tr>' in own
    assert "reef-pi setup" in own
    assert "<thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th><th>reason</th></tr></thead>" in refused
    assert (
        '<tr><td>bad name</td><td>env</td><td class="id">test -n &quot;$X&quot; &amp;&amp; echo 1 &lt; 2</td><td></td>'
        "<td>requires[0].name must be a non-empty string matching ^[A-Za-z0-9][A-Za-z0-9._-]*$</td></tr>" in refused
    )
    assert (
        '<tr><td>&quot;SLACK_WEBHOOK&quot;</td><td></td><td class="id"></td><td></td>'
        "<td>requires[0] must be an object with a name and a kind</td></tr>" in refused
    )
    assert (
        '<tr><td>phone</td><td>sms</td><td class="id"></td><td>The number to text, with the &lt;country&gt; code</td>'
        "<td>kind must be one of (&#x27;permission&#x27;, &#x27;env&#x27;, &#x27;service&#x27;)</td></tr>" in refused
    )
    # The backend's records first, then the method's.
    assert refused.index("bad name") < refused.index("SLACK_WEBHOOK") < refused.index("phone")
    assert "echo 1 < 2" not in page.partition("<script")[0]
    # With nothing to set up, the refused table still closes the section.
    only_refused = {**row, "metrics": {"selected": True, "proposal_notes": {"refused_requires": by_method}}}
    setup = _section(build_release_page(1, [creation, only_refused]), "Setup")
    assert "nothing to set up" in setup and "<td>phone</td>" in setup
    assert setup.index("nothing to set up") < setup.index("<h3>refused by the step</h3>")
