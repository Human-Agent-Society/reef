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
import html
import json
import re
from pathlib import Path
from urllib.parse import urlencode

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
from reef.service.page_chrome import status_label
from reef.service.release_page import before_release_id, build_release_page, result_of, served_step
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend import Mutation
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

MODULE = Path(__file__).parents[2] / "reef" / "service" / "release_page.py"
CHROME = Path(__file__).parents[2] / "reef" / "service" / "page_chrome.py"

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
    return re.findall(r"<h2>([^<]+)</h2>", page)


def _data(page: str) -> dict:
    _, _, tail = page.partition('<script id="data" type="application/json">')
    block, _, _ = tail.partition("</script>")
    assert "<" not in block
    return json.loads(block)


def _section(page: str, name: str) -> str:
    """One card's body: from its heading to the close of the section it sits in."""
    _, _, tail = page.partition(f"<h2>{name}</h2>")
    body, _, _ = tail.partition("</section>")
    return body


def _sub(page: str) -> str:
    """The line under the title: the release id, the served marker and the commit time."""
    _, _, tail = page.partition('<p class="subtitle">')
    body, _, _ = tail.partition("</p>")
    return body


def _hero(page: str) -> str:
    """The status pill beside the title, which carries the step's result."""
    _, _, tail = page.partition('<div class="status" role="status">')
    body, _, _ = tail.partition("</div>")
    return body


def test_the_page_for_a_published_step_carries_the_request_the_result_and_the_chain(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 1)
        assert page.startswith("<!doctype html>") and "<title>Harness v1</title>" in page
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
        assert "<h1>Harness v1</h1>" in page
        assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
        assert FIRST in _section(page, "Why") and "3f1c2a9d0b7e" in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag operation-update">update</span><span class="node-id">r1</span>' in changed
        assert '<span class="tag">rules</span>' in changed and "<pre>marker rules</pre>" in changed
        selection_result = _section(page, "Result")
        assert '<span class="selected">Published</span>' in selection_result
        assert "<dt>Wins</dt><dd>1</dd>" in selection_result and "<dt>Losses</dt><dd>0</dd>" in selection_result
        assert (
            "<dt>Candidate score</dt><dd>1.0</dd>" in selection_result
            and "<dt>Current score</dt><dd>0.0</dd>" in selection_result
        )
        assert "<dt>Episode failures</dt><dd>0</dd>" in selection_result
        assert "nothing to set up" in _section(page, "Setup")
        chain = _section(page, "Chain")
        assert rows[0]["release_id"] in chain and rows[1]["release_id"] in chain
        # The pending extension update was evaluated against this head, so it is this release's child.
        assert f">v4</a><span class=\"id\">{rows[4]['release_id']}</span>" in chain
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
        assert "<title>Harness v4</title>" in page
        assert FOURTH.replace("<", "&lt;") in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag operation-update">update</span><span class="node-id">ext</span>' in changed
        assert '<span class="tag">code_extension</span>' in changed
        # A unified diff of the rendered file against the head's copy, the added line marked.
        assert f"--- pi-agent/extensions/hello.ts ({rows[1]['release_id'][:8]})" in changed
        assert f"+++ pi-agent/extensions/hello.ts ({rows[4]['release_id'][:8]})" in changed
        assert '<span class="add">+  pi.on(&quot;session_start&quot;, () =&gt; {});</span>' in changed
        assert "if (1 &lt; 2) return;" in changed
        # The rules bump that made the composite win rides beside it.
        assert "<pre>marker marker rules</pre>" in changed
        selection_result = _section(page, "Result")
        assert '<span class="pending">Ready for review</span>' in selection_result
        assert "waits for a promote" in selection_result
        chain = _section(page, "Chain")
        assert f'<dt>Parent</dt><dd class="id">{rows[1]["release_id"]}</dd>' in chain
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
        assert '<span class="rejected">Not selected</span>' in selection_result
        assert "Did not pass the checks" in selection_result and "The head stayed" in selection_result
        assert "<dt>Wins</dt><dd>0</dd>" in selection_result and "<dt>Losses</dt><dd>1</dd>" in selection_result
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
        assert "A failure in the batch" in _section(page, "Why")
        changed = _section(page, "What changed")
        assert '<span class="tag operation-create">create</span><span class="node-id">s1</span>' in changed
        assert '<span class="tag">skill</span>' in changed
        assert "<pre># notes</pre>" in changed and "notes" in changed
        assert '<span class="rejected">Not selected</span>' in _section(page, "Result")
        # The trainer writes training_request on every request step's row; an automatic step has none.
        assert "training_request" not in _data(page)["metrics"]
    finally:
        dispatcher.close()


def test_the_chain_lists_the_steps_evaluated_against_a_release_as_its_children(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 0)
        assert "<title>Harness v0</title>" in page
        assert "no step made it" in _section(page, "Why")
        assert "the seed as the recipe rendered it" in _section(page, "What changed")
        assert '<span class="creation">Starting point</span>' in _section(page, "Result")
        chain = _section(page, "Chain")
        # Only the win ran on the seed; the two rejected candidates ran on the win, whose id their rows carry.
        assert chain.count('class="step-name"') == 1
        assert (
            f'>v1</a><span class="id">{rows[1]["release_id"]}</span><span class="selected">Published</span>' in chain
        )
        assert '<span class="rejected">' not in chain
        # Each child links the page of its own step, opened the way this one was.
        assert 'href="/reef/harness/releases/1/page"' in chain
        head = _section(_page(dispatcher, 1), "Chain")
        assert head.count('class="step-name"') == 3 and ">v1</a>" not in head
        for step in (2, 3):
            assert f'>v{step}</a><span class="id">{rows[1]["release_id"]}</span><span class="rejected"' in head
        assert (
            f'>v4</a><span class="id">{rows[4]["release_id"]}</span><span class="pending">Ready for review</span>'
            in head
        )
    finally:
        dispatcher.close()


def test_the_page_carries_the_shared_chrome_and_links_the_other_steps_with_the_query_it_was_opened_with() -> None:
    """The version page and the request page are one design: the same logo, header and status words.

    A person opens either from a browser, which sends no header, so the
    scenario and the token ride in the query; every link the page draws has
    to carry them on or the next page is a 401."""
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True},
    }
    query = {"scenario": "a b", "token": "t&<"}
    page = build_release_page(0, [creation, row], link_query=query)
    page.encode("ascii")
    logo = (MODULE.parents[2] / "docs" / "assets" / "reef-logo-light.svg").read_text().strip()
    # The same logo the request page carries, which its own test pins against the same file.
    assert logo in page
    assert page.startswith("<!doctype html>") and '<html lang="en">' in page
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert '<meta name="referrer" content="no-referrer">' in page
    # The scenario names the page in the header, and the token never shows.
    assert '<span class="context">a b</span>' in page and "t&<" not in page
    assert "<b>Versions</b>" in page
    assert 'href="/reef/harness/releases/1/page?scenario=a+b&amp;token=t%26%3C"' in _section(page, "Chain")
    # Without a query the links stay bare, as a curl with headers reads them.
    assert 'href="/reef/harness/releases/1/page"' in _section(build_release_page(0, [creation, row]), "Chain")


def test_the_top_bar_and_the_step_walk_lead_to_the_pages_a_browser_can_open() -> None:
    """Everything that looks clickable is: the logo and the crumb go to the served head, the arrows to the neighbours.

    Only the two ``/page`` routes take the token from the query, so those are
    the only places a link can lead; the catalog itself answers JSON and
    refuses a query token."""
    rows = [
        {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"},
        {"release_id": "rel-1", "parent_release_id": "rel-0", "operation": "training", "metrics": {"selected": True}},
        {"release_id": "rel-1", "parent_release_id": "rel-0", "operation": "training", "metrics": {"selected": False}},
    ]
    query = {"scenario": "agents", "token": "secret"}
    head = f"/reef/harness/releases/1/page?{urlencode(query)}"
    assert served_step(rows) == 1

    middle = build_release_page(2, rows, link_query=query)
    # The logo and the Harness crumb lead to the served head.
    assert f'<a class="brand" href="{html.escape(head)}" aria-label="Harness home">' in middle
    assert f'<a href="{html.escape(head)}">Harness</a>' in middle
    walk = middle.partition('<nav class="steps"')[2].partition("</nav>")[0]
    assert f'href="/reef/harness/releases/1/page?{html.escape(urlencode(query))}" rel="prev"' in walk
    assert "v2 of v2" in walk and 'rel="next"' not in walk
    # The catalog's ends have no neighbour that way, so the arrow is text, not a dead link.
    first = build_release_page(0, rows, link_query=query).partition('<nav class="steps"')[2].partition("</nav>")[0]
    assert 'rel="prev"' not in first and "<span>&#8592;</span>" in first
    assert f'href="/reef/harness/releases/1/page?{html.escape(urlencode(query))}" rel="next"' in first
    # On the served head's own page the crumb has nowhere to go, so it stays text.
    served = build_release_page(1, rows, link_query=query)
    assert '<div class="brand">' in served and "<span>Harness</span>" in served
    assert 'aria-label="Harness home"' not in served
    # Every link a page draws is a page route; nothing points at a JSON route a browser cannot open.
    for page in (first, middle, served):
        for href in re.findall(r'href="([^"]+)"', page):
            assert re.match(r"^/reef/harness/(releases/\d+|requests/[^/?]+)/page(\?|$)", href), href


def test_the_page_names_each_state_and_metric_in_the_words_the_request_page_uses() -> None:
    """One vocabulary over both pages: a person reads "Published", not the record's ``selected``."""
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    states = {
        "selected": ("selected", "Published"),
        "rejected": ("rejected", "Not selected"),
        "skipped": ("skipped", "No changes"),
        "pending": ("pending", "Ready for review"),
    }
    metrics = {
        "selected": {"selected": True},
        "rejected": {"selected": False},
        "skipped": {"skipped": "no proposal"},
        "pending": {"selected": True},
    }
    for state, (css_class, label) in states.items():
        row = {
            "release_id": "rel-1",
            "parent_release_id": "rel-0",
            "operation": "training",
            "pending": state == "pending",
            "metrics": metrics[state],
        }
        page = build_release_page(1, [creation, row])
        assert f'<span class="{css_class}">{label}</span>' in _hero(page)
        # The hero's tone drives the palette, so the card headline and the pill agree.
        assert f'<main class="tone-{css_class}">' in page
        assert status_label(state) == label
    # The creation row is no step's result; it reads as the starting point.
    assert '<span class="creation">Starting point</span>' in _hero(build_release_page(0, [creation]))


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
    CHROME.read_text(encoding="utf-8").encode("ascii")
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
    assert "<dt>Ties</dt><dd>1</dd>" in selection_result and "<dt>Episode failures</dt><dd>1</dd>" in selection_result
    assert (
        "<dt>Proposer input tokens</dt><dd>1200</dd>" in selection_result
        and "<dt>Proposer output tokens</dt><dd>80</dd>" in selection_result
    )
    assert "<dt>Evaluation tokens</dt><dd>950 in, 75 out</dd>" in selection_result
    assert (
        '<dd class="id">/srv/reef/steps/agents/1</dd>' in selection_result
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
    assert "reef-pi setup" in setup and "Carried from earlier steps" not in setup
    assert "Refused by the step" not in setup
    # No proposal notes on the row: no Design and no Review section.
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    data = _data(page)
    assert data["metrics"]["training_request"]["text"] == "caf\u00e9 <script>alert(1)</script>"
    assert data["metrics"]["training_request"]["requires"][1]["check"] == "osascript -e '1 < 2'"
    # The commit time reads as a date, the machine form beside it.
    assert '<time datetime="2025-08-28T16:53:20+00:00">28 Aug 2025, 16:53:20 UTC</time>' in _sub(page)


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
    assert twilio_row in own_only and "Carried from earlier steps" not in own_only
    candidate = _section(build_release_page(2, rows), "Setup")
    assert '<tr><td>never</td><td>service</td><td class="id"></td><td></td></tr>' in candidate
    assert "TWILIO_SID" not in candidate and "Carried from earlier steps" not in candidate
    pending = _section(build_release_page(3, rows), "Setup")
    own, _, carried = pending.partition("<h3>Carried from earlier steps</h3>")
    assert notify_row in own and twilio_row not in own
    assert twilio_row in carried and notify_row not in carried
    assert "reef-pi setup" in carried
    # The promote row names nothing itself; through its target it carries the whole chain.
    promoted = _section(build_release_page(4, rows), "Setup")
    own, _, carried = promoted.partition("<h3>Carried from earlier steps</h3>")
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
    assert '<span class="tag operation-update">update</span><span class="node-id">ext</span>' in without
    assert '<span class="tag">code_extension</span>' in without
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
        assert '<span class="chip">Currently served</span>' in _sub(_page(dispatcher, 1))
        for step in (0, 2, 3, 4):
            assert "Currently served" not in _sub(_page(dispatcher, step))
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
        assert result_of(rows[4], rows) == "promoted at v5"
        assert served_step(rows) == 5
        page = _page(dispatcher, 4)
        assert '<span class="promoted">Promoted at v5</span>' in _hero(page)
        assert "Currently served" not in _sub(page)
        selection_result = _section(page, "Result")
        assert '<span class="promoted">Promoted at v5</span>' in selection_result
        assert (
            "Passed the checks and was promoted at v5; the release that step published serves it" in selection_result
        )
        assert "waits for a promote" not in selection_result
        # The step still diffs against the head it was evaluated on; the promote's own page names it as the target.
        assert f"+++ pi-agent/extensions/hello.ts ({pending_id[:8]})" in _section(page, "What changed")
        promoted = _page(dispatcher, 5)
        assert '<span class="chip">Currently served</span>' in _sub(promoted)
        assert f"A person promoted release {pending_id} after reading it" in _section(promoted, "Why")
        head = _section(_page(dispatcher, 1), "Chain")
        assert '<span class="promoted">Promoted at v5</span>' in head
        # The promote was made on the head, so it is the head's child too.
        assert (
            f'>v5</a><span class="id">{rows[5]["release_id"]}</span>'
            '<span class="promote">Promoted by a person</span>' in head
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
    chain = _section(build_release_page(2, rows), "Chain")
    assert '<dt>Ran on</dt><dd class="id">rel-1 (the head at this step; nothing was evaluated)</dd>' in chain
    assert "none (the candidate published nothing)" in chain
    assert "<dt>Parent</dt>" not in chain and "<dt>This release</dt>" not in chain and "rel-0" not in chain
    published = _section(build_release_page(1, rows), "Chain")
    assert '<dt>Parent</dt><dd class="id">rel-0</dd>' in published
    assert '<dt>This release</dt><dd class="id">rel-1</dd>' in published
    # The skipped step ran on rel-1, so it is rel-1's child and not rel-0's, though its row names rel-0 as parent.
    assert '>v2</a><span class="id">rel-1</span><span class="skipped">No changes</span></li>' in published
    seed = _section(build_release_page(0, rows), "Chain")
    assert seed.count('class="step-name"') == 1 and ">v1</a>" in seed


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


def test_a_review_that_did_not_run_is_named_on_the_version_page() -> None:
    """A published step with no review is a step nothing checked; the page says why instead of omitting it."""
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    notes = {"design": "Add /away.", "review_failure": "the review reply carried no result object"}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True, "training_request": {"id": "q-1", "text": "away"}, "proposal_notes": notes},
    }
    page = build_release_page(1, [creation, row])
    review = _section(page, "Review")
    assert "did not run, so nothing checked whether they deliver it" in review
    assert "carried no result object" in review


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
    # A design that ends with a How to use section shows it as the card after the design.
    with_usage = dict(
        row, metrics={**row["metrics"], "proposal_notes": {"design": "Add /away.\n\n## How to use\n/away on"}}
    )
    usage_page = build_release_page(1, [creation, with_usage])
    assert _sections(usage_page)[:3] == ["Why", "Design", "How to use"]
    assert '<p class="text">Add /away.</p>' in _section(usage_page, "Design")
    assert '<p class="text">/away on</p>' in _section(usage_page, "How to use")
    review = _section(page, "Review")
    assert 'The proposer\'s review of its entries against the request: <span class="partial">Partial</span>' in review
    assert (
        '<h3>Covered</h3><ul class="review-list"><li>the /away command toggles the state</li>'
        "<li>the rule reads a &lt; b</li></ul>" in review
    )
    assert (
        '<h3>Uncovered</h3><ul class="review-list"><li>no &lt;b&gt;notice&lt;/b&gt; when the state clears</li></ul>'
        in review
    )
    assert (
        "The extension reads these and no requires item names them: "
        '<span class="id">SLACK_WEBHOOK, X&lt;Y</span>' in review
    )
    assert "<away>" not in page and "<b>notice" not in page and "X<Y" not in page
    assert _data(page)["metrics"]["proposal_notes"] == notes
    # A complete review with nothing left uncovered says so.
    complete = {"review": {"result": "complete", "covered": ["the command"], "uncovered": []}}
    review = _section(
        build_release_page(1, [creation, {**row, "metrics": {**row["metrics"], "proposal_notes": complete}}]), "Review"
    )
    assert '<span class="complete">Complete</span>' in review
    assert '<h3>Uncovered</h3><p class="empty">nothing left uncovered</p>' in review
    assert "no requires item names them" not in review
    # The review call failed: no result on record, but the undeclared variables still show.
    failed = {"design": "One command.", "undeclared_env": ["TWILIO_SID"]}
    page = build_release_page(1, [creation, {**row, "metrics": {**row["metrics"], "proposal_notes": failed}}])
    assert _sections(page) == ["Why", "Design", "What changed", "Review", "Result", "Setup", "Chain"]
    review = _section(page, "Review")
    assert '<p class="empty">no review on record</p>' in review and "<h3>Covered</h3>" not in review
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
    # A failure alone adds no Design or Review section; the Result section names it after the skip, escaped.
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    selection_result = _section(page, "Result")
    assert "<dt>Skipped</dt><dd>no proposal</dd>" in selection_result
    assert (
        "<h3>Proposer failure</h3><p>model call failed after "
        "58.2 s (max_tokens=16384): model endpoint returned &lt;non-text&gt; content</p>"
    ) in selection_result
    assert "<non-text>" not in page.partition("<script")[0]
    # Without the note, or with one that is not text, there is no such row.
    for notes in ({}, {"failure": "  "}, {"failure": 3}):
        without = {**row, "metrics": {**row["metrics"], "proposal_notes": notes}}
        assert "Proposer failure" not in build_release_page(1, [creation, without])
    # A design written before the reply came to nothing keeps its section beside the row.
    designed = {**row, "metrics": {**row["metrics"], "proposal_notes": {"design": "A tool.", "failure": failure}}}
    page = build_release_page(1, [creation, designed])
    assert _sections(page) == ["Why", "Design", "What changed", "Result", "Setup", "Chain"]
    assert "<h3>Proposer failure</h3>" in _section(page, "Result")


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
    assert '<span class="selected">Published</span>' in selection_result
    assert "<dt>Passed</dt><dd>1</dd>" in selection_result and "<dt>Failed</dt><dd>0</dd>" in selection_result
    assert (
        "<dt>Floor score</dt><dd>1.0</dd>" in selection_result
        and "<dt>Evaluation sides</dt><dd>candidate</dd>" in selection_result
    )
    assert (
        "<dt>Candidate score</dt><dd>1.0</dd>" in selection_result
        and "<dt>Episode failures</dt><dd>0</dd>" in selection_result
    )
    for absent in ("Current score", "Wins", "Losses", "Ties"):
        assert f"<dt>{absent}</dt>" not in selection_result
    # The floor fields sit between the comparison's and the scores.
    assert selection_result.index("<dt>Passed</dt>") < selection_result.index("<dt>Candidate score</dt>")
    assert "<dt>Evaluation tokens</dt><dd>300 in, 20 out</dd>" in selection_result
    assert "candidate met the floor on all 1 tasks" in selection_result
    both = {**row, "metrics": {**row["metrics"], "evaluation_sides": ["candidate", "current"], "ties": 0}}
    selection_result = _section(build_release_page(1, [creation, both]), "Result")
    assert "<dt>Evaluation sides</dt><dd>candidate, current</dd>" in selection_result
    assert selection_result.index("<dt>Ties</dt>") < selection_result.index("<dt>Passed</dt>")
    # The headline already says the step was published, so the record's own selected flag is no metric of its own.
    assert "<dt>Selected</dt>" not in selection_result


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
    own, _, refused = setup.partition("<h3>Refused by the step</h3>")
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
    assert setup.index("nothing to set up") < setup.index("<h3>Refused by the step</h3>")
