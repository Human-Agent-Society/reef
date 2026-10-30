"""The page per catalog step: why a version exists, what it changed, the evaluation's result, its setup and its chain.

``GET /reef/harness/releases/{step}/page`` renders it from the releases row,
the step being the row's position oldest first with the creation row as 0. The
chain driven here runs in ``training_mode: manual``: one published win and one
rejected candidate answering requests posted to ``POST /reef/train``, one
rejected automatic step from a scored report with no request, run after a switch
to ``auto``, and one inactive extension update, again from a request in
``manual``, whose page diffs the source file against the head's.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import harness_dispatcher, report_records, wait_for_rows
from tests.support.harness_steps import UNCHANGED, harness_recipe

from reef.core.plugins import PluginReference
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate
from reef.harness.rollouts.run import RolloutResult
from reef.harness.tree import HarnessTree, TreePatch
from reef.service.app import create_app
from reef.service.page_layout import status_label
from reef.service.release_page import before_release_id, build_release_page, result_of, served_step

MODULE = Path(__file__).parents[2] / "reef" / "service" / "release_page.py"
LAYOUT = Path(__file__).parents[2] / "reef" / "service" / "page_layout.py"

OLD_CODE = "export default function hello(pi) {\n  if (1 < 2) return;\n}\n"
NEW_CODE = 'export default function hello(pi) {\n  if (1 < 2) return;\n  pi.on("session_start", () => {});\n}\n'
EXTENSION_PATH = "extensions/hello.ts"
SEED = HarnessTree({"rules.md": "Answer briefly.\n", EXTENSION_PATH: OLD_CODE})

MARKER = {"rules.md": "marker rules\n"}
PLAIN = {"rules.md": "Answer briefly, with care.\n"}
TWO_MARKERS_AND_EXTENSION = {"rules.md": "marker marker rules\n", EXTENSION_PATH: NEW_CODE}
NOTES = {"skills/notes/SKILL.md": "# notes\n"}

FIRST = "run the tests before you answer"
SECOND = "answer with more care"
FOURTH = "log a note when a < b, before the turn ends"
# What the method answers each request with; an automatic step, with no request, writes NOTES from the failure.
ANSWERS = {FIRST: MARKER, SECOND: PLAIN, FOURTH: TWO_MARKERS_AND_EXTENSION}
SCENARIO = "agents"
# Hand-made catalog rows of a one-component scenario, where a row's artifact id is its release id: the creation,
# active until a step replaces it, and a step that made release rel-1 on it; a test sets ``active`` and the metrics.
CREATION = {
    "release_id": "rel-0",
    "artifact_id": "rel-0",
    "parent_release_id": None,
    "operation": "creation",
    "checkpoint": True,
    "active": True,
    "activated": True,
}
PUBLISHED = {
    "release_id": "rel-1",
    "artifact_id": "rel-1",
    "parent_release_id": "rel-0",
    "operation": "training",
    "checkpoint": True,
    "active": False,
    "activated": True,
    "metrics": {"selected": True},
}


def evaluate(task: str, result: RolloutResult) -> float:
    # Marker count, so a second marker still beats a head that carries one.
    return float(result.trajectory[-1]["rules"].count("marker"))


def answer(workspace, tools) -> TreePatch:
    request = workspace.request()
    return TreePatch(ANSWERS[request.text] if request is not None else NOTES)


def train_request(text: str, *, session: str = "3f1c2a9d0b7e", release_id: str = "rel-0") -> dict:
    return {"text": text, "session": session, "release_id": release_id}


def _dispatcher(tmp_path: Path) -> Dispatcher:
    activation = PluginReference(AlwaysActivate, {"review_kinds": ["extension"]})
    recipe = harness_recipe(tmp_path, answer, seed=SEED, score=evaluate, activation=activation)
    return harness_dispatcher(tmp_path, replace(recipe, training_mode="manual"))


def _chain(tmp_path: Path) -> Dispatcher:
    """Steps 1 to 4 on one scenario: published, rejected, rejected without a request, inactive extension."""
    dispatcher = _dispatcher(tmp_path)
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    assert scenario is not None

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        headers = {"x-reef-scenario": SCENARIO}
        try:
            # One step at a time, so the rows land in the order the pages are counted by.
            for count, body in ((2, train_request(FIRST)), (3, train_request(SECOND, session="s2"))):
                response = await client.post("/reef/train", headers=headers, json=body)
                assert response.status == 200, await response.text()
                await wait_for_rows(scenario, count)
            dispatcher.set_training_mode(SCENARIO, "auto")
            for record in report_records(SCENARIO, "3"):
                dispatcher.accept_record(record)
            await wait_for_rows(scenario, 4)
            dispatcher.set_training_mode(SCENARIO, "manual")
            body = train_request(FOURTH, session="s4", release_id="rel-1")
            response = await client.post("/reef/train", headers=headers, json=body)
            assert response.status == 200, await response.text()
            await wait_for_rows(scenario, 5)
        finally:
            await client.close()

    asyncio.run(run())
    rows = list(reversed(scenario.releases()))
    assert [result_of(row) for row in rows] == ["creation", "selected", "rejected", "rejected", "inactive"]
    assert [row["active"] for row in rows] == [False, True, False, False, False]
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
        # The patch file by file, each updated file diffed against the release the step ran on.
        assert '<span class="tag operation-update">update</span><span class="node-id">rules.md</span>' in changed
        assert (
            '<span class="del">-Answer briefly.</span>' in changed
            and '<span class="add">+marker rules</span>' in changed
        )
        selection_result = _section(page, "Result")
        assert '<span class="selected">Published</span>' in selection_result
        assert "<dt>Wins</dt><dd>1</dd>" in selection_result and "<dt>Losses</dt><dd>0</dd>" in selection_result
        assert (
            "<dt>Candidate score</dt><dd>1.0</dd>" in selection_result
            and "<dt>Current score</dt><dd>0.0</dd>" in selection_result
        )
        assert "<dt>Rollout failures</dt><dd>0</dd>" in selection_result
        assert "nothing to set up" in _section(page, "Setup")
        chain = _section(page, "Chain")
        assert rows[0]["release_id"] in chain and rows[1]["release_id"] in chain
        # The inactive extension update was evaluated against this head, so it is this release's child.
        assert f">v4</a><span class=\"id\">{rows[4]['release_id']}</span>" in chain
        data = _data(page)
        assert data["release_id"] == rows[1]["release_id"] and data["metrics"]["training_request"]["text"] == FIRST
        assert data["metrics"]["training_request"]["requires"] == []
    finally:
        dispatcher.close()


def test_the_page_for_an_inactive_extension_step_diffs_the_file_against_the_head(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 4)
        assert "<title>Harness v4</title>" in page
        assert FOURTH.replace("<", "&lt;") in _section(page, "Why")
        changed = _section(page, "What changed")
        assert (
            f'<span class="tag operation-update">update</span><span class="node-id">{EXTENSION_PATH}</span>' in changed
        )
        # A unified diff of the source file against the copy in the release it ran on, the added line marked.
        assert f"--- {EXTENSION_PATH} ({rows[1]['release_id'][:8]})" in changed
        assert f"+++ {EXTENSION_PATH} ({rows[4]['release_id'][:8]})" in changed
        assert '<span class="add">+  pi.on(&quot;session_start&quot;, () =&gt; {});</span>' in changed
        assert "if (1 &lt; 2) return;" in changed
        # The rules bump that made the patch win rides beside it.
        assert '<span class="add">+marker marker rules</span>' in changed
        selection_result = _section(page, "Result")
        assert '<span class="inactive">Waiting for activation</span>' in selection_result
        assert "a person activates it first" in selection_result
        chain = _section(page, "Chain")
        assert f'<dt>Parent</dt><dd class="id">{rows[1]["release_id"]}</dd>' in chain
        assert rows[4]["release_id"] in chain and "<li>" not in chain
        data = _data(page)
        assert data["active"] is False and data["metrics"]["patch"][EXTENSION_PATH] == NEW_CODE
    finally:
        dispatcher.close()


def test_the_page_for_a_rejected_step_names_the_head_it_ran_on_and_the_candidate(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        page = _page(dispatcher, 2)
        assert SECOND in _section(page, "Why") and "s2" in _section(page, "Why")
        # A rejected step ran on the head, so its patch reads against the head's files.
        changed = _section(page, "What changed")
        assert '<span class="del">-marker rules</span>' in changed
        assert '<span class="add">+Answer briefly, with care.</span>' in changed
        selection_result = _section(page, "Result")
        assert '<span class="rejected">Not selected</span>' in selection_result
        assert (
            "Did not pass the checks" in selection_result and "The active release did not change" in selection_result
        )
        assert "<dt>Wins</dt><dd>0</dd>" in selection_result and "<dt>Losses</dt><dd>1</dd>" in selection_result
        chain = _section(page, "Chain")
        assert rows[2]["release_id"] == rows[1]["release_id"]
        assert f'{rows[1]["release_id"]} (the active release at this step; the candidate published nothing)' in chain
        assert before_release_id(rows[2]) == rows[1]["release_id"]
    finally:
        dispatcher.close()


def test_the_page_for_an_automatic_step_without_a_request_says_a_failure_in_the_job_data(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        page = _page(dispatcher, 3)
        assert "A failure in the job data" in _section(page, "Why")
        changed = _section(page, "What changed")
        assert (
            '<span class="tag operation-create">create</span><span class="node-id">skills/notes/SKILL.md</span>'
            in changed
        )
        assert "<pre># notes\n</pre>" in changed
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
        assert "this release is the seed tree" in _section(page, "What changed")
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
            f'>v4</a><span class="id">{rows[4]["release_id"]}</span><span class="inactive">Waiting for activation</span>'
            in head
        )
    finally:
        dispatcher.close()


def test_the_page_carries_the_shared_layout_and_links_the_other_steps_with_the_query_it_was_opened_with() -> None:
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
    """Everything that looks clickable is: the logo and the crumb go to the active release, the arrows to the
    neighbours.

    Only the two ``/page`` routes take the token from the query, so those are
    the only places a link can lead; the catalog itself answers JSON and
    refuses a query token."""
    rows = [
        {**CREATION, "active": False},
        {**PUBLISHED, "active": True},
        {**PUBLISHED, "checkpoint": False, "active": False, "metrics": {"selected": False}},
    ]
    query = {"scenario": "agents", "token": "secret"}
    head = f"/reef/harness/releases/1/page?{urlencode(query)}"
    assert served_step(rows) == 1

    middle = build_release_page(2, rows, link_query=query)
    # The logo and the Harness crumb lead to the active release.
    assert f'<a class="brand" href="{html.escape(head)}" aria-label="Harness home">' in middle
    assert f'<a href="{html.escape(head)}">Harness</a>' in middle
    walk = middle.partition('<nav class="steps"')[2].partition("</nav>")[0]
    assert f'href="/reef/harness/releases/1/page?{html.escape(urlencode(query))}" rel="prev"' in walk
    assert "v2 of v2" in walk and 'rel="next"' not in walk
    # The catalog's ends have no neighbour that way, so the arrow is text, not a dead link.
    first = build_release_page(0, rows, link_query=query).partition('<nav class="steps"')[2].partition("</nav>")[0]
    assert 'rel="prev"' not in first and "<span>&#8592;</span>" in first
    assert f'href="/reef/harness/releases/1/page?{html.escape(urlencode(query))}" rel="next"' in first
    # On the page of the active release the crumb has nowhere to go, so it stays text.
    served = build_release_page(1, rows, link_query=query)
    assert '<div class="brand">' in served and "<span>Harness</span>" in served
    assert 'aria-label="Harness home"' not in served
    # Every link a page draws is a page route; nothing points at a JSON route a browser cannot open.
    for page in (first, middle, served):
        for href in re.findall(r'href="([^"]+)"', page):
            assert re.match(r"^/reef/harness/(releases/\d+|requests/[^/?]+)/page(\?|$)", href), href


def test_the_page_names_each_state_and_metric_in_the_words_the_request_page_uses() -> None:
    """One vocabulary over both pages: a person reads "Published", not the record's ``selected``."""
    creation = CREATION
    states = {
        "selected": ("selected", "Published"),
        "rejected": ("rejected", "Not selected"),
        "skipped": ("skipped", "No changes"),
        "inactive": ("inactive", "Waiting for activation"),
    }
    metrics = {
        "selected": {"selected": True},
        "rejected": {"selected": False},
        "skipped": {"skipped": UNCHANGED},
        "inactive": {"selected": True},
    }
    for state, (css_class, label) in states.items():
        row = {
            **PUBLISHED,
            "checkpoint": state != "rejected",
            "active": state == "selected",
            "activated": state != "inactive",
            "metrics": metrics[state],
        }
        served = {**creation, "active": state != "selected"}
        page = build_release_page(1, [served, row])
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
    LAYOUT.read_text(encoding="utf-8").encode("ascii")
    creation = {**CREATION, "active": False}
    row = {
        **PUBLISHED,
        "recorded_at": 1756400000.0,
        "metrics": {
            "steps": 1,
            "selected": True,
            "wins": 2,
            "losses": 0,
            "ties": 1,
            "current_score": 1.0,
            "candidate_score": 3.0,
            "rollout_failures": 1,
            "rollouts_run": 3,
            "rollouts_cached": 3,
            "method_rollouts": 2,
            "model_calls": 4,
            "model_seconds": 12.5,
            "model_input_tokens": 1200,
            "model_output_tokens": 80,
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
            "patch": {"rules.md": "<b>bold</b>"},
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
    assert "<dt>Ties</dt><dd>1</dd>" in selection_result and "<dt>Rollout failures</dt><dd>1</dd>" in selection_result
    # What the step cost: the rollouts the evaluation ran and reused, the method's own, and its model use.
    for label, value in (
        ("Rollouts run", "3"),
        ("Rollouts cached", "3"),
        ("Method rollouts", "2"),
        ("Model calls", "4"),
        ("Model seconds", "12.5"),
        ("Model input tokens", "1200"),
        ("Model output tokens", "80"),
    ):
        assert f"<dt>{label}</dt><dd>{value}</dd>" in selection_result
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
    assert "Reef does not check them" in setup and "Carried from earlier steps" not in setup
    assert "Refused by the step" not in setup
    # No method summary or notes on the row: no section for them.
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    data = _data(page)
    assert data["metrics"]["training_request"]["text"] == "caf\u00e9 <script>alert(1)</script>"
    assert data["metrics"]["training_request"]["requires"][1]["check"] == "osascript -e '1 < 2'"
    # The commit time reads as a date, the machine form beside it.
    assert '<time datetime="2025-08-28T16:53:20+00:00">28 Aug 2025, 16:53:20 UTC</time>' in _sub(page)


def test_the_setup_section_splits_the_steps_own_items_from_what_the_chain_carries() -> None:
    """The same union ``GET /reef/harness`` answers (required_by), shown by the step that named each item."""
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
    # The rejected candidate's row carries the active release's id; its own item never installed anywhere.
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
        "metrics": {"selected": True, "training_request": {"id": "q-3", "text": "notify", "requires": [notify]}},
    }
    # The next step names nothing itself; through its parent it carries the whole chain.
    third = {
        "release_id": "rel-3",
        "parent_release_id": "rel-2",
        "operation": "training",
        "metrics": {"selected": True},
    }
    rows = [creation, first, rejected, second, third]
    twilio_row = '<tr><td>TWILIO_SID</td><td>env</td><td class="id">TWILIO_SID</td><td></td></tr>'
    notify_row = '<tr><td>notify</td><td>permission</td><td class="id">test -d /</td><td>Allow notifications</td></tr>'

    assert "nothing to set up" in _section(build_release_page(0, rows), "Setup")
    own_only = _section(build_release_page(1, rows), "Setup")
    assert twilio_row in own_only and "Carried from earlier steps" not in own_only
    candidate = _section(build_release_page(2, rows), "Setup")
    assert '<tr><td>never</td><td>service</td><td class="id"></td><td></td></tr>' in candidate
    assert "TWILIO_SID" not in candidate and "Carried from earlier steps" not in candidate
    own, _, carried = _section(build_release_page(3, rows), "Setup").partition("<h3>Carried from earlier steps</h3>")
    assert notify_row in own and twilio_row not in own
    assert twilio_row in carried and notify_row not in carried
    assert "Reef does not check them" in carried
    own, _, carried = _section(build_release_page(4, rows), "Setup").partition("<h3>Carried from earlier steps</h3>")
    assert "nothing of its own" in own and twilio_row in carried and notify_row in carried
    assert carried.index(twilio_row) < carried.index(notify_row)


def test_each_file_of_the_patch_reads_against_the_release_the_step_ran_on() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    patch = {EXTENSION_PATH: NEW_CODE, "skills/notes/SKILL.md": "# notes\n", "rules.md": None}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {"selected": True, "patch": patch},
    }

    def change(page: str, operation: str, path: str) -> str:
        head = f'<span class="tag operation-{operation}">{operation}</span><span class="node-id">{path}</span>'
        assert head in page, (operation, path)
        return page.partition(f"{head}</div>")[2].partition('<div class="change">')[0]

    # The release the step ran on is not restorable: a written file shows its new text, an update or a creation.
    without = _section(build_release_page(1, [creation, row]), "What changed")
    assert change(without, "write", EXTENSION_PATH).startswith("<pre>export default function hello(pi) {")
    assert "+++ " not in without
    change(without, "remove", "rules.md")
    before = {EXTENSION_PATH: OLD_CODE, "rules.md": "Answer briefly.\n"}
    with_files = _section(build_release_page(1, [creation, row], before_files=before), "What changed")
    update = change(with_files, "update", EXTENSION_PATH)
    assert f"+++ {EXTENSION_PATH} (rel-1)" in update and '<span class="add">+  pi.on(' in update
    assert change(with_files, "create", "skills/notes/SKILL.md").startswith("<pre># notes")
    assert change(with_files, "remove", "rules.md") == "</div>"
    unchanged = build_release_page(1, [creation, row], before_files={EXTENSION_PATH: NEW_CODE})
    assert f"{EXTENSION_PATH} is unchanged" in _section(unchanged, "What changed")


def test_active_marks_the_active_release_and_not_the_newest_row(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        rows = list(reversed(dispatcher.get_or_create_scenario(SCENARIO).releases()))
        # The newest row made an inactive release, so it is not active.
        assert rows[4]["active"] is False and rows[1]["active"] is True
        assert served_step(rows) == 1
        assert '<span class="chip">Active release</span>' in _sub(_page(dispatcher, 1))
        for step in (0, 2, 3, 4):
            assert "Active release" not in _sub(_page(dispatcher, step))
    finally:
        dispatcher.close()


def test_an_activated_step_reads_selected_and_adds_no_row(tmp_path: Path) -> None:
    dispatcher = _chain(tmp_path)
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        inactive_id = list(reversed(scenario.releases()))[4]["release_id"]
        assert dispatcher.activate(SCENARIO, inactive_id).release_id == inactive_id
        rows = list(reversed(scenario.releases()))
        assert len(rows) == 5 and rows[4]["active"] is True and rows[1]["active"] is False
        assert result_of(rows[4]) == "selected"
        assert served_step(rows) == 4
        page = _page(dispatcher, 4)
        assert '<span class="selected">Published</span>' in _hero(page)
        assert '<span class="chip">Active release</span>' in _sub(page)
        assert "a person activates it first" not in _section(page, "Result")
        # The step still diffs against the release it was evaluated on.
        assert f"+++ {EXTENSION_PATH} ({inactive_id[:8]})" in _section(page, "What changed")
        assert "Active release" not in _sub(_page(dispatcher, 1))
        head = _section(_page(dispatcher, 1), "Chain")
        assert f'>v4</a><span class="id">{inactive_id}</span><span class="selected">Published</span>' in head
    finally:
        dispatcher.close()


def test_a_step_that_published_nothing_chains_to_the_release_it_ran_on() -> None:
    head = {**PUBLISHED, "active": True}
    skipped = {**PUBLISHED, "checkpoint": False, "active": False, "metrics": {"skipped": UNCHANGED}}
    rows = [{**CREATION, "active": False}, head, skipped]
    assert before_release_id(skipped) == "rel-1" and served_step(rows) == 1
    chain = _section(build_release_page(2, rows), "Chain")
    assert '<dt>Ran on</dt><dd class="id">rel-1 (the active release at this step; nothing was evaluated)</dd>' in chain
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
        "metrics": {"selected": True, "patch": {EXTENSION_PATH: "let n = 0;\n++n;\n"}},
    }
    page = build_release_page(1, [creation, row], before_files={EXTENSION_PATH: "let n = 0;\n--n;\n"})
    changed = _section(page, "What changed")
    assert f'<span class="hunk">--- {EXTENSION_PATH} (rel-0)</span>' in changed
    assert f'<span class="hunk">+++ {EXTENSION_PATH} (rel-1)</span>' in changed
    assert '<span class="del">---n;</span>' in changed and '<span class="add">+++n;</span>' in changed


def test_the_page_shows_the_method_summary_and_notes_and_escapes_them() -> None:
    """What the method returned shows as written: the summary as plain text with its line breaks after Why, and the
    notes as JSON after the Result; neither is read as HTML."""
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    summary = "Add an <away> command.\n\nHow to use: type /away on."
    notes = {"review": {"result": "partial", "uncovered": ["no <b>notice</b>"]}, "attempts": 2}
    row = {
        "release_id": "rel-1",
        "parent_release_id": "rel-0",
        "operation": "training",
        "metrics": {
            "selected": True,
            "training_request": {"id": "q-1", "text": "away mode"},
            "method_summary": summary,
            "method_notes": notes,
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    assert _sections(page) == ["Why", "Method summary", "What changed", "Result", "Method notes", "Setup", "Chain"]
    assert '<p class="text">Add an &lt;away&gt; command.\n\nHow to use: type /away on.</p>' in _section(
        page, "Method summary"
    )
    shown = _section(page, "Method notes")
    assert shown.startswith("\n<pre>{\n") and "&quot;attempts&quot;: 2" in shown
    assert "no &lt;b&gt;notice&lt;/b&gt;" in shown
    assert "<away>" not in page and "<b>notice" not in page
    assert _data(page)["metrics"]["method_notes"] == notes


def test_a_row_without_a_method_summary_or_notes_has_no_such_section() -> None:
    creation = {"release_id": "rel-0", "parent_release_id": None, "operation": "creation"}
    plain = ["Why", "What changed", "Result", "Setup", "Chain"]
    for metrics in (
        {"selected": True},
        # An empty summary and empty notes count as absent.
        {"selected": True, "method_summary": "  ", "method_notes": {}},
        {"skipped": UNCHANGED},
    ):
        row = {"release_id": "rel-1", "parent_release_id": "rel-0", "operation": "training", "metrics": metrics}
        assert _sections(build_release_page(1, [creation, row])) == plain


def test_a_step_that_failed_reads_as_failed_names_its_error_and_published_nothing() -> None:
    creation = CREATION
    error = "RuntimeError: model call failed after 58.2 s: model endpoint returned <non-text> content"
    row = {
        **CREATION,
        "operation": "training",
        "checkpoint": False,
        "active": False,
        "metrics": {
            "skipped": "instruction failed",
            "error": error,
            "training_request": {"id": "q-1", "text": "text me"},
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    assert result_of(row) == "failed" and served_step([creation, row]) == 0 and before_release_id(row) == "rel-0"
    assert '<span class="failed">Failed</span>' in page
    assert _sections(page) == ["Why", "What changed", "Result", "Setup", "Chain"]
    selection_result = _section(page, "Result")
    assert "see the error below before retrying" in selection_result
    assert "<dt>Skipped</dt><dd>instruction failed</dd>" in selection_result
    assert (
        "<h3>Error</h3><p>RuntimeError: model call failed after 58.2 s: model endpoint returned "
        "&lt;non-text&gt; content</p>"
    ) in selection_result
    assert "<non-text>" not in page.partition("<script")[0]
    # Without an error, or with one that is not text, the step skipped: nothing failed.
    for metrics in ({"skipped": "instruction failed"}, {"skipped": "instruction failed", "error": "  "}):
        without = {**row, "metrics": metrics}
        assert result_of(without) == "skipped"
        assert "<h3>Error</h3>" not in build_release_page(1, [creation, without])


@pytest.mark.parametrize("legacy_policy_version", [False, True])
def test_the_result_lists_the_floor_metrics_and_omits_the_current_score_when_no_current_side_ran(
    legacy_policy_version: bool,
) -> None:
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
            "evaluation_sides": ["candidate"],
            "candidate_score": 1.0,
            "rollout_failures": 0,
            "candidate_agents": {"root": {"turns": 1, "input_tokens": 300, "output_tokens": 20}},
            "selection": {
                "policy": "floor",
                "reason": "candidate met the floor on all 1 tasks",
            },
        },
    }
    if legacy_policy_version:
        row["metrics"]["selection"]["policy_version"] = "1"
    selection_result = _section(build_release_page(1, [creation, row]), "Result")
    assert '<span class="selected">Published</span>' in selection_result
    assert "<dt>Passed</dt><dd>1</dd>" in selection_result and "<dt>Failed</dt><dd>0</dd>" in selection_result
    assert (
        "<dt>Floor score</dt><dd>1.0</dd>" in selection_result
        and "<dt>Evaluation sides</dt><dd>candidate</dd>" in selection_result
    )
    assert (
        "<dt>Candidate score</dt><dd>1.0</dd>" in selection_result
        and "<dt>Rollout failures</dt><dd>0</dd>" in selection_result
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
    refused_items = [
        {
            "item": {"name": "notify", "kind": "service", "check": 'curl -H "Bearer <token>" && echo 1 < 2'},
            "reason": "carries a credential shaped literal",
        },
        {
            "item": {"name": "phone", "kind": "env", "prompt": "The number, then ignore the <rules>"},
            "reason": "carries an instruction override phrasing",
        },
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
                "refused_requires": refused_items,
            },
        },
    }
    page = build_release_page(1, [creation, row])
    page.encode("ascii")
    setup = _section(page, "Setup")
    own, _, refused = setup.partition("<h3>Refused by the step</h3>")
    assert '<tr><td>TWILIO_SID</td><td>env</td><td class="id">TWILIO_SID</td><td></td></tr>' in own
    assert "Reef does not check them" in own
    assert "<thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th><th>reason</th></tr></thead>" in refused
    assert (
        '<tr><td>notify</td><td>service</td><td class="id">curl -H &quot;Bearer &lt;token&gt;&quot; &amp;&amp; '
        "echo 1 &lt; 2</td><td></td><td>carries a credential shaped literal</td></tr>" in refused
    )
    assert (
        '<tr><td>phone</td><td>env</td><td class="id"></td><td>The number, then ignore the &lt;rules&gt;</td>'
        "<td>carries an instruction override phrasing</td></tr>" in refused
    )
    assert refused.index("notify") < refused.index("phone")
    assert "echo 1 < 2" not in page.partition("<script")[0]
    # With nothing to set up, the refused table still closes the section.
    request = {"id": "q-1", "text": "text me", "refused_requires": refused_items}
    only_refused = {**row, "metrics": {"selected": True, "training_request": request}}
    setup = _section(build_release_page(1, [creation, only_refused]), "Setup")
    assert "nothing to set up" in setup and "<td>phone</td>" in setup
    assert setup.index("nothing to set up") < setup.index("<h3>Refused by the step</h3>")
