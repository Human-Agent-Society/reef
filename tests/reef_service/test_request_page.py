"""The page per filed harness request: the step's state while it runs, the result once its row lands.

``GET /reef/harness/requests/{record_id}/page`` renders it from the request's
agent record, the catalog and the running step's progress, and a browser opens
it by a link that carries the scenario and a page key as query parameters. The
live chain here runs in ``training_mode: manual`` with a method that holds
its step open until the test has read the page mid-step.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from dataclasses import replace
from pathlib import Path
from threading import Event
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_service import harness_dispatcher
from tests.support.harness_steps import UNCHANGED, harness_recipe

from reef.core import AgentRecord, RequestType
from reef.harness.tree import TreePatch
from reef.service.app import create_app
from reef.service.auth import page_key_for_digest
from reef.service.release_page import build_release_page
from reef.service.request_page import (
    MAX_ACTIVITY_SHOWN,
    REFRESH_SECONDS,
    STATE_WORDS,
    build_request_page,
    settled_step,
)
from reef.train.harness_backend.capabilities import StepProgress

MODULE = Path(__file__).parents[2] / "reef" / "service" / "request_page.py"
REFRESH = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
RECORD_ID = "3f1c2a9d0b7e4c5d8e9f0a1b2c3d4e5f"
TEXT = "text me when the run is blocked"
SESSION = "3f1c2a9d0b7e"
#: The patch of the step that answered the request: new rules the scorer prefers.
PATCH = {"rules.md": "marker rules\n"}
SCENARIO = "agents"
#: What a page link the service hands out carries for the token "secret": the scenario and its page key.
KEY = page_key_for_digest(hashlib.sha256(b"secret").digest(), SCENARIO)
QUERY = {"scenario": SCENARIO, "key": KEY}


def _record(text: str = TEXT, requires: list | None = None) -> dict:
    """The agent record as ``Dispatcher.read_record`` answers it for a ``POST /reef/train`` instruction."""
    payload = {"text": text, "session": SESSION, "release_id": "rel-0", "requires": requires or []}
    return {
        "sequence": 1,
        "agent_record_id": RECORD_ID,
        "request_type": "train",
        "created_at": 1_000.0,
        "references": [],
        "artifact_ref": None,
        "score": None,
        "payload": payload,
    }


def _row(metrics: dict, *, release_id: str = "rel-1", parent: str | None = "rel-0", **rest) -> dict:
    return {
        "release_id": release_id,
        "parent_release_id": parent,
        "operation": "training",
        "pending": False,
        "recorded_at": 1_050.0,
        "metrics": metrics,
        **rest,
    }


CREATION = _row({}, release_id="rel-0", parent=None, operation="creation")


def request_page(record: dict, rows: list[dict], **options) -> str:
    """The request page of ``record`` in this module's scenario."""
    return build_request_page(record, rows, scenario=SCENARIO, **options)


#: The part of the pi harness tree each path the test patches write is, as the backend records it.
PARTS = {"rules.md": "rules", "extensions/notify.ts": "extension", "skills/old/SKILL.md": "skill"}


def _answered(**extra) -> dict:
    """The metrics of the row that answered the request, the trainer's ``training_request`` stamp included, and the
    part of each path its patch writes."""
    request = {"id": RECORD_ID, "text": TEXT, "session": SESSION, "release_id": "rel-0", "requires": []}
    parts = {path: PARTS[path] for path in extra.get("patch", {}) if path in PARTS}
    return {"steps": 1, "training_request": request, "patch_parts": parts, **extra}


def _sections(page: str) -> list[str]:
    return re.findall(r"<h2>([^<]+)</h2>", page)


def _section(page: str, name: str) -> str:
    _, _, tail = page.partition(f"<h2>{name}</h2>")
    body, _, _ = tail.partition("</section>")
    return body


def test_both_harness_pages_carry_the_readme_logo_from_the_shared_layout() -> None:
    logo = (MODULE.parents[2] / "docs" / "assets" / "reef-logo-light.svg").read_text().strip()
    page = request_page(_record(), [CREATION], now=1_042.0)
    assert logo in page
    # The version page draws with the same layout, so one logo serves both.
    assert logo in build_release_page(0, [CREATION])


def test_the_logo_leads_to_the_served_head_the_request_is_asked_against() -> None:
    """The top bar is navigable on both pages, and only ever to a page route a browser can open."""
    rows = [CREATION, _row(_answered(selected=True, patch=PATCH))]
    page = request_page(_record(), rows, link_query=QUERY, now=1_100.0)
    head = f"/reef/harness/releases/1/page?scenario=agents&amp;key={KEY}"
    assert f'<a class="brand" href="{head}" aria-label="Harness home">' in page
    assert f'<a href="{head}">Harness</a>' in page and "<b>Requests</b>" in page
    for href in re.findall(r'href="([^"]+)"', page):
        assert re.match(r"^/reef/harness/releases/\d+/page(\?|$)", href.replace("&amp;", "&")), href
    # A catalog with no served head leaves the crumb as text rather than a broken link.
    pending_only = [_row(_answered(selected=True), release_id="rel-0", parent=None, pending=True)]
    bare = request_page(_record(), pending_only, now=1_100.0)
    assert '<div class="brand">' in bare and "<span>Harness</span>" in bare


def test_a_queued_request_reloads_and_says_no_step_has_taken_it() -> None:
    page = request_page(_record(), [CREATION], now=1_042.0)
    page.encode("ascii")
    assert page.startswith("<!doctype html>") and '<html lang="en">' in page
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
    assert REFRESH in page and "<title>Harness request 3f1c2a9d</title>" in page
    assert "<h1>Request details</h1>" in page
    assert _sections(page) == ["Request", "Progress"]
    request = _section(page, "Request")
    assert f'<p class="text">{TEXT}</p>' in request
    assert f'<dt>Request ID</dt><dd class="id">{RECORD_ID}</dd>' in request
    assert f'<dt>Session</dt><dd class="id">{SESSION}</dd>' in request
    assert '<dt>Base release</dt><dd class="id">rel-0</dd>' in request
    assert '<time datetime="1970-01-01T00:16:40+00:00">01 Jan 1970, 00:16:40 UTC</time>' in request
    progress = _section(page, "Progress")
    assert '<span class="queued">Queued</span>' in progress
    assert "no step has taken the request yet" in progress
    assert "<dt>Waiting</dt><dd>42 s</dd>" in progress
    assert f"Updates every {REFRESH_SECONDS} seconds until the step settles." in progress
    assert 'aria-current="step"' in page
    assert "Step time" not in progress and "<h2>Result</h2>" not in page
    assert settled_step([CREATION], RECORD_ID) is None


def test_a_running_request_shows_the_steps_phase_its_elapsed_time_and_its_evaluation_size() -> None:
    evaluating = StepProgress(RECORD_ID, "evaluating", started_at=900.0, step_record="/work/steps/1", episodes_total=2)
    page = request_page(_record(), [CREATION], progress=evaluating, now=1_100.0)
    progress = _section(page, "Progress")
    assert REFRESH in page and '<span class="evaluating">Checking the harness</span>' in progress
    assert "the evaluation is running the candidate through its episodes" in progress
    assert "<dt>Step time</dt><dd>3 min 20 s into the step</dd>" in progress
    assert "<dt>Evaluation episodes</dt><dd>2 in the evaluation</dd>" in progress
    assert '<dt>Step record</dt><dd class="id">/work/steps/1</dd>' in progress
    assert "<dt>Waiting</dt>" not in progress

    proposing = StepProgress(RECORD_ID, "proposing", started_at=1_058.0, step_record="/work/steps/2")
    page = request_page(_record(), [CREATION], progress=proposing, now=1_100.0)
    progress = _section(page, "Progress")
    assert (
        '<span class="proposing">Designing the change</span>' in progress
        and "the method is making the change" in progress
    )
    assert "<dt>Step time</dt><dd>42 s into the step</dd>" in progress
    assert "Evaluation episodes" not in progress
    assert '<dt>Step record</dt><dd class="id">/work/steps/2</dd>' in progress

    # Another request's step says nothing about this one, which still waits.
    other = replace(evaluating, request_id="another")
    page = request_page(_record(), [CREATION], progress=other, now=1_100.0)
    assert '<span class="queued">Queued</span>' in page and "<dd>100 s</dd>" in page
    assert "Evaluation episodes" not in page
    page = request_page(_record(), [CREATION], consumed=True, now=1_100.0)
    assert '<span class="running">In progress</span>' in page and "its row follows" in page and REFRESH in page


def test_a_settled_selected_request_carries_the_result_the_patch_and_the_link_with_its_query() -> None:
    rows = [CREATION, _row(_answered(selected=True, patch=PATCH))]
    page = request_page(_record(), rows, link_query=QUERY, now=1_100.0)
    page.encode("ascii")
    assert REFRESH not in page and "<title>Harness request 3f1c2a9d</title>" in page
    assert _sections(page) == ["Request", "Result", "What changed"]
    selection_result = _section(page, "Result")
    assert '<span class="selected">Published</span>' in selection_result
    assert "Published as release rel-1. GET /reef/harness serves it from now on." in selection_result
    assert '<dt>Release</dt><dd class="id">rel-1</dd>' in selection_result
    assert f'href="/reef/harness/releases/1/page?scenario=agents&amp;key={KEY}">View v1' in selection_result
    assert "<h3>Error</h3>" not in selection_result
    changed = _section(page, "What changed")
    # Each file the patch wrote, with its part of the harness tree.
    assert '<span class="tag operation-write">write</span><span class="node-id">rules.md</span>' in changed
    assert '<span class="tag">rules</span>' in changed
    # The next action is the call that pulls the published files.
    assert "<h3>Pull it</h3><code>GET /reef/harness</code>" in selection_result
    assert settled_step(rows, RECORD_ID) == 1
    bare = request_page(_record(), rows, now=1_100.0)
    assert 'href="/reef/harness/releases/1/page">View v1' in bare

    # A rejected candidate is labeled as proposed, never as an applied change.
    rejected = _row(
        _answered(
            selected=False,
            patch={**PATCH, "extensions/notify.ts": "x", "skills/old/SKILL.md": None},
            selection={"reason": "candidate missed the floor on 1 of 1 tasks"},
        )
    )
    page = request_page(_record(), [CREATION, rejected], now=1_100.0)
    assert '<span class="rejected">Not selected</span>' in page
    assert "did not pass the checks (candidate missed the floor on 1 of 1 tasks). Nothing changed" in page
    assert "rephrase or split the request" in page
    assert "What changed" not in _sections(page)
    changed = _section(page, "Proposed changes")
    assert changed.count("<li>") == 3
    assert '<span class="tag operation-write">write</span><span class="node-id">extensions/notify.ts</span>' in changed
    assert '<span class="tag">extension</span>' in changed
    assert (
        '<span class="tag operation-remove">remove</span><span class="node-id">skills/old/SKILL.md</span>' in changed
    )
    assert "Pull it" not in page


def test_a_pending_request_names_the_promote_and_reads_promoted_once_a_promote_row_names_it() -> None:
    pending = _row(_answered(selected=True, patch=PATCH), pending=True)
    page = request_page(_record(), [CREATION, pending], now=1_100.0)
    assert '<span class="pending">Ready for review</span>' in page
    assert "Proposed changes" in _sections(page)
    assert "Release rel-1 is ready. The change touches files a person reviews before Reef serves them." in page
    assert REFRESH not in page
    # The next action is the promote call, runnable as it stands with the scenario header.
    promote_call = html.escape(f'POST /reef/scenarios/{SCENARIO}/promote {json.dumps({"release_id": "rel-1"})}')
    assert f"<h3>Read this page, then promote it</h3><code>{promote_call}</code>" in page
    promote = _row({}, release_id="rel-2", parent="rel-0", operation="promote", rollback_target_release_id="rel-1")
    page = request_page(_record(), [CREATION, pending, promote], now=1_100.0)
    assert '<span class="promoted">Promoted at v2</span>' in page
    assert "passed the checks and was promoted at v2; the release that step published serves it" in page
    assert "What changed" in _sections(page) and "promote it</h3>" not in page


def test_a_rejected_request_names_the_missed_episode_on_both_pages() -> None:
    selection = {"reason": "candidate missed the floor on 1 of 1 tasks", "metrics": {"floor_score": 1.0}}
    episode = {"task": "[health] echo", "score": 0.0, "failure": "no transcript was read", "reply": None}
    row = _row(_answered(selected=False, selection=selection, candidate_episodes=[episode], patch=PATCH))
    page = request_page(_record(), [CREATION, row], now=1_100.0)
    result = _section(page, "Result")
    assert "the task &#x27;[health] echo&#x27; failed: no transcript was read" in result or (
        "the task '[health] echo' failed: no transcript was read" in result
    )
    assert "rephrase" not in result and "the change itself was not judged" in result
    version = build_release_page(1, [CREATION, row])
    assert "Missed" in version and "no transcript was read" in version


def test_an_evaluation_that_never_ran_says_so_on_both_pages_with_no_candidate_score() -> None:
    """Every candidate episode failed before a score (the runner was not found): the checks judged nothing, so both
    pages say the evaluation could not run and quote the cause, and the version page shows no candidate score. Both
    pages say the floor tasks were set before the request and do not test what it asks for."""
    selection = {
        "policy": "floor",
        "reason": "candidate missed the floor on 1 of 1 tasks",
        "metrics": {"floor_score": 1},
    }
    cause = "harness binary reef-terminus not found"
    episode = {"task": "/checkout/tasks/health", "score": None, "failure": cause, "reply": None}
    metrics = _answered(
        selected=False, selection=selection, candidate_episodes=[episode], candidate_score=0.0, patch=PATCH
    )
    row = _row(metrics)
    result = _section(request_page(_record(), [CREATION, row], now=1_100.0), "Result")
    assert f"could not be evaluated: {cause}" in result and "did not pass the checks" not in result
    note = "The floor tasks (health) were set before this request"
    assert note in result
    version = build_release_page(1, [CREATION, row])
    assert "The evaluation could not run" in version and f"<dt>Could not run</dt><dd>{cause}</dd>" in version
    assert "Candidate score" not in version and "Missed" not in version and note in version
    # A scored miss keeps the checks' own words and its candidate score.
    scored = _row({**metrics, "candidate_episodes": [{**episode, "score": 0.0, "failure": None, "reply": "no"}]})
    version = build_release_page(1, [CREATION, scored])
    assert "The evaluation could not run" not in version and "Candidate score" in version


def test_the_next_action_of_a_published_release_names_what_it_requires_where_the_harness_runs() -> None:
    metrics = _answered(selected=True, patch=PATCH)
    metrics["training_request"]["requires"] = [{"name": "DEEPSEEK_API_KEY", "kind": "env", "prompt": "Your key"}]
    result = _section(request_page(_record(), [CREATION, _row(metrics)], now=1_100.0), "Result")
    assert "<code>GET /reef/harness</code>" in result
    assert "It requires DEEPSEEK_API_KEY where the harness runs." in result


def test_a_step_that_failed_during_its_evaluation_says_so_and_shows_its_candidate() -> None:
    """A skip row whose candidate reached its evaluation says where the step failed, on both pages, and lists the
    change the candidate made instead of saying no change was produced."""
    metrics = _answered(
        skipped="instruction failed",
        error="AttributeError: 'str' object has no attribute 'get'",
        failed_stage="evaluating",
        patch=PATCH,
    )
    failed = _row(metrics, release_id="rel-0")
    page = request_page(_record(), [CREATION, failed], now=1_100.0)
    result = _section(page, "Result")
    assert "The step failed during its evaluation" in result and "before evaluation" not in result
    assert "rules.md" in _section(page, "Proposed changes") and "No changes were produced" not in page
    version = build_release_page(1, [CREATION, failed])
    assert "The step failed during its evaluation" in version and "before evaluation" not in version


def test_a_settled_request_shows_what_the_method_said_and_why_a_step_failed() -> None:
    """The method's summary, its line breaks kept, sits before the change, and its notes after it; a step whose
    method raised shows the error in the Result."""
    summary = "Terminus has no session, so a mode cannot be entered.\n\nHow to use: nothing to use."
    skipped = _row(
        _answered(skipped=UNCHANGED, method_summary=summary, method_notes={"declined": True}), release_id="rel-0"
    )
    page = request_page(_record(), [CREATION, skipped], now=1_100.0)
    assert REFRESH not in page and '<span class="skipped">No changes</span>' in page
    assert _sections(page) == ["Request", "Result", "Method summary", "Proposed changes", "Method notes"]
    assert f"produced no change ({UNCHANGED}); nothing changed" in _section(page, "Result")
    assert _section(page, "Method summary").strip() == (
        "<p>Terminus has no session, so a mode cannot be entered.\n\nHow to use: nothing to use.</p>"
    )
    assert "No changes were produced by this step." in _section(page, "Proposed changes")
    assert "&quot;declined&quot;: true" in _section(page, "Method notes")
    # A published step without either shows the result and the change alone.
    plain = request_page(_record(), [CREATION, _row(_answered(selected=True, patch=PATCH))], now=1_100.0)
    assert _sections(plain) == ["Request", "Result", "What changed"]
    failed = _row(_answered(skipped="instruction failed", error="RuntimeError: poison method"), release_id="rel-0")
    page = request_page(_record(), [CREATION, failed], now=1_100.0)
    assert '<span class="failed">Failed</span>' in page
    assert "The step failed before evaluation. Nothing was published" in _section(page, "Result")
    assert "<h3>Error</h3><p>RuntimeError: poison method</p>" in page


def test_the_page_module_is_ascii_and_the_builder_escapes_the_request_the_method_text_and_the_link() -> None:
    MODULE.read_text(encoding="utf-8").encode("ascii")
    text = 'text me <script>alert(1)</script> & "quote" café'
    requires = [
        {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
        {"name": "<x>", "kind": "service", "prompt": "Sign in to <x>"},
    ]
    method = {"method_summary": "<b>failed</b>", "method_notes": {"uncovered": ["<i>off</i>"]}}
    row = _row(_answered(skipped=UNCHANGED, **method), release_id="rel-0")
    page = request_page(
        _record(text=text, requires=requires),
        [CREATION, row],
        link_query={"scenario": "a b", "key": "k&<"},
        now=1_100.0,
    )
    page.encode("ascii")
    assert "<script>" not in page and "<b>failed</b>" not in page and "<i>" not in page
    assert "text me &lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;quote&quot; caf&#233;" in page
    request = _section(page, "Request")
    assert "<summary>Needs from your machine (2)</summary>" in request
    assert "<thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th></tr></thead>" in request
    assert '<tr><td>TWILIO_SID</td><td>env</td><td class="id">TWILIO_SID</td><td></td></tr>' in request
    assert '<tr><td>&lt;x&gt;</td><td>service</td><td class="id"></td><td>Sign in to &lt;x&gt;</td></tr>' in request
    assert "Needs from your machine" not in request_page(_record(text=text), [CREATION], now=1_100.0)
    assert "&lt;b&gt;failed&lt;/b&gt;" in page and "&quot;&lt;i&gt;off&lt;/i&gt;&quot;" in page
    assert 'href="/reef/harness/releases/1/page?scenario=a+b&amp;key=k%26%3C"' in page
    assert '<meta name="referrer" content="no-referrer">' in page
    queued = request_page(_record(text=text), [CREATION], now=1_100.0)
    queued.encode("ascii")
    assert "<script>" not in queued and "caf&#233;" in queued


def holding_method(entered: Event, release: Event, failure: bool = False):
    """A method that holds its step open until ``release``, then writes ``PATCH``, or raises with ``failure``."""

    def method(workspace, services):
        entered.set()
        release.wait(30)
        if failure:
            raise RuntimeError("the E2B command connection was interrupted")
        return TreePatch(PATCH)

    return method


def test_a_running_request_lists_the_methods_activity_newest_first() -> None:
    activity = (
        {"at": 1_000.0, "kind": "method", "text": "the coding agent started on the request"},
        {"at": 1_065.0, "kind": "agent", "text": "write harness/extensions/speak.ts"},
        {"at": 1_130.0, "kind": "provider", "text": "/v1/audio/speech not/real -> 400: <no model>", "failed": True},
    )
    proposing = StepProgress(RECORD_ID, "proposing", started_at=1_000.0, step_record="/steps/1", activity=activity)
    page = request_page(_record(), [CREATION], progress=proposing, now=1_200.0)
    assert _sections(page) == ["Request", "Progress", "Activity"]
    listed = _section(page, "Activity")
    lines = re.findall(
        r"<li( class=\"failed\")?><span class=\"at\">([^<]+)</span><span class=\"kind\">(\w+)</span>", listed
    )
    assert lines == [(' class="failed"', "+2:10", "provider"), ("", "+1:05", "agent"), ("", "+0:00", "method")]
    # The newest line says how long ago it happened: a long wait shows as that. Text is escaped.
    assert '&lt;no model&gt; <span class="age">&middot; 70 s ago</span>' in listed
    assert "earlier line" not in listed

    many = tuple({"at": 1_000.0 + i, "kind": "model", "text": f"call {i}"} for i in range(MAX_ACTIVITY_SHOWN + 5))
    listed = _section(
        request_page(_record(), [CREATION], progress=replace(proposing, activity=many), now=1_200.0), "Activity"
    )
    assert (
        listed.count("<li") == MAX_ACTIVITY_SHOWN
        and f"call {MAX_ACTIVITY_SHOWN + 4}" in listed
        and "call 4<" not in listed
    )
    assert "5 earlier lines not shown" in listed

    quiet = request_page(_record(), [CREATION], progress=replace(proposing, activity=()), now=1_200.0)
    assert "Nothing yet: the method has not called a model or scored a tree." in _section(quiet, "Activity")
    # Another request's step lists nothing here.
    other = request_page(_record(), [CREATION], progress=replace(proposing, request_id="another"), now=1_200.0)
    assert _sections(other) == ["Request", "Progress"]


@pytest.mark.parametrize("failure", [False, True])
def test_the_page_follows_a_filed_request_from_proposing_to_its_result_by_a_browser_link(
    tmp_path: Path, failure: bool
) -> None:
    entered, release = Event(), Event()
    recipe = replace(harness_recipe(tmp_path, holding_method(entered, release, failure)), training_mode="manual")
    dispatcher = harness_dispatcher(tmp_path, recipe)
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    assert scenario is not None
    headers = {"x-reef-scenario": SCENARIO, "Authorization": "Bearer secret"}

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, tokens="secret")))
        await client.start_server()
        try:
            body = {"text": TEXT, "session": SESSION, "release_id": "rel-0"}
            response = await client.post("/reef/train", headers=headers, json=body)
            assert response.status == 200, await response.text()
            answer = await response.json()
            record_id = answer["agent_record_id"]
            link = f"/reef/harness/requests/{record_id}/page"
            # The service hands out the links a browser opens: the scenario and a page key, never the token.
            assert answer["page_path"] == f"{link}?{urlencode(QUERY)}"
            catalog = await (await client.get("/reef/harness/releases", headers=headers)).json()
            assert [row["page_path"] for row in catalog["releases"]] == [
                f"/reef/harness/releases/0/page?{urlencode(QUERY)}"
            ]
            assert await asyncio.to_thread(entered.wait, 10)

            # The link a browser opens: no header, the scenario and the page key in the query.
            response = await client.get(link, params=QUERY)
            page = await response.text()
            assert response.status == 200 and response.headers["content-type"].startswith("text/html"), page
            assert response.headers["Cache-Control"] == "no-store"
            page.encode("ascii")
            assert REFRESH in page and f"<title>Harness request {record_id[:8]}</title>" in page
            assert (
                '<span class="proposing">Designing the change</span>' in page and f'<p class="text">{TEXT}</p>' in page
            )
            assert "into the step" in page

            # The same reading as JSON, for a client with no browser: the phase the TUI's spinner names.
            progress_route = f"/reef/harness/requests/{record_id}/progress"
            response = await client.get(progress_route, headers=headers)
            progress = await response.json()
            assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
            assert progress["state"] == "proposing" and progress["settled"] is False and progress["step"] is None
            assert progress["request_id"] == record_id
            assert progress["meaning"] == STATE_WORDS["proposing"]
            assert progress["activity"] == []  # the holding method has called no model
            # A JSON route reads the headers alone: the page's key is refused here.
            assert (await client.get(progress_route, params=QUERY)).status == 401
            assert (await client.get("/reef/harness/requests/nope/progress", headers=headers)).status == 404

            # The version page opens the same way; the wrong key, no key, a key elsewhere or the token in the query
            # does not.
            response = await client.get("/reef/harness/releases/0/page", params=QUERY)
            assert response.status == 200 and "<title>Harness v0</title>" in await response.text()
            # The step the request is running has no row yet: its page says so and links the request, not a 404.
            response = await client.get("/reef/harness/releases/1/page", params=QUERY)
            running = await response.text()
            assert response.status == 200 and "This step is running" in running and REFRESH in running
            assert f'href="{link}?scenario=agents&amp;key={KEY}">Follow the request' in running
            assert (await client.get("/reef/harness/releases/2/page", params=QUERY)).status == 404
            response = await client.get(link, params={**QUERY, "key": "nope"})
            assert response.status == 401 and await response.text() == "invalid service token"
            assert (await client.get(link, params={"scenario": SCENARIO, "token": "secret"})).status == 401
            response = await client.get(link, params={"scenario": SCENARIO})
            assert response.status == 401
            response = await client.get("/reef/harness/releases", params=QUERY)
            assert response.status == 401
            # The header wins when present, and without a scenario from anywhere the page is a 400.
            response = await client.get(link, params=QUERY, headers={"Authorization": "Bearer nope"})
            assert response.status == 401
            response = await client.get(link, headers={"Authorization": "Bearer secret"})
            assert response.status == 400
            # The headers keep working, and win over a query scenario.
            response = await client.get(link, headers=headers, params={"scenario": "other"})
            assert (
                response.status == 200
                and '<span class="proposing">Designing the change</span>' in await response.text()
            )

            release.set()
            for _ in range(200):
                response = await client.get(link, params=QUERY)
                page = await response.text()
                if REFRESH not in page:
                    break
                await asyncio.sleep(0.05)
            assert response.status == 200 and REFRESH not in page, page
            # The settled request reads as settled on the JSON route too, naming the step its row landed as.
            settled = await (await client.get(progress_route, headers=headers)).json()
            assert settled["settled"] is True and settled["step"] == 1
            if failure:
                assert settled["state"] == "failed"
                assert '<span class="failed">Failed</span>' in page
                assert "the E2B command connection was interrupted" in page
                assert "Pull it" not in page
            else:
                assert settled["state"] == "selected"
                assert '<span class="selected">Published</span>' in page
                assert "Published as release " in page and "<code>GET /reef/harness</code>" in page
                assert '<span class="tag operation-write">write</span><span class="node-id">rules.md</span>' in page
            assert f'href="/reef/harness/releases/1/page?scenario=agents&amp;key={KEY}">View v1' in page

            # An unknown id, and a record that is no training instruction, are 404s naming the id.
            response = await client.get("/reef/harness/requests/nope/page", params=QUERY)
            assert response.status == 404 and "has no harness request 'nope'" in await response.text()
            inference = AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.INFERENCE,
                payload={"messages": [{"role": "user", "content": "q"}]},
                agent_record_id="i1",
            )
            await asyncio.to_thread(dispatcher.accept_record, inference)
            response = await client.get("/reef/harness/requests/i1/page", params=QUERY)
            assert response.status == 404 and "has no harness request 'i1'" in await response.text()
        finally:
            release.set()
            await client.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        dispatcher.close()


def test_a_config_file_shows_its_text_on_the_version_page() -> None:
    """An opencode agent's prompt holds non ASCII text: the step page shows the file as text, the non ASCII as
    character references a browser shows as the text itself."""
    agent = {"agent": {"chat": {"mode": "primary", "prompt": "Only search the web — nothing else."}}}
    patch = {"config/primary.json": json.dumps(agent, ensure_ascii=False, indent=2) + "\n"}
    step = _row(_answered(selected=True, patch=patch))
    version = build_release_page(1, [CREATION, step])
    version.encode("ascii")
    shown = html.unescape(version.split("<script", 1)[0])
    assert '"prompt": "Only search the web — nothing else."' in shown
