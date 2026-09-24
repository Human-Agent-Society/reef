"""The page per filed harness request: the step's state while it runs, the result once its row lands.

``GET /reef/harness/requests/{record_id}/page`` renders it from the request's
agent record, the catalog and the running step's progress, and a browser opens
it by a link that carries the scenario and the token as query parameters. The
live chain here runs in ``training_mode: manual`` with a proposer that holds
its step open until the test has read the page mid-step.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_proposals import _dispatcher, _recipe

from reef.core import AgentRecord, RequestType
from reef.service.app import create_app
from reef.service.release_page import build_release_page
from reef.service.request_page import (
    MAX_ACTIVITY_SHOWN,
    REFRESH_SECONDS,
    STATE_WORDS,
    build_request_page,
    settled_step,
)
from reef.train.cordis_backend import Mutation, StepProgress
from reef.train.cordis_backend.strategies import StepProposal

MODULE = Path(__file__).parents[2] / "reef" / "service" / "request_page.py"
REFRESH = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
RECORD_ID = "3f1c2a9d0b7e4c5d8e9f0a1b2c3d4e5f"
TEXT = "text me when the run is blocked"
SESSION = "3f1c2a9d0b7e"
MARKER = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})
SCENARIO = "agents"
QUERY = {"scenario": SCENARIO, "token": "secret"}


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
MUTATION = {"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}


def _answered(**extra) -> dict:
    """The metrics of the row that answered the request, the trainer's ``training_request`` stamp included."""
    request = {"id": RECORD_ID, "text": TEXT, "session": SESSION, "release_id": "rel-0", "requires": []}
    return {"steps": 1, "training_request": request, **extra}


def _sections(page: str) -> list[str]:
    return re.findall(r"<h2>([^<]+)</h2>", page)


def _section(page: str, name: str) -> str:
    _, _, tail = page.partition(f"<h2>{name}</h2>")
    body, _, _ = tail.partition("</section>")
    return body


def test_both_harness_pages_carry_the_readme_logo_from_the_shared_chrome() -> None:
    logo = (MODULE.parents[2] / "docs" / "assets" / "reef-logo-light.svg").read_text().strip()
    page = build_request_page(_record(), [CREATION], now=1_042.0)
    assert logo in page
    # The version page draws with the same chrome, so one logo serves both.
    assert logo in build_release_page(0, [CREATION])


def test_the_logo_leads_to_the_served_head_the_request_is_asked_against() -> None:
    """The top bar is navigable on both pages, and only ever to a page route a browser can open."""
    rows = [CREATION, _row(_answered(selected=True, mutation=MUTATION))]
    page = build_request_page(_record(), rows, link_query=QUERY, now=1_100.0)
    head = "/reef/harness/releases/1/page?scenario=agents&amp;token=secret"
    assert f'<a class="brand" href="{head}" aria-label="Harness home">' in page
    assert f'<a href="{head}">Harness</a>' in page and "<b>Requests</b>" in page
    for href in re.findall(r'href="([^"]+)"', page):
        assert re.match(r"^/reef/harness/releases/\d+/page(\?|$)", href.replace("&amp;", "&")), href
    # A catalog with no served head leaves the crumb as text rather than a broken link.
    pending_only = [_row(_answered(selected=True), release_id="rel-0", parent=None, pending=True)]
    bare = build_request_page(_record(), pending_only, now=1_100.0)
    assert '<div class="brand">' in bare and "<span>Harness</span>" in bare


def test_a_queued_request_reloads_and_says_no_step_has_taken_it() -> None:
    page = build_request_page(_record(), [CREATION], now=1_042.0)
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


def test_a_running_request_shows_the_steps_phase_its_elapsed_time_and_the_gates_size() -> None:
    evaluating = StepProgress(RECORD_ID, "evaluating", started_at=900.0, step_record="/work/steps/1", episodes_total=2)
    page = build_request_page(_record(), [CREATION], progress=evaluating, now=1_100.0)
    progress = _section(page, "Progress")
    assert REFRESH in page and '<span class="evaluating">Checking the harness</span>' in progress
    assert "the evaluation is running the candidate through its episodes" in progress
    assert "<dt>Step time</dt><dd>3 min 20 s into the step</dd>" in progress
    assert "<dt>Evaluation episodes</dt><dd>2 in the evaluation</dd>" in progress
    assert '<dt>Step record</dt><dd class="id">/work/steps/1</dd>' in progress
    assert "<dt>Waiting</dt>" not in progress

    proposing = StepProgress(RECORD_ID, "proposing", started_at=1_058.0, step_record=None)
    page = build_request_page(_record(), [CREATION], progress=proposing, now=1_100.0)
    progress = _section(page, "Progress")
    assert (
        '<span class="proposing">Designing the change</span>' in progress
        and "the proposer is writing the change" in progress
    )
    assert "<dt>Step time</dt><dd>42 s into the step</dd>" in progress
    assert "Evaluation episodes" not in progress and "Step record" not in progress

    # Another request's step says nothing about this one, which still waits.
    other = replace(evaluating, request_id="another")
    page = build_request_page(_record(), [CREATION], progress=other, now=1_100.0)
    assert '<span class="queued">Queued</span>' in page and "<dd>100 s</dd>" in page
    assert "Evaluation episodes" not in page
    page = build_request_page(_record(), [CREATION], consumed=True, now=1_100.0)
    assert '<span class="running">In progress</span>' in page and "its row follows" in page and REFRESH in page
    page = build_request_page(
        _record(),
        [CREATION],
        progress=StepProgress(RECORD_ID, "settling", started_at=1_099.0, step_record=None),
        now=1_100.0,
    )
    assert (
        '<span class="settling">Saving the result</span>' in page and "committing its row" in page and REFRESH in page
    )


def test_a_settled_selected_request_carries_the_result_the_mutation_and_the_link_with_its_query() -> None:
    rows = [CREATION, _row(_answered(selected=True, published=True, mutation=MUTATION))]
    page = build_request_page(_record(), rows, link_query=QUERY, now=1_100.0)
    page.encode("ascii")
    assert REFRESH not in page and "<title>Harness request 3f1c2a9d</title>" in page
    assert _sections(page) == ["Request", "Result", "What changed"]
    selection_result = _section(page, "Result")
    assert '<span class="selected">Published</span>' in selection_result
    assert (
        "Published as release rel-1. Your current session keeps its installed harness until you choose to update."
        in selection_result
    )
    assert '<dt>Release</dt><dd class="id">rel-1</dd>' in selection_result
    assert 'href="/reef/harness/releases/1/page?scenario=agents&amp;token=secret">View v1' in selection_result
    assert "<h3>Error</h3>" not in selection_result and "Proposer failure" not in selection_result
    changed = _section(page, "What changed")
    assert '<span class="tag operation-create">create</span><span class="node-id">r1</span>' in changed
    assert '<span class="tag">rules</span>' in changed
    assert "<code>/versions v1 install</code>" in selection_result
    assert settled_step(rows, RECORD_ID) == 1
    bare = build_request_page(_record(), rows, now=1_100.0)
    assert 'href="/reef/harness/releases/1/page">View v1' in bare

    # A rejected proposal is labeled as proposed, never as an applied change.
    second = {"op": "update", "id": "ext", "options": {"name": "code_extension", "config": {"code": "x"}}}
    rejected = _row(
        _answered(
            selected=False,
            mutations=[MUTATION, second],
            selection={"reason": "candidate missed the floor on 1 of 1 tasks"},
        )
    )
    page = build_request_page(_record(), [CREATION, rejected], now=1_100.0)
    assert '<span class="rejected">Not selected</span>' in page
    assert "did not pass the checks (candidate missed the floor on 1 of 1 tasks); nothing changed" in page
    assert "rephrase or split the request" in page
    assert "What changed" not in _sections(page)
    changed = _section(page, "Proposed changes")
    assert changed.count("<li>") == 2
    assert '<span class="tag operation-update">update</span><span class="node-id">ext</span>' in changed


def test_a_pending_request_names_the_promote_and_reads_promoted_once_a_promote_row_names_it() -> None:
    pending = _row(_answered(selected=True, mutation=MUTATION), pending=True)
    page = build_request_page(_record(), [CREATION, pending], now=1_100.0)
    assert '<span class="pending">Ready for review</span>' in page
    assert "Proposed changes" in _sections(page)
    assert "Release rel-1 is ready. This change includes an extension" in page
    assert REFRESH not in page
    assert "<code>/versions v1 install</code>" in page
    promote = _row({}, release_id="rel-2", parent="rel-0", operation="promote", rollback_target_release_id="rel-1")
    page = build_request_page(_record(), [CREATION, pending, promote], now=1_100.0)
    assert '<span class="promoted">Promoted at v2</span>' in page
    assert "passed the checks and was promoted at v2; the release that step published serves it" in page
    assert "What changed" in _sections(page)


def test_a_rejected_request_names_the_missed_episode_on_both_pages() -> None:
    selection = {"reason": "candidate missed the floor on 1 of 1 tasks", "metrics": {"floor_score": 1.0}}
    episode = {"task": "[health] echo", "score": 0.0, "failure": "no transcript was read", "reply": None}
    notes = {"review": {"result": "partial", "covered": [], "uncovered": ["no off switch"]}}
    row = _row(
        _answered(
            selected=False, selection=selection, candidate_episodes=[episode], mutation=MUTATION, proposal_notes=notes
        )
    )
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, row], now=1_100.0)
    result = _section(page, "Result")
    assert "the task &#x27;[health] echo&#x27; failed: no transcript was read" in result or (
        "the task '[health] echo' failed: no transcript was read" in result
    )
    assert "rephrase" not in result and "the change itself was not judged" in result
    review = _section(page, "Review")
    assert "Review notes" in review and "The checks decided this result" in review
    version = build_release_page(1, [CREATION, row])
    assert "Missed" in version and "no transcript was read" in version


def test_answers_the_proposer_wrote_again_show_under_review_on_both_pages() -> None:
    """An answer whose form slipped is written again; the Review says what the kept one replaced."""
    notes = {
        "review": {"result": "complete", "covered": ["chat"], "uncovered": []},
        "dropped_attempts": ["answer 1: the harness refused the entries: bad frontmatter"],
    }
    row = _row(_answered(selected=True, published=True, mutation=MUTATION, proposal_notes=notes))
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, row], now=1_100.0)
    review = _section(page, "Review")
    assert "Answers written again" in review and "the harness refused the entries: bad frontmatter" in review
    version = build_release_page(1, [CREATION, row])
    assert "Answers written again" in version and "bad frontmatter" in version


def test_off_pi_the_next_action_is_the_wrappers_command_in_a_terminal_not_a_pi_session() -> None:
    """hermes has no /versions and no reef-pi: a published release is installed with reef-hermes update, a
    release waiting for review is served with reef-hermes wait on the request, and the version page's Setup
    note names reef-hermes setup."""
    selected = _row(_answered(selected=True, published=True, mutation=MUTATION))
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, selected], now=1_100.0, adapter="hermes")
    result = _section(page, "Result")
    assert "<code>reef-hermes update</code>" in result and "start reef-hermes again" in result
    assert "/versions" not in page and "reef-pi" not in page
    pending = _row(_answered(selected=True, mutation=MUTATION), pending=True)
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, pending], now=1_100.0, adapter="hermes")
    assert f"<code>reef-hermes wait {RECORD_ID}</code>" in _section(page, "Result")
    requires = [{"name": "DEEPSEEK_API_KEY", "kind": "env", "prompt": "Your DeepSeek key"}]
    metrics = _answered(selected=True, published=True, mutation=MUTATION)
    metrics["training_request"]["requires"] = requires
    step = _row(metrics)
    version = build_release_page(1, [CREATION, step], adapter="dsh")
    assert "reef-dsh setup lists these" in version and "reef-pi setup" not in version


@pytest.mark.parametrize("review_key", ["result", "verdict"])
def test_a_skipped_request_shows_why_the_proposer_produced_nothing_and_what_the_review_left_uncovered(
    review_key,
) -> None:
    notes = {
        "design": "one rules entry",
        "failure": "model call failed after 60.0 s (max_tokens=16384): timeout",
        "review": {review_key: "partial", "covered": ["the trigger"], "uncovered": ["a way to turn it off"]},
    }
    skipped = _row(_answered(skipped="no proposal", proposal_notes=notes), release_id="rel-0")
    page = build_request_page(_record(), [CREATION, skipped], now=1_100.0)
    assert REFRESH not in page and '<span class="failed">Failed</span>' in page
    assert _sections(page) == ["Request", "Result", "Proposed changes", "Review", "Design"]
    assert "<p>one rules entry</p>" in _section(page, "Design")
    selection_result = _section(page, "Result")
    assert "The step failed before evaluation. Nothing was published" in selection_result
    assert (
        "<h3>Proposer failure</h3><p>model call failed after 60.0 s (max_tokens=16384): timeout</p>"
        in selection_result
    )
    assert "No changes were produced by this step." in _section(page, "Proposed changes")
    review = _section(page, "Review")
    assert '<span class="partial">Partial</span>' in review and "<li>a way to turn it off</li>" in review
    assert "the trigger" not in review
    complete = {"review": {"result": "complete", "covered": ["all of it"], "uncovered": []}}
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes=complete))
    page = build_request_page(_record(), [CREATION, row], now=1_100.0)
    assert '<span class="complete">Complete</span>' in page and "Nothing left uncovered." in page
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes={"design": "plan"}))
    page = build_request_page(_record(), [CREATION, row], now=1_100.0)
    assert "<h2>Review</h2>" not in page and _sections(page)[-1] == "Design" and "<h2>How to use</h2>" not in page
    failed = _row(_answered(skipped="instruction failed", error="RuntimeError: poison proposer"), release_id="rel-0")
    page = build_request_page(_record(), [CREATION, failed], now=1_100.0)
    assert '<span class="failed">Failed</span>' in page
    assert "<h3>Error</h3><p>RuntimeError: poison proposer</p>" in page


@pytest.mark.unit
def test_a_settled_request_ends_with_the_design_and_how_to_use_the_change() -> None:
    """The proposer's design closes with a How to use section; the page shows the two as its last cards, the plan
    and the usage of the change, escaped."""
    design = (
        "Restated: read each answer aloud.\n\nTrigger: the /speak command; state: the last answer, from the session.\n\n"
        "## How to use\n\nType /speak after an answer; it plays through <afplay>. /speak off stops it."
    )
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes={"design": design}))
    page = build_request_page(_record(), [CREATION, row], now=1_100.0)
    assert _sections(page) == ["Request", "Result", "What changed", "Design", "How to use"]
    assert _section(page, "Design").strip() == (
        "<p>Restated: read each answer aloud.\n\nTrigger: the /speak command; state: the last answer, from the session.</p>"
    )
    assert _section(page, "How to use").strip() == (
        "<p>Type /speak after an answer; it plays through &lt;afplay&gt;. /speak off stops it.</p>"
    )
    # A plain "How to use:" line splits the same way, wherever the proposer put the heading marks.
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes={"design": "A plan.\nHow to use:\nRun /x."}))
    page = build_request_page(_record(), [CREATION, row], now=1_100.0)
    assert "<p>A plan.</p>" in _section(page, "Design") and "<p>Run /x.</p>" in _section(page, "How to use")


@pytest.mark.unit
def test_a_step_whose_review_did_not_run_says_so_instead_of_dropping_the_section() -> None:
    """The review is the one check of whether the entries deliver the request; when its call fails the page says
    that nothing checked, rather than reading like a step that simply had no review."""
    notes = {"design": "plan", "review_failure": "model call failed after 91.9 s: non-text content"}
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes=notes))
    page = build_request_page(_record(), [CREATION, row], now=1_100.0)
    review = _section(page, "Review")
    assert "did not run, so nothing checked whether they deliver it" in review
    assert "non-text content" in review


def test_the_page_module_is_ascii_and_the_builder_escapes_the_request_the_notes_and_the_link() -> None:
    MODULE.read_text(encoding="utf-8").encode("ascii")
    text = 'text me <script>alert(1)</script> & "quote" caf\u00e9'
    requires = [
        {"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"},
        {"name": "<x>", "kind": "service", "prompt": "Sign in to <x>"},
    ]
    notes = {"failure": "<b>failed</b>", "review": {"result": "partial", "covered": [], "uncovered": ["<i>off</i>"]}}
    row = _row(_answered(skipped="no proposal", proposal_notes=notes), release_id="rel-0")
    page = build_request_page(
        _record(text=text, requires=requires),
        [CREATION, row],
        link_query={"scenario": "a b", "token": "t&<"},
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
    assert "Needs from your machine" not in build_request_page(_record(text=text), [CREATION], now=1_100.0)
    assert "&lt;b&gt;failed&lt;/b&gt;" in page and "<li>&lt;i&gt;off&lt;/i&gt;</li>" in page
    assert 'href="/reef/harness/releases/1/page?scenario=a+b&amp;token=t%26%3C"' in page
    assert '<meta name="referrer" content="no-referrer">' in page
    queued = build_request_page(_record(text=text), [CREATION], now=1_100.0)
    queued.encode("ascii")
    assert "<script>" not in queued and "caf&#233;" in queued


def _propose_holding(entered: Event, release: Event, failure: bool = False):
    def propose(nodes, samples, models, *, requests=()):
        entered.set()
        release.wait(30)
        if failure:
            return StepProposal((), {"failure": "the E2B command connection was interrupted"})
        return MARKER

    return propose


def test_a_running_request_lists_the_proposers_activity_newest_first() -> None:
    activity = (
        {"at": 1_000.0, "kind": "proposer", "text": "the coding agent started on the request"},
        {"at": 1_065.0, "kind": "agent", "text": "write harness/extensions/speak.ts"},
        {"at": 1_130.0, "kind": "provider", "text": "/v1/audio/speech not/real -> 400: <no model>", "failed": True},
    )
    proposing = StepProgress(RECORD_ID, "proposing", started_at=1_000.0, step_record=None, activity=activity)
    page = build_request_page(_record(), [CREATION], progress=proposing, now=1_200.0)
    assert _sections(page) == ["Request", "Progress", "Activity"]
    listed = _section(page, "Activity")
    lines = re.findall(
        r"<li( class=\"failed\")?><span class=\"at\">([^<]+)</span><span class=\"kind\">(\w+)</span>", listed
    )
    assert lines == [(' class="failed"', "+2:10", "provider"), ("", "+1:05", "agent"), ("", "+0:00", "proposer")]
    # The newest line says how long ago it happened: a long wait shows as that. Text is escaped.
    assert '&lt;no model&gt; <span class="age">&middot; 70 s ago</span>' in listed
    assert "earlier line" not in listed

    many = tuple({"at": 1_000.0 + i, "kind": "model", "text": f"call {i}"} for i in range(MAX_ACTIVITY_SHOWN + 5))
    listed = _section(
        build_request_page(_record(), [CREATION], progress=replace(proposing, activity=many), now=1_200.0), "Activity"
    )
    assert (
        listed.count("<li") == MAX_ACTIVITY_SHOWN
        and f"call {MAX_ACTIVITY_SHOWN + 4}" in listed
        and "call 4<" not in listed
    )
    assert "5 earlier lines not shown" in listed

    quiet = build_request_page(_record(), [CREATION], progress=replace(proposing, activity=()), now=1_200.0)
    assert "Nothing yet: the proposer has not called a model." in _section(quiet, "Activity")
    # Another request's step lists nothing here.
    other = build_request_page(_record(), [CREATION], progress=replace(proposing, request_id="another"), now=1_200.0)
    assert _sections(other) == ["Request", "Progress"]


@pytest.mark.parametrize("failure", [False, True])
def test_the_page_follows_a_filed_request_from_proposing_to_its_result_by_a_browser_link(
    tmp_path: Path, failure: bool
) -> None:
    entered, release = Event(), Event()
    recipe = replace(_recipe(tmp_path, _propose_holding(entered, release, failure)), training_mode="manual")
    dispatcher = _dispatcher(tmp_path, recipe)
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
            record_id = (await response.json())["agent_record_id"]
            link = f"/reef/harness/requests/{record_id}/page"
            assert await asyncio.to_thread(entered.wait, 10)

            # The link a browser opens: no header, the scenario and the token in the query.
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
            assert progress["activity"] == []  # the holding proposer has called no model
            # A JSON route reads the headers alone: the page's query token is refused here.
            assert (await client.get(progress_route, params=QUERY)).status == 401
            assert (await client.get("/reef/harness/requests/nope/progress", headers=headers)).status == 404

            # The version page opens the same way; the wrong token, no token or a token elsewhere does not.
            response = await client.get("/reef/harness/releases/0/page", params=QUERY)
            assert response.status == 200 and "<title>Harness v0</title>" in await response.text()
            # The step the request is running has no row yet: its page says so and links the request, not a 404.
            response = await client.get("/reef/harness/releases/1/page", params=QUERY)
            running = await response.text()
            assert response.status == 200 and "This step is running" in running and REFRESH in running
            assert f'href="{link}?scenario=agents&amp;token=secret">Follow the request' in running
            assert (await client.get("/reef/harness/releases/2/page", params=QUERY)).status == 404
            response = await client.get(link, params={**QUERY, "token": "nope"})
            assert response.status == 401 and await response.text() == "invalid service token"
            response = await client.get(link, params={"scenario": SCENARIO})
            assert response.status == 401
            response = await client.get("/reef/harness/releases", params=QUERY)
            assert response.status == 401
            # The header wins when present, and without a scenario from anywhere the page is a 400.
            response = await client.get(link, params=QUERY, headers={"Authorization": "Bearer nope"})
            assert response.status == 401
            response = await client.get(link, params={"token": "secret"})
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
                assert "/versions v1 install" not in page
            else:
                assert settled["state"] == "selected"
                assert '<span class="selected">Published</span>' in page
                assert "Published as release " in page and "/versions v1 install" in page
                assert '<span class="tag operation-create">create</span><span class="node-id">r1</span>' in page
            assert 'href="/reef/harness/releases/1/page?scenario=agents&amp;token=secret">View v1' in page

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
