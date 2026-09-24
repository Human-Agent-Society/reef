"""One HTML page per filed harness request: where its step stands, then the result once the step settles.

``GET /reef/harness/requests/{record_id}/page`` builds it from the request's
agent record, the scenario's catalog rows and the running step's progress.
The catalog row whose ``metrics.training_request.id`` is the record id
settles the request: the page then shows that row's result as the version
page words it, the mutations, what the review left uncovered, why the
proposer produced nothing when the step recorded that, and links the
version page; it ends with the proposer's design and its How to use
section, the plan and the usage of the change, when the step recorded them. Until then the page names the state the request is in
(``queued`` before a step takes it, ``proposing`` and ``evaluating`` from the
backend's progress, ``running`` while the trainer holds the request and the
backend reports no phase, ``settling`` while the row that consumed the
record lands), lists what the proposer has done so far (the step's
activity: model calls, and an agent's tool calls, checks and trials), and
reloads itself every ``REFRESH_SECONDS``, so a person opens the link right
after asking and watches. The chrome, the palette and
the status wording come from :mod:`reef.service.page_chrome`, which the
version page draws with too; this module adds only its own sections' style.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from reef.core.training_request import missed_episode_text, missed_episodes
from reef.harness.episodes.version_check import ships_version_check
from reef.service.page_chrome import document, escape, requires_table, stamp, status_span
from reef.service.release_page import design_sections, mutations_of, result_of, served_step, step_href
from reef.train.cordis_backend.contracts import StepProgress

#: Seconds between the page's own reloads while the request is not settled.
REFRESH_SECONDS = 5

# What the shared chrome does not draw: the progress strip, the two-column layout and the mutation list.
STYLE = """
main{max-width:1200px;margin:auto;padding:48px 40px 24px}
.journey{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:0 0 28px;padding:24px 28px;
list-style:none;background:var(--card);border:1px solid var(--line);border-radius:12px}
.journey li{position:relative;display:flex;gap:11px;align-items:center;min-width:0;color:var(--mute)}
.journey li:not(:last-child):after{content:"";height:1px;background:var(--line);flex:1;margin:0 20px 0 9px}
.stage-icon{width:28px;height:28px;display:grid;place-items:center;flex:none;border:1px solid var(--line);
border-radius:50%;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.stage-copy{font-size:12px;font-weight:550;white-space:nowrap}
.stage-copy small{display:block;font-size:11px;color:var(--mute);font-weight:400;margin-top:1px}
.journey .done .stage-icon{color:var(--good);background:var(--good-bg);border-color:transparent}
.journey .current{color:var(--status)}.journey .current .stage-icon{background:var(--status-bg);border-color:var(--status)}
.layout{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);gap:24px;align-items:start}
.request-card{grid-column:1;grid-row:1}.outcome-card{grid-column:2;grid-row:1 / span 3}
.changes-card,.review-card,.design-card,.usage-card{grid-column:1}
.design-card p,.usage-card p{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.7;margin:0}
.text{font-size:19px;
line-height:1.7;letter-spacing:-.3px;margin:0 0 30px;padding-left:20px;border-left:2px solid var(--accent)}
.metadata{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:0;padding-top:20px;border-top:1px solid var(--line)}
.record-id{grid-column:1 / -1}.outcome-summary{border-radius:8px;background:var(--status-bg);padding:18px;margin-bottom:22px}
.outcome-summary .status{padding:0;background:none;margin-bottom:10px;font-weight:650}
.outcome-summary p{font-size:13px;margin:0;line-height:1.8;overflow-wrap:anywhere}
.fact-list{margin:0}.fact-list>div{display:flex;justify-content:space-between;gap:18px;padding:12px 0;border-bottom:1px solid var(--line)}
.fact-list dt{font-size:12px;flex:none;margin:0}.fact-list dd{text-align:right;min-width:0}
.live-note{font-size:11px;color:var(--mute);display:flex;align-items:flex-start;gap:8px;margin:18px 0 0}
.live-dot{width:6px;height:6px;flex:none;border-radius:50%;background:var(--accent);margin-top:6px}
.version-link{display:flex;align-items:center;justify-content:space-between;margin-top:22px;border-radius:7px;
padding:11px 14px;background:var(--ink);color:var(--card);font-size:12px;font-weight:550;gap:12px}
.next-action{margin-top:24px;padding-top:22px;border-top:1px solid var(--line)}
.next-action h3{margin-bottom:10px}.next-action code{display:block;background:var(--code);border:1px solid var(--line);
border-radius:7px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace}
.next-action p{font-size:12px;color:var(--mute);margin:10px 0 0}
.version-link:hover{opacity:.88;text-decoration:none}
.mutations{list-style:none;padding:0;margin:0}.mutations li{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
padding:14px 0;border-top:1px solid var(--line)}.mutations li:first-child{border-top:0;padding-top:0}
.mutations li:last-child{padding-bottom:0}.mutations .node-id{flex:1;overflow-wrap:anywhere;min-width:0;font-size:13px}
.review-card p{font-size:13px;color:var(--mute)}
.review-card ul{margin:0;padding-left:18px;font-size:13px}.review-card li{padding:5px 0;overflow-wrap:anywhere}
.review-card h3{margin:22px 0 8px}
.request-meta{border-top:1px solid var(--line);padding-top:20px}.request-meta .metadata{border:0;padding-top:20px}
.requirements{margin-top:20px;border-top:1px solid var(--line);padding-top:18px}
.activity-card{grid-column:1}.activity{list-style:none;margin:0;padding:0;font-size:12px;max-height:560px;overflow:auto}
.activity li{display:grid;grid-template-columns:64px 72px minmax(0,1fr);gap:10px;padding:8px 0;border-top:1px solid var(--line)}
.activity li:first-child{border-top:0;padding-top:0}.activity .at{font:11px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute)}
.activity .kind{font-size:10px;line-height:1.9;color:var(--mute);text-transform:uppercase;letter-spacing:.5px}
.activity .what{overflow-wrap:anywhere;line-height:1.6}.activity .failed .what{color:var(--bad)}
.activity .age{color:var(--mute);white-space:nowrap}.activity-empty{font-size:13px;color:var(--mute);margin:0}
@media(min-width:1500px){main{padding-top:64px}}
@media(max-width:800px){main{padding:32px 24px 24px}
.layout{grid-template-columns:1fr}.request-card,.outcome-card,.changes-card,.review-card,.design-card,.usage-card,.activity-card{grid-column:auto;grid-row:auto}
.journey{padding:20px}.journey li:not(:last-child):after{margin:0 10px}.stage-copy small{display:none}}
@media(max-width:480px){main{padding:28px 16px 20px}.text{font-size:17px;padding-left:16px}
.journey{padding:18px 12px;gap:6px}.journey li{flex-direction:column;gap:7px}.journey li:not(:last-child):after{position:absolute;
left:calc(50% + 21px);right:calc(-50% + 15px);top:14px;margin:0}.stage-copy{font-size:11px}.metadata{gap:18px 12px}
.mutations li{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:start}
.mutations li>.tag:last-child{grid-column:2;justify-self:start}
.activity li{grid-template-columns:52px minmax(0,1fr)}.activity .kind{grid-column:2}.activity .what{grid-column:2}
.layout{gap:16px}}
"""

#: What each state means, in the words the page prints beside it.
STATE_WORDS = {
    "queued": "no step has taken the request yet; the trainer runs one step per instruction, oldest first",
    "proposing": "the proposer is writing the change; the activity shows what it has done so far",
    "evaluating": "the evaluation is running the candidate through its episodes",
    "running": "the step holds the request and reports no phase; its row follows",
    "settling": "the step that consumed the request is committing its row",
}


def settled_step(rows: Sequence[Mapping[str, object]], record_id: str) -> int | None:
    """The step whose row answered the request: the one whose ``metrics.training_request.id`` is ``record_id``."""
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
        if isinstance(request, Mapping) and request.get("id") == record_id:
            return index
    return None


def elapsed(seconds: float) -> str:
    """Seconds as a person reads them: ``42 s`` under two minutes, else ``3 min 05 s``."""
    whole = max(0, int(seconds))
    if whole < 120:
        return f"{whole} s"
    return f"{whole // 60} min {whole % 60:02d} s"


#: The most activity lines the page lists, newest first; the step record keeps every call.
MAX_ACTIVITY_SHOWN = 80


def step_clock(seconds: float) -> str:
    """Seconds into the step as a clock: ``+4:05``, or ``+1:02:05`` past an hour."""
    whole = max(0, int(seconds))
    hours, rest = divmod(whole, 3600)
    return f"+{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"+{rest // 60}:{rest % 60:02d}"


def activity_html(progress: StepProgress, now: float) -> str:
    """What the proposer has done so far, newest first, each line at its time into the step; the newest says how
    long ago it happened, so a long wait on one model call shows as that."""
    lines = list(progress.activity)[-MAX_ACTIVITY_SHOWN:]
    if not lines:
        return '<p class="activity-empty">Nothing yet: the proposer has not called a model.</p>'
    items = []
    for index, line in enumerate(reversed(lines)):
        at = line.get("at")
        at = float(at) if isinstance(at, (int, float)) else progress.started_at
        age = f' <span class="age">&middot; {escape(elapsed(now - at))} ago</span>' if index == 0 else ""
        failed = ' class="failed"' if line.get("failed") else ""
        items.append(
            f'<li{failed}><span class="at">{escape(step_clock(at - progress.started_at))}</span>'
            f'<span class="kind">{escape(line.get("kind"))}</span>'
            f'<span class="what">{escape(line.get("text"))}{age}</span></li>'
        )
    earlier = len(progress.activity) - len(lines)
    more = (
        f'<p class="live-note">{earlier} earlier line{"s" if earlier != 1 else ""} not shown; the step record keeps '
        "every call.</p>"
        if earlier > 0
        else ""
    )
    return '<ol class="activity" aria-label="Proposer activity, newest first">' + "".join(items) + "</ol>" + more


def request_state(record: Mapping[str, object], progress: StepProgress | None, consumed: bool) -> str:
    """Read the backend phase or reserved batch; an unreserved request is queued."""
    if progress is not None and progress.request_id == record["agent_record_id"]:
        return "evaluating" if progress.phase == "gating" else progress.phase
    if consumed:
        return "running"
    return "queued"


def request_html(record: Mapping[str, object]) -> str:
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    parts = [
        f'<p class="text">{escape(payload.get("text"))}</p>',
        '<details class="request-meta"><summary>Session and request details</summary><dl class="metadata">',
        f'<div><dt>Session</dt><dd class="id">{escape(payload.get("session") or "Not provided")}</dd></div>',
        f'<div><dt>Base release</dt><dd class="id">{escape(payload.get("release_id") or "Not provided")}</dd></div>',
        f'<div class="record-id"><dt>Request ID</dt><dd class="id">{escape(record["agent_record_id"])}</dd></div>',
    ]
    filed = record.get("created_at")
    if isinstance(filed, (int, float)):
        parts.append(f'<div class="record-id"><dt>Submitted</dt><dd>{stamp(filed)}</dd></div>')
    parts.append("</dl></details>")
    requires = payload.get("requires")
    items = [item for item in requires if isinstance(item, Mapping)] if isinstance(requires, Sequence) else []
    if items:
        parts.append(
            f'<details class="requirements"><summary>Needs from your machine ({len(items)})</summary>'
            f'<div class="table-scroll" role="region" aria-label="Machine requirements" tabindex="0">'
            f"{requires_table(items)}</div></details>"
        )
    return "\n".join(parts)


def progress_html(record: Mapping[str, object], state: str, progress: StepProgress | None, now: float) -> str:
    summary = (
        f'<div class="outcome-summary"><div class="status">{status_span(state)}</div>'
        f"<p>{escape(STATE_WORDS.get(state, state))}</p></div>"
    )
    lines = []
    if progress is not None and progress.request_id == record["agent_record_id"]:
        lines.append(
            f"<div><dt>Step time</dt><dd>{escape(elapsed(now - progress.started_at))} into the step</dd></div>"
        )
        if progress.episodes_total is not None:
            lines.append(
                f"<div><dt>Evaluation episodes</dt><dd>{progress.episodes_total} in the evaluation</dd></div>"
            )
        if progress.step_record:
            lines.append(f'<div><dt>Step record</dt><dd class="id">{escape(progress.step_record)}</dd></div>')
    else:
        filed = record.get("created_at")
        if isinstance(filed, (int, float)):
            lines.append(f"<div><dt>Waiting</dt><dd>{escape(elapsed(now - filed))}</dd></div>")
    return (
        summary + '<dl class="fact-list">' + "".join(lines) + "</dl>\n"
        f'<p class="live-note"><span class="live-dot" aria-hidden="true"></span>'
        f"Updates every {REFRESH_SECONDS} seconds until the step settles.</p>"
    )


def meaning(selection_result: str, row: Mapping[str, object], metrics: Mapping[str, object]) -> str:
    """What the result means for the person who asked, with the next action; the words the session prints."""
    release = str(row.get("release_id") or "-")[:8]
    if selection_result == "selected":
        return f"Published as release {release}. Your current session keeps its installed harness until you choose to update."
    if selection_result == "pending":
        return (
            f"Release {release} is ready. This change includes an extension, "
            "so it needs your review before installation."
        )
    if selection_result.startswith("promoted"):
        return f"passed the checks and was {selection_result}; the release that step published serves it"
    if selection_result == "rejected":
        selection = metrics.get("selection")
        reason = selection.get("reason") if isinstance(selection, Mapping) else None
        missed = missed_episodes(metrics)
        if not missed:
            return (
                f"did not pass the checks ({reason or 'the checks failed'}); nothing changed: rephrase or split the "
                "request"
            )
        # An episode that failed was the harness's run, not the request: no rephrasing would change it.
        advice = (
            "the episode failed, so the change itself was not judged"
            if any(episode.get("failure") for episode in missed)
            else "rephrase or split the request"
        )
        return (
            f"did not pass the checks ({reason or 'the checks failed'}): {missed_episode_text(missed[0])}; "
            f"nothing changed: {advice}"
        )
    if selection_result == "skipped":
        return f"produced no change ({metrics.get('skipped')}); nothing changed"
    if selection_result == "failed":
        return "The step failed before evaluation. Nothing was published; see the error below before retrying."
    return f"the step ended as {selection_result}"


def next_action(adapter: str, step: int, selection_result: str, record_id: str) -> tuple[str, str, str] | None:
    """The next action a settled step offers: its heading, the command and where to run it; ``None`` when a
    rejected or skipped step offers none. pi installs from its own session (``/versions``); another adapter's
    person runs its wrapper in a terminal."""
    if selection_result not in ("pending", "selected"):
        return None
    if ships_version_check(adapter):
        return (
            "Read this page, then install" if selection_result == "pending" else "Install when ready",
            f"/versions v{step} install",
            f"Run this in your reef-{adapter} session. You can keep chatting until you are ready.",
        )
    if selection_result == "pending":
        return (
            "Read this page, then serve it",
            f"reef-{adapter} wait {record_id}",
            "Run this in a terminal: it asks whether to serve this release, then installs it.",
        )
    return (
        "Install when ready",
        f"reef-{adapter} update",
        f"Run this in a terminal, then start reef-{adapter} again to use the new version.",
    )


def result_html(
    step: int,
    rows: Sequence[Mapping[str, object]],
    link_query: Mapping[str, str] | None,
    adapter: str = "pi",
    record_id: str = "",
) -> str:
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    selection_result = result_of(row, rows)
    parts = [
        f'<div class="outcome-summary"><div class="status">{status_span(selection_result)}</div>'
        f"<p>{escape(meaning(selection_result, row, metrics))}</p></div>",
        '<dl class="fact-list">',
        f"<div><dt>Version</dt><dd>v{step}</dd></div>",
        f'<div><dt>Release</dt><dd class="id">{escape(row.get("release_id"))}</dd></div>',
        "</dl>",
    ]
    if metrics.get("error"):
        parts.append(f'<div class="failure"><h3>Error</h3><p>{escape(metrics["error"])}</p></div>')
    notes = metrics.get("proposal_notes")
    failure = notes.get("failure") if isinstance(notes, Mapping) else None
    if isinstance(failure, str) and failure.strip():
        parts.append(f'<div class="failure"><h3>Proposer failure</h3><p>{escape(failure)}</p></div>')
    # Carry the scenario and authentication to the version page without displaying the token.
    href = step_href(step, link_query)
    offered = next_action(adapter, step, selection_result, record_id)
    if offered is not None:
        action, command, where = offered
        parts.append(
            f'<div class="next-action"><h3>{escape(action)}</h3><code>{escape(command)}</code>'
            f"<p>{escape(where)}</p></div>"
        )
    parts.append(
        f'<a class="version-link" href="{escape(href)}">View v{step}<span aria-hidden="true">&#8599;</span></a>'
    )
    return "\n".join(parts)


def what_changed(metrics: Mapping[str, object]) -> str:
    mutations = mutations_of(metrics)
    if not mutations:
        return '<p class="empty">No changes were produced by this step.</p>'
    items = []
    for mutation in mutations:
        options = mutation.get("options")
        kind = options.get("name") if isinstance(options, Mapping) else None
        operation = str(mutation.get("op") or "?")
        operation_class = operation if operation in ("create", "update", "delete") else "other"
        items.append(
            f'<li><span class="tag operation-{operation_class}">{escape(operation)}</span>'
            f'<span class="node-id">{escape(mutation.get("id") or "?")}</span>'
            f'<span class="tag">{escape(kind or "?")}</span></li>'
        )
    return '<ul class="mutations">' + "".join(items) + "</ul>"


def review_html(metrics: Mapping[str, object], rejected: bool = False) -> str:
    """The Review section: the result and what the entries left uncovered, or, when the review call failed, the
    reason it did not run, so a step never quietly publishes with nothing checking that it delivers the request."""
    notes = metrics.get("proposal_notes")
    notes = notes if isinstance(notes, Mapping) else {}
    review = notes.get("review")
    reasons = notes.get("dropped_attempts")
    dropped = [item for item in reasons if isinstance(item, str)] if isinstance(reasons, list) else []
    # Answers the proposer wrote again because their form slipped (broken JSON, refused entries), so the page says
    # what the kept one replaced.
    again = (
        "<h3>Answers written again</h3><ul>" + "".join(f"<li>{escape(item)}</li>" for item in dropped) + "</ul>\n"
        if dropped
        else ""
    )
    if not isinstance(review, Mapping):
        failure = notes.get("review_failure")
        if not isinstance(failure, str) or not failure.strip():
            return f'<section class="card review-card">\n<h2>Review</h2>\n{again}</section>\n' if again else ""
        return (
            f'<section class="card review-card">\n<h2>Review</h2>\n<p>The review of the entries against your '
            f"request did not run, so nothing checked whether they deliver it: {escape(failure)}</p>\n{again}"
            "</section>\n"
        )
    uncovered = review.get("uncovered")
    items = [item for item in uncovered if isinstance(item, str)] if isinstance(uncovered, Sequence) else []
    listed = "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>" if items else ""
    return (
        f'<section class="card review-card">\n<h2>Review</h2>\n<p>Coverage of the request: '
        f'{status_span(str(review.get("result", review.get("verdict")) or "unknown"))}</p>\n'
        + (
            (
                "<h3>Review notes</h3><p>The checks decided this result; these are the review's notes on the change."
                f"</p>{listed}\n"
                if rejected
                else f"<h3>Still uncovered</h3>{listed}\n"
            )
            if items
            else '<p class="empty">Nothing left uncovered.</p>\n'
        )
        + again
        + "</section>\n"
    )


def design_html(metrics: Mapping[str, object]) -> str:
    """The Design and How to use sections, only when the step recorded a design: the plan for the request, then
    how the person uses the change."""
    notes = metrics.get("proposal_notes")
    design, usage = design_sections(notes if isinstance(notes, Mapping) else {})
    cards = (
        f'<section class="card design-card">\n<h2>Design</h2>\n<p>{escape(design)}</p>\n</section>\n' if design else ""
    )
    if usage:
        cards += f'<section class="card usage-card">\n<h2>How to use</h2>\n<p>{escape(usage)}</p>\n</section>\n'
    return cards


def build_request_page(
    record: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    progress: StepProgress | None = None,
    consumed: bool = False,
    link_query: Mapping[str, str] | None = None,
    now: float | None = None,
    adapter: str = "pi",
) -> str:
    """The page for the request stored as ``record``, against the catalog ``rows`` oldest first.

    ``record`` is the agent record as ``Dispatcher.read_record`` answers it
    (``agent_record_id``, ``created_at`` and the
    ``POST /reef/train`` payload). ``progress`` is the training backend's
    running step, counted only when it names this request; ``consumed`` says
    whether the trainer's reserved batch carries the request. ``link_query``
    is carried to the version page link. ``now`` is the clock the elapsed
    times are read against. ``adapter`` names the wrapper the next action runs.
    """
    record_id = str(record["agent_record_id"])
    now = time.time() if now is None else now
    step = settled_step(rows, record_id)
    state = result_of(rows[step], rows) if step is not None else request_state(record, progress, consumed)
    head = "" if step is not None else f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">\n'
    if step is None:
        body = f'<section class="card outcome-card">\n<h2>Progress</h2>\n{progress_html(record, state, progress, now)}</section>\n'
        if progress is not None and progress.request_id == record_id:
            body += (
                f'<section class="card activity-card">\n<h2>Activity</h2>\n{activity_html(progress, now)}</section>\n'
            )
        subtitle = "Follow your request from instruction to outcome."
        current_stage = {"queued": 0, "proposing": 1, "evaluating": 1, "running": 1, "settling": 2}.get(state, 1)
    else:
        metrics = rows[step].get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        change_label = "Proposed changes" if state in ("pending", "rejected", "skipped", "failed") else "What changed"
        body = (
            f'<section class="card outcome-card">\n<h2>Result</h2>\n{result_html(step, rows, link_query, adapter, record_id)}</section>\n'
            f'<section class="card changes-card">\n<h2>{change_label}</h2>\n{what_changed(metrics)}</section>\n'
            f"{review_html(metrics, rejected=state == 'rejected')}{design_html(metrics)}"
        )
        subtitle = "Your request has a result. Review the outcome below."
        current_stage = 3
    stages = (
        ("Received", "Request accepted"),
        ("Processing", "Proposal & evaluation"),
        ("Recording", "Save result"),
        ("Outcome", "Final result"),
    )
    journey = []
    for index, (label, description) in enumerate(stages):
        if index < current_stage:
            stage_class, marker, current = "done", "&#10003;", ""
        elif index == current_stage:
            stage_class, marker, current = "current", f"0{index + 1}", ' aria-current="step"'
        else:
            stage_class, marker, current = "", f"0{index + 1}", ""
        journey.append(
            f'<li class="{stage_class}"{current}><span class="stage-icon" aria-hidden="true">{marker}</span>'
            f'<span class="stage-copy">{label}<small>{escape(description)}</small></span></li>'
        )
    scenario = link_query.get("scenario", "") if link_query else ""
    served = served_step(rows)
    return document(
        title=f"Harness request {record_id[:8]}",
        style=STYLE,
        breadcrumb="Requests",
        context=scenario,
        state=state,
        eyebrow="Harness evolution",
        heading="Request details",
        subtitle=escape(subtitle),
        # The served head is this scenario's home, the version this request is asked against.
        home="" if served is None else step_href(served, link_query),
        head=head,
        body=(
            '<ol class="journey" aria-label="Request progress">' + "".join(journey) + "</ol>\n"
            '<div class="layout"><section class="card request-card">\n'
            f"<h2>Request</h2>\n{request_html(record)}</section>\n"
            f"{body}</div>"
        ),
    )


__all__ = [
    "MAX_ACTIVITY_SHOWN",
    "REFRESH_SECONDS",
    "STATE_WORDS",
    "activity_html",
    "build_request_page",
    "request_state",
    "settled_step",
]
