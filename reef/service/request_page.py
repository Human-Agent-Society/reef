"""One HTML page per filed harness request: where its step stands, then the result once the step settles.

``GET /reef/harness/requests/{record_id}/page`` builds it from the request's
agent record, the scenario's catalog rows and the running step's progress.
The catalog row whose ``metrics.training_request.id`` is the record id
settles the request: the page then shows that row's result as the version
page words it, the mutations, what the review left uncovered, why the
proposer produced nothing when the step recorded that, and links the
version page. Until then the page names the state the request is in
(``queued`` before a step takes it, ``proposing`` and ``evaluating`` from the
backend's progress, ``running`` while the trainer holds the request and the
backend reports no phase, ``settling`` while the row that consumed the
record lands) and reloads itself every ``REFRESH_SECONDS``, so a person
opens the link right after asking and watches. Like the version page it
loads no asset and is pure ASCII.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from urllib.parse import urlencode

from reef.service.release_page import _esc, _requires_table, mutations_of, result_of
from reef.train.cordis_backend.contracts import StepProgress

#: Seconds between the page's own reloads while the request is not settled.
REFRESH_SECONDS = 5

# Inline the README logo so installed wheels need no docs checkout or external asset request.
LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 268 80" role="img" aria-labelledby="title">
  <title id="title">Reef</title>
  <g fill="none" stroke="#a03729" stroke-linecap="round" stroke-linejoin="round" stroke-width="3.5">
    <path d="M14 23.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 26 38.5 26s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
    <path d="M14 36.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 39 38.5 39s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
    <path d="M14 49.5c2.5 0 4 2.5 7 2.5s5.5-4.5 8.5-4.5S35.5 52 38.5 52s6-4.5 9-4.5c2.5 0 4 2 6.5 2"/>
  </g>
  <text x="78" y="53" fill="#14110e" font-family="Arial, Helvetica, sans-serif" font-size="38" font-weight="700" letter-spacing="7">REEF</text>
</svg>
"""

# The request page is self-contained, including its responsive and system-theme styles.
STYLE = """
:root{color-scheme:light;--bg:#f8f7f5;--card:#fff;--ink:#25221e;--mute:#716b64;--line:#e6e2dc;
--accent:#a03729;--soft:#f8eee9;--good:#24715a;--good-bg:#edf6f1;--warn:#926319;--warn-bg:#fcf4e5;
--status:#a03729;--status-bg:#f8eee9;--bad:#ab3e45;--bad-bg:#fceff0;--code:#f5f3f0;--shadow:0 2px 4px #25221e03,0 12px 32px #25221e03}
@media(prefers-color-scheme:dark){.brand svg g{stroke:#d99183}.brand svg text{fill:#f7f4f0}:root{color-scheme:dark;--bg:#141310;--card:#1c1a17;--ink:#eeeae4;
--mute:#b1a99f;--line:#37322d;--accent:#e8a092;--soft:#33231f;--good:#9ed3bc;--good-bg:#1f3028;
--warn:#e6c084;--warn-bg:#352b1d;--bad:#f0a3a8;--bad-bg:#392327;--code:#24211d;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
a:focus-visible,summary:focus-visible,.table-scroll:focus-visible{outline:2px solid var(--accent);outline-offset:5px;border-radius:4px}
header{background:var(--card);border-bottom:1px solid var(--line)}
.topbar{max-width:1200px;min-height:72px;margin:auto;padding:16px 40px;display:flex;align-items:center;gap:28px}
.brand{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:750;letter-spacing:.16em}
.brand svg{display:block;width:107.2px;height:32px}.divider{height:20px;width:1px;background:var(--line)}
.breadcrumb{display:flex;gap:12px;align-items:center;color:var(--mute);font-size:13px}.breadcrumb b{font-weight:500;color:var(--ink)}
.topbar .context{margin-left:auto;color:var(--mute);font-size:12px;max-width:240px;overflow-wrap:anywhere}
main{max-width:1200px;margin:auto;padding:48px 40px 24px}
h1,h2,h3,p{margin-top:0}h1{font-size:34px;line-height:1.2;letter-spacing:-1.2px;font-weight:650;margin-bottom:12px}
h2{font-size:15px;letter-spacing:-.2px;margin:0 0 22px;font-weight:650}h3{font-size:12px;font-weight:600;color:var(--mute)}
.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:650;color:var(--mute);margin-bottom:12px}
.hero{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:30px}
.subtitle{color:var(--mute);font-size:14px;margin:0}.status{display:inline-flex;align-items:center;gap:8px;
border:1px solid var(--line);border-radius:100px;padding:6px 12px;font-size:12px;background:var(--card);white-space:nowrap}
.status:before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.tone-selected,.tone-promoted{--status:var(--good);--status-bg:var(--good-bg)}
.tone-pending,.tone-skipped{--status:var(--warn);--status-bg:var(--warn-bg)}
.tone-rejected{--status:var(--bad);--status-bg:var(--bad-bg)}
.tone-queued,.tone-proposing,.tone-evaluating,.tone-running,.tone-settling{--status:var(--accent);--status-bg:var(--soft)}
.status{color:var(--status);background:var(--status-bg);border-color:transparent}
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
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:28px;min-width:0;box-shadow:var(--shadow)}
.request-card{grid-column:1;grid-row:1}.outcome-card{grid-column:2;grid-row:1 / span 3}
.changes-card,.review-card{grid-column:1}.text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:19px;
line-height:1.7;letter-spacing:-.3px;margin:0 0 30px;padding-left:20px;border-left:2px solid var(--accent)}
.metadata{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:0;padding-top:20px;border-top:1px solid var(--line)}
dt{font-size:11px;color:var(--mute);margin-bottom:5px}dd{margin:0;overflow-wrap:anywhere;font-size:12px}
.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
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
.next-action p{font-size:12px;color:var(--mute);margin:10px 0 0}.hero>div{min-width:0}
.version-link:hover{opacity:.88;text-decoration:none}.failure{border-left:2px solid var(--bad);padding-left:12px;margin:20px 0}
.failure h3{color:var(--bad);margin-bottom:5px}.failure p{font-size:12px;overflow-wrap:anywhere;margin-bottom:0}
.mutations{list-style:none;padding:0;margin:0}.mutations li{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
padding:14px 0;border-top:1px solid var(--line)}.mutations li:first-child{border-top:0;padding-top:0}
.mutations li:last-child{padding-bottom:0}.mutations .node-id{flex:1;overflow-wrap:anywhere;min-width:0;font-size:13px}
.tag{display:inline-block;font-size:11px;border:1px solid var(--line);padding:2px 7px;border-radius:4px;
color:var(--mute);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere;max-width:100%}
.operation-create{color:var(--good);background:var(--good-bg);border-color:transparent}
.operation-update{color:var(--warn);background:var(--warn-bg);border-color:transparent}
.operation-delete{color:var(--bad);background:var(--bad-bg);border-color:transparent}
.empty{font-size:13px;color:var(--mute);margin:0}.review-card p{font-size:13px;color:var(--mute)}
.review-card ul{margin:0;padding-left:18px;font-size:13px}.review-card li{padding:5px 0;overflow-wrap:anywhere}
.complete{color:var(--good)}.partial{color:var(--warn)}.review-card h3{margin:22px 0 8px}
.request-meta{border-top:1px solid var(--line);padding-top:20px}.request-meta .metadata{border:0;padding-top:20px}
.requirements{margin-top:20px;border-top:1px solid var(--line);padding-top:18px}
summary{cursor:pointer;font-size:12px;font-weight:550}summary::marker{color:var(--mute)}
.table-scroll{overflow-x:auto;margin-top:16px}table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;vertical-align:top;padding:10px 8px;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
th{font-size:11px;color:var(--mute);font-weight:550}td{min-width:80px;max-width:220px}
footer{display:flex;justify-content:space-between;gap:16px;margin-top:36px;padding-top:18px;border-top:1px solid var(--line);
font-size:11px;color:var(--mute)}footer p{margin:0}.footer-brand{letter-spacing:.1em;font-weight:600}
@media(min-width:1500px){main{padding-top:64px}}
@media(max-width:800px){.topbar{padding:16px 24px;gap:18px}main{padding:32px 24px 24px}
.layout{grid-template-columns:1fr}.request-card,.outcome-card,.changes-card,.review-card{grid-column:auto;grid-row:auto}
.journey{padding:20px}.journey li:not(:last-child):after{margin:0 10px}.stage-copy small{display:none}}
@media(max-width:480px){.topbar{padding:14px 20px;min-height:62px;gap:16px}.topbar .context,.breadcrumb b,.breadcrumb .slash{display:none}
main{padding:28px 16px 20px}.hero{align-items:flex-start;flex-direction:column;gap:16px}h1{font-size:27px;letter-spacing:-.8px}
.subtitle{font-size:12px}.hero .status{font-size:11px;padding:5px 8px}.card{padding:22px}.text{font-size:17px;padding-left:16px}
.journey{padding:18px 12px;gap:6px}.journey li{flex-direction:column;gap:7px}.journey li:not(:last-child):after{position:absolute;
left:calc(50% + 21px);right:calc(-50% + 15px);top:14px;margin:0}.stage-copy{font-size:11px}.metadata{gap:18px 12px}
.mutations li{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:start}
.mutations li>.tag:last-child{grid-column:2;justify-self:start}
.layout{gap:16px}footer{font-size:11px}}
"""

#: What each state means, in the words the page prints beside it.
STATE_WORDS = {
    "queued": "no step has taken the request yet; the trainer runs one step per instruction, oldest first",
    "proposing": "the proposer is writing the change: the served model reads the request and the tree",
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


STATUS_LABELS = {
    "queued": "Queued",
    "proposing": "Designing the change",
    "evaluating": "Checking the harness",
    "running": "In progress",
    "settling": "Saving the result",
    "selected": "Published",
    "pending": "Ready for review",
    "rejected": "Not selected",
    "skipped": "No changes",
    "complete": "Complete",
    "partial": "Partial",
}


def span(state: str) -> str:
    label = STATUS_LABELS.get(state, state.capitalize())
    return f'<span class="{_esc(state.split(" ")[0])}">{_esc(label)}</span>'


def request_state(record: Mapping[str, object], progress: StepProgress | None, consumed: bool) -> str:
    """The unsettled request's state: the backend's phase for it, the trainer's hold on it, else the record's.

    A record that is not compacted waits for its step; a compacted one was
    consumed by a commit whose row is about to show, since the row that
    names the request lands in the same commit as the compaction."""
    if progress is not None and progress.request_id == record["agent_record_id"]:
        return "evaluating" if progress.phase == "gating" else progress.phase
    if consumed:
        return "running"
    return "queued" if record.get("compacted_at") is None else "settling"


def request_html(record: Mapping[str, object]) -> str:
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    parts = [
        f'<p class="text">{_esc(payload.get("text"))}</p>',
        '<details class="request-meta"><summary>Session and request details</summary><dl class="metadata">',
        f'<div><dt>Session</dt><dd class="id">{_esc(payload.get("session") or "Not provided")}</dd></div>',
        f'<div><dt>Base release</dt><dd class="id">{_esc(payload.get("release_id") or "Not provided")}</dd></div>',
        f'<div class="record-id"><dt>Request ID</dt><dd class="id">{_esc(record["agent_record_id"])}</dd></div>',
    ]
    filed = record.get("created_at")
    if isinstance(filed, (int, float)):
        filed_at = datetime.fromtimestamp(filed, timezone.utc)
        parts.append(
            f'<div class="record-id"><dt>Submitted</dt><dd><time datetime="{filed_at.isoformat()}">'
            f'{filed_at.strftime("%d %b %Y, %H:%M:%S UTC")}</time></dd></div>'
        )
    parts.append("</dl></details>")
    requires = payload.get("requires")
    items = [item for item in requires if isinstance(item, Mapping)] if isinstance(requires, Sequence) else []
    if items:
        parts.append(
            f'<details class="requirements"><summary>Needs from your machine ({len(items)})</summary>'
            f'<div class="table-scroll" role="region" aria-label="Machine requirements" tabindex="0">'
            f"{_requires_table(items)}</div></details>"
        )
    return "\n".join(parts)


def progress_html(record: Mapping[str, object], state: str, progress: StepProgress | None, now: float) -> str:
    summary = (
        f'<div class="outcome-summary"><div class="status">{span(state)}</div>'
        f"<p>{_esc(STATE_WORDS.get(state, state))}</p></div>"
    )
    lines = []
    if progress is not None and progress.request_id == record["agent_record_id"]:
        lines.append(f"<div><dt>Step time</dt><dd>{_esc(elapsed(now - progress.started_at))} into the step</dd></div>")
        if progress.episodes_total is not None:
            lines.append(
                f"<div><dt>Evaluation episodes</dt><dd>{progress.episodes_total} in the evaluation</dd></div>"
            )
        if progress.step_record:
            lines.append(f'<div><dt>Step record</dt><dd class="id">{_esc(progress.step_record)}</dd></div>')
    else:
        filed = record.get("created_at")
        if isinstance(filed, (int, float)):
            lines.append(f"<div><dt>Waiting</dt><dd>{_esc(elapsed(now - filed))}</dd></div>")
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
        return f"did not pass the checks ({reason or 'the checks failed'}); nothing changed: rephrase or split the request"
    if selection_result == "skipped":
        return f"produced no change ({metrics.get('skipped')}); nothing changed"
    return f"the step ended as {selection_result}"


def result_html(step: int, rows: Sequence[Mapping[str, object]], link_query: Mapping[str, str] | None) -> str:
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    selection_result = result_of(row, rows)
    parts = [
        f'<div class="outcome-summary"><div class="status">{span(selection_result)}</div>'
        f"<p>{_esc(meaning(selection_result, row, metrics))}</p></div>",
        '<dl class="fact-list">',
        f"<div><dt>Step</dt><dd>{step}</dd></div>",
        f'<div><dt>Release</dt><dd class="id">{_esc(row.get("release_id"))}</dd></div>',
        "</dl>",
    ]
    if metrics.get("error"):
        parts.append(f'<div class="failure"><h3>Error</h3><p>{_esc(metrics["error"])}</p></div>')
    notes = metrics.get("proposal_notes")
    failure = notes.get("failure") if isinstance(notes, Mapping) else None
    if isinstance(failure, str) and failure.strip():
        parts.append(f'<div class="failure"><h3>Proposer failure</h3><p>{_esc(failure)}</p></div>')
    href = f"/reef/harness/releases/{step}/page"
    if link_query:
        # Carry the scenario and authentication to the version page without displaying the token.
        href += "?" + urlencode(dict(link_query))
    if selection_result == "pending":
        command = f"/reef-versions {step} promote"
        action = "Review, then promote"
    elif selection_result == "selected":
        command = f"/reef-versions {step} install"
        action = "Install when ready"
    else:
        command = ""
        action = ""
    if command:
        parts.append(
            f'<div class="next-action"><h3>{action}</h3><code>{command}</code>'
            "<p>Run this in your reef-pi session. You can keep chatting until you are ready.</p></div>"
        )
    parts.append(
        f'<a class="version-link" href="{_esc(href)}">View step {step}<span aria-hidden="true">&#8599;</span></a>'
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
            f'<li><span class="tag operation-{operation_class}">{_esc(operation)}</span>'
            f'<span class="node-id">{_esc(mutation.get("id") or "?")}</span>'
            f'<span class="tag">{_esc(kind or "?")}</span></li>'
        )
    return '<ul class="mutations">' + "".join(items) + "</ul>"


def review_html(metrics: Mapping[str, object]) -> str:
    """The Review section, only when the step recorded one: the result and what the entries left uncovered."""
    notes = metrics.get("proposal_notes")
    review = notes.get("review") if isinstance(notes, Mapping) else None
    if not isinstance(review, Mapping):
        return ""
    uncovered = review.get("uncovered")
    items = [item for item in uncovered if isinstance(item, str)] if isinstance(uncovered, Sequence) else []
    listed = "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>" if items else ""
    return (
        f'<section class="card review-card">\n<h2>Review</h2>\n<p>Coverage of the request: '
        f'{span(str(review.get("result", review.get("verdict")) or "unknown"))}</p>\n'
        + (f"<h3>Still uncovered</h3>{listed}\n" if items else '<p class="empty">Nothing left uncovered.</p>\n')
        + "</section>\n"
    )


def build_request_page(
    record: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    progress: StepProgress | None = None,
    consumed: bool = False,
    link_query: Mapping[str, str] | None = None,
    now: float | None = None,
) -> str:
    """The page for the request stored as ``record``, against the catalog ``rows`` oldest first.

    ``record`` is the agent record as ``Dispatcher.read_record`` answers it
    (``agent_record_id``, ``created_at``, ``compacted_at`` and the
    ``POST /reef/train`` payload). ``progress`` is the training backend's
    running step, counted only when it names this request; ``consumed`` says
    whether the trainer's reserved batch carries the request. ``link_query``
    is carried to the version page link. ``now`` is the clock the elapsed
    times are read against. Pure ASCII out: other characters leave as
    numeric references.
    """
    record_id = str(record["agent_record_id"])
    now = time.time() if now is None else now
    step = settled_step(rows, record_id)
    state = result_of(rows[step], rows) if step is not None else request_state(record, progress, consumed)
    title = f"Harness request {record_id[:8]}"
    head = "" if step is not None else f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">\n'
    state_class = state.split(" ")[0]
    if step is None:
        body = f'<section class="card outcome-card">\n<h2>Progress</h2>\n{progress_html(record, state, progress, now)}</section>\n'
        subtitle = "Follow your request from instruction to outcome."
        current_stage = {"queued": 0, "proposing": 1, "evaluating": 1, "running": 1, "settling": 2}.get(state, 1)
    else:
        metrics = rows[step].get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        change_label = "Proposed changes" if state in ("pending", "rejected", "skipped") else "What changed"
        body = (
            f'<section class="card outcome-card">\n<h2>Result</h2>\n{result_html(step, rows, link_query)}</section>\n'
            f'<section class="card changes-card">\n<h2>{change_label}</h2>\n{what_changed(metrics)}</section>\n'
            f"{review_html(metrics)}"
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
            f'<span class="stage-copy">{label}<small>{_esc(description)}</small></span></li>'
        )
    scenario = link_query.get("scenario", "") if link_query else ""
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="referrer" content="no-referrer">\n'
        f"{head}<title>{_esc(title)}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n"
        f'<header><div class="topbar"><div class="brand">{LOGO}</div>'
        '<span class="divider" aria-hidden="true"></span>'
        '<div class="breadcrumb"><span>Harness</span><span class="slash" aria-hidden="true">/</span>'
        "<b>Requests</b></div>"
        f'<span class="context">{_esc(scenario)}</span></div></header>\n'
        f'<main class="tone-{_esc(state_class)}">\n'
        '<div class="hero"><div><p class="eyebrow">Harness evolution</p><h1>Request details</h1>'
        f'<p class="subtitle">{subtitle}</p></div><div class="status" role="status">{span(state)}</div></div>\n'
        '<ol class="journey" aria-label="Request progress">' + "".join(journey) + "</ol>\n"
        '<div class="layout"><section class="card request-card">\n'
        f"<h2>Request</h2>\n{request_html(record)}</section>\n"
        f"{body}</div>"
        '<footer><p><span class="footer-brand">REEF</span> &nbsp; / &nbsp; Harness evolution</p>'
        "<p>Built with Reef</p></footer>\n"
        "</main>\n</body>\n</html>\n"
    )
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


__all__ = ["REFRESH_SECONDS", "build_request_page", "settled_step"]
