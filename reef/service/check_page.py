"""Show requested behavior and independent evaluation results on harness pages."""

from collections.abc import Mapping, Sequence

from reef.service.page_chrome import escape

STYLE = """
.evaluation-card{grid-column:1 / -1}.evaluation-card h2{margin-bottom:8px}
.evaluation-intro,.evaluation-note{color:var(--mute);font-size:12px;line-height:1.7;margin:0 0 16px}
.evaluation-overview{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
padding:14px 16px;background:var(--code);border-radius:8px;margin:20px 0 12px}
.evaluation-overview p{font-size:12px;color:var(--mute);margin:4px 0 0}.evaluation-count{font-size:12px;color:var(--mute)}
.evaluation-groups{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:24px}
.evaluation-groups a{display:flex;flex-direction:column;gap:9px;padding:12px;border:1px solid var(--line);
border-radius:8px;color:var(--ink);font-size:12px}.evaluation-groups a:hover{background:var(--code);text-decoration:none}
.evaluation-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;white-space:nowrap}
.evaluation-badge.pass{color:var(--good)}.evaluation-badge.fail,.evaluation-badge.invalid{color:var(--bad)}
.evaluation-badge.running{color:var(--accent)}.evaluation-badge.pending,.evaluation-badge.not_run{color:var(--warn)}
.evaluation-spinner{display:inline-block;width:11px;height:11px;border:2px solid var(--line);
border-top-color:currentColor;border-radius:50%;animation:evaluation-spin 1s linear infinite}
@keyframes evaluation-spin{to{transform:rotate(360deg)}}
.evaluation-group{border-top:1px solid var(--line);padding:18px 0 0;margin-top:18px;scroll-margin-top:20px}
.evaluation-group>summary{display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:14px;
list-style:none}.evaluation-group>summary:before{content:"+";color:var(--mute)}
.evaluation-group[open]>summary:before{content:"-"}.evaluation-group>summary>span:first-child{flex:1}
.evaluation-group>summary~*{margin-top:16px}.evaluation-heading{display:flex;align-items:center;
justify-content:space-between;gap:12px;margin-bottom:12px}.evaluation-heading h3{color:var(--ink);font-size:14px;margin:0}
.evaluation-check{padding:16px 0;border-top:1px solid var(--line)}.evaluation-check:first-child{border-top:0;padding-top:0}
.evaluation-check p,.evaluation-check li{font-size:13px;line-height:1.7;overflow-wrap:anywhere}
.evaluation-label{color:var(--mute);font-size:11px;margin-bottom:5px;display:block}
.evaluation-task{background:var(--code);border-radius:6px;padding:10px 12px;margin-bottom:16px}.evaluation-task p{margin:0}
.evaluation-requirements{margin:0 0 12px;padding-left:22px}.evaluation-requirements li{padding:5px 0 5px 4px}
.evaluation-observed{margin:12px 0 0;white-space:pre-wrap;overflow-wrap:anywhere}
.evaluation-findings{margin-top:14px}.evaluation-findings p{white-space:pre-wrap;margin-bottom:0}
.evaluation-comparison{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.evaluation-comparison>div{
border:1px solid var(--line);border-radius:6px;padding:10px 14px;min-width:150px}.evaluation-comparison strong{font-size:13px}
.evaluation-details{margin-top:22px;padding-top:16px;border-top:1px solid var(--line)}
.evaluation-details dl{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:14px 0 0}
.evaluation-details table{font-size:12px}.evaluation-details th,.evaluation-details td{overflow-wrap:anywhere}
@media(max-width:700px){.evaluation-groups{grid-template-columns:repeat(2,minmax(0,1fr))}
.evaluation-heading{align-items:flex-start}.evaluation-details dl{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){.evaluation-spinner{animation:none}}
"""

GROUPS = (
    ("request", "Requested behavior"),
    ("health", "Basic agent functions"),
    ("regression", "Existing tasks"),
    ("review", "Independent change review"),
)


def check_status(checks: Sequence[Mapping[str, object]]) -> str:
    if not checks:
        return "not_run"
    statuses = {str(check.get("status")) for check in checks}
    for status in ("invalid", "fail", "not_run", "running", "pending"):
        if status in statuses:
            return status
    return "pass" if statuses == {"pass"} else "invalid"


def status_html(status: str) -> str:
    labels = {
        "pass": "Passed",
        "fail": "Failed",
        "invalid": "Could not evaluate",
        "not_run": "Not run",
        "running": "Running",
        "pending": "Waiting",
    }
    if status not in labels:
        status = "invalid"
    marker = '<span class="evaluation-spinner" aria-hidden="true"></span>' if status == "running" else ""
    return f'<span class="evaluation-badge {status}" data-status="{status}">{marker}{labels[status]}</span>'


def check_html(
    check: Mapping[str, object],
    title: str,
    *,
    requirements: Sequence[str] = (),
    prompt: str = "",
    plan_check: Mapping[str, object] | None = None,
) -> str:
    status = check_status((check,))
    body = (
        f'<div class="evaluation-check" data-check-id="{escape(check.get("id"))}">'
        f'<div class="evaluation-heading"><h3>{escape(title)}</h3>{status_html(status)}</div>'
    )
    if prompt:
        body += (
            '<div class="evaluation-task"><span class="evaluation-label">Task given to the candidate agent</span>'
            f"<p>{escape(prompt)}</p></div>"
        )
    if plan_check is not None:
        plan_status = check_status((plan_check,))
        plan_labels = {"pass": "Ready", "running": "Being prepared", "pending": "Not ready yet"}
        body += (
            f'<p class="evaluation-note" data-check-id="{escape(plan_check.get("id"))}">Check plan: '
            f'{escape(plan_labels.get(plan_status, "Unavailable"))}.</p>'
        )
        if plan_status in {"fail", "invalid", "not_run"}:
            body += f'<p class="evaluation-observed">{escape(plan_check.get("reason"))}</p>'
    expected = check.get("expected")
    if check.get("id") == "request-behavior" and plan_check is not None and not requirements:
        expected = plan_check.get("observed") or expected
    if requirements:
        body += '<span class="evaluation-label">What the evaluator checks</span><ol class="evaluation-requirements">'
        body += "".join(f"<li>{escape(requirement)}</li>" for requirement in requirements) + "</ol>"
    elif expected:
        # Live progress carries the plan as text; keep its clauses intact without inventing individual results.
        expected = expected.replace("; ", ";\n") if isinstance(expected, str) else expected
        body += (
            '<span class="evaluation-label">What the evaluator checks</span>'
            f'<p class="evaluation-observed">{escape(expected)}</p>'
        )
    if check.get("id") == "request-behavior":
        body += (
            '<p class="evaluation-note">The status above is the result for the whole task. '
            "These requirements are checked together; this record does not contain separate results "
            "for each requirement.</p>"
        )
    if check.get("group") == "regression":
        comparison = []
        for label, key in (("Current release", "current_score"), ("Candidate", "score")):
            score = check.get(key)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                if score == 1:
                    result = "Passed"
                elif score == 0:
                    result = "Failed"
                else:
                    result = f"Score: {score:g}"
            else:
                result = "Not scored"
            comparison.append(f'<div><span class="evaluation-label">{label}</span><strong>{result}</strong></div>')
        body += f'<div class="evaluation-comparison">{"".join(comparison)}</div>'
    observed = check.get("reason") or check.get("observed")
    if isinstance(observed, str) and observed.strip():
        if status == "pass" and len(observed) > 240:
            body += (
                '<details class="evaluation-findings"><summary>Read the observed result</summary>'
                f"<p>{escape(observed)}</p></details>"
            )
        else:
            body += (
                '<span class="evaluation-label">Observed result / reason</span>'
                f'<p class="evaluation-observed">{escape(observed)}</p>'
            )
    return body + "</div>"


def checks_html(checks: object, report: object = None, reason: object = None, *, result: str = "") -> str:
    if not isinstance(checks, (list, tuple)) or not checks:
        return ""
    records = [check for check in checks if isinstance(check, Mapping)]
    if not records:
        return ""
    status = check_status(records)
    titles = {
        "pass": "Recorded checks passed",
        "running": "Evaluation is running",
        "pending": "Evaluation is waiting",
        "fail": "A required check failed",
        "invalid": "A check could not be evaluated",
        "not_run": "A required check did not run",
    }
    outcomes = {
        "pending": "Checks passed. This release is waiting for your review.",
        "selected": "Checks passed. This release was published.",
        "rejected": "This update was rejected. The current release stays in use.",
        "failed": "The step failed. This update was not published.",
    }
    counts = []
    for check_status_name, label in (("pass", "passed"), ("running", "running"), ("pending", "waiting")):
        count = sum(check.get("status") == check_status_name for check in records)
        if count:
            counts.append(f"{count} {label}")
    failed = sum(str(check.get("status")) not in {"pass", "running", "pending"} for check in records)
    if failed:
        counts.append(f"{failed} need attention")
    body = (
        '<section class="card evaluation-card"><h2>Independent evaluation</h2>'
        '<p class="evaluation-intro">Check the requested behavior first, then the existing capabilities '
        "and independent review. These results decide whether the harness update can be published.</p>"
        f'<div class="evaluation-overview"><div><strong>{titles[status]}</strong>'
        f'<p>{escape(outcomes.get(result, "Each check below shows its recorded result."))}</p></div>'
        f'<span class="evaluation-count">{escape(" / ".join(counts))}</span></div>'
    )
    grouped = [(group, label, [check for check in records if check.get("group") == group]) for group, label in GROUPS]
    body += '<nav class="evaluation-groups" aria-label="Evaluation check groups">'
    for group, label, members in grouped:
        body += f'<a href="#evaluation-{group}"><span>{label}</span>{status_html(check_status(members))}</a>'
    body += "</nav>"
    plan = report.get("plan") if isinstance(report, Mapping) else None
    requirements = plan.get("checks") if isinstance(plan, Mapping) else None
    requirements = [item for item in requirements if isinstance(item, str)] if isinstance(requirements, list) else []
    prompt = plan.get("prompt") if isinstance(plan, Mapping) else None
    prompt = prompt if isinstance(prompt, str) else ""
    for group, label, members in grouped:
        group_status = check_status(members)
        if group == "request":
            body += f'<section class="evaluation-group" id="evaluation-{group}">'
            plan_check = next((check for check in members if check.get("id") == "request-plan"), None)
            behavior = next((check for check in members if check.get("id") == "request-behavior"), None)
            if behavior is not None:
                body += check_html(
                    behavior, "Requested behavior", requirements=requirements, prompt=prompt, plan_check=plan_check
                )
            elif plan_check is not None:
                body += check_html(plan_check, "Prepare the behavior check")
            else:
                body += '<p class="evaluation-note">No requested-behavior result was recorded.</p>'
            for check in members:
                if check.get("id") not in {"request-plan", "request-behavior"}:
                    body += check_html(check, "Additional behavior check")
            body += "</section>"
        else:
            expanded = " open" if group_status != "pass" else ""
            body += (
                f'<details class="evaluation-group" id="evaluation-{group}"{expanded}>'
                f"<summary><span>{label}</span>{status_html(group_status)}</summary>"
            )
            for index, check in enumerate(members):
                check_title = f"Protected task {index + 1}" if group == "regression" else label
                body += check_html(check, check_title)
            if not members:
                body += '<p class="evaluation-note">No result was recorded for this group.</p>'
            body += "</details>"
    known_groups = {group for group, label in GROUPS}
    for check in records:
        if str(check.get("group")) not in known_groups:
            body += check_html(check, "Additional check")
    body += '<details class="evaluation-details"><summary>Technical details and publication reason</summary>'
    if reason:
        body += f"<p>{escape(reason)}</p>"
    if isinstance(report, Mapping):
        if report.get("same_model") is True:
            mode = "same model ID, fresh context"
        elif report.get("same_model") is False:
            mode = "separate model ID, fresh context"
        else:
            mode = "reviewer context not recorded"
        body += (
            f'<dl><div><dt>Reviewer</dt><dd>{escape(report.get("reviewer_model"))} ({mode})</dd></div>'
            f'<div><dt>Current release</dt><dd class="id">{escape(report.get("current_release_id"))}</dd></div></dl>'
        )
    body += '<div class="table-scroll"><table><thead><tr><th>Check ID</th><th>Status</th>'
    body += "<th>Current score</th><th>Candidate score</th><th>Seconds</th></tr></thead><tbody>"
    for check in records:
        values = (
            check.get("id"),
            check.get("status"),
            check.get("current_score"),
            check.get("score"),
            check.get("seconds"),
        )
        body += "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"
    return body + "</tbody></table></div></details></section>"


def evaluation_html(metrics: Mapping[str, object], *, result: str = "") -> str:
    report = metrics.get("reefine_evaluation")
    if not isinstance(report, Mapping):
        return ""
    selection = metrics.get("selection")
    reason = selection.get("reason") if isinstance(selection, Mapping) else None
    return checks_html(report.get("checks"), report, reason, result=result)
