"""Render structured evaluation checks without importing the training stack."""

from collections.abc import Mapping

from reef.service.page_chrome import escape


def checks_html(checks: object, report: object = None, reason: object = None) -> str:
    if not isinstance(checks, (list, tuple)) or not checks:
        return ""
    rows = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        cells = []
        for column_index, value in enumerate(
            (
                check.get("group"),
                check.get("id"),
                check.get("status"),
                check.get("current_score"),
                check.get("score"),
                check.get("expected"),
                check.get("reason") or check.get("observed"),
            )
        ):
            if column_index in {5, 6} and isinstance(value, str) and len(value) > 160:
                content = f"<details><summary>{escape(value[:80])}...</summary><p>{escape(value)}</p></details>"
            else:
                content = escape(value)
            cells.append(f"<td>{content}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    identity = ""
    if isinstance(report, Mapping):
        mode = "same model ID, fresh context" if report.get("same_model") else "separate model ID, fresh context"
        identity = f"<p>Reviewer: {escape(report.get('reviewer_model'))} ({mode}). Current release: {escape(report.get('current_release_id'))}.</p>"
    return (
        '<section class="card checks-card"><h2>Independent evaluation</h2>'
        + identity
        + (f"<p>{escape(reason)}</p>" if reason else "")
        + '<div style="overflow-x:auto"><table><thead><tr><th>Group</th><th>Check</th><th>Status</th><th>Current</th><th>Candidate</th><th>Expected</th><th>Observed / reason</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def evaluation_html(metrics: Mapping[str, object]) -> str:
    report = metrics.get("reefine_evaluation")
    if not isinstance(report, Mapping):
        return ""
    selection = metrics.get("selection")
    reason = selection.get("reason") if isinstance(selection, Mapping) else None
    return checks_html(report.get("checks"), report, reason)
