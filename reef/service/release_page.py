"""One HTML page per catalog step: why the version exists, what it changed, the evaluation's result, its setup, its chain.

``GET /reef/harness/releases/{step}/page`` builds it from the releases row
plus, for an extension update, the file the release replaced. A step whose
method recorded ``proposal_notes`` (Reefine's design, review, refused
requires and undeclared variables) also gets a Design section after Why and
a Review section after What changed, and why the proposer produced nothing,
when the step recorded that, is a row of the Result table. The page loads
no asset and carries its data inline, so one curl with the scenario header
is the whole read. The step is the row's position in the catalog oldest
first, the creation row being 0: a rejected step publishes nothing and its
row carries the head's release id, so only the step names it.
"""

from __future__ import annotations

import difflib
import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.requirements import required_by

#: The evaluation numbers the Result section lists, in this order, when the row carries them: a comparison writes
#: wins, losses and ties; a floor writes passed, failed and floor_score, and evaluation_sides when it ran one side only.
RESULT_FIELDS = (
    "selected",
    "wins",
    "losses",
    "ties",
    "passed",
    "failed",
    "floor_score",
    "evaluation_sides",
    "current_score",
    "candidate_score",
    "episode_failures",
    "proposer_input_tokens",
    "proposer_output_tokens",
)


def evaluation_token_counts(metrics: Mapping[str, Any]) -> tuple[int, int] | None:
    """Input and output tokens over both sides' agents, or None when no agent reported any."""
    inputs = outputs = 0
    for side in ("candidate_agents", "current_agents"):
        agents = metrics.get(side)
        if not isinstance(agents, Mapping):
            continue
        for counts in agents.values():
            if isinstance(counts, Mapping):
                inputs += int(counts.get("input_tokens", 0) or 0)
                outputs += int(counts.get("output_tokens", 0) or 0)
    return (inputs, outputs) if inputs or outputs else None


#: Node kinds whose config carries the change as ``text``; the page shows that text instead of the config JSON.
TEXT_KINDS = ("rules", "skill", "agent_command")

STYLE = """
:root{--bg:#f6f4ee;--ink:#1f2321;--mute:#6b6f6a;--line:#d9d5c9;--card:#fffdf8;--accent:#0a6f5c;--warn:#a1521a;--bad:#9b2c2c;--good:#2e7d4f;--code:#efece3}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171a;--ink:#e8e6df;--mute:#9a9d97;--line:#2e3236;--card:#1c1f23;--accent:#4fc3a8;--warn:#e0a05a;--bad:#e07474;--good:#7fcf9a;--code:#22262b}}
:root[data-theme="dark"]{--bg:#15171a;--ink:#e8e6df;--mute:#9a9d97;--line:#2e3236;--card:#1c1f23;--accent:#4fc3a8;--warn:#e0a05a;--bad:#e07474;--good:#7fcf9a;--code:#22262b}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0;padding:0 0 4rem}
main{max-width:960px;margin:0 auto;padding:1.5rem 1.25rem}
h1{font-size:1.5rem;margin:0 0 .25rem}h2{font-size:1.05rem;margin:2rem 0 .75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
h3{font-size:.95rem;margin:1rem 0 .5rem;font-weight:600}
.sub{color:var(--mute);margin:0 0 1rem}.id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85rem}
.text{white-space:pre-wrap}
.tag{display:inline-block;font-size:.72rem;padding:0 .35rem;border:1px solid var(--line);border-radius:3px;margin-right:.25rem;color:var(--mute)}
.selected,.promoted,.complete{color:var(--good)}.rejected{color:var(--bad)}.pending,.skipped,.partial{color:var(--warn)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:.5rem;border-radius:4px;max-height:32rem;overflow:auto;font-size:.8rem}
.add{color:var(--good)}.del{color:var(--bad)}.hunk{color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
.empty{color:var(--mute);font-style:italic}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _short(release_id: Any) -> str:
    return str(release_id)[:8] if release_id else "-"


def _notes(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """What the method recorded beside its proposal, ``proposal_notes``; empty when the step carries none."""
    notes = metrics.get("proposal_notes")
    return notes if isinstance(notes, Mapping) else {}


def _strings(value: Any) -> list[str]:
    """The items of a JSON list as text, in order; none when ``value`` is not a list."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item if isinstance(item, str) else json.dumps(item, sort_keys=True) for item in value]
    return []


def mutations_of(metrics: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The step's mutations: ``mutation`` for one, ``mutations`` for a composite, none for a skip or a recheck."""
    if not metrics:
        return []
    single = metrics.get("mutation")
    if isinstance(single, Mapping):
        return [single]
    many = metrics.get("mutations")
    if isinstance(many, Sequence) and not isinstance(many, str):
        return [mutation for mutation in many if isinstance(mutation, Mapping)]
    return []


def result_of(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] = ()) -> str:
    """The row's result: pending, selected, rejected, skipped, else the operation (creation, promote, rollback).

    A pending row stays pending in the catalog after a person promotes it;
    the promote is a later row naming it in ``rollback_target_release_id``,
    so with ``rows`` given such a row reads ``promoted at step N``."""
    if row.get("pending"):
        release_id = row.get("release_id")
        for index, other in enumerate(rows):
            if other.get("operation") == "promote" and other.get("rollback_target_release_id") == release_id:
                return f"promoted at step {index}"
        return "pending"
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        if isinstance(metrics.get("selected"), bool):
            return "selected" if metrics["selected"] else "rejected"
        if metrics.get("skipped"):
            return "skipped"
    return str(row.get("operation") or "unknown")


def _class(selection_result: str) -> str:
    return selection_result.split(" ")[0]


def _span(selection_result: str) -> str:
    return f'<span class="{_esc(_class(selection_result))}">{_esc(selection_result)}</span>'


def served_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """The step whose release serves: the newest row that is neither pending nor a rejected or skipped step.

    The catalog's own ``current`` flag sits on the newest row, which a
    pending win or a failed evaluation makes the wrong one: those rows publish
    nothing, and a rejected or skipped row carries the head's id."""
    for index in range(len(rows) - 1, -1, -1):
        if result_of(rows[index]) not in ("pending", "rejected", "skipped"):
            return index
    return None


def before_release_id(row: Mapping[str, Any]) -> str | None:
    """The release the step ran on: the parent of a release that won, the head a rejected or skipped step ran on.

    A rejected or skipped step's row carries the head's own release id, so
    its parent would be one release too far back."""
    selection_result = result_of(row)
    if selection_result in ("selected", "pending"):
        parent = row.get("parent_release_id")
        return str(parent) if parent else None
    if selection_result in ("rejected", "skipped"):
        return str(row.get("release_id") or "") or None
    return None


def _why(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    operation = row.get("operation")
    if operation != "training":
        target = row.get("rollback_target_release_id")
        words = {
            "creation": "the tree this scenario started from; no step made it",
            "promote": f"a person promoted release {target} after reading it",
            "rollback": f"a person rolled the head back to release {target}",
            "recovery": "the head this process recovered at boot",
        }
        return f"<p>{_esc(words.get(str(operation), f'a {operation} commit'))}</p>"
    request = metrics.get("training_request")
    if isinstance(request, Mapping) and request.get("text"):
        return (
            f"<p class=\"text\">{_esc(request['text'])}</p>"
            f"<p class=\"sub\">request <span class=\"id\">{_esc(request.get('id'))}</span> from session "
            f"<span class=\"id\">{_esc(request.get('session'))}</span> on release "
            f"<span class=\"id\">{_esc(request.get('release_id'))}</span></p>"
        )
    proposal = metrics.get("proposal")
    if isinstance(proposal, Mapping) and proposal.get("reason"):
        return (
            f"<p>{_esc(proposal['reason'])}</p>"
            f"<p class=\"sub\">an agent's proposal <span class=\"id\">{_esc(proposal.get('id'))}</span> from session "
            f"<span class=\"id\">{_esc(proposal.get('session'))}</span></p>"
        )
    return "<p>a failure in the batch</p>"


def _design(metrics: Mapping[str, Any]) -> str:
    """The Design section: the proposer's plan for the request, when the method recorded ``proposal_notes.design``."""
    design = _notes(metrics).get("design")
    if not isinstance(design, str) or not design.strip():
        return ""
    return f'<h2>Design</h2>\n<p class="text">{_esc(design)}</p>\n'


def _diff_block(path: str, before: str, after: str, before_id: str | None, release_id: Any) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{path} ({_short(before_id)})",
        tofile=f"{path} ({_short(release_id)})",
        lineterm="",
    )
    rendered = []
    # By position, not prefix: the first two lines are the file headers, after which "++count;" is an addition.
    for index, line in enumerate(lines):
        if index < 2:
            rendered.append(f'<span class="hunk">{_esc(line)}</span>')
        elif line.startswith("+"):
            rendered.append(f'<span class="add">{_esc(line)}</span>')
        elif line.startswith("-"):
            rendered.append(f'<span class="del">{_esc(line)}</span>')
        elif line.startswith("@@"):
            rendered.append(f'<span class="hunk">{_esc(line)}</span>')
        else:
            rendered.append(_esc(line))
    if not rendered:
        return f'<p class="empty">{_esc(path)} is unchanged</p>'
    return "<pre>" + "\n".join(rendered) + "</pre>"


def _mutation_block(
    mutation: Mapping[str, Any],
    row: Mapping[str, Any],
    before_entries: Mapping[str, Mapping[str, Any]],
    before_files: Mapping[str, str] | None,
    node_paths: Mapping[str, str],
) -> str:
    op = str(mutation.get("op") or "?")
    entry_id = str(mutation.get("id") or "?")
    options = mutation.get("options")
    options = options if isinstance(options, Mapping) else {}
    previous = before_entries.get(entry_id, {})
    kind = str(options.get("name") or previous.get("name") or "?")
    head = f'<h3><span class="tag">{_esc(op)}</span>{_esc(entry_id)} <span class="tag">{_esc(kind)}</span></h3>'
    if op == "remove":
        return head
    config = options.get("config")
    config = config if isinstance(config, Mapping) else {}
    if kind == "code_extension":
        code = config.get("code")
        if not isinstance(code, str):
            return head + '<p class="empty">this mutation changes no code</p>'
        previous_config = previous.get("config")
        previous_config = previous_config if isinstance(previous_config, Mapping) else {}
        name = config.get("name") or previous_config.get("name")
        template = node_paths.get("code_extension")
        if op == "update" and before_files is not None and template and name:
            path = template.format(name=name)
            before = before_files.get(path)
            if before is not None:
                return head + _diff_block(path, before, code, before_release_id(row), row.get("release_id"))
        return head + f"<pre>{_esc(code)}</pre>"
    if kind in TEXT_KINDS and isinstance(config.get("text"), str):
        rest = {key: value for key, value in options.items() if key not in ("config", "name")}
        rest.update({key: value for key, value in config.items() if key != "text"})
        note = f'<p class="sub">{_esc(json.dumps(rest, sort_keys=True))}</p>' if rest else ""
        return head + note + f"<pre>{_esc(config['text'])}</pre>"
    return head + f"<pre>{_esc(json.dumps(options, indent=2, sort_keys=True))}</pre>"


def _what_changed(
    row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    before_entries: Mapping[str, Mapping[str, Any]],
    before_files: Mapping[str, str] | None,
    node_paths: Mapping[str, str],
) -> str:
    mutations = mutations_of(metrics)
    if mutations:
        return "".join(_mutation_block(m, row, before_entries, before_files, node_paths) for m in mutations)
    if metrics.get("skipped"):
        return f'<p class="empty">nothing: the step skipped ({_esc(metrics["skipped"])})</p>'
    if metrics.get("recheck"):
        return '<p class="empty">nothing new: a recheck of the last good tree against the published one</p>'
    if row.get("operation") == "training":
        return '<p class="empty">no mutation on record</p>'
    if row.get("operation") == "creation":
        return '<p class="empty">nothing: the seed as the recipe rendered it</p>'
    return '<p class="empty">no mutation: the head moved without a step</p>'


def _listed(items: Sequence[str], empty: str) -> str:
    if not items:
        return f'<p class="empty">{_esc(empty)}</p>'
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"


def _review(metrics: Mapping[str, Any]) -> str:
    """The Review section: the proposer's reading of its entries against the request, then what it left undeclared.

    ``proposal_notes.review`` is absent when the method's review call failed;
    the ``undeclared_env`` line shows all the same, being the warning the
    person needs. Empty when the step recorded neither."""
    notes = _notes(metrics)
    review = notes.get("review")
    undeclared = _strings(notes.get("undeclared_env"))
    if not isinstance(review, Mapping) and not undeclared:
        return ""
    parts = []
    if isinstance(review, Mapping):
        review_result = _span(str(review.get("result", review.get("verdict")) or "unknown"))
        parts.append(f"<p>the proposer's review of its entries against the request: {review_result}</p>")
        parts.append("<h3>covered</h3>" + _listed(_strings(review.get("covered")), "nothing listed as covered"))
        parts.append("<h3>uncovered</h3>" + _listed(_strings(review.get("uncovered")), "nothing left uncovered"))
    else:
        parts.append('<p class="empty">no review on record</p>')
    if undeclared:
        parts.append(
            "<p>the extension reads these variables and no requires item names them: "
            f'<span class="id">{_esc(", ".join(undeclared))}</span></p>'
        )
    return "<h2>Review</h2>\n" + "".join(parts) + "\n"


def result_html(row: Mapping[str, Any], metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    selection_result = result_of(row, rows)
    notes = {
        "pending": "passed the checks; waits for a promote before any session installs it",
        "selected": "passed the checks and was published",
        "rejected": "failed the checks; the head stayed",
        "skipped": "no candidate reached the evaluation",
    }
    if _class(selection_result) == "promoted":
        notes[selection_result] = (
            f"passed the checks and was {selection_result}; the release that step published serves it"
        )
    lines = [f'<tr><th>result</th><td class="{_esc(_class(selection_result))}">{_esc(selection_result)}</td></tr>']
    if selection_result in notes:
        lines.append(f"<tr><th>meaning</th><td>{_esc(notes[selection_result])}</td></tr>")
    if metrics.get("skipped"):
        lines.append(f"<tr><th>skipped</th><td>{_esc(metrics['skipped'])}</td></tr>")
    failure = _notes(metrics).get("failure")
    if isinstance(failure, str) and failure.strip():
        # Why the proposer produced nothing: a failed model call, a reply with no entry.
        lines.append(f"<tr><th>proposer failure</th><td>{_esc(failure)}</td></tr>")
    for field in RESULT_FIELDS:
        legacy_field = "gate_sides" if field == "evaluation_sides" else field
        if field in metrics or legacy_field in metrics:
            value = metrics.get(field, metrics.get(legacy_field))
            if isinstance(value, bool):
                shown = json.dumps(value)
            elif isinstance(value, Sequence) and not isinstance(value, str):
                shown = ", ".join(_strings(value))
            else:
                shown = str(value)
            lines.append(f"<tr><th>{_esc(field.replace('_', ' '))}</th><td>{_esc(shown)}</td></tr>")
    evaluation_tokens = evaluation_token_counts(metrics)
    if evaluation_tokens is not None:
        lines.append(
            f"<tr><th>evaluation tokens</th><td>{evaluation_tokens[0]} in, {evaluation_tokens[1]} out</td></tr>"
        )
    selection = metrics.get("selection")
    if isinstance(selection, Mapping) and selection.get("reason"):
        lines.append(f"<tr><th>reason</th><td>{_esc(selection['reason'])}</td></tr>")
    if metrics.get("step_record"):
        lines.append(f'<tr><th>step record</th><td class="id">{_esc(metrics["step_record"])}</td></tr>')
    return "<table><tbody>" + "".join(lines) + "</tbody></table>"


def _requires_table(items: Sequence[Mapping[str, Any]]) -> str:
    """The ``requires`` items as setup reads them: name, kind, the check as written and the prompt setup shows."""
    rows = "".join(
        f"<tr><td>{_esc(item.get('name'))}</td><td>{_esc(item.get('kind'))}</td>"
        f"<td class=\"id\">{_esc(item.get('check') or '')}</td><td>{_esc(item.get('prompt') or '')}</td></tr>"
        for item in items
    )
    return (
        "<table><thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _refused_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """The ``refused_requires`` records: each item as written (name, kind, check, prompt) and why it was dropped."""
    rows = []
    for entry in entries:
        # The backend and the method record {item, reason}; a record without "item" is read as the item itself.
        item = entry.get("item", entry)
        if isinstance(item, Mapping):
            cells = (item.get("name"), item.get("kind"), item.get("check") or "", item.get("prompt") or "")
        else:
            # A malformed item need not be an object at all; its JSON stands where the name would.
            cells = (json.dumps(item, sort_keys=True), "", "", "")
        rows.append(
            f'<tr><td>{_esc(cells[0])}</td><td>{_esc(cells[1])}</td><td class="id">{_esc(cells[2])}</td>'
            f"<td>{_esc(cells[3])}</td><td>{_esc(entry.get('reason'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th><th>reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _setup(row: Mapping[str, Any], metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """The step's own ``training_request.requires`` items, then what its release carries from earlier steps.

    The install script, ``reef-<adapter> setup`` and the update notice read
    the union over the release's chain (``required_by``), so the page lists
    the same items split by the step that named them. A rejected or skipped
    row carries the head's id and published nothing, so only its own items
    show. The items the step dropped, the backend's under
    ``training_request.refused_requires`` and the method's under
    ``proposal_notes.refused_requires``, close the section with their reason."""
    request = metrics.get("training_request")
    request = request if isinstance(request, Mapping) else {}
    requires = request.get("requires")
    own = [item for item in requires if isinstance(item, Mapping)] if isinstance(requires, Sequence) else []
    carried: list[Mapping[str, Any]] = []
    if result_of(row) not in ("rejected", "skipped"):
        names = {item.get("name") for item in own}
        carried = [item for item in required_by(rows, row.get("release_id")) if item.get("name") not in names]
    refused = [
        entry
        for source in (request.get("refused_requires"), _notes(metrics).get("refused_requires"))
        if isinstance(source, Sequence) and not isinstance(source, str)
        for entry in source
        if isinstance(entry, Mapping)
    ]
    tail = f"<h3>refused by the step</h3>{_refused_table(refused)}" if refused else ""
    if not own and not carried:
        return '<p class="empty">this step names nothing to set up</p>' + tail
    parts = [_requires_table(own) if own else '<p class="empty">this step names nothing of its own</p>']
    if carried:
        parts.append(f"<h3>carried from earlier steps</h3>{_requires_table(carried)}")
    parts.append('<p class="sub">reef-pi setup lists these and runs a check only after you confirm it</p>')
    return "".join(parts) + tail


def _ran_on(other: Mapping[str, Any], release_id: Any) -> bool:
    """Whether ``other`` is a child of ``release_id``: a step evaluated on it, or a promote or rollback made on it."""
    if not release_id:
        return False
    if other.get("operation") in ("promote", "rollback"):
        return other.get("parent_release_id") == release_id
    # Not the parent: a rejected or skipped row copies the head's ref, so its parent is the grandparent.
    return before_release_id(other) == release_id


def _chain(step: int, row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    release_id = row.get("release_id")
    selection_result = result_of(row)
    if selection_result in ("rejected", "skipped"):
        # The row carries the head's id and published nothing, so the head's parent and children are not its own.
        ran_on = _esc(before_release_id(row) or "-")
        if selection_result == "rejected":
            ran_on += " (the head at this step; the candidate published nothing)"
        else:
            ran_on += " (the head at this step; nothing was evaluated)"
        return (
            "<table><tbody>"
            f'<tr><th>ran on</th><td class="id">{ran_on}</td></tr>'
            '<tr><th>children</th><td><span class="empty">none (the candidate published nothing)</span></td></tr>'
            "</tbody></table>"
        )
    children = [
        f'<li>step {index} <span class="id">{_esc(other.get("release_id"))}</span> {_span(result_of(other, rows))}</li>'
        for index, other in enumerate(rows)
        if index != step and _ran_on(other, release_id)
    ]
    listed = "<ul>" + "".join(children) + "</ul>" if children else '<span class="empty">none</span>'
    return (
        "<table><tbody>"
        f'<tr><th>parent</th><td class="id">{_esc(row.get("parent_release_id") or "-")}</td></tr>'
        f'<tr><th>this release</th><td class="id">{_esc(release_id)}</td></tr>'
        f"<tr><th>children</th><td>{listed}</td></tr>"
        "</tbody></table>"
    )


def build_release_page(
    step: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    before_entries: Sequence[Mapping[str, Any]] = (),
    before_files: Mapping[str, str] | None = None,
    node_paths: Mapping[str, str] | None = None,
) -> str:
    """The page for ``rows[step]``, the rows oldest first as ``GET /reef/harness/releases`` lists them.

    ``before_entries`` and ``before_files`` describe the release an update is
    read against (see ``before_release_id``); ``node_paths`` is the adapter's
    render template per kind, which names an extension's file. Without them an
    extension update shows its new text instead of a diff. Design and Review
    appear only when the row's ``proposal_notes`` carry them. Pure ASCII out:
    other characters leave as numeric references.
    """
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    entries = {str(entry["id"]): entry for entry in before_entries if isinstance(entry, Mapping) and "id" in entry}
    selection_result = result_of(row, rows)
    recorded = row.get("recorded_at")
    title = f"Harness step {step}"
    sub = f'release <span class="id">{_esc(row.get("release_id"))}</span> | {_span(selection_result)}'
    if served_step(rows) == step:
        sub += " | current"
    if isinstance(recorded, (int, float)):
        sub += f" | recorded at {recorded:.0f}"
    # Every "<" leaves the JSON as \u003c: a code text holding "</script>" would otherwise close the data block.
    data = json.dumps(row, ensure_ascii=True, sort_keys=True).replace("<", "\\u003c")
    page = (
        f"<title>{title}</title>\n<style>{STYLE}</style>\n<main>\n"
        f'<h1>{title}</h1>\n<p class="sub">{sub}</p>\n'
        f"<h2>Why</h2>\n{_why(row, metrics)}\n"
        f"{_design(metrics)}"
        f"<h2>What changed</h2>\n{_what_changed(row, metrics, entries, before_files, node_paths or {})}\n"
        f"{_review(metrics)}"
        f"<h2>Result</h2>\n{result_html(row, metrics, rows)}\n"
        f"<h2>Setup</h2>\n{_setup(row, metrics, rows)}\n"
        f"<h2>Chain</h2>\n{_chain(step, row, rows)}\n"
        "</main>\n"
        f'<script id="data" type="application/json">{data}</script>\n'
    )
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


# Compatibility aliases for existing imports.
VERDICT_FIELDS = RESULT_FIELDS
verdict_of = result_of

__all__ = [
    "RESULT_FIELDS",
    "VERDICT_FIELDS",
    "before_release_id",
    "build_release_page",
    "mutations_of",
    "result_of",
    "served_step",
    "verdict_of",
]
