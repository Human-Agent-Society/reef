"""One HTML page per catalog step: why the version exists, what it changed, the evaluation's result, its setup, its chain.

``GET /reef/harness/releases/{step}/page`` builds it from the releases row
plus, for an extension update, the file the release replaced. A step whose
method recorded ``proposal_notes`` (Reefine's design, review, refused
requires and undeclared variables) also gets a Design section after Why and
a Review section after What changed, and why the proposer produced nothing,
when the step recorded that, is a row of the Result section. The step is the
row's position in the catalog oldest first, the creation row being 0: a
rejected step publishes nothing and its row carries the head's release id,
so only the step names it.

The chrome, the palette and the status wording come from
:mod:`reef.service.page_chrome`, which the request page draws with too, so a
person moving between the two pages by their links reads one design. The
page carries its data inline and loads no asset, so one curl with the
scenario header is the whole read.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode

from reef.core.requirements import required_by
from reef.core.training_request import floor_tasks_note, missed_episode_text, missed_episodes, unscored_failures
from reef.harness.adapters import get_adapter
from reef.service.page_chrome import document, escape, requires_table, stamp, status_label, status_span, tone

#: The evaluation numbers the Result section lists, in this order, when the row carries them: a comparison writes
#: wins, losses and ties; a floor writes passed, failed and floor_score, and evaluation_sides when it ran one side only.
#: ``selected`` is not among them: the result headline above the numbers already says whether the step was published.
RESULT_FIELDS = (
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

#: Each evaluation number in the words a person reads, in place of the record's own key.
RESULT_LABELS = {
    "wins": "Wins",
    "losses": "Losses",
    "ties": "Ties",
    "passed": "Passed",
    "failed": "Failed",
    "floor_score": "Floor score",
    "evaluation_sides": "Evaluation sides",
    "current_score": "Current score",
    "candidate_score": "Candidate score",
    "episode_failures": "Episode failures",
    "proposer_input_tokens": "Proposer input tokens",
    "proposer_output_tokens": "Proposer output tokens",
}

#: What each result means for the person reading the step, under the headline.
RESULT_WORDS = {
    "pending": "Passed the checks and waits for a promote; no session installs it until then.",
    "selected": "Passed the checks and was published as the served head.",
    "rejected": "Did not pass the checks. The head stayed where it was.",
    "skipped": "No candidate reached the evaluation, so nothing changed.",
    "failed": "The step failed before evaluation. Nothing was published; see the error below before retrying.",
    "creation": "The tree this scenario started from, before any step ran.",
    "promote": "A person promoted a pending release, which now serves.",
    "rollback": "A person moved the head back to an earlier release.",
    "recovery": "The head this process recovered at boot.",
}


def failed_words(metrics: Mapping[str, Any]) -> str:
    """What a failed step means, by the phase it failed in: a step whose candidate reached its evaluation failed
    there, not before it."""
    if metrics.get("failed_stage") == "evaluating":
        return (
            "The step failed during its evaluation: the change was proposed and its episodes ran, but the "
            "evaluation raised. Nothing was published; see the error below before retrying."
        )
    return RESULT_WORDS["failed"]


#: Node kinds whose config carries the change as ``text``; the page shows that text instead of the config JSON.
TEXT_KINDS = ("rules", "skill", "agent_command")

# What the shared chrome does not draw: the reading column, the diff, the metric grid and the chain list.
STYLE = """
main{max-width:1040px;margin:auto;padding:48px 40px 24px}
.stack{display:grid;gap:24px}
.headline{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:14px}
.headline .id{font-size:12px}
.chip{display:inline-flex;align-items:center;gap:6px;border-radius:100px;padding:4px 10px;font-size:11px;
font-weight:600;background:var(--good-bg);color:var(--good)}
.steps{display:flex;align-items:center;gap:8px;margin-bottom:24px;font-size:12px}
.steps a,.steps span{border:1px solid var(--line);border-radius:7px;padding:7px 12px;background:var(--card)}
.steps span{color:var(--mute);opacity:.55}.steps a:hover{border-color:var(--accent);text-decoration:none}
.steps .here{margin-left:auto;border:0;background:none;padding:7px 0}
.text{font-size:16px;line-height:1.75;margin:0;padding-left:18px;border-left:2px solid var(--accent)}
.origin{font-size:12px;color:var(--mute);margin:16px 0 0;display:grid;gap:4px}
.origin span{overflow-wrap:anywhere}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);border:1px solid var(--line);
border-radius:8px;padding:14px;margin:0;max-height:34rem;overflow:auto;
font:12px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace}
.change{border-top:1px solid var(--line);padding-top:20px;margin-top:20px}
.change:first-child{border-top:0;padding-top:0;margin-top:0}
.change-head{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:12px}
.change-head .node-id{font-size:13px;font-weight:600;overflow-wrap:anywhere;min-width:0}
.diff .add{color:var(--good);background:var(--good-bg);display:block}
.diff .del{color:var(--bad);background:var(--bad-bg);display:block}
.diff .hunk{color:var(--mute);display:block}
.note{font-size:11px;color:var(--mute);margin:0 0 10px;overflow-wrap:anywhere}
.outcome-summary{border-radius:8px;background:var(--status-bg);padding:18px;margin-bottom:22px}
.outcome-summary .status{padding:0;background:none;margin-bottom:10px;font-weight:650}
.outcome-summary p{font-size:13px;margin:0;line-height:1.8;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:18px;margin:0 0 22px}
.metrics dd{font-size:20px;font-weight:600;letter-spacing:-.5px;line-height:1.2}
.details{margin:0;display:grid;gap:14px}
.details>div{display:flex;justify-content:space-between;gap:18px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.details>div:last-child{padding-bottom:0;border-bottom:0}
.details dt{font-size:12px;flex:none;margin:0}.details dd{text-align:right;min-width:0;font-size:12px}
.chain{list-style:none;margin:20px 0 0;padding:0;display:grid;gap:10px}
.chain li{display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-size:12px}
.chain .step-name{font-weight:600}
.review-list{margin:0;padding-left:18px;font-size:13px}.review-list li{padding:5px 0;overflow-wrap:anywhere}
.card h3{margin:22px 0 10px}.card h3:first-child{margin-top:0}
@media(min-width:1500px){main{padding-top:64px}}
@media(max-width:800px){main{padding:32px 24px 24px}}
@media(max-width:480px){main{padding:28px 16px 20px}.text{font-size:15px;padding-left:14px}
.metrics{grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:14px}.metrics dd{font-size:17px}
.details>div{flex-direction:column;gap:4px}.details dd{text-align:left}}
"""


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
    """The row's result, including failed proposals that published no release.

    A pending row stays pending in the catalog after a person promotes it;
    the promote is a later row naming it in ``rollback_target_release_id``,
    so with ``rows`` given such a row reads ``promoted at vN``."""
    if row.get("pending"):
        release_id = row.get("release_id")
        for index, other in enumerate(rows):
            if other.get("operation") == "promote" and other.get("rollback_target_release_id") == release_id:
                return f"promoted at v{index}"
        return "pending"
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        if isinstance(metrics.get("selected"), bool):
            return "selected" if metrics["selected"] else "rejected"
        if metrics.get("skipped"):
            notes = metrics.get("proposal_notes")
            failure = notes.get("failure") if isinstance(notes, Mapping) else None
            error = metrics.get("error")
            if any(isinstance(value, str) and value.strip() for value in (failure, error)):
                return "failed"
            return "skipped"
    return str(row.get("operation") or "unknown")


def served_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """The step whose release serves: the newest row that is neither pending nor a rejected or skipped step.

    The catalog's own ``current`` flag sits on the newest row, which a
    pending win or a failed evaluation makes the wrong one: those rows publish
    nothing, and a rejected or skipped row carries the head's id."""
    for index in range(len(rows) - 1, -1, -1):
        if result_of(rows[index]) not in ("pending", "rejected", "skipped", "failed"):
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
    if selection_result in ("rejected", "skipped", "failed"):
        return str(row.get("release_id") or "") or None
    return None


def step_href(step: int, link_query: Mapping[str, str] | None) -> str:
    """The page of another step, opened the way this one was: the scenario and the page key or token travel in the
    query."""
    href = f"/reef/harness/releases/{step}/page"
    if link_query:
        href += "?" + urlencode(dict(link_query))
    return href


def _card(name: str, body: str) -> str:
    return f'<section class="card">\n<h2>{name}</h2>\n{body}</section>\n'


def _why(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    operation = row.get("operation")
    if operation != "training":
        target = row.get("rollback_target_release_id")
        words = {
            "creation": "The tree this scenario started from; no step made it.",
            "promote": f"A person promoted release {target} after reading it.",
            "rollback": f"A person rolled the head back to release {target}.",
            "recovery": "The head this process recovered at boot.",
        }
        return f"<p>{escape(words.get(str(operation), f'A {operation} commit.'))}</p>"
    request = metrics.get("training_request")
    if isinstance(request, Mapping) and request.get("text"):
        return (
            f"<p class=\"text\">{escape(request['text'])}</p>"
            f'<p class="origin"><span>Request <span class="id">{escape(request.get("id"))}</span></span>'
            f'<span>Session <span class="id">{escape(request.get("session"))}</span></span>'
            f'<span>Filed on release <span class="id">{escape(request.get("release_id"))}</span></span></p>'
        )
    proposal = metrics.get("proposal")
    if isinstance(proposal, Mapping) and proposal.get("reason"):
        return (
            f"<p class=\"text\">{escape(proposal['reason'])}</p>"
            f'<p class="origin"><span>An agent\'s proposal <span class="id">{escape(proposal.get("id"))}</span>'
            f'</span><span>Session <span class="id">{escape(proposal.get("session"))}</span></span></p>'
        )
    return "<p>A failure in the batch; no request asked for this step.</p>"


#: The heading that ends a design and starts its How to use section: a markdown heading or a labelled line.
#: The How to use heading: a line of its own, or ``How to use:`` with the usage after it on the same line.
USAGE_HEADING = re.compile(r"^(?:#{1,6}\s*)?how to use(?:\s*:[ \t]*|\s*$)", re.IGNORECASE | re.MULTILINE)


def design_sections(notes: Mapping[str, Any]) -> tuple[str, str]:
    """The design the method recorded as ``proposal_notes.design`` and its How to use section, each stripped;
    empty where the record has none. The proposer writes the usage under a ``How to use`` heading at the end."""
    design = notes.get("design")
    if not isinstance(design, str) or not design.strip():
        return "", ""
    match = USAGE_HEADING.search(design)
    if match is None:
        return design.strip(), ""
    return design[: match.start()].strip(), design[match.end() :].strip()


def _design(metrics: Mapping[str, Any]) -> str:
    """The Design section and, when the design ends with one, the How to use section, when the method recorded
    ``proposal_notes.design``."""
    design, usage = design_sections(_notes(metrics))
    cards = _card("Design", f'<p class="text">{escape(design)}</p>\n') if design else ""
    if usage:
        cards += _card("How to use", f'<p class="text">{escape(usage)}</p>\n')
    return cards


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
            rendered.append(f'<span class="hunk">{escape(line)}</span>')
        elif line.startswith("+"):
            rendered.append(f'<span class="add">{escape(line)}</span>')
        elif line.startswith("-"):
            rendered.append(f'<span class="del">{escape(line)}</span>')
        elif line.startswith("@@"):
            rendered.append(f'<span class="hunk">{escape(line)}</span>')
        else:
            rendered.append(escape(line))
    if not rendered:
        return f'<p class="empty">{escape(path)} is unchanged</p>'
    return '<pre class="diff">' + "\n".join(rendered) + "</pre>"


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
    operation_class = op if op in ("create", "update", "delete", "remove") else "other"
    head = (
        f'<div class="change"><div class="change-head">'
        f'<span class="tag operation-{escape(operation_class)}">{escape(op)}</span>'
        f'<span class="node-id">{escape(entry_id)}</span><span class="tag">{escape(kind)}</span></div>'
    )
    if op == "remove":
        return head + "</div>"
    config = options.get("config")
    config = config if isinstance(config, Mapping) else {}
    if kind == "code_extension":
        code = config.get("code")
        if not isinstance(code, str):
            return head + '<p class="empty">this mutation changes no code</p></div>'
        previous_config = previous.get("config")
        previous_config = previous_config if isinstance(previous_config, Mapping) else {}
        name = config.get("name") or previous_config.get("name")
        template = node_paths.get("code_extension")
        if op == "update" and before_files is not None and template and name:
            path = template.format(name=name)
            before = before_files.get(path)
            if before is not None:
                return head + _diff_block(path, before, code, before_release_id(row), row.get("release_id")) + "</div>"
        return head + f"<pre>{escape(code)}</pre></div>"
    if kind in TEXT_KINDS and isinstance(config.get("text"), str):
        rest = {key: value for key, value in options.items() if key not in ("config", "name")}
        rest.update({key: value for key, value in config.items() if key != "text"})
        note = f'<p class="note">{escape(json.dumps(rest, sort_keys=True))}</p>' if rest else ""
        return head + note + f"<pre>{escape(config['text'])}</pre></div>"
    return head + f"<pre>{escape(json.dumps(options, indent=2, sort_keys=True))}</pre></div>"


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
        return f'<p class="empty">nothing: the step skipped ({escape(metrics["skipped"])})</p>'
    if metrics.get("recheck"):
        return '<p class="empty">nothing new: a recheck of the last good tree against the published one</p>'
    if row.get("operation") == "training":
        return '<p class="empty">no mutation on record</p>'
    if row.get("operation") == "creation":
        return '<p class="empty">nothing: the seed as the recipe rendered it</p>'
    return '<p class="empty">no mutation: the head moved without a step</p>'


def _listed(items: Sequence[str], empty: str) -> str:
    if not items:
        return f'<p class="empty">{escape(empty)}</p>'
    return '<ul class="review-list">' + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _review(metrics: Mapping[str, Any]) -> str:
    """The Review section: the proposer's reading of its entries against the request, then what it left undeclared.

    ``proposal_notes.review`` is absent when the method's review call failed,
    and ``review_failure`` then says why: the one check of whether the entries
    deliver the request did not run, which the page says rather than leaving
    the section out. The ``undeclared_env`` line shows all the same, being the
    warning the person needs. Empty when the step recorded none of them."""
    notes = _notes(metrics)
    review = notes.get("review")
    failure = notes.get("review_failure")
    undeclared = _strings(notes.get("undeclared_env"))
    dropped = _strings(notes.get("dropped_attempts"))
    if not isinstance(review, Mapping) and not isinstance(failure, str) and not undeclared and not dropped:
        return ""
    parts = []
    if isinstance(review, Mapping):
        review_result = status_span(str(review.get("result", review.get("verdict")) or "unknown"))
        parts.append(f"<p>The proposer's review of its entries against the request: {review_result}</p>")
        kept = kept_answer(notes)
        if kept is not None:
            parts.append(f"<p>{escape(kept)}; the others are in the step record</p>")
        parts.append("<h3>Covered</h3>" + _listed(_strings(review.get("covered")), "nothing listed as covered"))
        parts.append("<h3>Uncovered</h3>" + _listed(_strings(review.get("uncovered")), "nothing left uncovered"))
        limits = _strings(review.get("limits"))
        if limits:
            # What the harness's notes say no answer can deliver there: not a gap in this change.
            parts.append("<h3>Out of reach on this harness</h3>" + _listed(limits, ""))
    elif isinstance(failure, str) and failure.strip():
        parts.append(
            "<p>The proposer's review of its entries against the request did not run, so nothing checked "
            f"whether they deliver it: {escape(failure)}</p>"
        )
    else:
        parts.append('<p class="empty">no review on record</p>')
    if undeclared:
        parts.append(
            '<div class="failure"><h3>Undeclared variables</h3><p>The extension reads these and no requires item '
            f'names them: <span class="id">{escape(", ".join(undeclared))}</span></p></div>'
        )
    if dropped:
        parts.append("<h3>Answers written again</h3>" + _listed(dropped, ""))
    return _card("Review", "".join(parts) + "\n")


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


def result_html(row: Mapping[str, Any], metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """The Result section: the headline and what it means, then the evaluation's numbers and the step's record."""
    selection_result = result_of(row, rows)
    words = {**RESULT_WORDS, "failed": failed_words(metrics)}
    if tone(selection_result) == "promoted":
        words[selection_result] = (
            f"Passed the checks and was {selection_result}; the release that step published serves it."
        )
    # Every candidate episode failed before a score: the checks judged nothing, so no score stands for the change.
    unscored = unscored_failures(metrics) if selection_result == "rejected" else []
    if unscored:
        words["rejected"] = (
            "The evaluation could not run: every candidate episode failed before it was scored, so nothing judged "
            "the change. The head stayed where it was."
        )
    summary = (
        f'<div class="outcome-summary"><div class="status">{status_span(selection_result)}</div>'
        f"<p>{escape(words.get(selection_result, f'The step ended as {selection_result}.'))}</p></div>"
    )
    numbers = []
    for field in RESULT_FIELDS:
        legacy_field = "gate_sides" if field == "evaluation_sides" else field
        if unscored and field == "candidate_score":
            continue
        if field in metrics or legacy_field in metrics:
            value = metrics.get(field, metrics.get(legacy_field))
            if isinstance(value, Sequence) and not isinstance(value, str):
                shown = ", ".join(_strings(value))
            else:
                shown = str(value)
            numbers.append(f"<div><dt>{escape(RESULT_LABELS[field])}</dt><dd>{escape(shown)}</dd></div>")
    evaluation_tokens = evaluation_token_counts(metrics)
    if evaluation_tokens is not None:
        numbers.append(
            f"<div><dt>Evaluation tokens</dt><dd>{evaluation_tokens[0]} in, {evaluation_tokens[1]} out</dd></div>"
        )
    grid = f'<dl class="metrics">{"".join(numbers)}</dl>' if numbers else ""
    details = []
    if metrics.get("skipped"):
        details.append(f"<div><dt>Skipped</dt><dd>{escape(metrics['skipped'])}</dd></div>")
    selection = metrics.get("selection")
    if isinstance(selection, Mapping) and selection.get("reason"):
        details.append(f"<div><dt>Reason</dt><dd>{escape(selection['reason'])}</dd></div>")
    if unscored:
        details.extend(f"<div><dt>Could not run</dt><dd>{escape(cause)}</dd></div>" for cause in unscored)
    elif selection_result == "rejected":
        # What the checks saw: each missed episode's task, and why it failed or the reply that was graded.
        details.extend(
            f"<div><dt>Missed</dt><dd>{escape(missed_episode_text(episode))}</dd></div>"
            for episode in missed_episodes(metrics)
        )
    if metrics.get("step_record"):
        details.append(f'<div><dt>Step record</dt><dd class="id">{escape(metrics["step_record"])}</dd></div>')
    listed = f'<dl class="details">{"".join(details)}</dl>' if details else ""
    note = floor_tasks_note(metrics)
    if note is not None:
        listed += f'<p class="note">{escape(note)}</p>'
    failure = _notes(metrics).get("failure")
    if isinstance(failure, str) and failure.strip():
        # Why the proposer produced nothing: a failed model call, a reply with no entry.
        listed += f'<div class="failure"><h3>Proposer failure</h3><p>{escape(failure)}</p></div>'
    return summary + grid + listed


def _scrolled(table: str, label: str) -> str:
    return f'<div class="table-scroll" role="region" aria-label="{escape(label)}" tabindex="0">{table}</div>'


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
            f'<tr><td>{escape(cells[0])}</td><td>{escape(cells[1])}</td><td class="id">{escape(cells[2])}</td>'
            f"<td>{escape(cells[3])}</td><td>{escape(entry.get('reason'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th><th>reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def reef_installs(adapter: str) -> bool:
    """Whether Reef installs ``adapter``'s harness with a wrapper that sets up and updates it; terminus, a batch
    runner, has none: its tree is served by ``GET /reef/harness`` and a run's Harbor task holds what it needs."""
    return get_adapter(adapter).install is not None


def kept_answer(notes: Mapping[str, Any]) -> str | None:
    """Which of the proposer's answers the step kept, when it wrote more than one and kept one: ``kept_attempt``,
    or the last answer when the notes name none; nothing for a step whose answers all failed."""
    attempts = notes.get("attempts")
    if not isinstance(attempts, int) or attempts < 2 or notes.get("failure"):
        return None
    kept = notes.get("kept_attempt", attempts)
    return f"Kept answer {kept} of {attempts}" if isinstance(kept, int) and 1 <= kept <= attempts else None


def _setup(
    row: Mapping[str, Any], metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: str = "pi"
) -> str:
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
    if result_of(row) not in ("rejected", "skipped", "failed"):
        names = {item.get("name") for item in own}
        carried = [item for item in required_by(rows, row.get("release_id")) if item.get("name") not in names]
    refused = [
        entry
        for source in (request.get("refused_requires"), _notes(metrics).get("refused_requires"))
        if isinstance(source, Sequence) and not isinstance(source, str)
        for entry in source
        if isinstance(entry, Mapping)
    ]
    tail = (
        f'<h3>Refused by the step</h3>{_scrolled(_refused_table(refused), "Refused requirements")}' if refused else ""
    )
    if not own and not carried:
        return '<p class="empty">this step names nothing to set up</p>' + tail
    parts = [
        (
            _scrolled(requires_table(own), "Requirements named by this step")
            if own
            else '<p class="empty">this step names nothing of its own</p>'
        )
    ]
    if carried:
        parts.append(
            f'<h3>Carried from earlier steps</h3>{_scrolled(requires_table(carried), "Inherited requirements")}'
        )
    parts.append(
        f'<p class="note">reef-{escape(adapter)} setup lists these and runs a check only after you confirm it</p>'
        if reef_installs(adapter)
        else '<p class="note">these must hold in the Harbor task a run uses; nothing checks them</p>'
    )
    return "".join(parts) + tail


def _ran_on(other: Mapping[str, Any], release_id: Any) -> bool:
    """Whether ``other`` is a child of ``release_id``: a step evaluated on it, or a promote or rollback made on it."""
    if not release_id:
        return False
    if other.get("operation") in ("promote", "rollback"):
        return other.get("parent_release_id") == release_id
    # Not the parent: a rejected or skipped row copies the head's ref, so its parent is the grandparent.
    return before_release_id(other) == release_id


def _chain(
    step: int, row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], link_query: Mapping[str, str] | None
) -> str:
    release_id = row.get("release_id")
    selection_result = result_of(row)
    if selection_result in ("rejected", "skipped", "failed"):
        # The row carries the head's id and published nothing, so the head's parent and children are not its own.
        if selection_result == "rejected":
            ran_on = "the head at this step; the candidate published nothing"
        else:
            ran_on = "the head at this step; nothing was evaluated"
        return (
            '<dl class="details">'
            f'<div><dt>Ran on</dt><dd class="id">{escape(before_release_id(row) or "-")} ({escape(ran_on)})</dd></div>'
            "<div><dt>Children</dt><dd>none (the candidate published nothing)</dd></div>"
            "</dl>"
        )
    children = [
        f'<li><a class="step-name" href="{escape(step_href(index, link_query))}">v{index}</a>'
        f'<span class="id">{escape(other.get("release_id"))}</span>{status_span(result_of(other, rows))}</li>'
        for index, other in enumerate(rows)
        if index != step and _ran_on(other, release_id)
    ]
    listed = (
        f'<h3>Children</h3><ul class="chain">{"".join(children)}</ul>'
        if children
        else '<h3>Children</h3><p class="empty">none</p>'
    )
    return (
        '<dl class="details">'
        f'<div><dt>Parent</dt><dd class="id">{escape(row.get("parent_release_id") or "-")}</dd></div>'
        f'<div><dt>This release</dt><dd class="id">{escape(release_id)}</dd></div>'
        "</dl>" + listed
    )


def _steps_nav(step: int, rows: Sequence[Mapping[str, Any]], link_query: Mapping[str, str] | None) -> str:
    """The walk along the catalog: the neighbouring steps, a dead end shown as text, and where this step sits."""
    parts = []
    if step > 0:
        parts.append(f'<a href="{escape(step_href(step - 1, link_query))}" rel="prev">&#8592; v{step - 1}</a>')
    else:
        parts.append("<span>&#8592;</span>")
    if step + 1 < len(rows):
        parts.append(f'<a href="{escape(step_href(step + 1, link_query))}" rel="next">v{step + 1} &#8594;</a>')
    else:
        parts.append("<span>&#8594;</span>")
    parts.append(f'<span class="here">v{step} of v{len(rows) - 1}</span>')
    return f'<nav class="steps" aria-label="Catalog steps">{"".join(parts)}</nav>\n'


def build_running_step_page(
    step: int, request_id: str, link_query: Mapping[str, str] | None = None, *, refresh_seconds: int = 5
) -> str:
    """The page of step ``step`` while the request ``request_id`` runs it: the catalog has no row for it yet, so the
    page says the step is running, links the request's page, which follows it live, and reloads until the row lands."""
    href = f"/reef/harness/requests/{quote(request_id, safe='')}/page"
    if link_query:
        href += "?" + urlencode(dict(link_query))
    body = (
        '<div class="stack">\n'
        + _card(
            "Running",
            "<p>This step is running: the catalog records it when it settles, and this page then shows it.</p>"
            f'<p><a class="version-link" href="{escape(href)}">Follow the request</a></p>\n',
        )
        + "</div>\n"
    )
    return document(
        title=f"Harness v{step}",
        style=STYLE,
        breadcrumb="Versions",
        context=link_query.get("scenario", "") if link_query else "",
        state="running",
        eyebrow="Harness evolution",
        heading=f"Harness v{step}",
        subtitle="Running",
        body=body,
        head=f'<meta http-equiv="refresh" content="{refresh_seconds}">\n',
    )


def build_release_page(
    step: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    before_entries: Sequence[Mapping[str, Any]] = (),
    before_files: Mapping[str, str] | None = None,
    node_paths: Mapping[str, str] | None = None,
    link_query: Mapping[str, str] | None = None,
    adapter: str = "pi",
) -> str:
    """The page for ``rows[step]``, the rows oldest first as ``GET /reef/harness/releases`` lists them.

    ``before_entries`` and ``before_files`` describe the release an update is
    read against (see ``before_release_id``); ``node_paths`` is the adapter's
    render template per kind, which names an extension's file. Without them an
    extension update shows its new text instead of a diff. ``link_query`` is
    carried to the Chain's links, so a page opened through query parameters
    links pages that open the same way. Design and Review appear only when the
    row's ``proposal_notes`` carry them. ``adapter`` names the wrapper the
    Setup section's commands run.
    """
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    entries = {str(entry["id"]): entry for entry in before_entries if isinstance(entry, Mapping) and "id" in entry}
    selection_result = result_of(row, rows)
    recorded = row.get("recorded_at")
    headline = [f'<span class="id">{escape(row.get("release_id"))}</span>']
    if served_step(rows) == step:
        headline.append('<span class="chip">Currently served</span>')
    subtitle = f'<span class="headline">{"".join(headline)}</span>'
    if isinstance(recorded, (int, float)):
        subtitle += f"Recorded {stamp(recorded)}"
    else:
        subtitle += f"{escape(status_label(selection_result))} in this scenario's catalog"
    # Every "<" leaves the JSON as \\u003c: a code text holding "</script>" would otherwise close the data block.
    data = json.dumps(row, ensure_ascii=True, sort_keys=True).replace("<", "\\u003c")
    served = served_step(rows)
    return document(
        title=f"Harness v{step}",
        style=STYLE,
        breadcrumb="Versions",
        context=link_query.get("scenario", "") if link_query else "",
        state=selection_result,
        eyebrow="Harness evolution",
        heading=f"Harness v{step}",
        subtitle=subtitle,
        # The served head is this scenario's home; on its own page the crumb stays text.
        home="" if served is None or served == step else step_href(served, link_query),
        body=_steps_nav(step, rows, link_query)
        + '<div class="stack">\n'
        + _card("Why", f"{_why(row, metrics)}\n")
        + _design(metrics)
        + _card("What changed", f"{_what_changed(row, metrics, entries, before_files, node_paths or {})}\n")
        + _review(metrics)
        + _card("Result", f"{result_html(row, metrics, rows)}\n")
        + _card("Setup", f"{_setup(row, metrics, rows, adapter)}\n")
        + _card("Chain", f"{_chain(step, row, rows, link_query)}\n")
        + "</div>\n",
        tail=f'<script id="data" type="application/json">{data}</script>\n',
    )


# Compatibility aliases for existing imports.
VERDICT_FIELDS = RESULT_FIELDS
verdict_of = result_of

__all__ = [
    "RESULT_FIELDS",
    "RESULT_LABELS",
    "VERDICT_FIELDS",
    "before_release_id",
    "build_release_page",
    "build_running_step_page",
    "failed_words",
    "mutations_of",
    "result_of",
    "served_step",
    "step_href",
    "verdict_of",
]
