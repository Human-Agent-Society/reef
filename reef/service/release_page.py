"""One HTML page per catalog step: why the version exists, what it changed, the evaluation's result, its setup, its chain.

``GET /reef/harness/releases/{step}/page`` builds it from the releases row
plus the source files of the release the step ran on, which each updated
file of the step's patch is read against. A step whose method returned a
summary (``method_summary``) gets a Method summary section after Why, and a
step whose method returned notes (``method_notes``) gets a Method notes
section after Result. The step is the row's position in the catalog oldest
first, the creation row being 0: a rejected step publishes nothing and its
row carries the head's release id, so only the step names it.

The row readers here serve the request page too: the result, what the
method said, the episodes a rejected step missed and an evaluation that
could not run, so both pages name one cause the same way. The layout, the
palette and the status wording come from :mod:`reef.service.page_layout`,
which the request page draws with too, so a person moving between the two
pages by their links reads one design. The page carries its data inline and
loads no asset, so one curl with the scenario header is the whole read.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode

from reef.core.requirements import required_by
from reef.service.page_layout import document, escape, requires_table, stamp, status_label, status_span, tone

#: The evaluation numbers the Result section lists, in this order, when the row carries them: a comparison writes
#: wins, losses and ties, or the two means; a floor writes passed, failed and floor_score, and evaluation_sides when it
#: ran one side only; a step that ran its method writes the method's episodes and model use.
#: ``selected`` is not among them: the result headline above the numbers already says whether the step was published.
RESULT_FIELDS = (
    "wins",
    "losses",
    "ties",
    "candidate_mean",
    "current_mean",
    "passed",
    "failed",
    "floor_score",
    "evaluation_sides",
    "current_score",
    "candidate_score",
    "episode_failures",
    "episodes_run",
    "episodes_cached",
    "method_episodes",
    "model_calls",
    "model_seconds",
    "model_input_tokens",
    "model_output_tokens",
)

#: Each evaluation number in the words a person reads, in place of the record's own key.
RESULT_LABELS = {
    "wins": "Wins",
    "losses": "Losses",
    "ties": "Ties",
    "candidate_mean": "Candidate mean",
    "current_mean": "Current mean",
    "passed": "Passed",
    "failed": "Failed",
    "floor_score": "Floor score",
    "evaluation_sides": "Evaluation sides",
    "current_score": "Current score",
    "candidate_score": "Candidate score",
    "episode_failures": "Episode failures",
    "episodes_run": "Episodes run",
    "episodes_cached": "Episodes cached",
    "method_episodes": "Method episodes",
    "model_calls": "Model calls",
    "model_seconds": "Model seconds",
    "model_input_tokens": "Model input tokens",
    "model_output_tokens": "Model output tokens",
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
    "import": "A person published the harness from their machine, which now serves. Reef did not evaluate it.",
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


# What the shared layout does not draw: the reading column, the diff, the metric grid and the chain list.
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
.card h3{margin:22px 0 10px}.card h3:first-child{margin-top:0}
@media(min-width:1500px){main{padding-top:64px}}
@media(max-width:800px){main{padding:32px 24px 24px}}
@media(max-width:480px){main{padding:28px 16px 20px}.text{font-size:15px;padding-left:14px}
.metrics{grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:14px}.metrics dd{font-size:17px}
.details>div{flex-direction:column;gap:4px}.details dd{text-align:left}}
"""


def short_id(release_id: Any) -> str:
    return str(release_id)[:8] if release_id else "-"


def text_items(value: Any) -> list[str]:
    """The items of a JSON list as text, in order; none when ``value`` is not a list."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item if isinstance(item, str) else json.dumps(item, sort_keys=True) for item in value]
    return []


def recorded_patch(metrics: Mapping[str, Any] | None) -> dict[str, str | None]:
    """The patch a step recorded as ``patch``: tree path to new text, or to ``None`` for a removal.

    Empty for a step that recorded none, such as a skip before a candidate or the creation row. The record clips
    and redacts long texts, so a text here can differ from the released file.
    """
    patch = metrics.get("patch") if metrics else None
    if not isinstance(patch, Mapping):
        return {}
    return {str(path): value for path, value in patch.items() if value is None or isinstance(value, str)}


def result_of(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] = ()) -> str:
    """The row's result: a step that skipped with an error failed, and published no release.

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
            error = metrics.get("error")
            return "failed" if isinstance(error, str) and error.strip() else "skipped"
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
    """The page of another step, opened the way this one was: the scenario and the page key travel in the
    query."""
    href = f"/reef/harness/releases/{step}/page"
    if link_query:
        href += "?" + urlencode(dict(link_query))
    return href


def method_summary(metrics: Mapping[str, Any]) -> str:
    """The plain text the method returned about its step (``method_summary``), stripped; empty without one."""
    summary = metrics.get("method_summary")
    return summary.strip() if isinstance(summary, str) else ""


def method_notes(metrics: Mapping[str, Any]) -> str:
    """The notes the method returned about its step (``method_notes``) as indented JSON with sorted keys; empty
    without any."""
    notes = metrics.get("method_notes")
    if not isinstance(notes, Mapping) or not notes:
        return ""
    return json.dumps(notes, indent=2, sort_keys=True, ensure_ascii=False)


def missed_episodes(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The candidate episodes a step's evaluation names that failed or scored below its floor, from the row's
    ``candidate_episodes`` summary (``task``, ``score``, ``failure``, ``reply``); none when the step recorded none."""
    episodes = metrics.get("candidate_episodes")
    if not isinstance(episodes, list):
        return []
    selection = metrics.get("selection")
    decided = selection.get("metrics") if isinstance(selection, Mapping) else None
    floor = decided.get("floor_score") if isinstance(decided, Mapping) else None
    missed = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        score = episode.get("score")
        below = isinstance(floor, (int, float)) and isinstance(score, (int, float)) and score < floor
        if episode.get("failure") or score is None or below:
            missed.append(episode)
    return missed


def unscored_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Why the evaluation could not run: the distinct failures, when every candidate episode failed before it was
    scored (a launch that found no binary, say), so the step judged nothing; none when any episode was scored or
    the step recorded no episodes."""
    episodes = metrics.get("candidate_episodes")
    if not isinstance(episodes, list) or not episodes:
        return []
    causes: list[str] = []
    for episode in episodes:
        if not isinstance(episode, Mapping) or episode.get("score") is not None or not episode.get("failure"):
            return []
        cause = str(episode["failure"])
        if cause not in causes:
            causes.append(cause)
    return causes


def floor_tasks_note(metrics: Mapping[str, Any]) -> str | None:
    """For a step that answered a request under a floor: its floor tasks were set before the request and check that
    the changed harness still passes them, not what the request asks for; ``None`` for any other step."""
    selection = metrics.get("selection")
    if not isinstance(metrics.get("training_request"), Mapping) or not isinstance(selection, Mapping):
        return None
    if selection.get("policy") != "floor":
        return None
    episodes = metrics.get("candidate_episodes")
    tasks: list[str] = []
    for episode in episodes if isinstance(episodes, list) else ():
        task = str(episode.get("task") or "").strip() if isinstance(episode, Mapping) else ""
        # A Harbor task directory shows as its name, a prompt as its start.
        name = task.rstrip("/").rsplit("/", 1)[-1] if task.startswith("/") else task
        name = name if len(name) <= 60 else f"{name[:57]}..."
        if name and name not in tasks:
            tasks.append(name)
    named = f" ({'; '.join(tasks)})" if tasks else ""
    return (
        f"The floor tasks{named} were set before this request: they check that the changed harness still passes "
        "them, not what the request asks for."
    )


def missed_episode_text(episode: Mapping[str, Any]) -> str:
    """One missed episode in words: the task, then why it failed, or its score and the reply that was graded."""
    task = str(episode.get("task") or "").strip()
    failure = episode.get("failure")
    if failure:
        return f"the task '{task}' failed: {failure}"
    reply = episode.get("reply")
    if episode.get("transcript_read") is False:
        graded = "no transcript was read from the episode's session log, so no reply was graded"
    elif reply:
        graded = f"the reply graded was '{str(reply).strip()}'"
    else:
        graded = "no reply was graded"
    return f"the task '{task}' scored {episode.get('score')}; {graded}"


def card(name: str, body: str) -> str:
    return f'<section class="card">\n<h2>{name}</h2>\n{body}</section>\n'


def why_html(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    operation = row.get("operation")
    if operation != "training":
        target = row.get("rollback_target_release_id")
        words = {
            "creation": "The tree this scenario started from; no step made it.",
            "promote": f"A person promoted release {target} after reading it.",
            "rollback": f"A person rolled the head back to release {target}.",
            "import": "A person published the harness from their machine; Reef checked its files and did not "
            "evaluate it.",
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
    return "<p>A failure in the job data; no request asked for this step.</p>"


def summary_html(metrics: Mapping[str, Any]) -> str:
    """The Method summary section, when the method returned a summary: its plain text with the line breaks kept."""
    summary = method_summary(metrics)
    return card("Method summary", f'<p class="text">{escape(summary)}</p>\n') if summary else ""


def notes_html(metrics: Mapping[str, Any]) -> str:
    """The Method notes section, when the method returned notes: the notes as JSON."""
    notes = method_notes(metrics)
    return card("Method notes", f"<pre>{escape(notes)}</pre>\n") if notes else ""


def diff_block(path: str, before: str, after: str, before_id: str | None, release_id: Any) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{path} ({short_id(before_id)})",
        tofile=f"{path} ({short_id(release_id)})",
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


def changed_file_html(
    path: str, after: str | None, row: Mapping[str, Any], before_files: Mapping[str, str] | None
) -> str:
    """One file of a step's patch: a removal, a diff against the release the step ran on, or the new file's text.

    Without ``before_files`` (the release the step ran on is not restorable), a written file shows its new text
    under ``write``, since the page cannot tell an update from a new file.
    """
    if after is None:
        operation, body = "remove", ""
    elif before_files is None:
        operation, body = "write", f"<pre>{escape(after)}</pre>"
    elif path in before_files:
        before = before_files[path]
        operation, body = "update", diff_block(path, before, after, before_release_id(row), row.get("release_id"))
    else:
        operation, body = "create", f"<pre>{escape(after)}</pre>"
    return (
        '<div class="change"><div class="change-head">'
        f'<span class="tag operation-{operation}">{operation}</span>'
        f'<span class="node-id">{escape(path)}</span></div>{body}</div>'
    )


def what_changed_html(
    row: Mapping[str, Any], metrics: Mapping[str, Any], before_files: Mapping[str, str] | None
) -> str:
    patch = recorded_patch(metrics)
    if patch:
        return "".join(changed_file_html(path, after, row, before_files) for path, after in sorted(patch.items()))
    if metrics.get("skipped"):
        return f'<p class="empty">nothing: the step skipped ({escape(metrics["skipped"])})</p>'
    if row.get("operation") == "training":
        return '<p class="empty">no patch on record</p>'
    if row.get("operation") == "creation":
        return '<p class="empty">nothing: this release is the seed tree</p>'
    return '<p class="empty">nothing: the head moved without a step</p>'


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
        if unscored and field == "candidate_score":
            continue
        if field in metrics:
            value = metrics[field]
            if isinstance(value, Sequence) and not isinstance(value, str):
                shown = ", ".join(text_items(value)) or "none"
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
    error = metrics.get("error")
    if isinstance(error, str) and error.strip():
        # Why the step failed: what its method, or its evaluation, raised.
        listed += f'<div class="failure"><h3>Error</h3><p>{escape(error)}</p></div>'
    return summary + grid + listed


def scrolled(table: str, label: str) -> str:
    return f'<div class="table-scroll" role="region" aria-label="{escape(label)}" tabindex="0">{table}</div>'


def refused_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """The ``refused_requires`` records, each ``{item, reason}``: the item as written (name, kind, check, prompt)
    and why it was refused."""
    rows = []
    for entry in entries:
        item = entry["item"]
        rows.append(
            f'<tr><td>{escape(item.get("name"))}</td><td>{escape(item.get("kind"))}</td>'
            f'<td class="id">{escape(item.get("check") or "")}</td><td>{escape(item.get("prompt") or "")}</td>'
            f"<td>{escape(entry.get('reason'))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>name</th><th>kind</th><th>check</th><th>prompt</th><th>reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def setup_html(row: Mapping[str, Any], metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """The step's own ``training_request.requires`` items, then what its release carries from earlier steps.

    ``GET /reef/harness`` answers the union over the release's chain
    (``required_by``) as ``requires``, so the page lists the same items split
    by the step that named them. A rejected or skipped row carries the head's
    id and published nothing, so only its own items show. The items the step
    refused, under ``training_request.refused_requires``, close the section
    with their reason."""
    request = metrics.get("training_request")
    request = request if isinstance(request, Mapping) else {}
    requires = request.get("requires")
    own = [item for item in requires if isinstance(item, Mapping)] if isinstance(requires, Sequence) else []
    carried: list[Mapping[str, Any]] = []
    if result_of(row) not in ("rejected", "skipped", "failed"):
        names = {item.get("name") for item in own}
        carried = [item for item in required_by(rows, row.get("release_id")) if item.get("name") not in names]
    refused_requires = request.get("refused_requires")
    refused = (
        [entry for entry in refused_requires if isinstance(entry, Mapping) and isinstance(entry.get("item"), Mapping)]
        if isinstance(refused_requires, Sequence) and not isinstance(refused_requires, str)
        else []
    )
    tail = f'<h3>Refused by the step</h3>{scrolled(refused_table(refused), "Refused requirements")}' if refused else ""
    if not own and not carried:
        return '<p class="empty">this step names nothing to set up</p>' + tail
    parts = [
        (
            scrolled(requires_table(own), "Requirements named by this step")
            if own
            else '<p class="empty">this step names nothing of its own</p>'
        )
    ]
    if carried:
        parts.append(
            f'<h3>Carried from earlier steps</h3>{scrolled(requires_table(carried), "Inherited requirements")}'
        )
    parts.append('<p class="note">these must hold where the harness runs; Reef does not check them</p>')
    return "".join(parts) + tail


def ran_on(other: Mapping[str, Any], release_id: Any) -> bool:
    """Whether ``other`` is a child of ``release_id``: a step evaluated on it, or a promote, rollback or import made
    on it."""
    if not release_id:
        return False
    if other.get("operation") in ("promote", "rollback", "import"):
        return other.get("parent_release_id") == release_id
    # Not the parent: a rejected or skipped row copies the head's ref, so its parent is the grandparent.
    return before_release_id(other) == release_id


def chain_html(
    step: int, row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], link_query: Mapping[str, str] | None
) -> str:
    release_id = row.get("release_id")
    selection_result = result_of(row)
    if selection_result in ("rejected", "skipped", "failed"):
        # The row carries the head's id and published nothing, so the head's parent and children are not its own.
        if selection_result == "rejected":
            words = "the head at this step; the candidate published nothing"
        else:
            words = "the head at this step; nothing was evaluated"
        return (
            '<dl class="details">'
            f'<div><dt>Ran on</dt><dd class="id">{escape(before_release_id(row) or "-")} ({escape(words)})</dd></div>'
            "<div><dt>Children</dt><dd>none (the candidate published nothing)</dd></div>"
            "</dl>"
        )
    children = [
        f'<li><a class="step-name" href="{escape(step_href(index, link_query))}">v{index}</a>'
        f'<span class="id">{escape(other.get("release_id"))}</span>{status_span(result_of(other, rows))}</li>'
        for index, other in enumerate(rows)
        if index != step and ran_on(other, release_id)
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


def steps_nav(step: int, rows: Sequence[Mapping[str, Any]], link_query: Mapping[str, str] | None) -> str:
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
        + card(
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
    before_files: Mapping[str, str] | None = None,
    link_query: Mapping[str, str] | None = None,
) -> str:
    """The page for ``rows[step]``, the rows oldest first as ``GET /reef/harness/releases`` lists them.

    ``before_files`` is the source tree of the release the step ran on (see
    ``before_release_id``), which each updated file of the patch is read
    against; without it, a written file shows its new text instead of a diff.
    ``link_query`` is carried to the Chain's links, so a page opened through
    query parameters links pages that open the same way. Method summary and
    Method notes appear only when the row carries ``method_summary`` and
    ``method_notes``.
    """
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
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
        body=steps_nav(step, rows, link_query)
        + '<div class="stack">\n'
        + card("Why", f"{why_html(row, metrics)}\n")
        + summary_html(metrics)
        + card("What changed", f"{what_changed_html(row, metrics, before_files)}\n")
        + card("Result", f"{result_html(row, metrics, rows)}\n")
        + notes_html(metrics)
        + card("Setup", f"{setup_html(row, metrics, rows)}\n")
        + card("Chain", f"{chain_html(step, row, rows, link_query)}\n")
        + "</div>\n",
        tail=f'<script id="data" type="application/json">{data}</script>\n',
    )


__all__ = [
    "RESULT_FIELDS",
    "RESULT_LABELS",
    "before_release_id",
    "build_release_page",
    "build_running_step_page",
    "failed_words",
    "floor_tasks_note",
    "method_notes",
    "method_summary",
    "missed_episode_text",
    "missed_episodes",
    "recorded_patch",
    "result_of",
    "served_step",
    "step_href",
    "unscored_failures",
]
