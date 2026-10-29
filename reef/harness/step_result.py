"""What a settled harness step means for the person who asked, as the harness pages say it.

The request and version pages read a catalog row's metrics through these
functions, so a rejected step names the same cause on either page. The
request page also offers the next action for a release, from
``next_action``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def result_of(row: Mapping[str, Any]) -> str:
    """The result of one ``GET /reef/harness/releases`` row, one of:

    - ``rejected``: ``metrics.selected`` is false.
    - ``inactive``: ``metrics.selected`` is true, the row made an artifact
      (``checkpoint``) and that artifact was never active (``activated`` is
      false). A person must activate it.
    - ``selected``: ``metrics.selected`` is true otherwise.
    - ``skipped``: ``metrics.selected`` is absent and ``metrics.skipped`` is set.
    - the row's ``operation`` (``creation``, ``training`` or ``recovery``) for
      any other row, ``unknown`` when the row names none.

    The harness pages and the tutorial driver read this function."""
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    selected = metrics.get("selected")
    if selected is False:
        return "rejected"
    if selected is True:
        return "inactive" if row.get("checkpoint") and not row.get("activated") else "selected"
    if metrics.get("skipped"):
        return "skipped"
    return str(row.get("operation") or "unknown")


def missed_rollouts(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The candidate rollouts a step's evaluation names that failed or scored below its floor, from the row's
    ``candidate_rollouts`` summary (``task``, ``score``, ``failure``, ``reply``); none when the step recorded none."""
    rollouts = metrics.get("candidate_rollouts")
    if not isinstance(rollouts, list):
        return []
    selection = metrics.get("selection")
    decided = selection.get("metrics") if isinstance(selection, Mapping) else None
    floor = decided.get("floor_score") if isinstance(decided, Mapping) else None
    missed = []
    for rollout in rollouts:
        if not isinstance(rollout, Mapping):
            continue
        score = rollout.get("score")
        below = isinstance(floor, (int, float)) and isinstance(score, (int, float)) and score < floor
        if rollout.get("failure") or score is None or below:
            missed.append(rollout)
    return missed


def unscored_failures(metrics: Mapping[str, Any]) -> list[str]:
    """Why the evaluation could not run: the distinct failures, when every candidate rollout failed before it was
    scored (a launch that found no binary, say), so the step judged nothing; none when any rollout was scored or
    the step recorded no rollouts."""
    rollouts = metrics.get("candidate_rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        return []
    causes: list[str] = []
    for rollout in rollouts:
        if not isinstance(rollout, Mapping) or rollout.get("score") is not None or not rollout.get("failure"):
            return []
        cause = str(rollout["failure"])
        if cause not in causes:
            causes.append(cause)
    return causes


def floor_tasks_note(metrics: Mapping[str, Any]) -> str | None:
    """For a step that answered a request under a floor: its floor tasks were set before the request and check that
    the changed harness still passes them, not what the request asks for, which only the review reads; ``None``
    for any other step."""
    selection = metrics.get("selection")
    if not isinstance(metrics.get("training_request"), Mapping) or not isinstance(selection, Mapping):
        return None
    if selection.get("policy") != "floor":
        return None
    rollouts = metrics.get("candidate_rollouts")
    tasks: list[str] = []
    for rollout in rollouts if isinstance(rollouts, list) else ():
        task = str(rollout.get("task") or "").strip() if isinstance(rollout, Mapping) else ""
        # A Harbor task directory shows as its name, a prompt as its start.
        name = task.rstrip("/").rsplit("/", 1)[-1] if task.startswith("/") else task
        name = name if len(name) <= 60 else f"{name[:57]}..."
        if name and name not in tasks:
            tasks.append(name)
    named = f" ({'; '.join(tasks)})" if tasks else ""
    return (
        f"The floor tasks{named} were set before this request: they check that the changed harness still passes "
        "them, not what the request asks for, which only the review reads."
    )


def missed_rollout_text(rollout: Mapping[str, Any]) -> str:
    """One missed rollout in words: the task, then why it failed, or its score and the reply that was graded."""
    task = str(rollout.get("task") or "").strip()
    failure = rollout.get("failure")
    if failure:
        return f"the task '{task}' failed: {failure}"
    reply = rollout.get("reply")
    if rollout.get("transcript_read") is False:
        graded = "no transcript was read from the rollout's session log, so no reply was graded"
    elif reply:
        graded = f"the reply graded was '{str(reply).strip()}'"
    else:
        graded = "no reply was graded"
    return f"the task '{task}' scored {rollout.get('score')}; {graded}"


#: The How to use heading that ends a design: a markdown heading or a line of its own, or ``How to use:`` with the
#: usage after it on the same line, bold or not, with an ASCII or a full width colon (a design in Chinese writes
#: ``How to use\uff1a``); in the middle of a line only right after a sentence ends and with its colon, so a sentence
#: that merely says how to use something is no heading.
USAGE_HEADING = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*|__)?how to use(?:\*\*|__)?(?:\s*[:\uff1a](?:\*\*|__)?[ \t]*|\s*$)"
    r"|(?<=[.!?\u3002])[ \t]+(?:\*\*|__)?how to use(?:\*\*|__)?\s*[:\uff1a](?:\*\*|__)?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def design_sections(notes: Mapping[str, Any]) -> tuple[str, str]:
    """The design a step recorded as ``proposal_notes.design`` and its How to use section, each stripped, the
    usage starting with a capital; empty where the record has none. The proposer writes the usage under a
    ``How to use`` heading at the end, which the pages read."""
    design = notes.get("design")
    if not isinstance(design, str) or not design.strip():
        return "", ""
    match = USAGE_HEADING.search(design)
    if match is None:
        return design.strip(), ""
    usage = design[match.end() :].strip()
    return design[: match.start()].strip(), usage[:1].upper() + usage[1:]


def rejection_text(metrics: Mapping[str, Any]) -> str:
    """Why a rejected step published nothing and what the person can do, as one sentence without its subject: the
    evaluation's cause when no candidate rollout was scored, else the selection's reason and the first missed
    rollout. A rollout that failed was the harness's run, not the request, so no rephrasing is advised then."""
    unscored = unscored_failures(metrics)
    if unscored:
        return (
            f"could not be evaluated: {'; '.join(unscored)}. Nothing judged the change and nothing was published; "
            "fix that and ask again."
        )
    selection = metrics.get("selection")
    reason = (selection.get("reason") if isinstance(selection, Mapping) else None) or "no reason recorded"
    missed = missed_rollouts(metrics)
    if not missed:
        return f"did not pass the checks ({reason}). Nothing changed; rephrase or split the request."
    if any(rollout.get("failure") for rollout in missed):
        advice = "the rollout failed, so the change itself was not judged"
    elif any(rollout.get("transcript_read") is False for rollout in missed):
        advice = "no transcript was read, so the change itself was not judged"
    else:
        advice = "rephrase or split the request"
    return f"did not pass the checks ({reason}): {missed_rollout_text(missed[0])}. Nothing changed; {advice}."


@dataclass(frozen=True)
class NextAction:
    """What a settled step offers the person who asked.

    Values: ``heading`` names the action. ``commands`` are calls to this Reef
    service, in order, each runnable as it stands with the scenario's header.
    ``place`` says what the calls do.
    Purpose: ``next_action`` makes it; the request page renders it.
    """

    heading: str
    commands: tuple[str, ...]
    place: str


def next_action(scenario: str, release_id: str, selection_result: str, requires: Sequence[str]) -> NextAction | None:
    """The next action of a settled step: activate a release that is not active, or pull an active one.

    ``requires`` names what the release needs where the harness runs (see
    ``reef.core.requirements.required_by``). ``None`` for a rejected,
    skipped or failed step: nothing waits on the person then.
    """
    if selection_result == "inactive":
        return NextAction(
            "Read this page, then activate it",
            (f'POST /reef/scenarios/{scenario}/activate {{"artifact_id": "{release_id}"}}',),
            "The scenario's activation rule did not make this release active; activating it serves it.",
        )
    if selection_result != "selected":
        return None
    needs = f" It requires {', '.join(requires)} where the harness runs." if requires else ""
    return NextAction(
        "Pull it",
        ("GET /reef/harness",),
        f"Reef serves this release's files now; a client gets them from there.{needs}",
    )


__all__ = [
    "USAGE_HEADING",
    "NextAction",
    "design_sections",
    "floor_tasks_note",
    "missed_rollout_text",
    "missed_rollouts",
    "next_action",
    "rejection_text",
    "result_of",
    "unscored_failures",
]
