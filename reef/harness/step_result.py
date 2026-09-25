"""What a settled harness step means for the person who asked, as the wrapper and the harness pages say it.

The wrapper prints one result line and the next commands in a terminal; the
request and version pages render the same result and next action in HTML.
Both read a catalog row's metrics through these functions, so a rejected
step names the same cause and a release offers the same install commands in
either place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.harness.adapters import get_adapter
from reef.harness.episodes.version_check import ships_version_check


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
    the changed harness still passes them, not what the request asks for, which only the review reads; ``None``
    for any other step."""
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
        "them, not what the request asks for, which only the review reads."
    )


def missed_episode_text(episode: Mapping[str, Any]) -> str:
    """One missed episode in words: the task, then why it failed, or its score and the reply that was graded."""
    task = str(episode.get("task") or "").strip()
    failure = episode.get("failure")
    if failure:
        return f"the task '{task}' failed: {failure}"
    reply = episode.get("reply")
    graded = f"the reply graded was '{str(reply).strip()}'" if reply else "no reply was graded"
    return f"the task '{task}' scored {episode.get('score')}; {graded}"


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
    ``How to use`` heading at the end, which the pages and the wrapper's result lines read."""
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
    evaluation's cause when no candidate episode was scored, else the selection's reason and the first missed
    episode. An episode that failed was the harness's run, not the request, so no rephrasing is advised then."""
    unscored = unscored_failures(metrics)
    if unscored:
        return (
            f"could not be evaluated: {'; '.join(unscored)}. Nothing judged the change and nothing was published; "
            "fix that and ask again."
        )
    selection = metrics.get("selection")
    reason = (selection.get("reason") if isinstance(selection, Mapping) else None) or "no reason recorded"
    missed = missed_episodes(metrics)
    if not missed:
        return f"did not pass the checks ({reason}). Nothing changed; rephrase or split the request."
    advice = (
        "the episode failed, so the change itself was not judged"
        if any(episode.get("failure") for episode in missed)
        else "rephrase or split the request"
    )
    return f"did not pass the checks ({reason}): {missed_episode_text(missed[0])}. Nothing changed; {advice}."


def reef_installs(adapter: str) -> bool:
    """Whether Reef installs ``adapter``'s harness with a wrapper that sets up and updates it; terminus, a batch
    runner, has none: its tree is served by ``GET /reef/harness`` and a run's Harbor task holds what it needs."""
    return get_adapter(adapter).install is not None


@dataclass(frozen=True)
class NextAction:
    """What a settled step offers next.

    ``commands`` are what the page shows, each runnable as it stands, and
    ``place`` says where to run them under ``heading``. ``terminal`` are the
    wrapper commands that take the same step from a terminal, in order;
    empty where no wrapper runs (terminus). On an adapter with the update
    notice, ``commands`` is its in-session install and ``terminal`` the
    wrapper's alternative."""

    heading: str
    commands: tuple[str, ...]
    place: str
    terminal: tuple[str, ...]


def next_action(
    adapter: str, step: int, selection_result: str, record_id: str, requires: Sequence[str]
) -> NextAction | None:
    """The next action of a settled step, ``None`` for a rejected or skipped one. ``requires`` names the items the
    release still needs set up: the install refuses a release whose items are not set up, so setup comes first."""
    if selection_result not in ("pending", "selected"):
        return None
    if not reef_installs(adapter):
        # A batch runner with no wrapper: nothing to set up or update on this machine.
        if selection_result != "selected":
            return None
        return NextAction(
            "Run it",
            ("GET /reef/harness",),
            "It serves this release's tree now; a run gets it from there. What the release requires must hold in "
            "the Harbor task the run uses.",
            (),
        )
    install = (f"reef-{adapter} setup", f"reef-{adapter} update") if requires else (f"reef-{adapter} update",)
    if ships_version_check(adapter):
        return NextAction(
            "Read this page, then install" if selection_result == "pending" else "Install when ready",
            (f"/versions v{step} install",),
            f"Run this in your reef-{adapter} session. You can keep chatting until you are ready.",
            install,
        )
    if selection_result == "pending":
        serve = (f"reef-{adapter} wait {record_id}",)
        return NextAction(
            "Read this page, then serve it",
            serve,
            "Run this in a terminal: it asks whether to serve this release, then installs it.",
            serve,
        )
    if requires:
        return NextAction(
            "Set up, then install",
            install,
            f"Run these in a terminal, in this order: setup asks for what this release needs ({', '.join(requires)}), "
            f"update installs it; then start reef-{adapter} again to use the new version.",
            install,
        )
    return NextAction(
        "Install when ready",
        install,
        f"Run this in a terminal, then start reef-{adapter} again to use the new version.",
        install,
    )


__all__ = [
    "USAGE_HEADING",
    "NextAction",
    "design_sections",
    "floor_tasks_note",
    "missed_episode_text",
    "missed_episodes",
    "next_action",
    "reef_installs",
    "rejection_text",
    "unscored_failures",
]
