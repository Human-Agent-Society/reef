"""A user's training instruction and the session and release it came from."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.requirements import parse_requires

#: The commands a client reports as on its PATH or not, so a proposer builds for the machine the change runs on,
#: not the one it is tried in: players, notifiers, openers and clipboards per platform, and common tools.
CLIENT_COMMANDS = (
    "afplay",
    "say",
    "osascript",
    "open",
    "pbcopy",
    "terminal-notifier",
    "xdg-open",
    "notify-send",
    "paplay",
    "pw-play",
    "aplay",
    "wl-copy",
    "xclip",
    "powershell.exe",
    "wslview",
    "ffplay",
    "ffmpeg",
    "mpv",
    "mpg123",
    "sox",
    "espeak",
    "curl",
    "git",
    "gh",
    "python3",
    "node",
    "brew",
    "apt-get",
)
#: At most this many commands in one report.
MAX_CLIENT_COMMANDS = 64
_CLIENT_WORD = re.compile(r"[A-Za-z0-9._+\- ()]{1,64}")
_COMMAND_NAME = re.compile(r"[A-Za-z0-9._+\-]{1,40}")


def parse_client(value: Any) -> dict[str, Any]:
    """A client's report of its machine: ``platform``, ``arch`` and ``release`` as short words, and ``commands``
    mapping a command name to whether it is on the PATH, at most ``MAX_CLIENT_COMMANDS``. The report only
    informs a proposer, so what does not fit that shape is dropped rather than refusing the request; it is the
    client's word, data a proposer reads, never an instruction."""
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, Any] = {}
    for key in ("platform", "arch", "release"):
        word = value.get(key)
        if isinstance(word, str) and _CLIENT_WORD.fullmatch(word.strip()):
            parsed[key] = word.strip()
    commands = value.get("commands")
    if isinstance(commands, Mapping):
        kept = {
            name: present
            for name, present in commands.items()
            if isinstance(name, str) and _COMMAND_NAME.fullmatch(name) and isinstance(present, bool)
        }
        if kept:
            parsed["commands"] = dict(list(kept.items())[:MAX_CLIENT_COMMANDS])
    return parsed


@dataclass(frozen=True)
class TrainingRequest:
    """A training instruction, independent of inference batches and feedback.

    ``id`` is filled from the enclosing AgentRecord when it becomes a batch.
    Session and release identify the request's source; they do not select an inference batch.
    ``requires`` is what the change needs from the person's machine, at most
    ``MAX_REQUIRES`` ``{name, kind, check}`` items of the shape
    ``reef.core.requirements.parse_requires`` admits; default none.
    ``client`` is the requesting client's report of its machine (see
    :func:`parse_client`); empty when it sent none.
    """

    text: str
    session: str
    release_id: str
    id: str = ""
    # Out of the hash: the items are dicts, and the frozen contract is what the other fields carry.
    requires: tuple[Mapping[str, Any], ...] = field(default=(), hash=False)
    client: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if len(self.text) > 4000:
            raise ValueError("text must not exceed 4000 characters")
        if not isinstance(self.session, str) or not isinstance(self.release_id, str):
            raise ValueError("session and release_id must be strings")
        object.__setattr__(self, "requires", tuple(parse_requires(self.requires)))
        object.__setattr__(self, "client", parse_client(self.client))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingRequest:
        fields: dict[str, str] = {}
        for key in ("text", "session", "release_id"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            fields[key] = value
        requires = payload.get("requires")
        return cls(**fields, requires=() if requires is None else requires, client=parse_client(payload.get("client")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "session": self.session,
            "release_id": self.release_id,
            "requires": [dict(item) for item in self.requires],
            # Only when the client reported one, so a request without it keeps its earlier shape.
            **({"client": dict(self.client)} if self.client else {}),
        }


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
