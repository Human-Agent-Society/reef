"""ACE's two model roles: the Reflector and the Curator, one call each per run.

Each function builds the role's prompt, calls the model once, and checks the reply. A reply without the asked JSON
gives ``None``. Text from the step's material enters a prompt only inside ``untrusted_text``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.core.prompts import untrusted_text
from reef.core.trajectories import atif_messages
from reef.harness.tasks.model_binding import ModelBinding
from reef.train.harness_backend import Trace

from .playbook import Addition, Playbook, normalized, one_line

PROMPTS_DIR = Path(__file__).with_name("prompts")
REFLECTOR_PROMPT = (PROMPTS_DIR / "reflector.md").read_text(encoding="utf-8")
CURATOR_PROMPT = (PROMPTS_DIR / "curator.md").read_text(encoding="utf-8")
#: The tags that a reflection gives a bullet.
TAGS = ("helpful", "harmful", "neutral")
#: The characters of one trace that the Reflector reads; a longer trace keeps its start and its end.
TRACE_CHARS = 30_000


@dataclass(frozen=True)
class Reflection:
    """The Reflector's analysis of one run.

    Values: ``analysis`` is the JSON object of the reply, with a ``key_insight`` key. ``tags`` maps the id of each
    bullet that the Reflector judged to one of ``TAGS``.
    Purpose: ``reflect`` makes it; ``ACE.step`` adds ``tags`` to the bullet counts, and ``curate`` shows
    ``analysis`` to the Curator.
    """

    analysis: Mapping[str, object]
    tags: Mapping[str, str]

    def __post_init__(self) -> None:
        if "key_insight" not in self.analysis:
            raise ValueError("a reflection's analysis must have a key_insight")
        if any(tag not in TAGS for tag in self.tags.values()):
            raise ValueError(f"a reflection's tags must be among {', '.join(TAGS)}")


@dataclass(frozen=True)
class Curation:
    """The Curator's answer to one reflection.

    Values: ``additions`` are the ``ADD`` operations of the reply that name a configured section and have text, in
    their order. ``ignored`` ≥ 0 counts the other operations of the reply.
    Purpose: ``curate`` makes it; ``ACE.step`` merges ``additions`` into the playbook and counts ``ignored``.
    """

    additions: tuple[Addition, ...]
    ignored: int

    def __post_init__(self) -> None:
        if self.ignored < 0:
            raise ValueError("Curation.ignored must be at least 0")


def reflect(model: ModelBinding, playbook: Playbook, sections: Sequence[str], trace: Trace) -> Reflection | None:
    """The Reflector's analysis of ``trace`` against ``playbook``; ``None`` when the reply has no JSON object with a
    ``key_insight`` and a ``bullet_tags`` list (or none).

    The tags keep each ``bullet_tags`` entry whose ``id`` names a bullet of ``playbook``, with or without the
    brackets of ``rules.md``, and whose ``tag`` is one of ``TAGS``. Of two entries for one bullet, the later one
    counts. Other entries have no effect.
    """
    prompt = (
        f"Playbook:\n{playbook.text(sections) or '(empty)'}\n\n"
        f"Trace:\n{untrusted_text(trace_text(trace), 'recorded trace')}"
    )
    reply = model.chat([{"role": "system", "content": REFLECTOR_PROMPT}, {"role": "user", "content": prompt}])
    analysis = reply_object(reply, "key_insight")
    entries = analysis.get("bullet_tags", []) if analysis is not None else None
    if analysis is None or not isinstance(entries, list):
        return None
    ids = {bullet.id for bullet in playbook.bullets}
    tags: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        bullet_id, tag = str(entry.get("id")).strip("[] "), entry.get("tag")
        if bullet_id in ids and isinstance(tag, str) and tag in TAGS:
            tags[bullet_id] = tag
    return Reflection(analysis, tags)


def curate(
    model: ModelBinding,
    playbook: Playbook,
    sections: Sequence[str],
    max_bullets: int,
    reflection: Reflection,
    trace: Trace,
) -> Curation | None:
    """The Curator's new bullets for ``reflection``; ``None`` when the reply has no JSON object with an
    ``operations`` list.

    An operation becomes an ``Addition`` when it is an object whose ``type`` is ``ADD``, whose ``section`` is one of
    ``sections`` by ``normalized``, and whose ``content`` is text with a non-space character. The addition takes
    the configured spelling of the section and the ``one_line`` form of the text. Each other operation is ignored.
    """
    analysis = json.dumps(reflection.analysis, ensure_ascii=False, indent=2)
    prompt = (
        f"Sections: {json.dumps(list(sections), ensure_ascii=False)}\n"
        f"The playbook holds {len(playbook.bullets)} bullets and keeps at most {max_bullets}.\n\n"
        f"Playbook:\n{playbook.text(sections) or '(empty)'}\n\n"
        f"Reflection:\n{untrusted_text(analysis, 'reflection')}\n\n"
        f"Task:\n{untrusted_text(trace.task or '(none)', 'recorded task')}"
    )
    reply = model.chat([{"role": "system", "content": CURATOR_PROMPT}, {"role": "user", "content": prompt}])
    curation = reply_object(reply, "operations")
    operations = curation["operations"] if curation is not None else None
    if not isinstance(operations, list):
        return None
    by_name = {normalized(section): section for section in sections}
    additions = []
    for operation in operations:
        if not isinstance(operation, Mapping) or operation.get("type") != "ADD":
            continue
        section = by_name.get(normalized(str(operation.get("section"))))
        content = operation.get("content")
        if section is not None and isinstance(content, str) and content.strip():
            additions.append(Addition(section, one_line(content)))
    return Curation(tuple(additions), len(operations) - len(additions))


def reply_object(reply: str, key: str) -> dict[str, object] | None:
    """The first JSON object in ``reply`` that has ``key``, with any text or code fence around it; ``None`` when
    the reply has no such object."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(reply):
        if char != "{":
            continue
        try:
            value = decoder.raw_decode(reply, start)[0]
        except ValueError:
            continue
        if isinstance(value, dict) and key in value:
            return value
    return None


def trace_text(trace: Trace) -> str:
    """What the Reflector reads of ``trace``.

    The text starts with the trace's score, feedback and failure, each when the trace has one. Then it has the
    conversation without the system prompt: one line per message, and one line per tool call. A trace of a run,
    which has no trajectory, shows its task and its reply instead. A text longer than ``TRACE_CHARS`` keeps its
    first and its last ``TRACE_CHARS // 2`` characters.
    """
    lines = [
        f"{name}: {value}"
        for name, value in (("score", trace.score), ("feedback", trace.feedback), ("failure", trace.failure))
        if value is not None
    ]
    trajectory = trace.trajectory()
    if trajectory is None:
        lines.extend((f"[user] {trace.task or ''}", f"[assistant] {trace.reply or ''}"))
    else:
        for message in atif_messages(trajectory):
            if message["role"] == "system":
                continue
            text = content_text(message["content"])
            if text:
                lines.append(f"[{message['role']}] {text}")
            lines.extend(
                f"[tool call] {call['function']['name']} {call['function']['arguments']}"
                for call in message.get("tool_calls", ())
            )
    text = "\n".join(lines)
    if len(text) <= TRACE_CHARS:
        return text
    half = TRACE_CHARS // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters cut ...]\n{text[-half:]}"


def content_text(content: object) -> str:
    """The text of a message's content: the text itself, or the text of each part, with ``[<type>]`` for a part
    without text, such as an image."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", f"[{part.get('type')}]")) for part in content if isinstance(part, Mapping)
        )
    return ""


__all__ = ["Curation", "Reflection", "curate", "reflect"]
