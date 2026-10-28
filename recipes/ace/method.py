"""ACE, Agentic Context Engineering (Zhang et al., arXiv 2510.04618), as a Reef harness method.

ACE keeps a playbook of short lessons, called bullets, in the context of the agent. The served agent is ACE's
Generator: it reads the playbook in ``rules.md``, which every session reads. In each step, each run (a trace with
a score or a reply) gets two model calls. The Reflector analyses the run and tags the bullets that applied as
helpful, harmful or neutral. The Curator turns the analysis into new bullets. Code, not a model, merges the
result: it counts the tags, adds the new bullets, and then removes the harmful bullets and the bullets past the
cap.

``state/playbook.json`` holds the playbook. Each step renders it into one marked block of ``rules.md`` and keeps
the rest of that file unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.prompts import untrusted_text
from reef.core.trajectories import atif_messages
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import TreePatch
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.train.harness_backend import Method, StepContext, Trace

PROMPTS_DIR = Path(__file__).with_name("prompts")
REFLECTOR_PROMPT = (PROMPTS_DIR / "reflector.md").read_text(encoding="utf-8")
CURATOR_PROMPT = (PROMPTS_DIR / "curator.md").read_text(encoding="utf-8")
PLAYBOOK_FILE = "playbook.json"
RULES_FILE = "rules.md"
#: The lines around the method's block in rules.md. Each step replaces the block and keeps the text outside it.
BLOCK_BEGIN = "<!-- ace playbook: begin -->"
BLOCK_END = "<!-- ace playbook: end -->"
BLOCK = re.compile(re.escape(BLOCK_BEGIN) + ".*?" + re.escape(BLOCK_END), re.DOTALL)
#: What the served agent reads above the bullets.
PLAYBOOK_INTRO = (
    "Lessons from earlier sessions: use the ones that fit your task. "
    "The counts say how often a review found a lesson helpful or harmful."
)
BULLET_ID = re.compile(r"ace-(\d{5,})")
#: The tags that a reflection gives a bullet.
TAGS = ("helpful", "harmful", "neutral")
#: The default sections of the playbook.
SECTIONS = ("Strategies and insights", "Code and commands", "Common mistakes to avoid", "Others")
#: The characters of one trace that the Reflector reads; a longer trace keeps its start and its end.
TRACE_CHARS = 30_000


def normalized(text: str) -> str:
    """``text`` in lower case, with each run of whitespace as one space: the form in which two texts are compared."""
    return " ".join(text.casefold().split())


@dataclass(frozen=True)
class ACESettings(PluginSettings):
    """The config of ``ACE``.

    Values: ``model`` is a non-empty name that ``ACE.step`` looks up in ``Tools.models``: ``served`` or an
    ``evolution.models`` entry. ``sections`` holds one or more names; each name is non-empty text on one line, with single spaces, and no
    two names are equal by ``normalized``. ``max_bullets`` ≥ 1 and ``prune_margin`` ≥ 1.
    ``max_traces_per_step`` ≥ 0, where 0 sets no limit.
    Purpose: ``ACE.from_config`` parses ``evolution.method.config`` into it; ``ACE.step`` reads it.
    """

    model: str = config_option("served", help="The model binding of the Reflector and the Curator.")
    sections: tuple[str, ...] = config_option(SECTIONS, help="The playbook's sections; each new bullet goes in one.")
    max_bullets: int = config_option(100, help="The most bullets that the playbook keeps.")
    prune_margin: int = config_option(
        2, help="Remove a bullet when its harmful count exceeds its helpful count by this many or more."
    )
    max_traces_per_step: int = config_option(
        0, help="The most runs (traces with a score or a reply) that one step reads. 0 sets no limit."
    )

    def __post_init__(self) -> None:
        sections = tuple(" ".join(section.split()) for section in self.sections)
        if not self.model.strip():
            raise ValueError("model must name a model binding: served or an evolution.models entry")
        if not sections or not all(sections) or len({normalized(section) for section in sections}) < len(sections):
            raise ValueError("sections must be one or more names, each non-empty and different from the others")
        if self.max_bullets < 1 or self.prune_margin < 1:
            raise ValueError("max_bullets and prune_margin must be at least 1")
        if self.max_traces_per_step < 0:
            raise ValueError("max_traces_per_step must be at least 0 (0 sets no limit)")
        # Each name becomes one heading line of rules.md.
        object.__setattr__(self, "sections", sections)


@dataclass(frozen=True)
class Bullet:
    """One lesson of the playbook.

    Values: ``id`` is ``ace-`` and a number of five or more digits. ``section`` and ``content`` are non-empty
    text on one line, with single spaces between words. ``helpful`` and ``harmful`` ≥ 0 count the reflections
    that tagged the bullet so. ``step`` ≥ 1 is the step that added the bullet.
    Purpose: a ``Playbook`` holds the bullets; ``rules.md`` and the prompts show each bullet as its ``line``.
    """

    id: str
    section: str
    content: str
    helpful: int
    harmful: int
    step: int

    def __post_init__(self) -> None:
        if BULLET_ID.fullmatch(self.id) is None:
            raise ValueError(f"bullet id {self.id!r} must be 'ace-' and a number of five or more digits")
        if any(not text or text != " ".join(text.split()) for text in (self.section, self.content)):
            raise ValueError("a bullet's section and content must be non-empty text on one line")
        if self.helpful < 0 or self.harmful < 0 or self.step < 1:
            raise ValueError("a bullet's counts must be at least 0 and its step at least 1")

    @property
    def number(self) -> int:
        """The number in the bullet's id."""
        return int(self.id.removeprefix("ace-"))

    def line(self) -> str:
        """The bullet as one Markdown list item: its id, its counts, then its text."""
        return f"- [{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"


@dataclass(frozen=True)
class Playbook:
    """The playbook: its bullets in the order they were added, and the number for the next bullet's id.

    Values: no two ``bullets`` have the same id, and no two have the same text by ``normalized``.
    ``next_number`` is above the number of every bullet. It only grows, so the id of a removed bullet is never
    given again.
    Purpose: ``ACE.step`` reads it from ``state/playbook.json``, changes it with the operations below, writes it
    back, and renders it into ``rules.md``. Each operation returns a new playbook.
    """

    bullets: tuple[Bullet, ...] = ()
    next_number: int = 1

    def __post_init__(self) -> None:
        if len({bullet.id for bullet in self.bullets}) < len(self.bullets):
            raise ValueError("the bullets of a playbook must have different ids")
        if len({normalized(bullet.content) for bullet in self.bullets}) < len(self.bullets):
            raise ValueError("the bullets of a playbook must have different texts")
        if any(bullet.number >= self.next_number for bullet in self.bullets):
            raise ValueError("next_number must be above the number of every bullet")

    @classmethod
    def from_json(cls, text: str) -> Playbook:
        """The playbook that ``to_json`` wrote as ``text``."""
        data = json.loads(text)
        return cls(tuple(Bullet(**row) for row in data["bullets"]), data["next_number"])

    def to_json(self) -> str:
        """The playbook as indented JSON: ``next_number``, and the ``bullets`` with all their fields."""
        rows = [asdict(bullet) for bullet in self.bullets]
        return json.dumps({"next_number": self.next_number, "bullets": rows}, indent=2, ensure_ascii=False) + "\n"

    def holds(self, content: str) -> bool:
        """Whether a bullet has the text ``content``, compared by ``normalized``."""
        return normalized(content) in {normalized(bullet.content) for bullet in self.bullets}

    def added(self, section: str, content: str, step: int) -> Playbook:
        """The playbook with one more bullet at the end: ``content`` on one line, in ``section``, with counts 0,
        added in ``step``, and the id that ``next_number`` gives. ``ValueError`` when the playbook ``holds`` the
        text."""
        bullet = Bullet(f"ace-{self.next_number:05d}", section, " ".join(content.split()), 0, 0, step)
        return Playbook((*self.bullets, bullet), self.next_number + 1)

    def tagged(self, tags: Mapping[str, str]) -> Playbook:
        """The playbook with a bullet's helpful or harmful count one higher when ``tags`` maps its id to that tag."""
        bullets = tuple(
            replace(
                bullet,
                helpful=bullet.helpful + int(tags.get(bullet.id) == "helpful"),
                harmful=bullet.harmful + int(tags.get(bullet.id) == "harmful"),
            )
            for bullet in self.bullets
        )
        return replace(self, bullets=bullets)

    def pruned(self, margin: int) -> Playbook:
        """The playbook without the bullets whose harmful count exceeds their helpful count by ``margin`` or more."""
        return replace(
            self, bullets=tuple(bullet for bullet in self.bullets if bullet.harmful - bullet.helpful < margin)
        )

    def capped(self, size: int) -> Playbook:
        """The playbook with at most ``size`` bullets. The bullets with the lowest helpful minus harmful count go
        first, and of two bullets with equal counts, the older one goes first. The kept bullets keep their order."""
        excess = len(self.bullets) - size
        if excess <= 0:
            return self
        # The sort is stable, so of equal counts the bullet added first comes first.
        ranked = sorted(self.bullets, key=lambda bullet: bullet.helpful - bullet.harmful)
        dropped = {bullet.id for bullet in ranked[:excess]}
        return replace(self, bullets=tuple(bullet for bullet in self.bullets if bullet.id not in dropped))

    def text(self, sections: Sequence[str]) -> str:
        """The bullets as Markdown, ``""`` when there are none.

        Each section that has a bullet gets a ``###`` heading with its bullets below it, in the order they were
        added. The sections of ``sections`` come first, in their order, then each other section that a bullet
        names.
        """
        groups = []
        for section in dict.fromkeys((*sections, *(bullet.section for bullet in self.bullets))):
            lines = [bullet.line() for bullet in self.bullets if bullet.section == section]
            if lines:
                groups.append("\n".join((f"### {section}", *lines)))
        return "\n\n".join(groups)


class ACE(Method):
    """ACE's Reflector, Curator and merge as a harness method; the served agent is ACE's Generator.

    Values: ``settings`` is the parsed config.
    Purpose: a deployment names the class in ``evolution.method``, and the backend calls ``step`` once per step.
    A step reads the runs of its material, the traces with a score or a reply, in order and at most
    ``max_traces_per_step`` of them. For each run, the Reflector's tags add to the counts of the bullets, and the
    Curator's ``ADD`` operations add bullets. Then the step removes the harmful bullets and the bullets past the
    cap, writes ``state/playbook.json``, and renders the playbook into ``rules.md``. The step's notes count what
    happened:

    ``traces``, ``runs``, ``read``
        The traces of the step's material, the runs among them, and the runs that the Reflector read.
    ``skipped``
        Each read run whose Reflector or Curator reply was not the asked JSON, as ``{trace, reason}``.
    ``helpful_tags``, ``harmful_tags``
        The counts that the reflections added.
    ``added``, ``duplicates``, ``refused``, ``ignored``
        The Curator's operations: new bullets; texts that a bullet has; texts with the shape of a credential or
        of an instruction override, or with a marker line of the playbook block; and operations that are not an
        ``ADD`` of text to a configured section.
    ``pruned``, ``capped``, ``bullets``
        The bullets removed for their harmful count, the bullets removed by the cap, and the bullets kept.
    """

    def __init__(self, settings: ACESettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(ACESettings.parse(config, plugin_class=cls))

    def step(self, context: StepContext) -> None:
        workspace, tools = context.workspace, context.tools
        settings = self.settings
        model = tools.models[settings.model]
        path = workspace.state / PLAYBOOK_FILE
        playbook = Playbook.from_json(path.read_text(encoding="utf-8")) if path.is_file() else Playbook()
        traces = workspace.traces()
        # A model call that ends on a tool call is the middle of a run: it has no reply, and the call that ends with
        # the reply carries the whole run.
        runs = [trace for trace in traces if trace.score is not None or trace.reply]
        read = runs[: settings.max_traces_per_step or len(runs)]
        counts: Counter[str] = Counter()
        skipped: list[dict[str, str]] = []
        for trace in read:
            reflection = self.reflect(model, playbook, trace)
            if reflection is None:
                skipped.append({"trace": trace.record, "reason": "the Reflector's reply is not the asked JSON"})
                continue
            tags = bullet_tags(reflection.get("bullet_tags", []), playbook)
            playbook = playbook.tagged(tags)
            counts.update(f"{tag}_tags" for tag in tags.values())
            operations = self.curate(model, playbook, reflection, trace)
            if operations is None:
                skipped.append({"trace": trace.record, "reason": "the Curator's reply is not the asked JSON"})
                continue
            playbook, outcomes = merged(playbook, operations, settings.sections, workspace.step)
            counts.update(outcomes)
        pruned = playbook.pruned(settings.prune_margin)
        kept = pruned.capped(settings.max_bullets)
        notes: dict[str, int | list[dict[str, str]]] = {
            "traces": len(traces),
            "runs": len(runs),
            "read": len(read),
            "skipped": skipped,
            "helpful_tags": counts["helpful_tags"],
            "harmful_tags": counts["harmful_tags"],
            "added": counts["added"],
            "duplicates": counts["duplicates"],
            "refused": counts["refused"],
            "ignored": counts["ignored"],
            "pruned": len(playbook.bullets) - len(pruned.bullets),
            "capped": len(pruned.bullets) - len(kept.bullets),
            "bullets": len(kept.bullets),
        }
        tree = workspace.tree()
        rules = rules_with(tree.files.get(RULES_FILE, ""), kept.text(settings.sections))
        candidate = tree.apply(TreePatch({RULES_FILE: rules or None}))
        validation = tools.check(candidate)
        if not validation.passed:
            # The playbook stays as the last step left it, so state/ and rules.md still agree.
            workspace.write_output(
                f"kept the last playbook: the new rules fail validation: {validation.message()}", notes
            )
            return
        path.write_text(kept.to_json(), encoding="utf-8")
        workspace.write_tree(candidate)
        workspace.write_output(
            f"traces: {len(read)} of {len(traces)} read, {len(skipped)} skipped; tags: {counts['helpful_tags']} "
            f"helpful, {counts['harmful_tags']} harmful; bullets: {counts['added']} added, "
            f"{len(playbook.bullets) - len(kept.bullets)} removed, {len(kept.bullets)} kept",
            notes,
        )

    def reflect(self, model: ModelBinding, playbook: Playbook, trace: Trace) -> dict[str, Any] | None:
        """The Reflector's analysis of ``trace``: the JSON object of its reply that has a ``key_insight``, and
        whose ``bullet_tags`` is a list when present; ``None`` when the reply has no such object."""
        data = (
            f"Playbook:\n{playbook.text(self.settings.sections) or '(empty)'}\n\n"
            f"Trace:\n{untrusted_text(trace_text(trace), 'recorded trace')}"
        )
        reply = model.chat([{"role": "system", "content": REFLECTOR_PROMPT}, {"role": "user", "content": data}])
        reflection = reply_object(reply, "key_insight")
        if reflection is None or not isinstance(reflection.get("bullet_tags", []), list):
            return None
        return reflection

    def curate(
        self, model: ModelBinding, playbook: Playbook, reflection: Mapping[str, Any], trace: Trace
    ) -> list[object] | None:
        """The Curator's operations for ``reflection``: the ``operations`` list of the JSON object in its reply, or
        ``None`` when the reply has no such object."""
        settings = self.settings
        data = (
            f"Sections: {json.dumps(list(settings.sections), ensure_ascii=False)}\n"
            f"The playbook holds {len(playbook.bullets)} bullets and keeps at most {settings.max_bullets}.\n\n"
            f"Playbook:\n{playbook.text(settings.sections) or '(empty)'}\n\n"
            f"Reflection:\n{untrusted_text(json.dumps(reflection, ensure_ascii=False, indent=2), 'reflection')}\n\n"
            f"Task:\n{untrusted_text(trace.task or '(none)', 'recorded task')}"
        )
        reply = model.chat([{"role": "system", "content": CURATOR_PROMPT}, {"role": "user", "content": data}])
        curation = reply_object(reply, "operations")
        return curation["operations"] if curation is not None and isinstance(curation["operations"], list) else None


def bullet_tags(entries: Sequence[object], playbook: Playbook) -> dict[str, str]:
    """The tag of each bullet that the Reflector's ``entries`` judged.

    An entry counts when it is an object whose ``id`` names a bullet of ``playbook``, with or without the
    brackets of ``rules.md``, and whose ``tag`` is one of ``TAGS``. Of two entries for one bullet, the later one
    counts.
    """
    ids = {bullet.id for bullet in playbook.bullets}
    tags: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        bullet_id, tag = str(entry.get("id")).strip("[] "), entry.get("tag")
        if bullet_id in ids and tag in TAGS:
            tags[bullet_id] = tag
    return tags


def merged(
    playbook: Playbook, operations: Sequence[object], sections: Sequence[str], step: int
) -> tuple[Playbook, Counter[str]]:
    """``playbook`` with a new bullet for each operation that adds one, and the count of each outcome.

    An operation adds a bullet (``added``) when it is an object whose ``type`` is ``ADD``, whose ``section`` is
    one of ``sections`` by ``normalized``, and whose ``content`` is non-empty text, unless the text has the
    shape of a credential or of an instruction override, or holds ``BLOCK_BEGIN`` or ``BLOCK_END`` once its
    whitespace runs are single spaces (``refused``), or a bullet has the text (``duplicates``). Every other
    operation is ``ignored``.
    """
    by_name = {normalized(section): section for section in sections}
    outcomes: Counter[str] = Counter()
    for operation in operations:
        if not isinstance(operation, Mapping):
            outcomes["ignored"] += 1
            continue
        section = by_name.get(normalized(str(operation.get("section"))))
        content = operation.get("content")
        if operation.get("type") != "ADD" or section is None or not isinstance(content, str) or not content.strip():
            outcomes["ignored"] += 1
        elif (
            secret_shaped(content)
            or directive_shaped(content)
            # A bullet line that holds a marker would end the block early when the next step reads rules.md.
            or any(marker in " ".join(content.split()) for marker in (BLOCK_BEGIN, BLOCK_END))
        ):
            outcomes["refused"] += 1
        elif playbook.holds(content):
            outcomes["duplicates"] += 1
        else:
            playbook = playbook.added(section, content, step)
            outcomes["added"] += 1
    return playbook, outcomes


def reply_object(reply: str, key: str) -> dict[str, Any] | None:
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
    conversation without the system prompt: one line per message, and one line per tool call. A rollout, whose
    trace has no trajectory, shows its task and its reply instead. A text longer than ``TRACE_CHARS`` keeps its
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


def rules_with(rules: str, playbook_text: str) -> str:
    """``rules`` with a playbook block made from ``playbook_text``.

    ``rules`` comes back unchanged when its first marked block is already that block, or when it has no block
    and ``playbook_text`` is empty. Otherwise the new block replaces the first marked block of ``rules``, or
    follows the end of ``rules`` when it has none, and an empty ``playbook_text`` removes the block. The text
    outside the block stays, with one blank line between it and the block. The result is ``""`` when no text
    remains.
    """
    found = BLOCK.search(rules)
    block = f"{BLOCK_BEGIN}\n## Playbook\n\n{PLAYBOOK_INTRO}\n\n{playbook_text}\n{BLOCK_END}" if playbook_text else ""
    if block == ("" if found is None else found.group(0)):
        return rules
    before, after = (rules, "") if found is None else (rules[: found.start()], rules[found.end() :])
    parts = [part.strip("\n") for part in (before, block, after) if part.strip()]
    return "\n\n".join(parts) + "\n" if parts else ""


__all__ = ["ACE", "ACESettings", "Bullet", "Playbook"]
