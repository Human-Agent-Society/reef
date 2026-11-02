"""ACE's playbook: the bullets, the merge that code runs on them, and their block in ``rules.md``.

No model call happens here. ``ACE.step`` gives this module what the Reflector and the Curator returned, and the
playbook changes by fixed rules: tags add to counts, new texts become bullets, and the refine step removes
bullets by their counts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from reef.harness.tree.checks import directive_shaped, secret_shaped

BULLET_ID = re.compile(r"ace-(\d{5,})")
#: The lines around the playbook's block in rules.md. Each step replaces the block and keeps the text outside it.
BLOCK_BEGIN = "<!-- ace playbook: begin -->"
BLOCK_END = "<!-- ace playbook: end -->"
BLOCK = re.compile(re.escape(BLOCK_BEGIN) + ".*?" + re.escape(BLOCK_END), re.DOTALL)
#: What the served agent reads above the bullets.
PLAYBOOK_INTRO = (
    "Lessons from earlier sessions: use the ones that fit your task. "
    "The counts say how often a review found a lesson helpful or harmful."
)


def normalized(text: str) -> str:
    """``text`` in lower case, with each run of whitespace as one space: the form in which two texts are compared."""
    return " ".join(text.casefold().split())


def one_line(text: str) -> str:
    """``text`` with each run of whitespace, line breaks included, as one space, and no space at either end."""
    return " ".join(text.split())


@dataclass(frozen=True)
class Bullet:
    """One lesson of the playbook.

    Values: ``id`` is ``ace-`` and a number of five or more digits. ``section`` and ``content`` are non-empty
    text equal to their ``one_line`` form. ``helpful`` and ``harmful`` ≥ 0 count the reflections that tagged the
    bullet so. ``step`` ≥ 1 is the step that added the bullet; only readers of ``state/playbook.json`` use it.
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
        if any(not text or text != one_line(text) for text in (self.section, self.content)):
            raise ValueError("a bullet's section and content must be non-empty text on one line")
        if self.helpful < 0 or self.harmful < 0 or self.step < 1:
            raise ValueError("a bullet's counts must be at least 0 and its step at least 1")

    @property
    def number(self) -> int:
        """The number in the bullet's id."""
        return int(self.id.removeprefix("ace-"))

    @property
    def score(self) -> int:
        """``helpful`` minus ``harmful``: the order in which the cap removes bullets, lowest first."""
        return self.helpful - self.harmful

    def line(self) -> str:
        """The bullet as one Markdown list item: its id, its counts, then its text."""
        return f"- [{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"


@dataclass(frozen=True)
class Addition:
    """One new bullet that the Curator asks for.

    Values: ``section`` is one of the configured sections. ``content`` is non-empty text equal to its
    ``one_line`` form.
    Purpose: ``roles.curate`` makes it from one ``ADD`` operation; ``Playbook.merged`` adds it or turns it down.
    """

    section: str
    content: str

    def __post_init__(self) -> None:
        if any(not text or text != one_line(text) for text in (self.section, self.content)):
            raise ValueError("an addition's section and content must be non-empty text on one line")


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

    def tagged(self, tags: Mapping[str, str]) -> Playbook:
        """The playbook with a bullet's helpful or harmful count one higher when ``tags`` maps its id to that tag.

        A ``neutral`` tag changes no count.
        """
        bullets = tuple(
            replace(
                bullet,
                helpful=bullet.helpful + int(tags.get(bullet.id) == "helpful"),
                harmful=bullet.harmful + int(tags.get(bullet.id) == "harmful"),
            )
            for bullet in self.bullets
        )
        return replace(self, bullets=bullets)

    def merged(self, additions: Sequence[Addition], step: int) -> tuple[Playbook, Counter[str]]:
        """The playbook with a new bullet for each addition that passes the screens, and the count of each outcome.

        Each addition, in order, has one outcome:

        ``refused``
            Its text has the shape of a credential or of an instruction override, or it holds ``BLOCK_BEGIN`` or
            ``BLOCK_END``, which would end the block early when the next step reads ``rules.md``.
        ``duplicates``
            A bullet already has its text, by ``normalized``.
        ``added``
            A new bullet at the end, with counts 0, ``step``, and the id that ``next_number`` gives.
        """
        playbook, outcomes = self, Counter[str]()
        for addition in additions:
            content = addition.content
            if secret_shaped(content) or directive_shaped(content) or BLOCK_BEGIN in content or BLOCK_END in content:
                outcomes["refused"] += 1
            elif normalized(content) in {normalized(bullet.content) for bullet in playbook.bullets}:
                outcomes["duplicates"] += 1
            else:
                bullet = Bullet(f"ace-{playbook.next_number:05d}", addition.section, content, 0, 0, step)
                playbook = Playbook((*playbook.bullets, bullet), playbook.next_number + 1)
                outcomes["added"] += 1
        return playbook, outcomes

    def pruned(self, margin: int) -> Playbook:
        """The playbook without the bullets whose harmful count exceeds their helpful count by ``margin`` or more."""
        return replace(self, bullets=tuple(bullet for bullet in self.bullets if -bullet.score < margin))

    def capped(self, size: int) -> Playbook:
        """The playbook with at most ``size`` bullets. The bullets with the lowest ``score`` go first, and of two
        bullets with equal scores, the older one goes first. The kept bullets keep their order."""
        excess = len(self.bullets) - size
        if excess <= 0:
            return self
        # The sort is stable, so of equal scores the bullet added first comes first.
        dropped = {bullet.id for bullet in sorted(self.bullets, key=lambda bullet: bullet.score)[:excess]}
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


def rules_with(rules: str, playbook_text: str) -> str:
    """``rules`` with a playbook block made from ``playbook_text``.

    The new block replaces the first marked block of ``rules``, or follows the end of ``rules`` when it has none.
    An empty ``playbook_text`` removes the block. The text outside the block stays, with one blank line between it
    and the block, and the result is ``""`` when no text remains. When the block does not change, ``rules`` comes
    back byte for byte, so the step leaves the tree unchanged.
    """
    found = BLOCK.search(rules)
    block = f"{BLOCK_BEGIN}\n## Playbook\n\n{PLAYBOOK_INTRO}\n\n{playbook_text}\n{BLOCK_END}" if playbook_text else ""
    if found is None:
        before, current, after = rules, "", ""
    else:
        before, current, after = rules[: found.start()], found.group(0), rules[found.end() :]
    if block == current:
        return rules
    parts = [part.strip("\n") for part in (before, block, after) if part.strip()]
    return "\n\n".join(parts) + "\n" if parts else ""


__all__ = ["Addition", "Bullet", "Playbook", "normalized", "one_line", "rules_with"]
