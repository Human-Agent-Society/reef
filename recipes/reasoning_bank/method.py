"""ReasoningBank: a memory of strategies and lessons that an agent distills from its own runs.

Based on Ouyang et al., "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory" (arXiv 2509.25140).
Each step reads the runs among the step's traces. A run's outcome is its score against ``success_score``, or the
label of a judge model when the run has no score. An extractor model then writes memory items from the run:
strategies from a success, lessons from a failure. The bank of items is the method's state, ``state/memory.json``.
Each item is served as the Agent Skill ``skills/rb-<slug>/SKILL.md``, so the harness's own skill list does the
retrieval. The README describes the mapping and where it differs from the paper.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any, Self

import yaml

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.prompts import untrusted_text
from reef.core.trajectories import atif_messages
from reef.harness.tree import HarnessTree
from reef.harness.tree.checks import directive_shaped, secret_shaped
from reef.train.harness_backend import Method, StepContext, Trace

PROMPTS = Path(__file__).parent / "prompts"
#: The judge's prompt; ``$trace`` is the run.
JUDGE_PROMPT = Template((PROMPTS / "judge.md").read_text(encoding="utf-8"))
#: The extractor's prompt: ``$focus`` says what to look for in a run of its outcome, ``$max_items`` is the most
#: items it may write, ``$titles`` lists the bank's titles and ``$trace`` is the run.
EXTRACT_PROMPT = Template((PROMPTS / "extract.md").read_text(encoding="utf-8"))
#: The outcomes of a run, and what the extractor looks for in a run of each (``prompts/<outcome>.md``).
OUTCOMES = ("success", "failure")
FOCUS = {outcome: (PROMPTS / f"{outcome}.md").read_text(encoding="utf-8").strip() for outcome in OUTCOMES}
#: The bank's file in ``state/``.
MEMORY_FILE = "memory.json"
#: The tree directory of the method's skills; the method leaves every other file of the tree as it is.
SKILL_PREFIX = "rb-"
SKILL_ROOT = f"skills/{SKILL_PREFIX}"
#: The Agent Skills format allows a skill name of at most 64 characters and a description of at most 1024.
MAX_SLUG_CHARS = 64 - len(SKILL_PREFIX)
MAX_DESCRIPTION_CHARS = 1024
#: A slug is lowercase ASCII letters and digits in groups joined by single hyphens, as a skill name must be.
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: The most characters of one run that a prompt holds; a longer run loses the middle of its transcript.
MAX_TRACE_CHARS = 30_000


@dataclass(frozen=True)
class MemoryItem:
    """One item of the bank: a strategy from a successful run, or a lesson from a failed one.

    Values: ``slug`` matches ``SLUG`` and has at most ``MAX_SLUG_CHARS`` characters. ``title`` and
    ``description`` are non-empty single lines without leading, trailing or repeated whitespace; the description
    has at most ``MAX_DESCRIPTION_CHARS`` characters. ``content`` is non-empty text without leading or trailing
    whitespace. Every text encodes as UTF-8. ``outcome`` is one of ``OUTCOMES``, the outcome of the run the item
    came from, and ``step`` ≥ 1 is the step that wrote it.
    Purpose: ``state/memory.json`` holds the bank's items, oldest first. Each step adds the items the extractor
    writes and writes each item into the tree as the skill at ``path``, with the text ``skill()``.
    """

    slug: str
    title: str
    description: str
    content: str
    outcome: str
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.slug, str) or not SLUG.fullmatch(self.slug) or len(self.slug) > MAX_SLUG_CHARS:
            raise ValueError(f"an item slug must match {SLUG.pattern} in at most {MAX_SLUG_CHARS} characters")
        for name, line in (("title", self.title), ("description", self.description)):
            if not isinstance(line, str) or not line or line != " ".join(line.split()):
                raise ValueError(f"an item {name} must be one non-empty line")
        if len(self.description) > MAX_DESCRIPTION_CHARS:
            raise ValueError(f"an item description has at most {MAX_DESCRIPTION_CHARS} characters")
        if not isinstance(self.content, str) or not self.content or self.content != self.content.strip():
            raise ValueError("an item content must be non-empty text without leading or trailing whitespace")
        # A JSON escape can make a lone surrogate, which has no UTF-8 form: UnicodeEncodeError is a ValueError.
        "".join((self.title, self.description, self.content)).encode("utf-8")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"an item outcome must be one of {', '.join(OUTCOMES)}")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("an item step must be a positive integer")

    @property
    def path(self) -> str:
        """The tree path of the item's skill: ``skills/rb-<slug>/SKILL.md``."""
        return f"{SKILL_ROOT}{self.slug}/SKILL.md"

    def skill(self) -> str:
        """The item's ``SKILL.md``: ``name`` and ``description`` frontmatter, then the title as a heading and the
        content."""
        # The width keeps the longest description on one line.
        header = yaml.safe_dump(
            {"name": SKILL_PREFIX + self.slug, "description": self.description},
            sort_keys=False,
            allow_unicode=True,
            width=4096,
        )
        return f"---\n{header}---\n\n# {self.title}\n\n{self.content}\n"


@dataclass(frozen=True)
class ReasoningBankSettings(PluginSettings):
    """The config of ``ReasoningBank``.

    Values: ``judge_model`` and ``extract_model`` are non-empty names of model bindings: ``served`` or an
    ``evolution.models`` entry. ``success_score`` is a finite number. ``max_items_per_trace`` ≥ 1.
    ``max_items`` ≥ 0 and ``max_traces_per_step`` ≥ 0, where 0 sets no limit.
    Purpose: ``ReasoningBank.from_config`` parses it from ``evolution.method.config``; each step reads it.
    """

    judge_model: str = config_option("served", help="The model that labels a run without a score.")
    extract_model: str = config_option("served", help="The model that writes the memory items of a run.")
    success_score: float = config_option(
        1.0, help="A run scored at least this is a success, and a run with a lower score is a failure."
    )
    max_items_per_trace: int = config_option(3, help="The most memory items that one run adds.")
    max_items: int = config_option(50, help="The most items the bank keeps; the oldest go first. 0 sets no limit.")
    max_traces_per_step: int = config_option(
        0, help="The most runs that one step learns from, in trace order. 0 sets no limit."
    )

    def __post_init__(self) -> None:
        if not self.judge_model.strip() or not self.extract_model.strip():
            raise ValueError("judge_model and extract_model must name a model: served or an evolution.models entry")
        if not math.isfinite(self.success_score):
            raise ValueError("success_score must be a finite number")
        if self.max_items_per_trace < 1:
            raise ValueError("max_items_per_trace must be at least 1")
        if self.max_items < 0 or self.max_traces_per_step < 0:
            raise ValueError("max_items and max_traces_per_step must be at least 0 (0 sets no limit)")


class ReasoningBank(Method):
    """ReasoningBank as a harness evolution method: each step adds the memory items of its runs to the bank.

    Values: ``settings`` is the method's config.
    Purpose: a deployment names ``recipes.reasoning_bank:ReasoningBank`` in ``evolution.method``; the backend
    calls ``step`` once per step.
    """

    def __init__(self, settings: ReasoningBankSettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(ReasoningBankSettings.parse(config, plugin_class=cls))

    def step(self, context: StepContext) -> None:
        """Label each run among the traces, add its memory items to the bank, and write the bank as the tree's
        ``rb-*`` skills.

        A reply that is not the asked JSON skips its run, and an item that carries a credential or an
        instruction override stays out of the bank; the notes say why. The bank and the tree change together:
        when the new tree fails ``services.check``, neither changes, so no later step starts from a bank that
        cannot render.
        """
        workspace, services = context.workspace, context.services
        settings = self.settings
        judge, extractor = services.models[settings.judge_model], services.models[settings.extract_model]
        bank_path = workspace.state / MEMORY_FILE
        bank = (
            [MemoryItem(**row) for row in json.loads(bank_path.read_text(encoding="utf-8"))]
            if bank_path.is_file()
            else []
        )
        known = {item.slug for item in bank}
        traces = workspace.traces()
        # A run is a trace with a task and a score or a final reply; a model call that ends on a tool call is the
        # middle of a run, so it has no reply, and the call that ends with the reply carries the whole run.
        runs = [trace for trace in traces if trace.task is not None and (trace.score is not None or trace.reply)]
        if settings.max_traces_per_step:
            runs = runs[: settings.max_traces_per_step]
        outcomes = dict.fromkeys(OUTCOMES, 0)
        learned: list[str] = []
        skipped: list[dict[str, str]] = []
        for trace in runs:
            text = trace_text(trace)
            if trace.score is not None:
                outcome = "success" if trace.score >= settings.success_score else "failure"
            else:
                reply = judge.chat([{"role": "user", "content": JUDGE_PROMPT.substitute(trace=text)}])
                try:
                    outcome = judged_outcome(reply)
                except ValueError as exc:
                    skipped.append(
                        {"record": trace.record, "reason": f"the judge's reply is not the asked JSON: {exc}"}
                    )
                    continue
            outcomes[outcome] += 1
            titles = "\n".join(f"- {item.title}" for item in bank) or "(the memory is empty)"
            prompt = EXTRACT_PROMPT.substitute(
                focus=FOCUS[outcome],
                max_items=settings.max_items_per_trace,
                titles=untrusted_text(titles, "memory titles"),
                trace=text,
            )
            reply = extractor.chat([{"role": "user", "content": prompt}])
            try:
                items = memory_items(reply, outcome=outcome, step=workspace.step, limit=settings.max_items_per_trace)
            except ValueError as exc:
                skipped.append(
                    {"record": trace.record, "reason": f"the extractor's reply is not the asked JSON: {exc}"}
                )
                continue
            for item in items:
                # The tree is served to every session: a credential would leak, and an instruction override would
                # speak to every later run.
                written = "\n".join((item.title, item.description, item.content))
                if secret_shaped(written) or directive_shaped(written):
                    refusal = "an item carries a credential or an instruction override"
                    skipped.append({"record": trace.record, "reason": refusal})
                    continue
                bank = [kept for kept in bank if kept.slug != item.slug] + [item]
                learned.append(item.slug)
        overflow = max(0, len(bank) - settings.max_items) if settings.max_items else 0
        dropped, bank = bank[:overflow], bank[overflow:]
        new = list(dict.fromkeys(learned))
        notes: dict[str, Any] = {
            "traces": len(traces),
            "runs": len(runs),
            "judged": sum(1 for trace in runs if trace.score is None),
            "outcomes": outcomes,
            "added": [slug for slug in new if slug not in known],
            "replaced": [slug for slug in new if slug in known],
            "dropped": [item.slug for item in dropped],
            "skipped": skipped,
            "items": len(bank),
        }
        others = {path: text for path, text in workspace.tree().files.items() if not path.startswith(SKILL_ROOT)}
        candidate = HarnessTree({**others, **{item.path: item.skill() for item in bank}})
        validation = services.check(candidate)
        if not validation.passed:
            summary = f"the memory skills fail validation, so the bank stays as it was: {validation.message()}"
            workspace.write_output(summary, {**notes, "problems": list(validation.problems)})
            return
        workspace.write_tree(candidate)
        rows = [asdict(item) for item in bank]
        bank_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = (
            f"learned from {len(runs)} run(s) ({outcomes['success']} success, {outcomes['failure']} failure); "
            f"{len(notes['added'])} item(s) added, {len(notes['replaced'])} replaced and {len(dropped)} dropped; "
            f"the bank holds {len(bank)}"
        )
        workspace.write_output(summary, notes)


def trace_text(trace: Trace) -> str:
    """What the judge and the extractor read of a run, fenced as data.

    It is one JSON object: the task, the transcript without the system prompt (a run's trace has none), the
    final reply, the feedback and the failure, each left out when empty. Past ``MAX_TRACE_CHARS`` characters, the
    text loses its middle.
    """
    trajectory = trace.trajectory()
    transcript = (
        [message for message in atif_messages(trajectory) if message["role"] != "system"]
        if trajectory is not None
        else []
    )
    view = {
        "task": trace.task,
        "transcript": transcript,
        "final_reply": trace.reply,
        "feedback": trace.feedback,
        "failure": trace.failure,
    }
    text = json.dumps({key: value for key, value in view.items() if value}, ensure_ascii=False)
    if len(text) > MAX_TRACE_CHARS:
        half = MAX_TRACE_CHARS // 2
        text = f"{text[:half]} [... {len(text) - 2 * half} characters cut ...] {text[-half:]}"
    return untrusted_text(text, "trace")


def judged_outcome(reply: str) -> str:
    """The outcome a judge reply ``{"reason": ..., "outcome": "success" | "failure"}`` names, in either case;
    ``ValueError`` for a reply of another form."""
    outcome = json_object(reply).get("outcome")
    if not isinstance(outcome, str) or outcome.strip().lower() not in OUTCOMES:
        raise ValueError(f"its outcome is not one of {', '.join(OUTCOMES)}")
    return outcome.strip().lower()


def memory_items(reply: str, *, outcome: str, step: int, limit: int) -> list[MemoryItem]:
    """The first ``limit`` items of an extractor reply ``{"items": [{"title", "description", "content"}, ...]}``,
    as items of ``outcome`` that ``step`` wrote; ``ValueError`` for a reply of another form or an empty field.

    A title and a description become single lines, and a description longer than ``MAX_DESCRIPTION_CHARS`` is
    cut. An item's slug comes from its title, so two titles that differ only in case or punctuation name one item.
    """
    rows = json_object(reply).get("items")
    if not isinstance(rows, list):
        raise ValueError("it has no items list")
    items = []
    for row in rows[:limit]:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) for key in ("title", "description", "content")
        ):
            raise ValueError("an item lacks a title, a description or a content text")
        title = " ".join(row["title"].split())
        description = " ".join(row["description"].split())[:MAX_DESCRIPTION_CHARS].rstrip()
        items.append(MemoryItem(slug_of(title), title, description, row["content"].strip(), outcome, step))
    return items


def json_object(reply: str) -> dict[str, Any]:
    """The JSON object in a model reply: the text from the first ``{`` to the last ``}`` after the reasoning a
    thinking model ends with ``</think>``, so that reasoning, a code fence or a sentence around the object does not
    matter; ``ValueError`` when the reply holds no JSON object."""
    answer = reply.rpartition("</think>")[2]
    start, end = answer.find("{"), answer.rfind("}")
    try:
        value = json.loads(answer[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError as exc:
        raise ValueError(f"it is not JSON ({exc.msg})") from None
    if not isinstance(value, dict):
        raise ValueError("it holds no JSON object")
    return value


def slug_of(title: str) -> str:
    """The slug of an item title: its letters and digits as lowercase ASCII, in groups joined by hyphens and cut to
    ``MAX_SLUG_CHARS``; a title without any is named by the start of its SHA-256 digest."""
    letters = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    slug = "-".join(re.findall(r"[a-z0-9]+", letters))[:MAX_SLUG_CHARS].strip("-")
    return slug or hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]


__all__ = ["MemoryItem", "ReasoningBank", "ReasoningBankSettings"]
