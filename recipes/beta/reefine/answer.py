"""Reefine's text answer: the served model answers a request with files, or learns one skill from failures.

A request is answered design first: the served model restates it, names what
triggers the behavior, what state the harness must know and what only the
user can provide, then writes the files; a second call reviews the files
against the request. What only the user can provide is declared as a
``requires`` item with a prompt sentence, never asked for by the change at
run time. A review that finds the answer short sends the request back, up to
``REQUEST_ATTEMPTS`` answers in all. Without an instruction the step learns
one skill from its failing traces. The prompts, their sections and the
words they take for each kind of harness are in ``prompts/``; the functions
here fill them in and build the short parts that depend on the request and
the adapter, such as the list of files a request may write. The backend owns
validation, evaluation, publication and review.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recipes.beta.reefine.harness_facts import harness_facts
from recipes.beta.reefine.prompts import prompt, prompt_words
from reef.core.job_data import TrajectorySample
from reef.core.prompts import untrusted_text
from reef.core.requirements import parse_requires
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import recorded_payload
from reef.harness.adapters import get_adapter
from reef.harness.adapters.parts import Format, TreeLayout
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, TreePatch
from reef.train.harness_backend import ModelCalls, Services, Trace

#: The one adapter whose extension API Reefine knows; the others take rules, skills and commands from it.
EXTENSION_ADAPTER = "pi"

#: The pi extension API reference a request prompt and the agent read before they write an extension.
PI_EXTENSION_API = Path(__file__).with_name("pi_extension_api.md")

#: How much of each file the request prompt shows: enough to recognize it. The rules file and the config files
#: show whole up to the larger cap, since an answer that changes one writes it again in full.
PREVIEW_CHARS = 240
WHOLE_FILE_CHARS = 8000
#: How much of a design the step records: a few paragraphs with its How to use section.
DESIGN_CHARS = 4000
#: How much of each long text in a refused answer a retry shows.
EARLIER_TEXT_CHARS = 400
#: The two words a review result may be.
REVIEW_RESULTS = ("complete", "partial")
#: How many covered or uncovered points a review keeps: the record is a summary, not a transcript.
REVIEW_ITEMS = 20
#: How many answers a request may get: the first, then one more each time the review finds the last one short.
REQUEST_ATTEMPTS = 3

#: A shell variable name: what an env item's check (else its name) must be, and what an extension reads.
VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: The ``$VAR`` and ``${VAR}`` references a shell check makes.
SHELL_VARIABLE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
#: A ``process.env.X`` or ``process.env["X"]`` read in an extension's code.
ENV_READ = re.compile(r"""process\.env(?:\.([A-Za-z_][A-Za-z0-9_]*)|\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\])""")
#: Variables pi or the shell sets for every session: an extension reading one needs nothing from the user.
SESSION_ENV = frozenset(
    {"PI_OFFLINE", "PI_CODING_AGENT_DIR", "HOME", "PATH", "USER", "SHELL", "TMPDIR", "LANG", "TERM"}
)

#: The Claude Code permission rules a request's config file may add: tools that read the web and run no shell.
CLAUDE_PREAPPROVED = ("WebSearch", "WebFetch")
CLAUDE_FETCH_DOMAIN = re.compile(r"WebFetch\(domain:[A-Za-z0-9.-]+\)")

#: The source path of the primary config target, the one config file a request may write.
PRIMARY_CONFIG = "config/primary.json"

#: The prompt that answers a person's request, and its sections. Braces doubled where the JSON shapes need them.
REQUEST_PROMPT = prompt("request")
#: What the request prompt says about extensions on pi: when to write one and how it must behave.
EXTENSIONS_SECTION = prompt("extensions")
#: What the request prompt says about an adapter Reefine writes no extension for, from its facts.
HARNESS_SECTION = prompt("harness")
#: The same about an adapter that takes no extension from Reefine and has no facts.
NO_EXTENSIONS_SECTION = prompt("no_extensions")
#: The platforms a change serves on a harness that runs on the person's machine.
PLATFORMS_SENTENCE = prompt("platforms")
#: The section of the request prompt and the agent's prompt carrying the failures a step in training_mode manual
#: hands over beside the request.
FAILURES_SECTION = prompt("failing_requests")
#: The request prompt's section carrying the extension API reference.
API_SECTION = prompt("api_reference")

#: How the request and review prompts say an env item's value reaches the change: ``extension`` on pi, whose
#: extensions read process.env, and ``files`` on another harness, whose files find the value in its environment.
ENV_WORDS = prompt_words("env_words")

#: The first of the calls a request takes: the steps the request names and which of them need a tool.
PLAN_PROMPT = prompt("plan")
#: The request prompt's section when the plan found steps the harness cannot perform, on pi.
PLAN_TOOLS_SECTION = prompt("plan_tools")
#: The same section on an adapter that takes no extension: the harness's own tools are all there is.
PLAN_NO_TOOL_SECTION = prompt("plan_no_tool")

#: The prompt of the review call: the model reads its own files against the request and says what they cover.
REVIEW_PROMPT = prompt("review")
#: The review's check of new slash commands on pi, where a command is a prompt template or registered code.
PI_REVIEW_COMMANDS = prompt("review_commands_pi")
#: The same check on an adapter with facts: its own command surface and way of building a mode.
HARNESS_REVIEW_COMMANDS = prompt("review_commands_harness")
#: What the review prompt gets when its first reply carried no result object.
REVIEW_AGAIN = prompt("review_again")

#: The prompt section a request gets again after a review found the previous answer short.
RETRY_REVIEWED_SECTION = prompt("retry_reviewed")
#: What that section adds when the earlier answer only put a substitute in place of the behavior.
RETRY_UNDELIVERED = prompt("retry_undelivered")
#: The prompt section a request gets again after an answer that could not be used at all: JSON that does not parse,
#: files every one of which was dropped, files that change nothing, or files the harness's validation refuses.
RETRY_UNUSABLE_SECTION = prompt("retry_unusable")
#: What the unusable retry adds when the refused answer parsed: its design and files, so what it got right stays.
RETRY_EARLIER_ANSWER = prompt("retry_earlier_answer")

#: Why an answer is refused before its review on opencode: an agent it defines without a permission map is offered
#: every tool, so a mode built on it restricts nothing.
UNRESTRICTED_AGENT = prompt("unrestricted_agent")
#: What a request step records under ``declined`` when the reply is a design saying no file can deliver it.
DECLINED = "the design says no file this harness takes can deliver the request"

#: The prompt that learns one skill from failing traces, when no request drives the step.
FAILURES_PROMPT = prompt("failures")


def request_parts(adapter: str) -> tuple[str, ...]:
    """The parts a request may write on ``adapter``: the skills, rules and commands it takes, extensions on pi, and
    the primary config where the harness's own config can enforce a behavior (``HarnessFacts.config_keys``)."""
    facts = harness_facts(adapter)
    taken = tuple(part for part in ("skill", "rules", "command") if part in get_adapter(adapter).layout.parts)
    if adapter == EXTENSION_ADAPTER:
        extra: tuple[str, ...] = ("extension",)
    elif facts is not None and facts.config_keys:
        extra = ("config",)
    else:
        extra = ()
    return (*taken, *extra)


def request_config_keys(adapter: str) -> tuple[str, ...]:
    """The top level keys a request may change in ``adapter``'s primary config; none off the list."""
    facts = harness_facts(adapter)
    return () if facts is None else facts.config_keys


def file_name(path: str, fallback: str) -> str:
    return path.rsplit("/", 1)[-1] if path else fallback


def file_lines(adapter: str) -> str:
    """One prompt line per file a request may write on ``adapter``, with what it holds and where it lands."""
    descriptor = get_adapter(adapter)
    rules_part = descriptor.layout.parts.get("rules")
    rules = file_name(rules_part.renders or "" if rules_part else "", f"{adapter}'s rules file")
    primary = descriptor.config_targets.get("primary")
    lines = {
        "skill": (
            "- skills/<name>/SKILL.md: a skill; the text starts with YAML frontmatter "
            "(--- name: <name> / description: <one line> ---) followed by the skill's markdown\n"
        ),
        "rules": (
            f"- rules.md: the markdown every session reads ({rules}); write the whole file, keeping every rule "
            "that still holds\n"
        ),
        "command": (
            "- commands/<name>.md: the prompt template of the /<name> command\n"
            if adapter == EXTENSION_ADAPTER
            else "- commands/<name>.md: the command file, as the harness notes below describe\n"
        ),
        "extension": "- extensions/<name>.ts: a complete pi extension module\n",
        "config": (
            f"- {PRIMARY_CONFIG}: a JSON object merged into "
            f"{file_name(primary.path if primary else '', 'the primary config')}; write the whole file, and "
            f"change only the top level keys {', '.join(request_config_keys(adapter))}\n"
        ),
    }
    return "".join(lines[part] for part in request_parts(adapter))


def harness_section(adapter: str) -> str:
    """The request prompt's section on the harness's own surface, for an adapter other than pi."""
    facts = harness_facts(adapter)
    if facts is None:
        return NO_EXTENSIONS_SECTION.format(adapter=adapter)
    config = (
        f" {PRIMARY_CONFIG} may set {', '.join(facts.config_keys)}: {facts.config_example}."
        if facts.config_keys
        else ""
    )
    return HARNESS_SECTION.format(
        title=facts.title, command=facts.command, tools=facts.tools, mode=facts.mode, config=config
    )


def review_commands(adapter: str) -> str:
    """The review prompt's check of new commands for ``adapter``."""
    facts = harness_facts(adapter)
    if adapter == EXTENSION_ADAPTER or facts is None:
        return PI_REVIEW_COMMANDS
    return HARNESS_REVIEW_COMMANDS.format(title=facts.title, command=facts.command, mode=facts.mode)


@dataclass(frozen=True)
class Answer:
    """What a request step answers: the change, what it needs, and what the step records.

    Values: ``patch`` changes the head tree, or is the empty patch when the
    step changes nothing. ``requires`` are the requires items the change
    adds, each through the screens of ``screened_requires``; the empty patch
    adds none. ``notes`` are what the step records: the design, the review,
    what was refused or left undeclared. They hold ``failure`` (or
    ``declined``, for a design that says no file can deliver the request),
    why nothing changes, exactly when the patch is empty.
    Purpose: ``answer_request`` and the agent make it; ``ReefineMethod``
    writes ``patch`` into the step's tree and returns the rest.
    """

    patch: TreePatch
    requires: tuple[Mapping[str, Any], ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        empty = not self.patch.changes
        if empty != ("failure" in self.notes or "declined" in self.notes):
            raise ValueError("an Answer's notes say why under failure or declined exactly when its patch is empty")
        if empty and self.requires:
            raise ValueError("an Answer with the empty patch adds no requires item")


@dataclass(frozen=True)
class ParsedAnswer:
    """The usable part of one request reply.

    Values: ``design`` is the reply's design text, ``None`` when it wrote
    none. ``patch`` holds the reply's files that this harness takes, and
    ``dropped`` one line per file the parse refused, with the reason.
    Purpose: ``parse_answer`` makes it; ``answer_once`` validates and
    reviews ``patch`` and reports ``dropped``.
    """

    design: str | None
    patch: TreePatch
    dropped: tuple[str, ...] = ()


@dataclass(frozen=True)
class WrittenAnswer:
    """An answer whose files pass validation.

    Values: ``patch`` is the change, and the patched head tree passes
    validation; ``added`` are the requires items the reply added; ``notes``
    are what the step records (the design, the review, what was refused or
    undeclared).
    Purpose: ``answer_once`` makes one per usable answer; the loop keeps
    the one with the fewest uncovered points, and the step answers with it.
    """

    patch: TreePatch
    added: list[dict[str, Any]]
    notes: dict[str, Any]


@dataclass(frozen=True)
class UnusableAnswer:
    """An answer the loop may ask again.

    Values: ``reason`` is why it could not be used; ``design`` is
    the design it carried, if any; ``files`` shows, when its files parsed and
    something refused them, those files in short, for the retry to start
    from; ``dropped`` names why the parser dropped each file, for the retry
    alone (the step's failure keeps the reason).
    Purpose: ``answer_once`` makes it; the loop writes the next retry
    section from it.
    """

    reason: str
    design: str | None = None
    files: str | None = None
    dropped: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeclinedAnswer:
    """A design that says no file of the parts this harness takes can deliver the request.

    Values: ``notes`` hold the design, ``declined`` and the review that
    names what is out of reach.
    Purpose: ``declined_answer`` makes it; the step records the notes
    with the empty patch.
    """

    notes: dict[str, Any]


@dataclass(frozen=True)
class FailedCall:
    """A call that ended the loop: a failed request, or a reply with nothing to apply.

    Values: ``notes`` say why, under ``failure``, with the design when the
    reply held one.
    Purpose: ``answer_once`` makes it; the loop stops and the step
    records the notes with the empty patch.
    """

    notes: dict[str, Any]


@dataclass
class RequestAnswers:
    """What the answers to one request have left so far, and the retry section each one asks for next.

    ``kept`` is the delivering answer with the fewest uncovered points and
    ``kept_attempt`` the answer it was; ``undelivered`` the notes of the last
    answer whose review found it only put a substitute in the behavior's
    place; ``declined`` the notes of the last design that said no file can
    deliver the request; ``unusable`` the last answer that could not be used
    and ``failed`` the call that ended the loop. ``dropped_attempts`` says
    why each unusable answer was dropped, and ``reviewed`` holds the last
    review's findings, which stay in every later retry.
    Purpose: the request loop updates one per request and makes the
    answer from it when the loop ends."""

    kept: WrittenAnswer | None = None
    kept_attempt: int = 0
    undelivered: dict[str, Any] | None = None
    declined: dict[str, Any] | None = None
    unusable: UnusableAnswer | None = None
    failed: FailedCall | None = None
    dropped_attempts: list[str] = field(default_factory=list)
    reviewed: str = ""

    def take_unusable(self, answer: UnusableAnswer, attempt: int, calls: ModelCalls) -> str:
        """A slip in the answer's form: the retry that names it, asked while attempts remain, and a line on
        ``calls``'s activity."""
        self.unusable = answer
        why = answer.reason if not answer.dropped else f"{answer.reason}: {'; '.join(answer.dropped)}"
        self.dropped_attempts.append(f"answer {attempt}: {why}")
        if attempt < REQUEST_ATTEMPTS:
            # The request page shows it while the step runs, not only once the step settles.
            calls.note("check", f"answer {attempt} written again: {why}", failed=True)
        retry = f"{self.reviewed}{RETRY_UNUSABLE_SECTION.format(reason=why)}\n\n"
        if answer.files is not None:
            earlier = RETRY_EARLIER_ANSWER.format(design=answer.design or "(none written)", files=answer.files)
            retry += f"{earlier}\n\n"
        return retry

    def take_declined(self, answer: DeclinedAnswer, attempt: int) -> str:
        """A design with no file: asked again when its review found points a file could deliver and nothing is
        kept yet (the design gave up early); otherwise the loop ends."""
        self.declined = answer.notes
        review = answer.notes.get("review")
        if review is None or not review["uncovered"] or attempt >= REQUEST_ATTEMPTS or self.kept is not None:
            return ""
        findings = "\n".join(f"- {point}" for point in review["uncovered"])
        design = answer.notes.get("design", "(none written)")
        self.reviewed = f"{RETRY_REVIEWED_SECTION.format(design=design, findings=findings, delivered='')}\n\n"
        return self.reviewed

    def take_written(self, answer: WrittenAnswer, attempt: int) -> str:
        """An answer with files: kept when it delivers with fewer uncovered points than the one kept, and the retry
        its review asks for; empty once the review finds nothing another answer could close."""
        review = answer.notes.get("review")
        if review is not None and review.get("delivers") is False:
            reason = review["uncovered"][0] if review["uncovered"] else "the files only imitate the behavior"
            self.undelivered = {
                **answer.notes,
                "failure": f"the change does not deliver the request: {reason}",
                "attempts": attempt,
            }
        elif self.kept is None or uncovered_count(answer.notes) < uncovered_count(self.kept.notes):
            self.kept = answer
            self.kept_attempt = attempt
        if review is None or (review["result"] == "complete" and review.get("delivers") is not False):
            return ""
        if review.get("limits") and not review["uncovered"] and review.get("delivers") is not False:
            # What remains is what the harness notes say no answer here can deliver: another answer changes nothing.
            return ""
        findings = [*review["uncovered"], *answer.notes.get("dropped", ())]
        section = RETRY_REVIEWED_SECTION.format(
            design=answer.notes.get("design", "(none written)"),
            findings="\n".join(f"- {point}" for point in findings) or "- (the review named no point)",
            delivered="" if review.get("delivers") is not False else f" {RETRY_UNDELIVERED}",
        )
        self.reviewed = f"{section}\n\n"
        return self.reviewed

    def unanswered(self, attempt: int) -> dict[str, Any]:
        """The step's notes when no answer was kept, with the reasons earlier answers were dropped. After a failed
        call an earlier substitute says more about the request than the failure does; otherwise a design that
        declined, then a substitute, then the last unusable answer's reason with its design."""
        if self.failed is not None:
            notes = dict(self.undelivered or self.failed.notes)
        elif self.declined is not None:
            notes = dict(self.declined)
        elif self.undelivered is not None:
            notes = dict(self.undelivered)
        else:
            failure = "no answer was written" if self.unusable is None else self.unusable.reason
            design = None if self.unusable is None else self.unusable.design
            notes = {**({} if design is None else {"design": kept_design(design)}), "failure": failure}
        if attempt > 1 and "attempts" not in notes:
            notes["attempts"] = attempt
        if self.dropped_attempts:
            notes["dropped_attempts"] = list(self.dropped_attempts)
        return notes


def answer_request(
    tree: HarnessTree,
    request: TrainingRequest,
    traces: Sequence[Trace],
    services: Services,
    adapter: str = EXTENSION_ADAPTER,
) -> Answer:
    """The served model's answer to one request: the files of the kept answer, with the notes the step records.

    A review that finds the answer short (partial, or not delivering the
    behavior at all) sends the request back with what it found, up to
    ``REQUEST_ATTEMPTS`` answers in all; a complete review ends the loop at
    once. The answer kept is the delivering one with the fewest uncovered
    points, and ``attempts`` in the notes counts the answers written when
    there was more than one. When no answer delivers the behavior, the
    answer has the empty patch and its notes say why under ``failure``.
    ``services`` are the step's: its models answer, its check validates each
    answer's files, and its activity names each answer written again.
    """
    prompt_text = request_prompt(tree, request, traces, services.models, adapter)
    answers = RequestAnswers()
    retry = ""
    attempt = 0
    while attempt < REQUEST_ATTEMPTS:
        attempt += 1
        asked = prompt_text if not retry else prompt_text.rstrip("\n") + "\n\n" + retry
        answer = answer_once(asked, tree, request, services, adapter)
        if isinstance(answer, UnusableAnswer):
            retry = answers.take_unusable(answer, attempt, services.calls)
        elif isinstance(answer, DeclinedAnswer):
            retry = answers.take_declined(answer, attempt)
        elif isinstance(answer, FailedCall):
            # A failed call or an empty reply ends the loop.
            answers.failed = answer
            retry = ""
        else:
            retry = answers.take_written(answer, attempt)
        if not retry:
            break
    written = answers.kept
    if written is None:
        return Answer(TreePatch({}), notes=answers.unanswered(attempt))
    notes = written.notes
    if attempt > 1:
        notes["attempts"] = attempt
        if answers.kept_attempt != attempt:
            # An earlier answer covered more than the later ones: the notes say which one the step kept.
            notes["kept_attempt"] = answers.kept_attempt
    if answers.dropped_attempts:
        notes["dropped_attempts"] = answers.dropped_attempts
    return Answer(written.patch, requires=tuple(written.added), notes=notes)


def answer_once(
    asked: str, tree: HarnessTree, request: TrainingRequest, services: Services, adapter: str
) -> WrittenAnswer | UnusableAnswer | DeclinedAnswer | FailedCall:
    """One answer and its review: the patch, the requires items the reply added and the notes; an
    ``UnusableAnswer`` the loop asks again (JSON that does not parse, every file dropped, files that change
    nothing, files ``services.check`` refuses); a ``DeclinedAnswer`` design that says no file can deliver the
    request; or a ``FailedCall`` whose notes say why there is nothing to apply (a failed call, a provider
    refusal)."""
    models = services.models
    # An extension is longer than a skill, and a thinking model reasons for tens of thousands of tokens before
    # it writes one, answering with no text when the budget ends inside that reasoning; the request path pays
    # for the room and the minutes, the failure path and the review keep their shorter budgets.
    reply, failure = ask(models, asked, max_tokens=max_tokens(65536), timeout_s=timeout_s(600.0))
    if reply is None:
        return FailedCall({"failure": failure})
    slipped = slipped_json(reply)
    layout = get_adapter(adapter).layout
    parsed = (
        None if slipped else parse_answer(reply, tree, layout, request_parts(adapter), request_config_keys(adapter))
    )
    design = design_text(reply)
    if parsed is None or not parsed.patch.changes:
        refusal = provider_refusal(models)
        if refusal is not None:
            return nothing_to_apply(reply, f"the provider refused the reply ({refusal})")
        if slipped is not None:
            return UnusableAnswer(slipped, design)
        if parsed is not None and parsed.dropped:
            # Files the parser dropped, one and all: an answer to write again, never a design that declined.
            return UnusableAnswer("the reply holds no usable file", design, dropped=parsed.dropped)
        if parsed is None and reply.strip():
            return UnusableAnswer("the reply holds no JSON object with files", design)
        if design is not None and adapter != EXTENSION_ADAPTER and harness_facts(adapter) is not None:
            return declined_answer(models, request, design, adapter)
        return nothing_to_apply(reply, "the reply holds no usable file")
    patch = parsed.patch
    written = files_in_short(patch, layout)
    candidate = tree.apply(patch)
    if candidate == tree:
        # The step skips a tree it left unchanged, so such an answer is written again instead.
        return UnusableAnswer("the files change nothing in the tree", design, written)
    validation = services.check(candidate)
    if not validation.passed:
        # The validation the step meets next, run here so a refused file is written again instead of losing the step.
        return UnusableAnswer(f"the harness refused the files: {validation.message()}", design, written)
    unrestricted = unrestricted_agents(tree, patch)
    if unrestricted:
        return UnusableAnswer(
            "; ".join(UNRESTRICTED_AGENT.format(name=name) for name in unrestricted), design, written
        )
    widened = widened_permissions(patch) if adapter == "claude" else []
    if widened:
        return UnusableAnswer("; ".join(widened), design, written)
    added, refused = parse_answer_requires(reply)
    own = [dict(item) for item in request.requires]
    notes: dict[str, Any] = {}
    if design is not None:
        notes["design"] = kept_design(design)
    if parsed.dropped:
        notes["dropped"] = list(parsed.dropped)
    # The review reads the whole design; the step records it cut to the record's size.
    review, review_failure = review_answer(models, request.text, design, patch, [*own, *added], adapter=adapter)
    if review is not None:
        notes["review"] = review
    else:
        notes["review_failure"] = review_failure or "the review did not run"
    if refused:
        notes["refused_requires"] = refused
    undeclared = undeclared_env(patch, [*own, *added], layout)
    if undeclared:
        notes["undeclared_env"] = undeclared
    return WrittenAnswer(patch, added, notes)


def declined_answer(models: ModelBindings, request: TrainingRequest, design: str, adapter: str) -> DeclinedAnswer:
    """A design that writes no file, on a harness whose notes name what no answer there can deliver: an answer with
    no change, reviewed like one, so those limits reach the pages and a point a file could still deliver sends the
    request back. On pi such a reply stays an answer with nothing to apply."""
    notes: dict[str, Any] = {"design": kept_design(design), "declined": DECLINED}
    own = [dict(item) for item in request.requires]
    review, review_failure = review_answer(models, request.text, design, TreePatch({}), own, adapter=adapter)
    if review is not None:
        notes["review"] = review
    else:
        notes["review_failure"] = review_failure or "the review did not run"
    return DeclinedAnswer(notes)


def nothing_to_apply(reply: str, reason: str) -> FailedCall:
    """A request step's notes when the reply gave no file: the reason, and the design when the model wrote one."""
    notes: dict[str, Any] = {}
    design = design_text(reply)
    if design is not None:
        notes["design"] = kept_design(design)
    notes["failure"] = reason if reply.strip() else "the reply is empty"
    return FailedCall(notes)


def clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


def files_in_short(patch: TreePatch, layout: TreeLayout) -> str:
    """A patch for a retry prompt: one line per path, so the model can rewrite from it.

    A config file shows its JSON object with every long text cut, so each of its keys stays in view; any other
    file shows the start of its text, as a JSON string.
    """
    lines = []
    for path, text in patch.changes.items():
        location = layout.locate(path)
        data = json_object_of(text) if location is not None and location.part.format is Format.CONFIG else {}
        if text is None:
            shown = "(removed)"
        elif data:
            shown = json.dumps(short_texts(data), ensure_ascii=False)
        else:
            shown = json.dumps(clipped(text, EARLIER_TEXT_CHARS), ensure_ascii=False)
        lines.append(f"- {path}: {shown}")
    return "\n".join(lines)


def short_texts(value: Any) -> Any:
    """``value`` with every string longer than ``EARLIER_TEXT_CHARS`` cut, at any depth."""
    if isinstance(value, str):
        return clipped(value, EARLIER_TEXT_CHARS)
    if isinstance(value, Mapping):
        return {key: short_texts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [short_texts(item) for item in value]
    return value


def files_view(tree: HarnessTree, layout: TreeLayout) -> str:
    """The tree as the request prompt shows it: each path, then its text, whole for the rules and config files up to
    ``WHOLE_FILE_CHARS`` and the start of every other file."""
    if not tree.files:
        return "(the tree holds no file yet)"
    parts = []
    for path, text in tree.files.items():
        location = layout.locate(path)
        whole = location is not None and (location.part.name == "rules" or location.part.format is Format.CONFIG)
        parts.append(f"--- {path}\n{clipped(text, WHOLE_FILE_CHARS if whole else PREVIEW_CHARS)}")
    return "\n".join(parts)


def json_object_of(text: str | None) -> Mapping[str, Any]:
    """The JSON object a config file holds; empty for no file or a text that is not one."""
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def config_agents(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """The agents a config object defines under ``agent`` (opencode's key), by name."""
    agents = data.get("agent")
    return agents if isinstance(agents, Mapping) else {}


def unrestricted_agents(tree: HarnessTree, patch: TreePatch) -> list[str]:
    """The agents the patch's primary config defines with no permission map, unless the tree's own primary config
    already gives that agent one: opencode offers such an agent every tool. Only a request's answer is held to this;
    a tree of the person's own may define an agent without a map."""
    if PRIMARY_CONFIG not in patch.changes:
        return []
    mapped = {
        name
        for name, agent in config_agents(json_object_of(tree.files.get(PRIMARY_CONFIG))).items()
        if isinstance(agent, Mapping) and isinstance(agent.get("permission"), Mapping) and agent["permission"]
    }
    names = []
    for name, agent in config_agents(json_object_of(patch.changes[PRIMARY_CONFIG])).items():
        permission = agent.get("permission") if isinstance(agent, Mapping) else None
        if not (isinstance(permission, Mapping) and permission) and name not in mapped:
            names.append(str(name))
    return names


def widened_permissions(patch: TreePatch) -> list[str]:
    """Why the patch's Claude Code permissions reach past what a request may grant: every session of the release
    runs under them, so a request may only add an allow rule for a tool in ``CLAUDE_PREAPPROVED`` (or a
    ``WebFetch(domain:<host>)`` rule), never a mode, a directory, a deny or ask edit, a shell or a wildcard."""
    data = json_object_of(patch.changes.get(PRIMARY_CONFIG))
    if "permissions" not in data:
        return []
    permissions = data["permissions"]
    if not isinstance(permissions, Mapping):
        return ["permissions must be an object holding an allow list"]
    reasons = [
        f"a request may not set permissions.{key}: it changes what every session does unasked"
        for key in sorted(set(permissions) - {"allow"})
    ]
    allow = permissions.get("allow", [])
    rules = allow if isinstance(allow, list) else [allow]
    reasons.extend(
        f"permissions.allow may name only WebSearch, WebFetch or WebFetch(domain:<host>), not {rule!r}"
        for rule in rules
        if not (isinstance(rule, str) and (rule in CLAUDE_PREAPPROVED or CLAUDE_FETCH_DOMAIN.fullmatch(rule)))
    )
    return reasons


def uncovered_count(notes: Mapping[str, Any]) -> int:
    """How many points an answer's review left uncovered; an answer without a review counts none."""
    review = notes.get("review")
    return 0 if review is None else len(review["uncovered"])


def client_text(request: TrainingRequest) -> str:
    """The machine the change will run on, as the request's client reported it, for a request prompt; without
    a report the change must serve every platform the harness runs on."""
    client = request.client
    if not client:
        return (
            "The user's machine is unknown (their client reported none): support macOS, Linux and Windows under "
            "WSL 2 alike.\n\n"
        )
    platform = " ".join(str(client[key]) for key in ("platform", "release", "arch") if client.get(key))
    reported = client.get("commands")
    commands: Mapping[str, Any] = reported if isinstance(reported, Mapping) else {}
    lines = [f"platform: {platform or 'not reported'}"]
    present = sorted(str(name) for name, found in commands.items() if found)
    absent = sorted(str(name) for name, found in commands.items() if not found)
    if present:
        lines.append("on its PATH: " + ", ".join(present))
    if absent:
        lines.append("not on its PATH: " + ", ".join(absent))
    return (
        "The machine the change will run on, as the user's client reported it (data):\n"
        f"{untrusted_text(chr(10).join(lines), 'client report')}\n"
        "Build for this machine; anything the change needs that it lacks is a requires item.\n\n"
    )


def failures_section(traces: Sequence[Trace]) -> str:
    """The prompt section that carries the step's traces, fenced as data since they are client text, with a blank
    line after it; empty for a step without traces, such as one in manual mode."""
    if not traces:
        return ""
    return f"{FAILURES_SECTION.format(text=untrusted_text(failures_text(traces)))}\n\n"


def request_prompt(
    tree: HarnessTree,
    request: TrainingRequest,
    traces: Sequence[Trace],
    models: ModelBindings,
    adapter: str = EXTENSION_ADAPTER,
) -> str:
    """The request prompt: the request fenced as data, the failures beside it when the step has any traces, every
    file of the tree, the steps the plan call found need a tool, and the extension API reference on pi."""
    request_text = untrusted_text(request.text, "user request")
    view = files_view(tree, get_adapter(adapter).layout)
    extensions = "extension" in request_parts(adapter)
    facts = harness_facts(adapter)
    # The plan call first: the steps the harness cannot perform get a tool written beside their rule on pi, and
    # off pi the harness's own tools, which the plan call is told about, before the step is called undone.
    steps_found = tool_steps(models, request_text, view, None if facts is None else facts.tools)
    steps = "\n".join(f"- {step}" for step in steps_found)
    words = ENV_WORDS["extension" if extensions else "files"]
    if not steps_found:
        plan = ""
    elif extensions:
        plan = f"{PLAN_TOOLS_SECTION.format(steps=steps)}\n\n"
    else:
        plan = f"{PLAN_NO_TOOL_SECTION.format(steps=steps)}\n\n"
    return REQUEST_PROMPT.format(
        request=request_text,
        env_value=words["env_value"],
        env_reader=words["env_reader"],
        means=words["means"],
        asker=words["asker"],
        # A harness that runs away from the person's machine says where; the client's report is not that place.
        machine="" if facts is not None and facts.machine else client_text(request),
        paths=file_lines(adapter),
        extensions=EXTENSIONS_SECTION if extensions else harness_section(adapter),
        platforms=(
            facts.machine
            if facts is not None and facts.machine
            else PLATFORMS_SENTENCE.format(platform="branch on process.platform, " if extensions else "")
        ),
        failures=failures_section(traces),
        files=view,
        plan=plan,
        api=f"{API_SECTION.format(text=PI_EXTENSION_API.read_text(encoding='utf-8'))}\n\n" if extensions else "",
    )


def parse_answer(
    reply: str, tree: HarnessTree, layout: TreeLayout, parts: Sequence[str], config_keys: Sequence[str]
) -> ParsedAnswer | None:
    """The design and the files of a request reply; ``None`` when the reply holds no object with ``files``.

    A file is kept when its path is a path of ``layout``, its part is one of
    ``parts``, and its text is a string or ``None`` (a removal). The primary
    config file is kept when it is the only config file the reply writes and
    every top level key it changes is in ``config_keys``. Every other file is
    dropped with a reason.
    """
    value = json_in(reply, openers=("{",))
    if not isinstance(value, Mapping) or not isinstance(value.get("files"), Mapping):
        return None
    kept: dict[str, str | None] = {}
    dropped: list[str] = []
    taken = ", ".join(layout.parts[part].path for part in parts if part in layout.parts)
    for path, text in value["files"].items():
        location = layout.locate(str(path))
        if location is None or location.part.name not in parts:
            dropped.append(f"{path} was dropped: a request may write {taken}")
        elif text is not None and (not isinstance(text, str) or not text.strip()):
            dropped.append(f"{path} was dropped: its text is empty or not a string")
        elif location.part.format is Format.CONFIG and (path != PRIMARY_CONFIG or text is None):
            dropped.append(f"{path} was dropped: a request may change {PRIMARY_CONFIG} only, and never remove it")
        elif location.part.format is Format.CONFIG:
            changed = changed_keys(json_object_of(tree.files.get(PRIMARY_CONFIG)), json_object_of(text))
            outside = sorted(changed - set(config_keys))
            if outside or not json_object_of(text):
                allowed = ", ".join(config_keys) or "no key"
                dropped.append(
                    f"{path} was dropped: it changes {', '.join(outside) or 'nothing'}; a request may change {allowed}"
                )
            else:
                kept[str(path)] = text
        else:
            kept[str(path)] = text
    design = value.get("design")
    return ParsedAnswer(
        design.strip() if isinstance(design, str) and design.strip() else None, TreePatch(kept), tuple(dropped)
    )


def changed_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> set[str]:
    """The top level keys whose value differs between two config objects, added and removed ones included."""
    return {key for key in set(before) | set(after) if before.get(key) != after.get(key)}


def review_answer(
    models: ModelBindings,
    request_text: str,
    design: str | None,
    patch: TreePatch,
    requires: Sequence[Mapping[str, Any]],
    *,
    adapter: str = EXTENSION_ADAPTER,
) -> tuple[dict[str, Any] | None, str | None]:
    """The served model's reading of its files against the request, ``{result, covered, uncovered}``, and no
    reason; or ``None`` and the reason the step has no review, which the step records so the page says the one
    check of whether the files deliver the request did not run. New commands are judged by ``adapter``'s own
    command surface."""
    written: dict[str, Any] = {"files": patch.to_json()}
    if requires:
        written["requires"] = [dict(item) for item in requires]
    notes_harness = adapter != EXTENSION_ADAPTER and harness_facts(adapter) is not None
    words = ENV_WORDS["extension" if "extension" in request_parts(adapter) else "files"]
    review_prompt = REVIEW_PROMPT.format(
        request=untrusted_text(request_text, "user request"),
        design="(none written)" if design is None else design,
        files=json.dumps(written, indent=2, ensure_ascii=False),
        env_check=words["env_check"],
        commands=review_commands(adapter),
        # The harness notes send a point no answer can deliver to limits: the object has the key for it.
        limits_key=', "limits": ["<one point per item>"]' if notes_harness else "",
    )
    # A reasoning model spends the budget on its reasoning first; a reply the reasoning ate is asked once more
    # with room for both.
    reply, reason = ask(models, review_prompt, max_tokens=max_tokens(16384), timeout_s=timeout_s(120.0))
    if reply is None and reason is not None and "non-text content" in reason:
        reply, reason = ask(models, review_prompt, max_tokens=max_tokens(16384) * 2, timeout_s=timeout_s(240.0))
    if reply is None:
        return None, reason
    review = parse_review(reply)
    if review is None and json_in(reply, openers=("{",)) is None and '"result"' in reply:
        # The model wrote the object but a stray quote broke its JSON: asked once more before the step records that
        # no review ran.
        reply, reason = ask(
            models, f"{review_prompt}\n\n{REVIEW_AGAIN}", max_tokens=max_tokens(16384), timeout_s=timeout_s(120.0)
        )
        if reply is None:
            return None, reason
        review = parse_review(reply)
    if review is None:
        return None, "the review reply carried no result object"
    return review, None


def parse_review(reply: str) -> dict[str, Any] | None:
    """The review object in the model's text; ``None`` when there is none or its result is not one of the two words."""
    value = json_in(reply, openers=("{",))
    if not isinstance(value, dict):
        return None
    review_result = str(value.get("result", "")).strip().lower()
    if review_result not in REVIEW_RESULTS:
        return None
    review: dict[str, Any] = {
        "result": review_result,
        "covered": strings_of(value.get("covered")),
        "uncovered": strings_of(value.get("uncovered")),
    }
    # What the harness notes declare out of reach, kept apart from the gaps a better answer could close.
    limits = strings_of(value.get("limits"))
    if limits:
        review["limits"] = limits
    # Only an explicit boolean decides delivery; a review that says nothing about it keeps the change.
    if isinstance(value.get("delivers"), bool):
        review["delivers"] = value["delivers"]
    return review


def strings_of(value: Any) -> list[str]:
    """The non-empty strings of a JSON list, at most ``REVIEW_ITEMS`` of them; none when ``value`` is not a list."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:REVIEW_ITEMS]


def design_text(reply: str) -> str | None:
    """The whole text of the reply's ``design``; ``None`` when it wrote none."""
    value = json_in(reply, openers=("{",))
    design = value.get("design") if isinstance(value, Mapping) else None
    return design.strip() if isinstance(design, str) and design.strip() else None


def kept_design(design: str) -> str:
    """The design as the step records it: whole up to ``DESIGN_CHARS``, and past it the start cut so the last
    paragraph, the How to use the person reads, stays whole."""
    if len(design) <= DESIGN_CHARS:
        return design
    head, _, last = design.rpartition("\n\n")
    if not head or len(last) >= DESIGN_CHARS // 2:
        return design[:DESIGN_CHARS]
    marker = "\n\n[...]\n\n"
    return head[: DESIGN_CHARS - len(last) - len(marker)].rstrip() + marker + last


def undeclared_env(patch: TreePatch, requires: Sequence[Mapping[str, Any]], layout: TreeLayout) -> list[str]:
    """The variables the patch's extensions read through ``process.env`` that no requires item names, in reading
    order; the ones pi and the shell set, and reef's own, are not needs of the user's."""
    declared = {str(item.get(key)) for item in requires for key in ("name", "check")}
    found: list[str] = []
    for path, text in patch.changes.items():
        location = layout.locate(path)
        if text is None or location is None or location.part.format is not Format.CODE:
            continue
        for dotted, bracketed in ENV_READ.findall(text):
            variable = dotted or bracketed
            if variable in declared or variable in found or variable in SESSION_ENV or variable.startswith("REEF_"):
                continue
            found.append(variable)
    return found


def tool_steps(models: ModelBindings, request_text: str, files_text: str, tools: str | None = None) -> list[str]:
    """The steps of a request the harness cannot perform, as the served model lists them in a first, short call;
    ``tools`` names the harness's own tools where Reefine knows them (off pi).

    A call that fails or answers without the JSON shape yields no steps: the request is then answered without
    the plan section."""
    plan_prompt = PLAN_PROMPT.format(
        request=request_text, files=files_text, tools="" if tools is None else f"Its own tools: {tools}.\n\n"
    )
    reply, _ = ask(models, plan_prompt, max_tokens=max_tokens(4096), timeout_s=timeout_s(60.0))
    if reply is None:
        return []
    value = json_in(reply)
    items = value if isinstance(value, list) else [value]
    steps: list[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("needs_tool") is not True:
            continue
        step = item.get("step")
        if isinstance(step, str) and step.strip():
            steps.append(step.strip()[:200])
    return steps


def timeout_s(default: float) -> float:
    """The budget of one model call: ``REEF_REEFINE_CALL_TIMEOUT_S`` when set, else the caller's default."""
    raw = os.environ.get("REEF_REEFINE_CALL_TIMEOUT_S", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logging.getLogger(__name__).warning("REEF_REEFINE_CALL_TIMEOUT_S=%r is not a number; using %s", raw, default)
        return default


def max_tokens(default: int) -> int:
    """The reply budget of one model call: ``REEF_REEFINE_CALL_MAX_TOKENS`` when set, else the caller's default.

    A thinking model spends the budget on its reasoning first, and a reply cut there has no text: the defaults are
    sized for that, and a local model may still need more."""
    raw = os.environ.get("REEF_REEFINE_CALL_MAX_TOKENS", "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        logging.getLogger(__name__).warning("REEF_REEFINE_CALL_MAX_TOKENS=%r is not a number; using %s", raw, default)
        return default


def failures_text(traces: Sequence[Trace]) -> str:
    """The step's traces as a prompt reads them: one object per trace with the request as served (the trace's last
    captured exchange; the task of a run, whose trace has no trajectory), the trace's score and its feedback
    verbatim (``null`` where there is none)."""
    views = []
    for trace in traces:
        document = trace.trajectory()
        served = trace.task if document is None else recorded_payload(TrajectorySample(document))
        views.append({"request": served, "score": trace.score, "feedback": trace.feedback})
    return json.dumps(views, indent=2, default=str)


#: What to add when the binding got a reply without text: a thinking model's reasoning took the budget.
NO_TEXT_HINT = (
    "; a thinking model may have spent the reply budget on its reasoning, raise REEF_REEFINE_CALL_MAX_TOKENS"
)


def ask(
    models: ModelBindings, text: str, *, max_tokens: int, timeout_s: float = 60.0
) -> tuple[str | None, str | None]:
    """One served model call: the reply and no reason, or ``None`` and a one-line reason when the endpoint failed.

    The reason names how long the call took and the reply budget, then the exception's text; it goes to the log,
    and a request step records it so the person sees why nothing changed."""
    started = time.monotonic()
    try:
        reply = models.served.chat([{"role": "user", "content": text}], timeout_s=timeout_s, max_tokens=max_tokens)
    except Exception as exc:
        elapsed = time.monotonic() - started
        reason = f"model call failed after {elapsed:.1f} s (max_tokens={max_tokens}): {exc}"
        if "non-text content" in str(exc):
            reason += NO_TEXT_HINT
        logging.getLogger(__name__).warning("reefine: served %s", reason)
        return None, reason
    return reply, None


def provider_refusal(models: ModelBindings) -> str | None:
    """Why the provider cut the last reply short, when its response says a filter or a refusal stopped it: an
    OpenAI ``finish_reason``, a Responses ``incomplete_details.reason`` or an Anthropic ``stop_reason``."""
    response = models.served.last_response()
    if not isinstance(response, Mapping):
        return None
    details = response.get("incomplete_details")
    reasons = [
        *(choice.get("finish_reason") for choice in response.get("choices") or () if isinstance(choice, Mapping)),
        details.get("reason") if isinstance(details, Mapping) else None,
        response.get("stop_reason"),
    ]
    return next((str(reason) for reason in reasons if reason in ("content_filter", "refusal")), None)


def parse_answer_requires(reply: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ``requires`` items of the reply's object that pass the shape check, and the refused ones with the reason."""
    value = json_in(reply, openers=("{",))
    items = value.get("requires") if isinstance(value, Mapping) else None
    return screened_requires(items if isinstance(items, list) else [])


def screened_requires(items: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``items`` through the requires shape check, an env check brought to its variable name and the prompt to
    one sentence first: the kept items, and the refused ones each with its reason."""
    kept: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for item in items:
        try:
            (parsed,) = parse_requires([trimmed_prompt(named_env_check(item))])
        except ValueError as error:
            refused.append({"item": item, "reason": str(error)})
            continue
        kept.append(parsed)
    return kept, refused


def trimmed_prompt(item: Any) -> Any:
    """An item's prompt stripped and cut at the cap the wire contract puts on it, dropped when empty; anything else
    stays as written, for ``parse_requires`` to refuse or keep by its own rule."""
    if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
        return item
    item_prompt = item["prompt"].strip()[:200].strip()
    if not item_prompt:
        return {key: value for key, value in item.items() if key != "prompt"}
    return {**item, "prompt": item_prompt}


def named_env_check(item: Any) -> Any:
    """An env item whose check is a shell test rather than a variable name, brought to the variable it tests.

    The one ``$VAR`` the check names becomes the check; when it names none or several, the check is dropped if
    ``name`` is itself a variable name, so the item still says which variable to set. Anything else is returned
    as written, for ``parse_requires`` to refuse with its reason."""
    if not isinstance(item, dict) or item.get("kind") != "env" or not isinstance(item.get("check"), str):
        return item
    check = item["check"].strip()
    if VARIABLE_NAME.fullmatch(check):
        return item
    named = list(dict.fromkeys(SHELL_VARIABLE.findall(check)))
    if len(named) == 1:
        return {**item, "check": named[0]}
    if isinstance(item.get("name"), str) and VARIABLE_NAME.fullmatch(item["name"]):
        return {key: value for key, value in item.items() if key != "check"}
    return item


def json_in(reply: str, openers: Sequence[str] = ("[", "{")) -> Any:
    """The JSON array or object inside the model's text, fences and prose around it dropped; ``None`` when none
    parses. ``openers`` says which to look for and in what order."""
    found = json_found(reply, openers)
    return None if found is None else found[1]


def json_found(reply: str, openers: Sequence[str] = ("[", "{")) -> tuple[int, Any] | None:
    """Where ``json_in``'s value starts in the reply, and the value; ``None`` when none parses."""
    decoder = json.JSONDecoder()
    # The first array, else the first object, decoded in place: prose after it (a bracketed citation, say) is ignored.
    for opener in openers:
        for at, char in enumerate(reply):
            if char != opener:
                continue
            try:
                return at, decoder.raw_decode(reply, at)[0]
            except ValueError:
                continue
    return None


def slipped_json(reply: str) -> str | None:
    """Why the reply's outermost JSON does not parse; ``None`` when it parses or the reply holds no bracket.

    The outermost value starts at the first bracket or brace. When it does not decode, a value that decodes later
    stands for the answer only when the text before it closed every bracket it opened (a stray bracket in prose),
    never when it sits inside the broken value."""
    starts = [at for at in (reply.find("["), reply.find("{")) if at >= 0]
    if not starts:
        return None
    start = min(starts)
    try:
        json.JSONDecoder().raw_decode(reply, start)
        return None
    except json.JSONDecodeError as error:
        failure = error
    # The answer is one object, so an array nested in it (its requires list) never stands for the reply.
    found = json_found(reply, openers=("{",))
    if found is not None and open_brackets(reply[start : found[0]]) <= 0:
        return None
    return f"the reply's JSON does not parse ({failure.msg} at line {failure.lineno} column {failure.colno})"


def open_brackets(text: str) -> int:
    """How many brackets and braces ``text`` leaves open, the ones inside JSON strings not counted."""
    depth = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
    return depth


def learn_from_failures(
    tree: HarnessTree, traces: Sequence[Trace], models: ModelBindings, layout: TreeLayout
) -> TreePatch | None:
    """One improved or new skill from the step's failing traces; ``None`` when the call fails or the reply holds
    no skill, and the reason goes to the log alone."""
    skills = {
        path: text
        for path, text in tree.files.items()
        if (location := layout.locate(path)) and location.part.name == "skill"
    }
    # The requests and their feedback are client text: fenced as data so nothing inside them can speak as this prompt.
    failures_prompt = FAILURES_PROMPT.format(
        failures=untrusted_text(failures_text(traces)), skills=json.dumps(skills, indent=2, ensure_ascii=False)
    )
    reply, _ = ask(models, failures_prompt, max_tokens=max_tokens(8192), timeout_s=timeout_s(60.0))
    if reply is None:
        return None
    parsed = parse_answer(reply, tree, layout, ("skill",), ())
    if parsed is None:
        return None
    first = next(((path, text) for path, text in parsed.patch.changes.items() if text is not None), None)
    if first is None:
        return None
    return TreePatch(dict([first]))


__all__ = [
    "EXTENSION_ADAPTER",
    "PI_EXTENSION_API",
    "Answer",
    "answer_request",
    "client_text",
    "failures_section",
    "kept_design",
    "learn_from_failures",
    "request_parts",
    "review_answer",
    "screened_requires",
    "undeclared_env",
]
