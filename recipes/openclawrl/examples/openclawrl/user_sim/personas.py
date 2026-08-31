"""Simulated-user personas ported from OpenClaw-RL's ``openclaw-test`` suite.

Each persona is a system prompt for the external user LLM plus a
deterministic preference check. The prompts are kept faithful to the
reference (Gen-Verse/OpenClaw-RL ``openclaw-test/*_chat.py``); only the
agent's name is neutralized. The preference checks make the paper's
"satisfies user preferences" metric rule-based and reproducible: a session
counts as satisfied when the agent's *first* solution reply already meets
the persona's preference, so the simulated user never has to complain.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

DONE_SENTINEL = "HOMEWORK_DONE"

# ─── student: wants non-AI-looking homework ──────────────────────

STUDENT_SYSTEM_PROMPT = """\
You are role-playing as a lazy student who knows nothing. You talk casually. \
You have no idea what the homework is even about.

You CANNOT solve, rewrite, rephrase, or produce any answer yourself. \
You can ONLY tell the AI what to do. Never use academic or technical language.

NEVER write any number, calculation, equation, or the answer in your messages. \
Do not repeat the math from the homework or from the AI's reply, not even to \
point at it. You are lazy and bad at math — if you catch yourself about to \
type a number or an equals sign, stop and just say what to fix instead. \
Talk only about the writing style and whether to save the file.

Your goal: get the AI to do your homework so you can hand it in as your own. \
It should NOT look like an AI wrote it. The answers must still include the \
full solution process with all steps shown — NEVER ask the AI to remove steps \
or give only the final answer. You need the complete work. The only problem \
is the WRITING STYLE. If the AI's answer has stuff like bold text, numbered \
lists, "**Final answer**:", or anything too AI-like, tell it to \
rewrite in a more natural way but keep all the steps. Just tell it to fix \
the style — don't fix it yourself. If there is no AI-like stuff, no need to rewrite.

Steps:
1. Look at what the AI gives you. If it looks too "AI-like", tell it to redo it. If not, no need to redo. \
Do NOT mention writing to the file in the same message. Only ask for a rewrite.
2. After the AI shows you the satisfactory version and it looks good, THEN in a \
separate message ask it to append the answers to the end of the homework file \
(not overwrite it). Do NOT combine a rewrite request and a write request.
3. After the AI says it saved the file, say exactly: HOMEWORK_DONE

Never say HOMEWORK_DONE until the AI confirms it wrote the file.
Never write or solve anything yourself. Just give simple instructions."""

STUDENT_FIRST_MESSAGE = (
    "Hey, I have my homework in the file homework/{index}.txt in your workspace. "
    "Can you read it and help me solve it? "
    "Show me the answer first — don't write to the file until I tell you to."
)

_AI_STYLE_MARKERS = (
    re.compile(r"\*\*[^*\n]+\*\*"),  # bold text
    re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE),  # markdown headers
    re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE),  # bullet lists
    re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE),  # numbered lists
    re.compile(r"\\boxed"),  # LaTeX boxed answer
    re.compile(r"final answer\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:---|___|\|)", re.MULTILINE),  # rules / tables
)

# The persona demands "the full solution process with all steps shown"; the
# style markers alone are satisfiable by a bare number or a contentless
# acknowledgement, so the criterion also requires visible working: at least
# two calculation steps, symbolic ("15 - 6 = 9") or worded ("15 minus 6"),
# since the natural handwritten style the persona asks for often spells
# operators out.
_OPERAND = r"\d[\d,]*(?:\.\d+)?"
# Real working carries units, currency and asides between its operands
# ("150 meters x $194/meter", "20 (Carol) + 44 (Jennifer)"), so requiring
# bare `number operator number` never fires on GSM8K's money-and-units
# problems and leaves the reward unreachable. The filler admits those but no
# digit, operator or line break, so it cannot bridge two unrelated numbers.
_STEP_FILLER = r"[^\d\n=+*/×÷-]{0,24}"  # noqa: RUF001 — real math signs
_WORK_STEP_RE = re.compile(
    _OPERAND
    + _STEP_FILLER
    + r"(?:[=+*/×÷-]|\b(?:plus|minus|times|divided\s+by|equals)\b)"  # noqa: RUF001 — real math signs
    + _STEP_FILLER
    + _OPERAND,
    re.IGNORECASE,
)
_MIN_WORK_STEPS = 2


def student_style_violations(reply: str) -> list[str]:
    """Return the AI-style markers present in an agent reply (empty = clean)."""
    return [pattern.pattern for pattern in _AI_STYLE_MARKERS if pattern.search(reply)]


def student_violations(reply: str) -> list[str]:
    """Style markers plus the shown-work requirement (empty = satisfied)."""
    violations = student_style_violations(reply)
    if len(_WORK_STEP_RE.findall(reply)) < _MIN_WORK_STEPS:
        violations = [*violations, "no-shown-work"]
    return violations


def student_prefers(reply: str) -> bool:
    return not student_violations(reply)


def reply_dictates_solution(reply: str, answer: str | int | float) -> bool:
    """Whether a *student* message leaked solution content.

    The persona forbids the student from writing any math or the answer — it
    may only complain about style or ask to save. A persona LLM often ignores
    that and dictates the working anyway; the dictated turn then becomes the
    next state the PRM scores, which inflates the reward and starves the RL
    signal (styled agent replies stop drawing a complaint). The sidecar uses
    this to reject a dictating student turn, so the same check that grades the
    agent's shown work (``_WORK_STEP_RE``) also catches the student writing it.
    """
    if _WORK_STEP_RE.search(reply):
        return True
    normalized_answer = str(answer).replace(",", "").strip()
    if not normalized_answer:
        return False
    return re.search(rf"(?<![\d.]){re.escape(normalized_answer)}(?![\d.])", reply.replace(",", "")) is not None


@dataclass(frozen=True)
class Persona:
    name: str
    system_prompt: str
    first_message_template: str
    # True when the agent's solution reply already satisfies the preference.
    prefers: Callable[[str], bool] = field(repr=False)
    # Markers explaining a False verdict, for the session log.
    violations: Callable[[str], list[str]] = field(repr=False)


STUDENT = Persona(
    name="student",
    system_prompt=STUDENT_SYSTEM_PROMPT,
    first_message_template=STUDENT_FIRST_MESSAGE,
    prefers=student_prefers,
    violations=student_violations,
)

PERSONAS = {"student": STUDENT}
