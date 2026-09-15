"""The Environment Designer: what it is asked, how its reply becomes an environment, and the smoke test.

The Designer is the served model in a second role (SPADE Sec. 4.1). It is asked for one environment per
call, adversarially: the request carries what the Reasoning Agent did on the last generation's
environments, split into the ones at the frontier (lost without the hint, won with it), the ones it wins
anyway and the ones out of its reach, so the next environment lands where the agent fails today and a
hint would help (the hint based regret of Sec. 4.2). The reply is one ``python`` block and one ``hint``
block; ``smoke_test`` runs the code in a child interpreter before anything else trusts it.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from recipes.beta.spade import environment_loader
from recipes.beta.spade.environment_loader import environment_class_name
from recipes.beta.spade.tasks import DEFAULT_MAX_TURNS, SKILL_PATTERN
from reef.train.cordis_backend.strategies import untrusted_text

DIFFICULTIES = ("easy", "medium", "hard")
MAX_EXPERIENCE_RECORDS = 12
CODE_EXCERPT_CHARS = 1200
GROUNDING_CHARS = 6000
SMOKE_TIMEOUT_S = 20.0
WIN_RETURN = 1.0
PYTHON_BLOCK = re.compile(r"```python[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.S)
HINT_BLOCK = re.compile(r"```hint[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.S)
HINT_LINE = re.compile(r"^HINT:[ \t]*(.+)$", re.M)

SYSTEM_PROMPT = (
    "You are an expert Python programmer and game designer. You write interactive, multi turn text games "
    "that train a language model agent by finding the edge of what it can do."
)


class DesignerReplyError(ValueError):
    """The Designer's reply holds no usable environment."""


@dataclass(frozen=True)
class PlayRecord:
    """What the agent did on one earlier environment: its episode return without and with the hint."""

    name: str
    skill: str
    return_without_hint: float
    return_with_hint: float
    code_excerpt: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a play record needs the environment's name")
        for label, value in (
            ("return_without_hint", self.return_without_hint),
            ("return_with_hint", self.return_with_hint),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{label} must be a number in [-1, 1]")

    @property
    def regret(self) -> float:
        """How much the hint helped: the with hint return minus the without hint return."""
        return float(self.return_with_hint) - float(self.return_without_hint)

    @property
    def outcome(self) -> str:
        """``easy`` when won without the hint, ``frontier`` when won only with it, else ``out_of_reach``."""
        if self.return_without_hint >= WIN_RETURN:
            return "easy"
        if self.return_with_hint >= WIN_RETURN:
            return "frontier"
        return "out_of_reach"


@dataclass(frozen=True)
class DesignerRequest:
    """One Designer call: the skill to test, how hard, what the agent did last time, and a grounding text."""

    skill: str
    skill_description: str
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_MAX_TURNS
    grounding: str | None = None
    experience: tuple[PlayRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        if not isinstance(self.skill_description, str) or not self.skill_description.strip():
            raise ValueError("skill_description must be non-empty text")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if isinstance(self.turn_limit, bool) or not isinstance(self.turn_limit, int) or self.turn_limit < 2:
            raise ValueError("turn_limit must be an integer of at least 2")
        if self.grounding is not None and (not isinstance(self.grounding, str) or not self.grounding.strip()):
            raise ValueError("grounding must be non-empty text when set")
        if not isinstance(self.experience, tuple) or not all(isinstance(r, PlayRecord) for r in self.experience):
            raise ValueError("experience must be a tuple of PlayRecord")
        if len(self.experience) > MAX_EXPERIENCE_RECORDS:
            raise ValueError(f"experience holds at most {MAX_EXPERIENCE_RECORDS} records; pick the most recent")


@dataclass(frozen=True)
class DesignerReply:
    """The two parts of a usable reply: the environment's code and the hint for the agent."""

    code: str
    hint: str


@dataclass(frozen=True)
class SmokeResult:
    """Whether the code runs as an environment; ``reason`` names the first contract break."""

    ok: bool
    reason: str
    first_observation: str = ""


def designer_messages(request: DesignerRequest) -> list[dict[str, str]]:
    """The chat messages for one Designer call, in the shape ``ModelBinding.chat`` takes."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": designer_prompt(request)}]


def designer_prompt(request: DesignerRequest) -> str:
    """The user turn of a Designer call: target, what the agent did last time, grounding, rules, output."""
    parts = [
        "Create ONE interactive, multi turn text game as a Python class that tests: "
        f"{request.skill} ({request.skill_description.strip()}).",
        f"DIFFICULTY: {request.difficulty}. The game lasts at most {request.turn_limit} turns; optimal play "
        "wins in fewer, random play loses.",
        experience_text(request.experience),
    ]
    if request.grounding is not None:
        parts.append(
            "GROUNDING: the game must make the agent execute a technique or operate a system from this "
            "document. Never mention the document in the game's text.\n"
            + untrusted_text(request.grounding.strip()[:GROUNDING_CHARS], "reference document")
        )
    parts.extend([RULES_TEXT.format(turn_limit=request.turn_limit), OUTPUT_TEXT])
    return "\n\n".join(parts)


def experience_text(experience: Sequence[PlayRecord]) -> str:
    """The agent's results on the last environments, sorted into what to write more of and what to avoid."""
    if not experience:
        return (
            "WHAT THE AGENT DID LAST TIME: nothing recorded yet. Aim for a game a careful player wins and a "
            "hasty one loses."
        )
    frontier = [r for r in experience if r.outcome == "frontier"]
    easy = [r for r in experience if r.outcome == "easy"]
    out_of_reach = [r for r in experience if r.outcome == "out_of_reach"]
    lines = [
        "WHAT THE AGENT DID LAST TIME (episode returns in [-1, 1]; a hint is a few sentences of strategy "
        "the agent was given on a second attempt):"
    ]
    if frontier:
        lines.append(
            "- At the frontier, lost without the hint and won with it. Write games like these, varied, not copies:"
        )
        lines.extend(record_lines(frontier, with_code=True))
    if easy:
        lines.append("- Won without any hint. Too easy; do not write games like these:")
        lines.extend(record_lines(easy, with_code=False))
    if out_of_reach:
        lines.append(
            "- Lost even with the hint. Out of reach or broken; do not write games like these, and make sure "
            "the observation gives the agent enough to act on:"
        )
        lines.extend(record_lines(out_of_reach, with_code=False))
    return "\n".join(lines)


def record_lines(records: Sequence[PlayRecord], *, with_code: bool) -> list[str]:
    lines = []
    for record in records:
        lines.append(
            f"  {record.name} ({record.skill}): without hint {record.return_without_hint:+.2f}, "
            f"with hint {record.return_with_hint:+.2f}"
        )
        if with_code and record.code_excerpt.strip():
            lines.append(untrusted_text(record.code_excerpt.strip()[:CODE_EXCERPT_CHARS], "earlier game code"))
    return lines


RULES_TEXT = """RULES:
- One class whose name ends in Env, standard library only (random, json, re, math), no input(), no files, no network, no printing.
- reset(self, seed=None) -> (observation: str, info: dict). All randomness comes from the seed: the same seed gives the same game. reset() generates ONE task for the episode; step() never generates a new one.
- step(self, action: str) -> (observation: str, reward: float, terminated: bool, truncated: bool, info: dict). Every code path returns that 5-tuple; info is a dict, never a string.
- step() receives the agent's answer as \\boxed{{action}}. Extract the action with re.search(r"\\\\boxed\\{{([^}}]*)\\}}", action); when there is no box, return a reminder of the format with reward 0.0 and do not end the episode. An action the game does not understand gets a specific error observation and does not end the episode.
- HIDDEN STATE: the goal cannot be reached in one action; the agent must probe, remember and plan. The observation never states the answer or the rule behind it.
- Every observation shows the current state, the result of the last action, what actions are possible, and reminds the agent to answer with \\boxed{{action}}.
- REWARD: a win returns 1.0 with terminated=True; a loss returns 0.0 with terminated=True; every other step returns 0.0; after {turn_limit} turns return truncated=True with 0.0.
- SELF CHECK before you answer: trace two different action sequences from reset(seed=0) and confirm the returns above."""

OUTPUT_TEXT = """OUTPUT exactly two fenced blocks and nothing else:
```python
<the complete environment code>
```
```hint
<one to three sentences for the agent: the key strategy and the answer format, without the answer itself; mention only what the agent can see>
```"""


def parse_designer_reply(text: str) -> DesignerReply:
    """The first ``python`` block and the hint of a reply; ``DesignerReplyError`` when either is missing."""
    if not isinstance(text, str) or not text.strip():
        raise DesignerReplyError("the reply is empty")
    code_match = PYTHON_BLOCK.search(text)
    if code_match is None:
        raise DesignerReplyError("the reply holds no ```python block")
    code = code_match.group(1).replace("\r\n", "\n").strip("\n") + "\n"
    if not code.strip():
        raise DesignerReplyError("the ```python block is empty")
    hint_match = HINT_BLOCK.search(text, code_match.end())
    if hint_match is None:
        hint_match = HINT_LINE.search(text, code_match.end())
    if hint_match is None:
        raise DesignerReplyError("the reply holds no ```hint block after the code")
    hint = " ".join(hint_match.group(1).split())
    if not hint:
        raise DesignerReplyError("the hint is empty")
    return DesignerReply(code=code, hint=hint)


SMOKE_SCRIPT = r'''"""Load an environment class, reset it twice with one seed, take one step, and print what held."""

import json
import sys

from env_loader import load_environment_class, make_environment


def main(path, seed, max_turns):
    environment_class = load_environment_class(path)
    first = make_environment(environment_class, max_turns).reset(seed=seed)
    if not (isinstance(first, tuple) and len(first) == 2):
        return {"ok": False, "reason": "reset(seed) must return (observation, info)"}
    observation, info = first
    if not isinstance(observation, str) or not observation.strip():
        return {"ok": False, "reason": "reset(seed) must return a non-empty observation string"}
    if not isinstance(info, dict):
        return {"ok": False, "reason": "reset(seed) must return a dict as info"}
    again, _ = make_environment(environment_class, max_turns).reset(seed=seed)
    if again != observation:
        return {"ok": False, "reason": "reset(seed) is not deterministic: two resets with one seed differ"}
    environment = make_environment(environment_class, max_turns)
    environment.reset(seed=seed)
    result = environment.step("\\boxed{probe}")
    if not (isinstance(result, tuple) and len(result) == 5):
        return {"ok": False, "reason": "step(action) must return (observation, reward, terminated, truncated, info)"}
    next_observation, reward, terminated, truncated, step_info = result
    if not isinstance(next_observation, str):
        return {"ok": False, "reason": "step(action) must return an observation string"}
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or reward != reward:
        return {"ok": False, "reason": "step(action) must return a finite number as reward"}
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        return {"ok": False, "reason": "step(action) must return bools for terminated and truncated"}
    if not isinstance(step_info, dict):
        return {"ok": False, "reason": "step(action) must return a dict as info"}
    return {"ok": True, "reason": "", "first_observation": observation[:2000]}


if __name__ == "__main__":
    try:
        report = main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    except BaseException as exc:
        report = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:500]}
    sys.stdout.write("\n" + json.dumps(report) + "\n")
'''


def smoke_test(
    code: str, *, seed: int = 0, max_turns: int = DEFAULT_MAX_TURNS, timeout_s: float = SMOKE_TIMEOUT_S
) -> SmokeResult:
    """Run ``code`` as an environment in a child interpreter: the class loads, reset is deterministic, step returns the 5-tuple."""
    try:
        environment_class_name(code)
    except ValueError as exc:
        return SmokeResult(ok=False, reason=str(exc))
    with tempfile.TemporaryDirectory(prefix="spade-smoke-") as directory:
        root = Path(directory)
        (root / "env.py").write_text(code, encoding="utf-8")
        (root / "env_loader.py").write_text(inspect.getsource(environment_loader), encoding="utf-8")
        (root / "smoke.py").write_text(SMOKE_SCRIPT, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-E", "-s", str(root / "smoke.py"), str(root / "env.py"), str(seed), str(max_turns)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SmokeResult(
                ok=False, reason=f"the environment did not finish reset and one step in {timeout_s:g} s"
            )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        return SmokeResult(
            ok=False, reason=f"the child exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        )
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError:
        return SmokeResult(ok=False, reason="the environment wrote to stdout and hid the smoke report")
    if not isinstance(report, dict) or not isinstance(report.get("ok"), bool):
        return SmokeResult(ok=False, reason="the smoke report is malformed")
    first_observation = report.get("first_observation", "")
    if not isinstance(first_observation, str):
        first_observation = ""
    return SmokeResult(ok=report["ok"], reason=str(report.get("reason", "")), first_observation=first_observation)
