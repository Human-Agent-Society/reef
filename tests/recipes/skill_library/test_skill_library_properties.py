"""Properties of the SkillLibrary inputs that come from a model reply or from ``state/skills.json``.

The extractor's reply is arbitrary text, a skill name can hold any character a JSON escape makes, and the library
file is read back on every step, so each of these must parse or refuse with ``ValueError`` and never fail otherwise.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml
from hypothesis import given
from hypothesis import strategies as st

from recipes.skill_library.method import (
    MAX_DESCRIPTION_CHARS,
    MAX_SLUG_CHARS,
    MAX_TRACE_CHARS,
    SKILL_FIELDS,
    SKILL_PREFIX,
    SLUG,
    ReplayOutcome,
    Skill,
    induced,
    slug_of,
    trace_text,
)
from reef.train.harness_backend import Trace

# Every code point, lone surrogates included and drawn often: a JSON escape such as "\ud800" makes one.
# st.text drops surrogates from any alphabet, so the text is joined from a list of characters.
ANY_CHAR = st.characters(exclude_categories=()) | st.characters(categories=("Cs",))
ANY_TEXT = st.lists(ANY_CHAR).map("".join)
JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | ANY_TEXT,
    lambda children: st.lists(children) | st.dictionaries(ANY_TEXT, children),
    max_leaves=30,
)
WORD = st.text(st.characters(exclude_categories=("Cs", "Z", "Cc")), min_size=1, max_size=12)
LINE = st.lists(WORD, min_size=1, max_size=8).map(" ".join)
TEXT = st.lists(LINE, min_size=1, max_size=4).map("\n".join)
DESCRIPTION = LINE.filter(lambda line: len(line) <= MAX_DESCRIPTION_CHARS)
FRONTMATTER = re.compile(r"---\n(.*?)\n---\n", re.DOTALL)


def parsed_or_refused(reply: str) -> None:
    """Parse ``reply`` with ``induced`` and accept a result or a ``ValueError``."""
    try:
        fields, implicated = induced(reply)
    except ValueError:
        return
    assert fields is None or (set(fields) == set(SKILL_FIELDS) and all(fields.values()))
    assert all(isinstance(name, str) for name in implicated)


@given(ANY_TEXT)
def test_induced_parses_or_refuses_any_text(reply: str) -> None:
    parsed_or_refused(reply)


@given(st.integers(min_value=0, max_value=200_000), st.sampled_from(["[", '{"a":']))
def test_induced_refuses_a_deeply_nested_reply(depth: int, opener: str) -> None:
    closer = "]" if opener == "[" else "}"
    parsed_or_refused('{"skill": ' + opener * depth + "null" + closer * depth + "}")


@given(JSON_VALUES, JSON_VALUES)
def test_induced_parses_or_refuses_any_json(skill: object, implicated: object) -> None:
    parsed_or_refused(json.dumps({"skill": skill, "implicated": implicated}))
    parsed_or_refused(json.dumps(skill))


@given(
    st.fixed_dictionaries({"name": LINE, "description": DESCRIPTION, "check": TEXT, "known_bad": TEXT}),
    TEXT | st.lists(LINE, min_size=1, max_size=5),
    st.lists(LINE, max_size=3),
    st.text(st.characters(exclude_categories=("Cs",)), max_size=40).filter(lambda think: "</think>" not in think),
)
def test_induced_round_trips_a_well_formed_reply(
    texts: dict[str, str], procedure: str | list[str], implicated: list[str], think: str
) -> None:
    reply = f"{think}</think>" + json.dumps({"skill": {**texts, "procedure": procedure}, "implicated": implicated})
    if isinstance(procedure, str):
        expected = procedure
    else:
        expected = "\n".join(line if line.startswith("-") else f"- {line}" for line in procedure)
    assert induced(reply) == ({**texts, "procedure": expected}, implicated)


@given(ANY_TEXT)
def test_slug_of_is_a_bounded_slug(name: str) -> None:
    slug = slug_of(name)
    assert SLUG.fullmatch(slug) is not None and len(slug) <= MAX_SLUG_CHARS
    assert slug_of(name) == slug


SKILLS = st.builds(
    Skill,
    slug=st.from_regex(SLUG, fullmatch=True).filter(lambda slug: len(slug) <= MAX_SLUG_CHARS),
    name=LINE,
    description=DESCRIPTION,
    procedure=TEXT,
    check=TEXT,
    known_bad=TEXT,
    verified=st.booleans(),
    wins=st.integers(min_value=0),
    losses=st.integers(min_value=0),
    step=st.integers(min_value=1),
)


@given(SKILLS)
def test_skill_survives_the_library_file(skill: Skill) -> None:
    rows = json.loads(json.dumps([asdict(skill)], indent=2, ensure_ascii=False))
    assert [Skill(**row) for row in rows] == [skill]


@given(SKILLS)
def test_skill_text_frontmatter_parses_back(skill: Skill) -> None:
    match = FRONTMATTER.match(skill.skill())
    assert match is not None
    assert yaml.safe_load(match.group(1)) == {"name": SKILL_PREFIX + skill.slug, "description": skill.description}


@given(
    st.text(max_size=50) | st.integers(0, 3 * MAX_TRACE_CHARS).map(lambda count: "x" * count),
    ANY_TEXT,
    st.floats(allow_nan=False, allow_infinity=False) | st.none(),
    st.none() | st.integers(0, 3 * MAX_TRACE_CHARS).map(lambda count: "é" * count),
)
def test_trace_text_is_bounded(task: str, reply: str, score: float | None, failure: str | None) -> None:
    trace = Trace(Path("unused"), "run", task, reply, score, None, failure, "record", "untagged")
    # The cut keeps MAX_TRACE_CHARS characters; the marker and the data fence add a short fixed text.
    assert len(trace_text(trace)) <= MAX_TRACE_CHARS + 120


SCORES = st.none() | st.floats(allow_nan=False, allow_infinity=False)


@given(SCORES, SCORES, st.floats(min_value=0, allow_nan=False, allow_infinity=False))
def test_replay_outcome_of_means_keeps_its_invariants(
    challenger: float | None, reference: float | None, margin: float
) -> None:
    outcome = ReplayOutcome.of_means(challenger, reference, margin)
    assert outcome.decided == (challenger is not None and reference is not None)
    assert outcome.better == (outcome.decided and challenger > reference + margin)  # type: ignore[operator]


@given(SCORES, SCORES, st.booleans(), st.booleans())
def test_replay_outcome_refuses_exactly_the_inconsistent_values(
    challenger: float | None, reference: float | None, decided: bool, better: bool
) -> None:
    consistent = decided == (challenger is not None and reference is not None) and (decided or not better)
    try:
        ReplayOutcome(challenger, reference, decided, better)
    except ValueError:
        assert not consistent
    else:
        assert consistent
