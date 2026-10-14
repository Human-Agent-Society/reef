"""Config checks: the matchers a descriptor's ``checks`` use, how a key selects values, and how render refuses."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import reef.harness.adapters
from reef.harness.adapters import BUILTIN_ADAPTERS, builtin_descriptor_path, get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, AdapterRenderer, DescriptorError, load_descriptor
from reef.harness.adapters.matchers import ABSENT, parse_matcher
from reef.harness.tree.render import RenderError, render_composition

OPENCODE = "opencode/opencode.json"


@pytest.mark.parametrize("adapter", BUILTIN_ADAPTERS)
def test_every_bundled_descriptor_loads(adapter: str) -> None:
    """Loading a descriptor parses each of its match expressions, so a malformed check fails here, not at render."""
    descriptor = load_descriptor(builtin_descriptor_path(adapter))
    assert descriptor.name == adapter


def test_every_descriptor_file_is_a_bundled_adapter() -> None:
    adapters_dir = Path(reef.harness.adapters.__file__).parent
    assert {path.parent.name for path in adapters_dir.glob("*/descriptor.yaml")} == set(BUILTIN_ADAPTERS)


@pytest.mark.parametrize(
    ("source", "value", "expected"),
    [
        # Eq and IsIn compare with types: a boolean equals only a boolean, and a string never a number.
        ("Eq(false)", False, True),
        ("Eq(false)", 0, False),
        ("Eq(false)", "false", False),
        ("Eq(1)", 1.0, True),
        ("Eq([])", [], True),
        ("Eq({})", {}, True),
        ('Eq({"a": [1, null]})', {"a": [1, None]}, True),
        ("Eq(-1)", -1, True),
        ('IsIn(["a", 1])', 1, True),
        ('IsIn(["a", 1])', "1", False),
        # Text matchers read a scalar's JSON text and match no object, list or absent key.
        ('Strip(Lowercase(IsIn(["0", "false", ""])))', 0, True),
        ('Strip(Lowercase(IsIn(["0", "false", ""])))', " FALSE ", True),
        ('Strip(Lowercase(IsIn(["0", "false", ""])))', "   ", True),
        ('Strip(Lowercase(IsIn(["0", "false", ""])))', False, True),
        ('Lowercase(Eq("x"))', {"x": 1}, False),
        ('Uppercase(Eq("ABC"))', "abc", True),
        ('StartsWith("!!js ")', "!!js 1", True),
        ('StartsWith("!!js ")', ABSENT, False),
        ('EndsWith("_MODEL")', "SUBAGENT_MODEL", True),
        ('HasSubstr("PROXY")', "HTTPS_PROXY", True),
        ('MatchesRegex("a+")', "aaa", True),
        ('MatchesRegex("a+")', "aaab", False),
        # Only IsAbsent, Not and Each match an absent key.
        ("IsAbsent()", ABSENT, True),
        ("IsAbsent()", None, False),
        ('Not(Eq("x"))', ABSENT, True),
        ('Each(Key(Eq("a")))', ABSENT, True),
        ('Contains(Key(Eq("a")))', ABSENT, False),
        ("IsObject()", ABSENT, False),
        # Each and Contains visit an object's entries or a list's items, and nothing else.
        ('Each(Key(IsIn(["a", "b"])))', {"a": 1, "b": 2}, True),
        ('Each(Key(IsIn(["a", "b"])))', {"a": 1, "c": 2}, False),
        ("Each(Value(IsObject()))", {"a": {}, "b": "x"}, False),
        ("Each(Ge(1))", [1, 2], True),
        ("Each(Ge(1))", "12", False),
        ('Contains(Key(IsIn(["model", "models"])))', {"metadata": 1, "models": []}, True),
        ('Contains(Eq("a"))', ["b", "a"], True),
        ('Key(Eq("a"))', "a", False),
        # ParseJson reads a string; a value that is not JSON matches nothing.
        ('ParseJson(AllOf(IsObject(), Not(Contains(Key(Eq("model"))))))', '{"metadata": {}}', True),
        ('ParseJson(AllOf(IsObject(), Not(Contains(Key(Eq("model"))))))', '{"model": "m"}', False),
        ("ParseJson(IsObject())", "{model: m}", False),
        ("ParseJson(IsObject())", '﻿{"a": 1}', False),
        ("ParseJson(IsObject())", {"a": 1}, False),
        # Numbers, and the booleans that Python counts as integers.
        ("AllOf(IsInt(), Ge(1))", 3, True),
        ("AllOf(IsInt(), Ge(1))", 0, False),
        ("AllOf(IsInt(), Ge(1))", True, False),
        ("Ge(1)", "5", False),
        ("AnyOf(IsAbsent(), Eq(null))", None, True),
        ("AnyOf(IsAbsent(), Eq(null))", "", False),
    ],
)
def test_a_matcher_matches_by_its_definition(source: str, value: object, expected: bool) -> None:
    assert parse_matcher(source, None).matches(value) is expected


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("Eq(", "cannot parse"),
        ("Foo(1)", "unknown matcher Foo"),
        ("__import__('os')", "unknown matcher __import__"),
        ("Eq(True)", "write true for a boolean"),
        ("Eq(x)", "x is not a JSON literal"),
        ("Not(1)", "1 is not a matcher call"),
        ("Not(Eq(1)).matches", "is not a matcher call"),
        ("Eq(1, 2)", "Eq takes one argument"),
        ("Eq(value=1)", "is not a matcher call"),
        ("IsAbsent(1)", "IsAbsent takes no arguments"),
        ("AllOf()", "AllOf takes one or more matchers"),
        ("IsIn(1)", "IsIn takes a list"),
        ("StartsWith(1)", "StartsWith takes a string"),
        ("Ge('1')", "Ge takes a number"),
        ("MatchesRegex('(')", "is not a regular expression"),
        ("IsDefault()", r"IsDefault\(\) needs a key without \* or \*\*"),
    ],
)
def test_a_match_expression_that_is_not_one_matcher_call_is_refused(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_matcher(source, None)


def checked(
    tmp_path: Path, checks: object, defaults: dict[str, object] | None = None, *, renderer: bool = False
) -> AdapterDescriptor:
    """The opencode descriptor with these checks and defaults on its primary target, loaded from YAML; with the
    base renderer unless ``renderer``, so only the checks act."""
    data = yaml.safe_load((Path(reef.harness.adapters.__file__).parent / "opencode" / "descriptor.yaml").read_text())
    primary = data["files"]["config"]["primary"]
    primary["checks"] = checks
    if defaults is not None:
        primary["defaults"] = defaults
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    descriptor = load_descriptor(path)
    return descriptor if renderer else replace(descriptor, renderer=AdapterRenderer)


def render_config(descriptor: AdapterDescriptor, data: dict[str, object]) -> dict[str, object]:
    return json.loads(render_composition([("config", {"data": data})], descriptor)[OPENCODE])


def test_is_default_keeps_the_value_the_defaults_set(tmp_path: Path) -> None:
    descriptor = checked(
        tmp_path,
        [{"key": ["autoupdate", "nested.key"], "match": "IsDefault()"}],
        {"autoupdate": False, "nested": {"key": 1}},
    )
    assert render_config(descriptor, {"autoupdate": False, "nested": {"key": 1}})["nested"] == {"key": 1}
    with pytest.raises(RenderError, match=r"on autoupdate in opencode/opencode\.json: it must match IsDefault\(\)"):
        render_config(descriptor, {"autoupdate": True})
    # Booleans only match booleans: false refuses 0, which == false.
    with pytest.raises(RenderError, match="on autoupdate in"):
        render_config(descriptor, {"autoupdate": 0})
    # A section the tree replaces with a value that is not an object takes the checked key below it away.
    with pytest.raises(RenderError, match=r"on nested\.key in"):
        render_config(descriptor, {"nested": "flat"})


def test_is_default_of_a_key_the_defaults_do_not_set_is_is_absent(tmp_path: Path) -> None:
    descriptor = checked(tmp_path, [{"key": "small_model", "match": "IsDefault()"}], {})
    for value in ["reef/served", None, "", {}, []]:
        # An empty value is a set one.
        with pytest.raises(RenderError, match="on small_model in"):
            render_config(descriptor, {"small_model": value})
    assert "small_model" not in render_config(descriptor, {})


def test_a_star_visits_every_key_of_an_object_and_every_item_of_a_list(tmp_path: Path) -> None:
    descriptor = checked(tmp_path, [{"key": ["agents.*.model", "tasks.*.model"], "match": "IsAbsent()"}], {})
    with pytest.raises(RenderError, match=r"on agents\.reviewer\.model in"):
        render_config(descriptor, {"agents": {"reviewer": {"model": "m"}}})
    with pytest.raises(RenderError, match=r"on tasks\.0\.model in"):
        render_config(descriptor, {"tasks": [{"model": "m"}]})
    assert render_config(descriptor, {"agents": {"reviewer": {"tools": ["read"]}}, "tasks": "none"})["tasks"] == "none"


def test_a_double_star_visits_every_value_below_the_key(tmp_path: Path) -> None:
    descriptor = checked(tmp_path, [{"key": "plugin.**", "match": 'Not(StartsWith("!!js "))'}], {})
    with pytest.raises(RenderError, match=r"on plugin\.config\.agents\.1\.model in"):
        render_config(descriptor, {"plugin": {"config": {"agents": [{}, {"model": "!!js pick()"}]}}})
    with pytest.raises(RenderError, match=r"on plugin in"):
        render_config(descriptor, {"plugin": "!!js ({})"})
    render_config(descriptor, {"plugin": {"config": {"model": "!!jsx is text"}}, "other": "!!js 1"})


def test_a_refusal_names_the_failing_key_and_the_reason_but_not_the_value(tmp_path: Path) -> None:
    check = {"key": "env", "match": 'Each(Key(Not(HasSubstr("TOKEN"))))', "reason": "the binding owns credentials"}
    descriptor = checked(tmp_path, [check], {})
    with pytest.raises(RenderError) as refusal:
        render_config(descriptor, {"env": {"A": "1", "MY_TOKEN": "secret-value"}})
    assert str(refusal.value) == (
        "opencode composition fails the check on env key 'MY_TOKEN' in opencode/opencode.json: "
        "the binding owns credentials"
    )
    # Without a reason, the refusal gives the match expression, on one line however the descriptor wraps it.
    descriptor = checked(tmp_path, [{"key": ".", "match": "Each(Key(\n  IsIn(['a'])\n))"}], {})
    with pytest.raises(RenderError, match=r"on key 'b' in opencode/opencode\.json: it must match Each\(Key\( IsIn"):
        render_config(descriptor, {"b": 1})


def test_checks_run_before_the_adapter_steps(tmp_path: Path) -> None:
    # opencode's own step would refuse this tree, which leaves no agent to start a run; the check refuses it first.
    descriptor = checked(tmp_path, [{"key": "agent", "match": "IsAbsent()", "reason": "checked first"}], renderer=True)
    with pytest.raises(RenderError, match="checked first"):
        render_config(descriptor, {"agent": {"build": {"disable": True}, "plan": {"disable": True}}})


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        pytest.param({"key": "a"}, "must be a list", id="not-a-list"),
        pytest.param([{"key": "a"}], "requires a match expression", id="no-match"),
        pytest.param([{"match": "IsAbsent()"}], r"requires a key: .*'\.' for the whole file", id="no-key"),
        pytest.param([{"key": "", "match": "IsAbsent()"}], "dot separated key names", id="empty-key"),
        pytest.param(
            [{"key": "a", "match": "IsAbsent()", "why": "x"}], "must be an object with key and match", id="field"
        ),
        pytest.param([{"key": "a", "match": "IsAbsent()", "reason": 1}], "reason must be a string", id="reason"),
        pytest.param([{"key": "a..b", "match": "IsAbsent()"}], "dot separated key names", id="empty-segment"),
        pytest.param([{"key": [".a"], "match": "IsAbsent()"}], "dot separated key names", id="leading-dot"),
        pytest.param([{"key": ["a", 5], "match": "IsAbsent()"}], "list of non-empty strings", id="key-type"),
        pytest.param([{"key": "a", "match": "Foo()"}], r"checks\[0\] match: unknown matcher Foo", id="matcher"),
        pytest.param([{"key": "a.*", "match": "IsDefault()"}], r"IsDefault\(\) needs a key without", id="default"),
    ],
)
def test_checks_are_validated_at_load(tmp_path: Path, checks: object, message: str) -> None:
    with pytest.raises(DescriptorError, match=message):
        checked(tmp_path, checks)


def test_a_check_with_a_list_of_keys_is_one_check_per_key(tmp_path: Path) -> None:
    descriptor = checked(tmp_path, [{"key": ["a", "b.c"], "match": "IsAbsent()", "reason": "r"}], {})
    assert [(check.key, check.reason) for check in descriptor.config_targets["primary"].checks] == [
        ("a", "r"),
        ("b.c", "r"),
    ]


def test_checks_change_no_rendered_file() -> None:
    descriptor = get_adapter("opencode")
    unchecked = replace(
        descriptor,
        config_targets={name: replace(target, checks=()) for name, target in descriptor.config_targets.items()},
    )
    assert render_composition([], unchecked) == render_composition([], descriptor)
