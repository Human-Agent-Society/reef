"""Matchers for the tree node checks a descriptor declares, borrowed from GoogleTest's matchers.

Each check names a ``key`` of the checked value and a ``match`` expression that
every value at that key must match, such as
``Not(Lowercase(IsIn(["0", "false"])))``. The key syntax and what each matcher
matches are the descriptor contract described in "Node checks" in
``docs/developer-guide/harness-adapters.rst``.

A match expression is parsed with Python's expression grammar, restricted to
matcher calls and JSON literals (``true``, ``false`` and ``null`` included), so
the descriptor needs no grammar of its own and the expression is never run.
"""

from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Absent:
    """The value of a key the config does not set."""

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT = Absent()


@dataclass(frozen=True)
class Entry:
    """One key of an object and its value, as ``Each`` and ``Contains`` visit an object."""

    key: object
    value: object


class Matcher(ABC):
    """One condition on a config value."""

    @abstractmethod
    def matches(self, value: object) -> bool: ...


def text_of(value: object) -> str | None:
    """A scalar as text matchers read it; None for an object, a list or ``ABSENT``."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float) or value is None:
        return json.dumps(value)
    return None


def json_equal(left: object, right: object) -> bool:
    """Whether two config values are equal with their types: a boolean equals only a boolean."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def elements_of(value: object) -> list[object] | None:
    """The entries of an object or the items of a list; None for any other value."""
    if isinstance(value, dict):
        return [Entry(key, item) for key, item in value.items()]
    if isinstance(value, list):
        return list(value)
    return None


@dataclass(frozen=True)
class Combining(Matcher):
    """A matcher over the results of several others."""

    matchers: tuple[Matcher, ...]


@dataclass(frozen=True)
class AllOf(Combining):
    def matches(self, value: object) -> bool:
        return all(matcher.matches(value) for matcher in self.matchers)


@dataclass(frozen=True)
class AnyOf(Combining):
    def matches(self, value: object) -> bool:
        return any(matcher.matches(value) for matcher in self.matchers)


@dataclass(frozen=True)
class Wrapping(Matcher):
    """A matcher that applies another to the value, or to a part of it."""

    matcher: Matcher


@dataclass(frozen=True)
class Not(Wrapping):
    def matches(self, value: object) -> bool:
        return not self.matcher.matches(value)


@dataclass(frozen=True)
class Each(Wrapping):
    """Every entry of an object, or every item of a list, matches; an absent key has none to fail."""

    def matches(self, value: object) -> bool:
        if value is ABSENT:
            return True
        elements = elements_of(value)
        return elements is not None and all(self.matcher.matches(element) for element in elements)

    def first_failure(self, value: object) -> str | None:
        """The key or the item that fails, to name in a refusal; None when the value has neither."""
        if isinstance(value, dict):
            return next(
                (f"key {key!r}" for key, item in value.items() if not self.matcher.matches(Entry(key, item))), None
            )
        if isinstance(value, list):
            return next((f"item {index}" for index, item in enumerate(value) if not self.matcher.matches(item)), None)
        return None


@dataclass(frozen=True)
class Contains(Wrapping):
    """At least one entry of an object, or one item of a list, matches."""

    def matches(self, value: object) -> bool:
        elements = elements_of(value)
        return elements is not None and any(self.matcher.matches(element) for element in elements)


@dataclass(frozen=True)
class Key(Wrapping):
    """The key of an object entry matches."""

    def matches(self, value: object) -> bool:
        return isinstance(value, Entry) and self.matcher.matches(value.key)


@dataclass(frozen=True)
class Value(Wrapping):
    """The value of an object entry matches."""

    def matches(self, value: object) -> bool:
        return isinstance(value, Entry) and self.matcher.matches(value.value)


@dataclass(frozen=True)
class ParseJson(Wrapping):
    """A string holding JSON whose value matches; a string that is not JSON matches nothing."""

    def matches(self, value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            return False
        return self.matcher.matches(parsed)


@dataclass(frozen=True)
class TextTransform(Wrapping):
    """A matcher applied to the value's text after a change, such as lowercase."""

    def matches(self, value: object) -> bool:
        text = text_of(value)
        return text is not None and self.matcher.matches(self.transform(text))

    @abstractmethod
    def transform(self, text: str) -> str: ...


@dataclass(frozen=True)
class Lowercase(TextTransform):
    def transform(self, text: str) -> str:
        return text.lower()


@dataclass(frozen=True)
class Uppercase(TextTransform):
    def transform(self, text: str) -> str:
        return text.upper()


@dataclass(frozen=True)
class Strip(TextTransform):
    """The text without the whitespace at its start and end."""

    def transform(self, text: str) -> str:
        return text.strip()


@dataclass(frozen=True)
class TextMatcher(Matcher):
    """A condition on the value's text and one string argument."""

    argument: str

    def matches(self, value: object) -> bool:
        text = text_of(value)
        return text is not None and self.matches_text(text)

    @abstractmethod
    def matches_text(self, text: str) -> bool: ...


@dataclass(frozen=True)
class StartsWith(TextMatcher):
    def matches_text(self, text: str) -> bool:
        return text.startswith(self.argument)


@dataclass(frozen=True)
class EndsWith(TextMatcher):
    def matches_text(self, text: str) -> bool:
        return text.endswith(self.argument)


@dataclass(frozen=True)
class HasSubstr(TextMatcher):
    def matches_text(self, text: str) -> bool:
        return self.argument in text


@dataclass(frozen=True)
class MatchesRegex(TextMatcher):
    """Text the whole of which the regular expression matches."""

    def matches_text(self, text: str) -> bool:
        return re.fullmatch(self.argument, text) is not None


@dataclass(frozen=True)
class IsAbsent(Matcher):
    def matches(self, value: object) -> bool:
        return value is ABSENT


@dataclass(frozen=True)
class IsObject(Matcher):
    def matches(self, value: object) -> bool:
        return isinstance(value, dict)


@dataclass(frozen=True)
class IsInt(Matcher):
    """An integer, not a boolean."""

    def matches(self, value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Ge(Matcher):
    """A number, not a boolean, at least ``bound``."""

    bound: int | float

    def matches(self, value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool) and value >= self.bound


@dataclass(frozen=True)
class Eq(Matcher):
    """A value equal to ``expected`` with its type; ``IsDefault()`` is an ``Eq`` of the defaults' value."""

    expected: object

    def matches(self, value: object) -> bool:
        return json_equal(value, self.expected)


@dataclass(frozen=True)
class IsIn(Matcher):
    """A value equal to one of ``choices`` with its type."""

    choices: tuple[object, ...]

    def matches(self, value: object) -> bool:
        return any(json_equal(value, choice) for choice in self.choices)


COMBINING: dict[str, type[Combining]] = {"AllOf": AllOf, "AnyOf": AnyOf}
WRAPPING: dict[str, type[Wrapping]] = {
    "Not": Not,
    "Each": Each,
    "Contains": Contains,
    "Key": Key,
    "Value": Value,
    "ParseJson": ParseJson,
    "Lowercase": Lowercase,
    "Uppercase": Uppercase,
    "Strip": Strip,
}
TEXT: dict[str, type[TextMatcher]] = {
    "StartsWith": StartsWith,
    "EndsWith": EndsWith,
    "HasSubstr": HasSubstr,
    "MatchesRegex": MatchesRegex,
}
NO_ARGUMENT: dict[str, type[IsAbsent | IsObject | IsInt]] = {
    "IsAbsent": IsAbsent,
    "IsObject": IsObject,
    "IsInt": IsInt,
}
#: The matchers that take literals, and ``IsDefault``, which the parser resolves to an ``Eq``.
LITERAL_MATCHERS = ("Eq", "IsIn", "Ge", "IsDefault")
JSON_NAMES = {"true": True, "false": False, "null": None}


@dataclass(frozen=True)
class KeyDefault:
    """The value a check's key has in the defaults, ``ABSENT`` when they do not set it."""

    value: object


def parse_literal(node: ast.expr) -> object:
    """A JSON value written in a match expression."""
    if isinstance(node, ast.Name) and node.id in JSON_NAMES:
        return JSON_NAMES[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        raise ValueError(f"write {str(node.value).lower()} for a boolean, not {node.value}")
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        number = parse_literal(node.operand)
        if isinstance(number, int | float) and not isinstance(number, bool):
            return -number
    if isinstance(node, ast.List):
        return [parse_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        entries: dict[str, object] = {}
        for key, item in zip(node.keys, node.values, strict=True):
            # A None key is a ** unpacking.
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError(f"{ast.unparse(node)} is not a JSON literal: its keys must be strings")
            entries[key.value] = parse_literal(item)
        return entries
    raise ValueError(f"{ast.unparse(node)} is not a JSON literal")


def parse_matcher(source: str, default: KeyDefault | None) -> Matcher:
    """The matcher a ``match`` expression names.

    ``default`` is what ``IsDefault()`` matches; None for a key with ``*`` or ``**``, where it names no one value.
    Raises ValueError for an expression that is not one matcher call.
    """
    try:
        expression = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            f"cannot parse {source!r}: {error.msg} at line {error.lineno}, column {error.offset}"
        ) from error

    def build(node: ast.expr) -> Matcher:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError(f"{ast.unparse(node)} is not a matcher call such as Eq(1)")
        name, args = node.func.id, node.args
        known = [*COMBINING, *WRAPPING, *TEXT, *NO_ARGUMENT, *LITERAL_MATCHERS]
        if name not in known:
            raise ValueError(f"unknown matcher {name}; the matchers are {', '.join(known)}")
        if name in COMBINING:
            if not args:
                raise ValueError(f"{name} takes one or more matchers")
            return COMBINING[name](tuple(build(arg) for arg in args))
        if name in NO_ARGUMENT or name == "IsDefault":
            if args:
                raise ValueError(f"{name} takes no arguments: {ast.unparse(node)}")
        elif len(args) != 1:
            raise ValueError(f"{name} takes one argument: {ast.unparse(node)}")
        if name in WRAPPING:
            return WRAPPING[name](build(args[0]))
        if name in NO_ARGUMENT:
            return NO_ARGUMENT[name]()
        if name == "IsDefault":
            if default is None:
                raise ValueError("IsDefault() needs a key without * or **, which names one value of the defaults")
            return Eq(default.value)
        value = parse_literal(args[0])
        if name in TEXT:
            if not isinstance(value, str):
                raise ValueError(f"{name} takes a string: {ast.unparse(node)}")
            if name == "MatchesRegex":
                try:
                    re.compile(value)
                except re.error as error:
                    raise ValueError(f"MatchesRegex pattern {value!r} is not a regular expression: {error}") from error
            return TEXT[name](value)
        if name == "Ge":
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"Ge takes a number: {ast.unparse(node)}")
            return Ge(value)
        if name == "IsIn":
            if not isinstance(value, list):
                raise ValueError(f"IsIn takes a list: {ast.unparse(node)}")
            return IsIn(tuple(value))
        return Eq(value)

    return build(expression)


def select(
    data: object, segments: tuple[str, ...], path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], object]]:
    """Each concrete key path that ``segments`` names in ``data``, with its value there or ``ABSENT``."""
    if not segments:
        return [(path, data)]
    first, rest = segments[0], segments[1:]
    children: list[tuple[str, object]] = []
    if isinstance(data, dict):
        children = [(str(key), item) for key, item in data.items()]
    elif isinstance(data, list):
        children = [(str(index), item) for index, item in enumerate(data)]
    if first == "**":
        return select(data, rest, path) + [
            found for key, item in children for found in select(item, segments, (*path, key))
        ]
    if first == "*":
        return [found for key, item in children for found in select(item, rest, (*path, key))]
    child = data.get(first, ABSENT) if isinstance(data, dict) else ABSENT
    return select(child, rest, (*path, first))
