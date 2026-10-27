"""Guarantees of the harness source tree: the tree and patch values, the layout, and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.tree import (
    HarnessTree,
    TreeKind,
    TreeLocation,
    TreePatch,
    ValidationResult,
    locate,
    source_path,
    validate_patch,
    validate_tree,
)
from reef.harness.tree.layout import ALWAYS_REVIEWED_KINDS
from reef.harness.tree.tree import normalized_path

PI = get_adapter("pi")
NATIVE = get_adapter("native")

TOOL = (
    "def run(args, workdir):\n    return str(args.get('text', '')).upper()\n\n\n"
    "NAME = 'shout'\nDESCRIPTION = 'Upper-case a string.'\n"
    "PARAMETERS = {'type': 'object', 'properties': {'text': {'type': 'string'}}}\nCAPABILITIES = []\n"
)
GRAPH = {
    "name": "main",
    "start": "think",
    "stages": {"think": {"kind": "model"}, "act": {"kind": "tools"}, "done": {"kind": "end"}},
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "done"},
        {"from": "act", "when": "done", "to": "think"},
    ],
}


def problems(files: dict[str, str], descriptor=PI) -> tuple[str, ...]:
    return validate_tree(HarnessTree(files), descriptor).problems


# -- the tree and the patch ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["skills/a/SKILL.md", "rules.md", "a"])
def test_a_relative_posix_path_is_a_tree_key(path: str) -> None:
    assert normalized_path(path) == path


@pytest.mark.parametrize("path", ["", "/etc/passwd", "a//b", "./a", "a/../b", "a\\b", "a\x00b", "a/", 3])
def test_any_other_path_is_refused(path) -> None:
    with pytest.raises(ValueError):
        normalized_path(path)


def test_a_tree_is_an_immutable_sorted_value_and_refuses_a_file_that_is_also_a_directory() -> None:
    tree = HarnessTree({"rules.md": "b", "config/primary.json": "{}"})
    assert list(tree.files) == ["config/primary.json", "rules.md"]
    with pytest.raises(TypeError):
        tree.files["x"] = "y"  # type: ignore[index]
    with pytest.raises(ValueError, match="is a file and also the directory"):
        HarnessTree({"skills": "x", "skills/a/SKILL.md": "y"})
    with pytest.raises(ValueError, match="must hold text"):
        HarnessTree({"rules.md": b"bytes"})  # type: ignore[dict-item]
    assert HarnessTree({}).files == {}


def test_apply_writes_and_removes_and_diff_is_its_smallest_inverse() -> None:
    head = HarnessTree({"rules.md": "old", "skills/a/SKILL.md": "a"})
    patch = TreePatch({"rules.md": "new", "skills/a/SKILL.md": None, "skills/b/SKILL.md": "b", "gone.md": None})
    after = head.apply(patch)
    assert dict(after.files) == {"rules.md": "new", "skills/b/SKILL.md": "b"}
    assert dict(head.files) == {"rules.md": "old", "skills/a/SKILL.md": "a"}
    diff = head.diff(after)
    assert dict(diff.changes) == {"rules.md": "new", "skills/a/SKILL.md": None, "skills/b/SKILL.md": "b"}
    assert head.apply(diff) == after
    assert not after.diff(after).changes


def test_a_patch_round_trips_through_json_and_refuses_another_shape() -> None:
    patch = TreePatch({"rules.md": "text", "skills/a/SKILL.md": None})
    assert TreePatch.from_json(json.loads(json.dumps(patch.to_json()))) == patch
    with pytest.raises(ValueError, match="JSON object"):
        TreePatch.from_json(["rules.md"])
    with pytest.raises(ValueError):
        TreePatch.from_json({"../escape": "x"})
    with pytest.raises(ValueError, match="must hold text"):
        TreePatch.from_json({"rules.md": 3})


def test_the_checksum_tells_trees_apart_by_path_and_text() -> None:
    one = HarnessTree({"a": "bc"})
    assert one.checksum() == HarnessTree({"a": "bc"}).checksum()
    assert one.checksum() != HarnessTree({"ab": "c"}).checksum()
    assert one.checksum() != HarnessTree({"a": "bd"}).checksum()


def test_read_and_write_round_trip_and_read_refuses_a_link_and_binary_text(tmp_path: Path) -> None:
    tree = HarnessTree({"rules.md": "rules\n", "skills/a/SKILL.md": "é\n"})
    tree.write(tmp_path / "out")
    assert HarnessTree.read(tmp_path / "out") == tree
    with pytest.raises(ValueError, match="not empty"):
        tree.write(tmp_path / "out")
    with pytest.raises(ValueError, match="does not exist"):
        HarnessTree.read(tmp_path / "missing")
    (tmp_path / "linked").mkdir()
    os.symlink(tmp_path / "out" / "rules.md", tmp_path / "linked" / "rules.md")
    with pytest.raises(ValueError, match="symbolic link"):
        HarnessTree.read(tmp_path / "linked")
    (tmp_path / "binary").mkdir()
    (tmp_path / "binary" / "rules.md").write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="not UTF-8"):
        HarnessTree.read(tmp_path / "binary")


# -- the layout ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "location"),
    [
        ("rules.md", TreeLocation(TreeKind.RULES, "")),
        ("config/primary.json", TreeLocation(TreeKind.CONFIG, "primary")),
        ("skills/notes/SKILL.md", TreeLocation(TreeKind.SKILL, "notes")),
        ("commands/summarize.md", TreeLocation(TreeKind.AGENT_COMMAND, "summarize")),
        ("extensions/a.b.ts", TreeLocation(TreeKind.CODE_EXTENSION, "a.b", ".ts")),
        ("native/tools/shout.py", TreeLocation(TreeKind.NATIVE_TOOL, "shout")),
        ("native/hooks/guard.py", TreeLocation(TreeKind.NATIVE_HOOK, "guard")),
        ("native/graphs/main.json", TreeLocation(TreeKind.NATIVE_GRAPH, "main")),
        ("native/agents/helper.json", TreeLocation(TreeKind.NATIVE_AGENT, "helper")),
        ("native/loops/main.py", TreeLocation(TreeKind.NATIVE_LOOP, "main")),
    ],
)
def test_every_layout_path_locates_to_its_kind_and_back(path: str, location: TreeLocation) -> None:
    assert locate(path) == location
    assert source_path(location) == path


@pytest.mark.parametrize(
    "path", ["RULES.md", "skills/notes.md", "skills/a/b/SKILL.md", "extensions/a.rb", "native/tools/a.ts", "x"]
)
def test_a_path_off_the_layout_locates_to_nothing(path: str) -> None:
    assert locate(path) is None


def test_a_location_holds_a_name_exactly_when_it_is_not_rules_and_a_suffix_only_for_an_extension() -> None:
    with pytest.raises(ValueError):
        TreeLocation(TreeKind.RULES, "x")
    with pytest.raises(ValueError):
        TreeLocation(TreeKind.SKILL, "")
    with pytest.raises(ValueError):
        TreeLocation(TreeKind.SKILL, "a/b")
    with pytest.raises(ValueError):
        TreeLocation(TreeKind.SKILL, "a", ".ts")
    with pytest.raises(ValueError):
        TreeLocation(TreeKind.CODE_EXTENSION, "a", ".rb")
    assert TreeKind.NATIVE_LOOP in ALWAYS_REVIEWED_KINDS


# -- validation ----------------------------------------------------------------------------------------------------


def test_a_seed_like_tree_passes_and_the_result_has_no_problems() -> None:
    result = validate_tree(HarnessTree({"rules.md": "Be brief.\n", "skills/notes/SKILL.md": "# notes\n"}), PI)
    assert result == ValidationResult() and result.passed and result.message() == ""


def test_validation_reports_every_problem_of_the_tree_at_once() -> None:
    found = problems(
        {
            "RULES.md": "x",
            "config/primary.json": "[1]",
            "config/nope.json": "{}",
            "native/tools/shout.py": TOOL,
            "skills/empty/SKILL.md": "  \n",
            "extensions/tracer.js": "export default function () {}",
        }
    )
    assert len(found) == 6
    assert found[0].startswith("RULES.md: not a path of the harness layout")
    assert "config/nope.json: adapter 'pi' declares no config target 'nope'" in found
    assert any(problem.startswith("config/primary.json: must hold a JSON object") for problem in found)
    assert "native/tools/shout.py: adapter 'pi' does not render native_tool files" in found
    assert "skills/empty/SKILL.md: the file is empty" in found
    assert "extensions/tracer.js: adapter 'pi' loads .ts extensions" in found


def test_validation_refuses_credentials_in_config_and_in_free_text() -> None:
    (config,) = problems({"config/primary.json": json.dumps({"providers": {"x": {"apiKey": "sk-123"}}})})
    assert "inline credential" in config and "sk-123" not in config
    (rules,) = problems({"rules.md": "Use sk-abcdef1234567890ABCDEFGH to call it."})
    assert rules.startswith("rules.md: the file carries an inline credential")


def test_validation_refuses_an_unguarded_console_write_in_an_extension() -> None:
    (problem,) = problems({"extensions/voice.ts": 'pi.on("agent_end", () => { console.log("done"); });\n'})
    assert problem.startswith("extensions/voice.ts: the extension writes to the harness's stdout")


def test_native_files_must_declare_their_own_name_and_no_reserved_tool_name() -> None:
    renamed = TOOL.replace("NAME = 'shout'", "NAME = 'yell'")
    (mismatch,) = problems({"native/tools/shout.py": renamed}, NATIVE)
    assert mismatch == "native/tools/shout.py: NAME 'yell' must be the file name 'shout'"
    reserved = TOOL.replace("'shout'", "'harness_try'")
    (problem,) = problems({"native/tools/harness_try.py": reserved}, NATIVE)
    assert problem == "native/tools/harness_try.py: tool name 'harness_try' is reserved for a built-in tool"
    (broken,) = problems({"native/tools/shout.py": "def run(:\n"}, NATIVE)
    assert broken.startswith("native/tools/shout.py: the module does not compile")
    (graph,) = problems({"native/graphs/main.json": json.dumps({**GRAPH, "name": "other"})}, NATIVE)
    assert graph == "native/graphs/main.json: the object's name 'other' must be the file name 'main'"


def test_native_references_must_exist_and_delegation_must_not_cycle() -> None:
    one = {"name": "one", "prompt": "First.", "tools": ["shout", "missing"], "then": ["two"]}
    two = {"name": "two", "prompt": "Second.", "then": ["one"], "graph": "nowhere"}
    graph = {
        **GRAPH,
        "stages": {**GRAPH["stages"], "act": {"kind": "tools", "allow": ["gone"]}},
    }
    found = problems(
        {
            "native/tools/shout.py": TOOL,
            "native/agents/one.json": json.dumps(one),
            "native/agents/two.json": json.dumps(two),
            "native/graphs/main.json": json.dumps(graph),
        },
        NATIVE,
    )
    assert "native graph 'main' stage 'act' allows missing tools: ['gone']" in found
    assert "native agent 'one' names missing tools: ['missing']" in found
    assert "native agent 'two' runs a missing graph 'nowhere'" in found
    assert any("is called in a cycle" in problem for problem in found)


def test_a_tree_holds_at_most_one_native_loop() -> None:
    loop = "def run_turn(ctx):\n    return ctx.model()\n"
    found = problems({"native/loops/a.py": loop, "native/loops/b.py": loop}, NATIVE)
    assert found == ("one native loop per tree: native/loops/a.py, native/loops/b.py",)


def test_a_tree_whose_files_pass_must_still_render() -> None:
    # pi's models target refuses a provider other than Reef's: the file passes its own checks, the render does not.
    found = problems({"config/models.json": json.dumps({"providers": {"other": {}}})})
    assert len(found) == 1 and "fails the check on providers" in found[0]


def test_a_patch_that_changes_nothing_is_a_problem_and_a_real_one_is_the_tree_it_makes() -> None:
    head = HarnessTree({"rules.md": "Be brief.\n"})
    for empty in (TreePatch({}), TreePatch({"rules.md": "Be brief.\n"}), TreePatch({"skills/x/SKILL.md": None})):
        assert validate_patch(head, empty, PI).problems == ("the patch changes nothing",)
    assert validate_patch(head, TreePatch({"rules.md": "Be briefer.\n"}), PI).passed
    assert validate_patch(head, TreePatch({"RULES.md": "x"}), PI).problems[0].startswith("RULES.md: not a path")


def test_a_validation_result_holds_distinct_non_empty_messages() -> None:
    assert ValidationResult(("a", "b")).message() == "a; b"
    with pytest.raises(ValueError):
        ValidationResult(("",))
    with pytest.raises(ValueError):
        ValidationResult(("a", "a"))
