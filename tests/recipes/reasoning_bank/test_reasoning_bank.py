"""ReasoningBank on the workspace protocol: runs become memory items in the bank and Agent Skills in the tree.

Each step runs on a real workspace (``Workspace.create``) with the backend's own services (``StepServices``), so every
model call goes through the step's budgeted bindings and ``services.check`` is the pi validation. The judge and the
extractor are canned models that answer in order and keep every prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_methods import QueueChat
from tests.support.harness_steps import MODEL, PI_SEED, step_services

from recipes.reasoning_bank import ReasoningBank
from recipes.reasoning_bank.method import FOCUS
from reef.core.job_data import TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, TreePatch, validate_tree
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2sample.traces import InferenceSamples
from reef.train.harness_backend import StepContext, Workspace
from reef.train.harness_backend.workspace import StepOutput

PI = get_adapter("pi")
EXAMPLE = Path(__file__).resolve().parents[3] / "recipes" / "reasoning_bank" / "examples" / "serve.yaml"
#: The pi seed with a skill of its own, which the method leaves as it is.
SEED = PI_SEED.apply(TreePatch({"skills/review/SKILL.md": "Review the diff before you commit.\n"}))


def recorded_call(record: str, task: str, score: float | None = None, *, calls_tool: bool = False) -> TrajectorySample:
    """One recorded model call on ``task``: it ends with a reply, or with a tool call when ``calls_tool``."""
    message: dict[str, Any] = {"role": "assistant", "content": f"my answer to {task}"}
    if calls_tool:
        message["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    payload = {"messages": [{"role": "user", "content": task}], "response": {"choices": [{"message": message}]}}
    return recorded_trajectory(record, payload, score)


def judged(outcome: str) -> str:
    return json.dumps({"reason": "the transcript shows it", "outcome": outcome})


def extracted(*titles: str) -> str:
    items = [{"title": title, "description": f"Use when {title} helps.", "content": f"- {title}."} for title in titles]
    return json.dumps({"items": items})


def run_step(
    tmp_path: Path,
    judge: QueueChat,
    extractor: QueueChat,
    samples: list[TrajectorySample],
    *,
    number: int = 1,
    tree: HarnessTree = SEED,
    state: dict[str, str] | None = None,
    **config: Any,
) -> tuple[Workspace, StepOutput]:
    """Step ``number`` of a ReasoningBank whose judge and extractor are the named models ``judge`` and ``extract``."""
    workspace = Workspace.create(
        tmp_path / "steps" / str(number),
        history_dir=tmp_path / "history",
        step=number,
        tree=tree,
        descriptor=PI,
        tasks=(),
        repeats=0,
        task_directories=False,
        samples=samples,
        state_files=state,
    )
    services = step_services(workspace, ModelBindings(served=MODEL, named={"judge": judge, "extract": extractor}))
    method = ReasoningBank.from_config({"judge_model": "judge", "extract_model": "extract", **config})
    method.step(StepContext(workspace, services))
    return workspace, workspace.read_output()


def bank_of(workspace: Workspace) -> list[dict[str, Any]]:
    return json.loads((workspace.state / "memory.json").read_text(encoding="utf-8"))


def test_a_step_turns_runs_into_skills_that_pi_accepts_and_judges_only_the_unscored_runs(tmp_path: Path) -> None:
    # A reply may wrap its JSON in reasoning or in a code fence.
    judge = QueueChat(f"<think>Is {{the parser}} fixed? No.</think>\n{judged('failure')}")
    extractor = QueueChat(
        f"```json\n{extracted('Check the answer format')}\n```", extracted("Run the tests", "Read the error")
    )
    samples = [
        recorded_call("i1", "sum the column", 1.0),
        recorded_call("i2", "fix the parser"),
        recorded_call("i3", "fix the parser", calls_tool=True),
    ]
    workspace, result = run_step(tmp_path, judge, extractor, samples)
    # The scored run needs no judge, and a call that ends with a tool call is the middle of a run.
    assert (
        len(judge.prompts) == 1 and "fix the parser" in judge.prompts[0] and "sum the column" not in judge.prompts[0]
    )
    assert FOCUS["success"] in extractor.prompts[0] and FOCUS["failure"] in extractor.prompts[1]
    assert (result.notes["traces"], result.notes["runs"], result.notes["judged"]) == (3, 2, 1)
    assert result.notes["outcomes"] == {"success": 1, "failure": 1}
    assert [(item["slug"], item["outcome"], item["step"]) for item in bank_of(workspace)] == [
        ("check-the-answer-format", "success", 1),
        ("run-the-tests", "failure", 1),
        ("read-the-error", "failure", 1),
    ]
    # The tree is the seed and one skill per item, and pi accepts it.
    tree = workspace.tree()
    skills = [f"skills/rb-{item['slug']}/SKILL.md" for item in bank_of(workspace)]
    assert set(skills) <= set(tree.files) and tree.apply(TreePatch(dict.fromkeys(skills))) == SEED
    assert validate_tree(tree, PI).passed
    _, header, body = tree.files["skills/rb-run-the-tests/SKILL.md"].split("---\n", 2)
    assert yaml.safe_load(header) == {"name": "rb-run-the-tests", "description": "Use when Run the tests helps."}
    assert body == "\n# Run the tests\n\n- Run the tests.\n"


def test_the_bank_carries_over_a_reused_title_replaces_its_item_and_the_oldest_leave_past_the_cap(
    tmp_path: Path,
) -> None:
    first, _ = run_step(
        tmp_path, QueueChat(), QueueChat(extracted("Alpha", "Beta")), [recorded_call("i1", "one", 1.0)], max_items=2
    )
    extractor = QueueChat(extracted("beta!", "Gamma"))
    second, result = run_step(
        tmp_path,
        QueueChat(),
        extractor,
        [recorded_call("i2", "two", 0.0)],
        number=2,
        tree=first.tree(),
        state=first.state_files(),
        max_items=2,
    )
    # The extractor saw the bank's titles, so it could reuse one to replace its item.
    assert "- Alpha\n- Beta" in extractor.prompts[0]
    assert [(item["slug"], item["title"], item["outcome"], item["step"]) for item in bank_of(second)] == [
        ("beta", "beta!", "failure", 2),
        ("gamma", "Gamma", "failure", 2),
    ]
    assert (result.notes["added"], result.notes["replaced"], result.notes["dropped"]) == (
        ["gamma"],
        ["beta"],
        ["alpha"],
    )
    assert sorted(path for path in second.tree().files if path.startswith("skills/rb-")) == [
        "skills/rb-beta/SKILL.md",
        "skills/rb-gamma/SKILL.md",
    ]


def test_a_reply_that_is_not_the_asked_json_skips_its_run_and_an_item_with_a_credential_stays_out(
    tmp_path: Path,
) -> None:
    credential = "sk-" + "a" * 24
    leaky = {"title": "Use the deploy key", "description": "Use when you deploy.", "content": f"Export {credential}."}
    sound = {"title": "Pin the version", "description": "Use when you install a package.", "content": "Pin it."}
    extractor = QueueChat(json.dumps({"items": [{"title": "No description"}]}), json.dumps({"items": [leaky, sound]}))
    samples = [
        recorded_call("i1", "deploy"),
        recorded_call("i2", "install", 1.0),
        recorded_call("i3", "upgrade", 0.0),
    ]
    workspace, result = run_step(tmp_path, QueueChat("It looks done to me."), extractor, samples)
    reasons = [(entry["record"], entry["reason"]) for entry in result.notes["skipped"]]
    assert reasons == [
        ("i1", "the judge's reply is not the asked JSON: it holds no JSON object"),
        ("i2", "the extractor's reply is not the asked JSON: an item lacks a title, a description or a content text"),
        ("i3", "an item carries a credential or an instruction override"),
    ]
    assert [item["slug"] for item in bank_of(workspace)] == ["pin-the-version"]
    assert validate_tree(workspace.tree(), PI).passed


def test_the_example_deploys_the_method_without_rollouts_and_a_bad_setting_is_refused() -> None:
    config = recipe_config_from_file_layout(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
    recipe = HarnessEvolutionRecipe.from_environment({}, config=config)
    assert isinstance(recipe, HarnessEvolutionRecipe) and isinstance(recipe.method, ReasoningBank)
    assert recipe.method.settings.max_items == 50
    assert (recipe.evaluation.policy, recipe.scorer) == ("none", None)
    assert recipe.pipeline_stages is not None and recipe.pipeline_stages.record2sample.plugin_class is InferenceSamples
    with pytest.raises(PluginConfigError, match="max_items_per_trace must be at least 1"):
        ReasoningBank.from_config({"max_items_per_trace": 0})
