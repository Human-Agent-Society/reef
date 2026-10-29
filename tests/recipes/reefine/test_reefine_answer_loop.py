"""The request answer loop keeps an answer whose form slipped: it asks again instead of losing the step.

An answer whose JSON does not parse, whose files were all dropped or which
the harness's validation refuses is written again while attempts remain, and
the reason each was dropped rides the notes; a review reply whose JSON a
stray quote broke is asked once more; the review reads the whole design,
and the record keeps the design's last paragraph, its How to use, whole.
In a whole step, the answer's tree is the candidate and its design and
review are the step's summary. An automatic step, with no request, learns
one skill from the failing traces.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.recipes.reefine.reefine_support import (
    DESIGN,
    REQUEST,
    REVIEW,
    SHORT,
    TREE,
    Model,
    ModelServices,
    Served,
    answer,
    designed,
    extension,
    failure_of,
    reefine_step,
    skill,
    step_workspace,
)
from tests.support._trajectories import recorded_trajectory

from recipes.beta.reefine import ReefineMethod
from recipes.beta.reefine import answer as reefine_answer
from reef.core.training_request import TrainingRequest
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBindings
from reef.train.types import JobData

#: An extension pi's validation refuses: it writes to the session's stdout while it has a UI.
PI_LAYOUT = get_adapter("pi").layout
LOUD = extension("loud", 'export default function (pi) { console.log("hi"); }\n')

#: The starter skill retired, as the prompt asks the model to remove a file.
RETIRE = {"skills/answer-style/SKILL.md": None}

#: A complete review that also says the answer delivers.
DELIVERED = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})


def test_a_request_that_retires_a_held_file_removes_it() -> None:
    """The text answer can retire a file by its path, so a request to stop a behavior takes the old file out
    instead of piling a contradicting rule beside it; the review reads the removal, and the prompt offers the form."""
    model = Model(designed(RETIRE), json.dumps(REVIEW))
    outcome = answer(model)
    assert outcome.patch.changes == RETIRE
    review_prompt = model.prompts[2]
    assert '"skills/answer-style/SKILL.md": null' in review_prompt
    assert 'map its path to null, as in {"skills/old/SKILL.md": null}' in model.prompts[1]


def test_a_removal_that_changes_nothing_and_a_path_off_the_tree_are_named_in_the_retry() -> None:
    """A removal of a file the tree does not hold changes nothing, and a path no part takes is dropped: either
    leaves the answer with nothing to apply, and the retry names why, so the model can write it again."""
    model = Model(designed({"skills/no-such-skill/SKILL.md": None}), designed(RETIRE), json.dumps(REVIEW))
    outcome = answer(model)
    assert outcome.patch.changes == RETIRE
    assert outcome.notes["dropped_attempts"] == ["answer 1: the files change nothing in the tree"]
    model = Model(designed({"rules/no such rule.md": None}), designed(RETIRE), json.dumps(REVIEW))
    outcome = answer(model)
    assert outcome.patch.changes == RETIRE
    (dropped,) = outcome.notes["dropped_attempts"]
    assert dropped == (
        "answer 1: the reply holds no usable file: rules/no such rule.md was dropped: "
        "a request may write skills/{name}/SKILL.md, rules.md, commands/{name}.md, extensions/{name}.ts"
    )
    assert "rules/no such rule.md was dropped" in model.prompts[2]


def test_files_the_harness_refuses_are_sent_back_and_written_again() -> None:
    model = Model(designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW))
    outcome = answer(model)
    assert list(outcome.patch.changes) == ["skills/run-tests/SKILL.md"]
    (dropped,) = outcome.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the harness refused the files: ") and "extensions/loud.ts" in dropped
    retry = model.prompts[2]
    assert "An earlier answer to this request could not be used: the harness refused the files:" in retry
    assert outcome.notes["attempts"] == 2
    # Refused on every attempt: the step says why, with each dropped answer.
    model = Model(designed(LOUD))
    outcome = answer(model)
    assert failure_of(outcome).startswith("the harness refused the files: ")
    assert outcome.notes["attempts"] == 3 and len(outcome.notes["dropped_attempts"]) == 3


#: An answer a stray quote broke: its JSON does not parse.
BROKEN = '{"design": "a "quoted" word", "files": {"x"'


def test_an_answer_whose_json_does_not_parse_goes_on_to_the_next_attempt() -> None:
    """A stray quote once ended the loop on the kept first answer; now the third answer gets its turn."""
    model = Model(designed(skill("first")), json.dumps(SHORT), BROKEN, designed(skill("third")), json.dumps(REVIEW))
    outcome = answer(model)
    assert list(outcome.patch.changes) == ["skills/third/SKILL.md"]
    assert outcome.notes["attempts"] == 3
    # The reason names where the JSON broke, so the retry prompt tells the model what to close or escape.
    assert outcome.notes["dropped_attempts"] == [
        "answer 2: the reply's JSON does not parse (Expecting ',' delimiter at line 1 column 16)"
    ]
    # A design with no file is an answer on pi, not a slip: it is not asked again.
    model = Model(designed())
    outcome = answer(model)
    assert outcome.notes == {"design": DESIGN, "failure": "the reply holds no usable file"}
    assert model.calls == 2


def test_a_retry_after_an_unusable_answer_keeps_the_last_reviews_findings_after_a_blank_line() -> None:
    """The review of answer 1 found a gap and answer 2 could not be used: answer 3 is asked with both, the review's
    findings first, and the retry starts after a blank line instead of running on from the prompt."""
    model = Model(designed(skill("first")), json.dumps(SHORT), BROKEN, designed(skill("third")), json.dumps(REVIEW))
    answer(model)
    (third,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert "The review found:\n- no retry\n" in third
    assert third.index("The review found:") < third.index("could not be used")
    assert "\n\nAn earlier answer to this request was reviewed and fell short." in third


def test_a_review_reply_a_stray_quote_broke_is_asked_once_more() -> None:
    broken = '{"result": "complete", "covered": ["the "tests" run first"], "uncovered": []}'
    model = Model(designed(skill("run-tests")), broken, json.dumps(REVIEW))
    outcome = answer(model)
    assert outcome.notes["review"] == REVIEW and "review_failure" not in outcome.notes
    assert model.calls == 4 and "Your previous reply held no JSON object" in model.prompts[3]


def test_the_review_reads_the_whole_design_in_its_own_script_and_the_record_keeps_how_to_use() -> None:
    how_to_use = "How to use: type /chat to enter chat mode; /chat off leaves it."
    chat = chr(0x804A) + chr(0x5929)  # two CJK characters, which the review prompt must keep as they are
    design = f"{chat} " + "x" * 5000 + "\n\n" + how_to_use
    text = f"---\nname: run-tests\ndescription: run the tests\n---\n# {chat}\n\nchat"
    model = Model(designed(skill("run-tests", text), design=design), json.dumps(REVIEW))
    outcome = answer(model)
    kept = outcome.notes["design"]
    assert len(kept) <= reefine_answer.DESIGN_CHARS and kept.endswith(how_to_use) and "[...]" in kept
    review_prompt = model.prompts[2]
    assert design in review_prompt  # the whole design, not the record's cut
    assert f"# {chat}" in review_prompt and "\\u804a" not in review_prompt


#: A claude answer as a model writes it: the design, a command, the rules and one requires item.
CLAUDE_FILES = {
    "commands/chat.md": "---\nallowed-tools: [WebSearch]\n---\nChat.",
    "rules.md": "While chat mode is on, only search the web.\n",
}
CLAUDE_REQUIRES = [{"name": "gh", "kind": "binary", "prompt": "Install the GitHub CLI"}]


def test_a_reply_whose_outer_json_broke_is_written_again_not_read_by_a_list_nested_in_it() -> None:
    """A recorded claude answer lacked its closing brace: its nested requires list decoded on its own. A value that
    decodes inside a broken outer one is a fragment, so the reply is a slip and the next answer is asked for, the
    reason naming where the JSON broke."""
    whole = designed(CLAUDE_FILES, requires=CLAUDE_REQUIRES)
    broken = whole[:-1]
    model = Model(broken, whole, DELIVERED)
    outcome = answer(model, "claude")
    assert outcome.patch.changes == CLAUDE_FILES
    assert [dict(item) for item in outcome.requires] == CLAUDE_REQUIRES
    (dropped,) = outcome.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the reply's JSON does not parse (")
    assert "close every object and array" in model.prompts[2]
    # A stray bracket in prose before a whole answer is no slip: the text before it closes what it opened.
    prose = "Notes [draft] follow.\n" + designed(CLAUDE_FILES)
    assert reefine_answer.slipped_json(prose) is None and reefine_answer.slipped_json(broken) is not None
    assert reefine_answer.slipped_json("no json here") is None


def test_a_stray_bracket_in_prose_before_an_answer_with_requires_is_no_slip() -> None:
    prose = "Notes [draft] follow.\n" + designed(CLAUDE_FILES, requires=CLAUDE_REQUIRES)
    assert reefine_answer.slipped_json(prose) is None


def test_a_design_that_says_no_file_can_deliver_is_answered_with_no_change_and_its_limits() -> None:
    """Off pi, a design with no file is an answer, not a failure: the review runs on it, so what the harness notes
    put out of reach reaches the pages, and the step records it under declined. A review that finds a point a file
    could still deliver sends the request back."""
    limits_only = json.dumps(
        {"result": "complete", "delivers": False, "covered": [], "uncovered": [], "limits": ["a mode to enter"]}
    )
    model = Model(designed(), limits_only)
    outcome = answer(model, "terminus")
    assert outcome.patch.changes == {} and "failure" not in outcome.notes
    assert outcome.notes["declined"] == reefine_answer.DECLINED and outcome.notes["design"] == DESIGN
    assert outcome.notes["review"]["limits"] == ["a mode to enter"]
    assert model.answered == 2  # the answer and its review, no retry
    short = json.dumps(
        {"result": "partial", "delivers": False, "covered": [], "uncovered": ["a skill could search with curl"]}
    )
    model = Model(designed(), short, designed(skill("search")), DELIVERED)
    outcome = answer(model, "terminus")
    assert list(outcome.patch.changes) == ["skills/search/SKILL.md"]
    assert "- a skill could search with curl" in model.prompts[3]


def test_an_answer_whose_files_were_all_dropped_is_written_again_with_the_reasons() -> None:
    """A claude config file is dropped by the parser, since a Claude Code plugin carries no settings, and it is the
    answer's only file: that answer is written again with the reason, never recorded as a design that declined."""
    hooks = {"config/primary.json": json.dumps({"hooks": {"x": []}})}
    rules = {"rules.md": "While chat mode is on, only search the web.\n"}
    model = Model(designed(hooks), designed(rules), DELIVERED)
    outcome = answer(model, "claude")
    assert outcome.patch.changes == rules and "declined" not in outcome.notes
    retry = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert retry and "config/primary.json was dropped: a request may write skills/{name}/SKILL.md" in retry[0]


#: An opencode chat mode: the agent's config without a permission map, the command that enters it, and the fix.
AGENT = {"chat": {"mode": "primary", "prompt": "Leave with /agents, choosing build."}}
UNMAPPED = {"config/primary.json": json.dumps({"agent": AGENT})}
COMMAND = {"commands/chat.md": "---\nagent: chat\n---\nChat."}
MAPPED = {
    "config/primary.json": json.dumps({"agent": {"chat": {**AGENT["chat"], "permission": {"*": "deny"}}}}),
}


def test_a_retry_for_a_refused_answer_carries_that_answers_design_and_files() -> None:
    """opencode refused answer 1 for its agent's missing permission map alone; the retry shows that answer's design
    and files, so what it got right (a correct leave path, say) is not lost when the model writes again."""
    model = Model(designed(UNMAPPED, COMMAND), designed(MAPPED, COMMAND), DELIVERED)
    outcome = answer(model, "opencode")
    assert outcome.patch.changes == {**MAPPED, **COMMAND}
    (retry,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert "The refused answer's design and files were" in retry
    assert "Leave with /agents, choosing build." in retry and DESIGN in retry


def test_a_retry_shows_every_key_of_a_refused_config_file_and_cuts_only_its_long_texts() -> None:
    """A refused opencode agent with a long prompt: the retry cuts the prompt, not the file, so the keys after it
    stay in view and the model sees where the missing permission map belongs."""
    agent = {"chat": {"prompt": "Answer from web search only. " * 120, "mode": "primary", "description": "Chat"}}
    unmapped = {"config/primary.json": json.dumps({"agent": agent})}
    mapped = {"chat": {**agent["chat"], "permission": {"*": "deny", "websearch": "allow"}}}
    model = Model(designed(unmapped, COMMAND), designed({"config/primary.json": json.dumps({"agent": mapped})}))
    answer(model, "opencode")
    (retry,) = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert '"mode": "primary", "description": "Chat"' in retry and "Answer from web search only. " * 20 not in retry


def test_a_refused_answer_is_on_the_activity_while_the_step_runs() -> None:
    """The Activity says an answer was written again, and why, before the step settles, not only in its record."""
    services = ModelServices(Model(designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW)))
    reefine_answer.answer_request(TREE, REQUEST, (), services, "pi")
    (line,) = services.calls.activity()
    assert (line["kind"], line["failed"]) == ("check", True)
    assert line["text"].startswith("answer 1 written again: the harness refused the files: ")


def test_the_request_prompt_is_told_the_machine_the_change_runs_on() -> None:
    reported = {"platform": "darwin", "release": "24.6.0", "commands": {"afplay": True, "mpv": False}}
    told = reefine_answer.client_text(TrainingRequest("speak", "s", "r", client=reported))
    assert "platform: darwin 24.6.0" in told and "on its PATH: afplay" in told and "not on its PATH: mpv" in told
    assert "requires item" in told
    unknown = reefine_answer.client_text(TrainingRequest("speak", "s", "r"))
    assert "unknown" in unknown and "WSL 2" in unknown


# -- whole steps: the answer as the step records it, and one skill learned from the failing traces ----------------


#: One failing report: the request as served, its score and the reporter's feedback.
FAILING = recorded_trajectory("a1", {"messages": [{"role": "user", "content": "[csv] median"}]}, 0.0, "wrong row")


def test_a_request_step_publishes_the_answer_with_its_design_and_review_in_plain_text(tmp_path: Path) -> None:
    gh = {"name": "gh", "kind": "binary", "prompt": "Install the GitHub CLI"}
    served = Served(designed(skill("run-tests"), requires=[gh]), json.dumps(REVIEW))
    result, _ = reefine_step(
        tmp_path, ReefineMethod(), JobData("demo:1", (), request=REQUEST), ModelBindings(served=served)
    )
    metrics = result.metrics
    assert metrics["selected"] is True and metrics["patch"] == skill("run-tests")
    assert metrics["method_summary"] == f"Design:\n{DESIGN}\n\nReview: complete\nCovered:\n- the tests run first"
    assert metrics["method_notes"]["design"] == DESIGN and metrics["method_notes"]["review"] == REVIEW
    assert [item["name"] for item in metrics["training_request"]["requires"]] == ["gh"]
    # The step record keeps the candidate the method left, and every model call it made.
    step_dir = Path(metrics["step_record"])
    assert (step_dir / "harness" / "skills" / "run-tests" / "SKILL.md").is_file()
    assert len(json.loads((step_dir / "calls.json").read_text())) == metrics["model_calls"] == 3


def test_a_request_without_a_usable_answer_leaves_the_head_and_says_why(tmp_path: Path) -> None:
    served = Served("I would rather not.")
    result, _ = reefine_step(
        tmp_path, ReefineMethod(), JobData("demo:1", (), request=REQUEST), ModelBindings(served=served)
    )
    metrics = result.metrics
    assert metrics["skipped"] == "the method left the head tree unchanged" and result.artifact is None
    assert metrics["method_summary"] == "No change: the reply holds no JSON object with files"
    assert len(metrics["method_notes"]["dropped_attempts"]) == 3 and len(served.prompts) == 4


def test_an_automatic_step_learns_one_skill_from_its_failing_traces(tmp_path: Path) -> None:
    """With no request, the method asks once for a skill that would make the failures pass, shows the traces fenced
    as data, and keeps the first skill file of the reply: one change per automatic step."""
    served = Served(json.dumps({"files": {**skill("csv-median"), **skill("second")}}))
    result, _ = reefine_step(tmp_path, ReefineMethod(), JobData("demo:1", (FAILING,)), ModelBindings(served=served))
    assert result.metrics["patch"] == skill("csv-median")
    assert result.metrics["method_summary"] == "Learned skills/csv-median/SKILL.md from 1 failing trace."
    (prompt,) = served.prompts
    assert "wrong row" in prompt and "[BEGIN recorded traffic" in prompt


def test_an_automatic_step_without_traces_or_a_usable_skill_changes_nothing(tmp_path: Path) -> None:
    """Nothing to learn from, a reply that writes no skill, one that is not JSON, and a failed call each leave the
    head tree, so the step skips; the empty batch costs no model call."""
    served = Served(json.dumps({"files": skill("unused")}))
    result, _ = reefine_step(tmp_path, ReefineMethod(), JobData("demo:1", ()), ModelBindings(served=served))
    assert result.metrics["skipped"] == "the method left the head tree unchanged" and served.prompts == []
    traces = step_workspace(tmp_path / "failing", request=None, samples=(FAILING,)).traces()
    for reply in (json.dumps({"files": {"rules.md": "Answer in one line.\n"}}), "no json at all"):
        assert reefine_answer.learn_from_failures(TREE, traces, Model(reply), PI_LAYOUT) is None
    assert (
        reefine_answer.learn_from_failures(TREE, traces, Model(failure=OSError("connection refused")), PI_LAYOUT)
        is None
    )
