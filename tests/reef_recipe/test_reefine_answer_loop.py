"""The request answer loop keeps an answer whose form slipped: it asks again instead of losing the step.

An answer whose JSON does not parse, whose files were all dropped or which
the harness's validation refuses is written again while attempts remain, and
the reason each was dropped rides the notes; a review reply whose JSON a
stray quote broke is asked once more; the review reads the whole design,
and the record keeps the design's last paragraph, its How to use, whole.
An automatic step, with no request, learns one skill from the failing
reports.
"""

from __future__ import annotations

import json

from tests.support._trajectories import recorded_trajectory
from tests.support.harness_methods import (
    DESIGN,
    PLAN_MARKER,
    REQUEST,
    REVIEW,
    SHORT,
    TREE,
    Model,
    answer,
    designed,
    extension,
    failure_of,
    skill,
)

from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBinding, ModelBindings
from reef.recipe.reefine import proposer
from reef.train.harness_backend import ProposalContext

#: An extension pi's validation refuses: it writes to the session's stdout while it has a UI.
LOUD = extension("loud", 'export default function (pi) { console.log("hi"); }\n')

#: The starter skill retired, as the prompt asks the model to remove a file.
RETIRE = {"skills/answer-style/SKILL.md": None}

#: A complete review that also says the answer delivers.
DELIVERED = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})


def test_a_request_that_retires_a_held_file_removes_it() -> None:
    """The text proposer can retire a file by its path, so a request to stop a behavior takes the old file out
    instead of piling a contradicting rule beside it; the review reads the removal, and the prompt offers the form."""
    model = Model(designed(RETIRE), json.dumps(REVIEW))
    proposal = answer(model)
    assert proposal.patch.changes == RETIRE
    review_prompt = model.prompts[2]
    assert '"skills/answer-style/SKILL.md": null' in review_prompt
    assert 'map its path to null, as in {"skills/old/SKILL.md": null}' in model.prompts[1]


def test_a_removal_that_changes_nothing_and_a_path_off_the_layout_are_named_in_the_retry() -> None:
    """A removal of a file the tree does not hold changes nothing, and a path off the layout is dropped: either
    leaves the answer with nothing to apply, and the retry names why, so the model can write it again."""
    model = Model(designed({"skills/no-such-skill/SKILL.md": None}), designed(RETIRE), json.dumps(REVIEW))
    proposal = answer(model)
    assert proposal.patch.changes == RETIRE
    assert proposal.notes["dropped_attempts"] == ["answer 1: the harness refused the files: the patch changes nothing"]
    model = Model(designed({"rules/no such rule.md": None}), designed(RETIRE), json.dumps(REVIEW))
    proposal = answer(model)
    assert proposal.patch.changes == RETIRE
    (dropped,) = proposal.notes["dropped_attempts"]
    assert dropped == (
        "answer 1: the reply holds no usable file: rules/no such rule.md was dropped: "
        "it is not a path of the harness layout"
    )
    assert "rules/no such rule.md was dropped" in model.prompts[2]


def test_files_the_harness_refuses_are_sent_back_and_written_again() -> None:
    model = Model(designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW))
    proposal = answer(model)
    assert list(proposal.patch.changes) == ["skills/run-tests/SKILL.md"]
    (dropped,) = proposal.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the harness refused the files: ") and "extensions/loud.ts" in dropped
    retry = model.prompts[2]
    assert "An earlier answer to this request could not be used: the harness refused the files:" in retry
    assert proposal.notes["attempts"] == 2
    # Refused on every attempt: the step says why, with each dropped answer.
    model = Model(designed(LOUD))
    proposal = answer(model)
    assert failure_of(proposal).startswith("the harness refused the files: ")
    assert proposal.notes["attempts"] == 3 and len(proposal.notes["dropped_attempts"]) == 3


#: An answer a stray quote broke: its JSON does not parse.
BROKEN = '{"design": "a "quoted" word", "files": {"x"'


def test_an_answer_whose_json_does_not_parse_goes_on_to_the_next_attempt() -> None:
    """A stray quote once ended the loop on the kept first answer; now the third answer gets its turn."""
    model = Model(designed(skill("first")), json.dumps(SHORT), BROKEN, designed(skill("third")), json.dumps(REVIEW))
    proposal = answer(model)
    assert list(proposal.patch.changes) == ["skills/third/SKILL.md"]
    assert proposal.notes["attempts"] == 3
    # The reason names where the JSON broke, so the retry prompt tells the model what to close or escape.
    assert proposal.notes["dropped_attempts"] == [
        "answer 2: the reply's JSON does not parse (Expecting ',' delimiter at line 1 column 16)"
    ]
    # A design with no file is an answer on pi, not a slip: it is not asked again.
    model = Model(designed())
    proposal = answer(model)
    assert proposal.notes == {"design": DESIGN, "failure": "the reply holds no usable file"}
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
    proposal = answer(model)
    assert proposal.notes["review"] == REVIEW and "review_failure" not in proposal.notes
    assert model.calls == 4 and "Your previous reply held no JSON object" in model.prompts[3]


def test_the_review_reads_the_whole_design_in_its_own_script_and_the_record_keeps_how_to_use() -> None:
    how_to_use = "How to use: type /chat to enter chat mode; /chat off leaves it."
    chat = chr(0x804A) + chr(0x5929)  # two CJK characters, which the review prompt must keep as they are
    design = f"{chat} " + "x" * 5000 + "\n\n" + how_to_use
    text = f"---\nname: run-tests\ndescription: run the tests\n---\n# {chat}\n\nchat"
    model = Model(designed(skill("run-tests", text), design=design), json.dumps(REVIEW))
    proposal = answer(model)
    kept = proposal.notes["design"]
    assert len(kept) <= proposer.DESIGN_CHARS and kept.endswith(how_to_use) and "[...]" in kept
    review_prompt = model.prompts[2]
    assert design in review_prompt  # the whole design, not the record's cut
    assert f"# {chat}" in review_prompt and "\\u804a" not in review_prompt


#: A claude answer as a model writes it: the design, a command, the rules, the web permission and one requires item.
CLAUDE_FILES = {
    "commands/chat.md": "---\nallowed-tools: [WebSearch]\n---\nChat.",
    "rules.md": "While chat mode is on, only search the web.\n",
    "config/primary.json": json.dumps({"permissions": {"allow": ["WebSearch", "WebFetch"]}}),
}
CLAUDE_REQUIRES = [{"name": "gh", "kind": "binary", "prompt": "Install the GitHub CLI"}]


def test_a_reply_whose_outer_json_broke_is_written_again_not_read_by_a_list_nested_in_it() -> None:
    """A recorded claude answer lacked its closing brace: its nested requires list decoded on its own. A value that
    decodes inside a broken outer one is a fragment, so the reply is a slip and the next answer is asked for, the
    reason naming where the JSON broke."""
    whole = designed(CLAUDE_FILES, requires=CLAUDE_REQUIRES)
    broken = whole[:-1]
    model = Model(broken, whole, DELIVERED)
    proposal = answer(model, "claude")
    assert proposal.patch.changes == CLAUDE_FILES
    assert [dict(item) for item in proposal.requires] == CLAUDE_REQUIRES
    (dropped,) = proposal.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the reply's JSON does not parse (")
    assert "close every object and array" in model.prompts[2]
    # A stray bracket in prose before a whole answer is no slip: the text before it closes what it opened.
    prose = "Notes [draft] follow.\n" + designed(CLAUDE_FILES)
    assert proposer.slipped_json(prose) is None and proposer.slipped_json(broken) is not None
    assert proposer.slipped_json("no json here") is None


def test_a_stray_bracket_in_prose_before_an_answer_with_requires_is_no_slip() -> None:
    prose = "Notes [draft] follow.\n" + designed(CLAUDE_FILES, requires=CLAUDE_REQUIRES)
    assert proposer.slipped_json(prose) is None


def test_a_design_that_says_no_file_can_deliver_is_answered_with_no_change_and_its_limits() -> None:
    """Off pi, a design with no file is an answer, not a failure: the review runs on it, so what the harness notes
    put out of reach reaches the pages, and the step records it under declined. A review that finds a point a file
    could still deliver sends the request back."""
    limits_only = json.dumps(
        {"result": "complete", "delivers": False, "covered": [], "uncovered": [], "limits": ["a mode to enter"]}
    )
    model = Model(designed(), limits_only)
    proposal = answer(model, "terminus")
    assert proposal.patch.changes == {} and "failure" not in proposal.notes
    assert proposal.notes["declined"] == proposer.DECLINED and proposal.notes["design"] == DESIGN
    assert proposal.notes["review"]["limits"] == ["a mode to enter"]
    assert model.answered == 2  # the answer and its review, no retry
    short = json.dumps(
        {"result": "partial", "delivers": False, "covered": [], "uncovered": ["a skill could search with curl"]}
    )
    model = Model(designed(), short, designed(skill("search")), DELIVERED)
    proposal = answer(model, "terminus")
    assert list(proposal.patch.changes) == ["skills/search/SKILL.md"]
    assert "- a skill could search with curl" in model.prompts[3]


def test_an_answer_whose_files_were_all_dropped_is_written_again_with_the_reasons() -> None:
    """A claude config file that sets hooks is dropped by the parser, the answer's only file: that answer is
    written again with the reason, never recorded as a design that declined."""
    hooks = {"config/primary.json": json.dumps({"hooks": {"x": []}})}
    rules = {"rules.md": "While chat mode is on, only search the web.\n"}
    model = Model(designed(hooks), designed(rules), DELIVERED)
    proposal = answer(model, "claude")
    assert proposal.patch.changes == rules and "declined" not in proposal.notes
    retry = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert retry and "config/primary.json was dropped: it changes hooks; a request may change permissions" in retry[0]


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
    proposal = answer(model, "opencode")
    assert proposal.patch.changes == {**MAPPED, **COMMAND}
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


class NotedBinding(ModelBinding):
    """A served model with canned replies that keeps the activity lines a method writes through it."""

    def chat(self, messages, *, timeout_s=None, **params) -> str:
        replies = self.__dict__.setdefault("replies", [])
        if PLAN_MARKER in messages[-1]["content"]:
            return "[]"
        return replies.pop(0) if len(replies) > 1 else replies[0]

    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        self.__dict__.setdefault("notes", []).append((kind, text, failed))


def test_a_refused_answer_is_on_the_activity_while_the_step_runs() -> None:
    """The Activity says an answer was written again, and why, before the step settles, not only in its record."""
    served = NotedBinding(base_url="http://127.0.0.1:1", model="m")
    served.__dict__["replies"] = [designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW)]
    proposer.answer_request(TREE, REQUEST, (), ModelBindings(served=served), "pi")
    ((kind, text, failed),) = served.__dict__["notes"]
    assert (kind, failed) == ("check", True)
    assert text.startswith("answer 1 written again: the harness refused the files: ")


# -- an automatic step: one skill learned from the failing reports ------------------------------------------------


#: One failing report: the request as served, its score and the reporter's feedback.
FAILING = recorded_trajectory("a1", {"messages": [{"role": "user", "content": "[csv] median"}]}, 0.0, "wrong row")


def test_an_automatic_step_learns_one_skill_from_the_failing_reports() -> None:
    """With no request, the text proposer asks once for a skill that would make the failures pass, shows the
    reports fenced as data, and keeps the first skill file of the reply: one change per automatic step."""
    model = Model(json.dumps({"files": {**skill("csv-median"), **skill("second")}}))
    proposal = proposer.learn_from_failures(TREE, (FAILING,), model)
    assert proposal is not None and list(proposal.patch.changes) == ["skills/csv-median/SKILL.md"]
    assert model.calls == 1 and "wrong row" in model.prompt and "answer-style" in model.prompt


def test_an_automatic_step_without_samples_or_a_usable_skill_proposes_nothing() -> None:
    """Nothing to learn from, a reply that writes no skill, one that is not JSON, and a failed call each give no
    proposal, so the step skips; the empty batch costs no model call."""
    model = Model(json.dumps({"files": skill("unused")}))
    empty_batch = ProposalContext(
        step=1,
        tree=TREE,
        descriptor=get_adapter("pi"),
        samples=(),
        sources=(),
        request=None,
        evaluation_tasks=("task",),
        models=model,
    )
    assert proposer.TextProposer().propose(empty_batch) is None and model.calls == 0
    for reply in (json.dumps({"files": {"rules.md": "Answer in one line.\n"}}), "no json at all"):
        assert proposer.learn_from_failures(TREE, (FAILING,), Model(reply)) is None
    assert proposer.learn_from_failures(TREE, (FAILING,), Model(failure=OSError("connection refused"))) is None
