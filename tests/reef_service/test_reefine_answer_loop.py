"""The request answer loop keeps an answer whose form slipped: it asks again instead of losing the step.

An answer whose JSON does not parse, whose entries were all dropped or which
the harness's admission refuses is written again while attempts remain, and
the reason each was dropped rides the notes; a review reply whose JSON a
stray quote broke is asked once more; the review reads the whole design,
and the record keeps the design's last paragraph, its How to use, whole.
"""

from __future__ import annotations

import json

from reef_service.test_harness_example import (
    DESIGN,
    ENTRIES,
    NODES,
    PLAN_MARKER,
    REQUEST,
    REVIEW,
    SHORT,
    Model,
    designed,
    extension,
    failure_of,
    skill,
)

from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.recipe.reefine import evolution

#: An extension pi's admission refuses: it writes to the session's stdout while it has a UI.
LOUD = extension("loud", 'export default function (pi) { console.log("hi"); }\n')


def test_entries_the_harness_refuses_are_sent_back_and_written_again() -> None:
    model = Model(designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [(m.op, m.id) for m in proposal.mutations] == [("create", "run-tests")]
    (dropped,) = proposal.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the harness refused the entries: ") and "loud" in dropped
    retry = model.prompts[2]
    assert "An earlier answer to this request could not be used: the harness refused the entries:" in retry
    assert proposal.notes["attempts"] == 2
    # Refused on every attempt: the step says why, with each dropped answer.
    model = Model(designed(LOUD))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert failure_of(proposal).startswith("the harness refused the entries: ")
    assert proposal.notes["attempts"] == 3 and len(proposal.notes["dropped_attempts"]) == 3


def test_an_answer_whose_json_does_not_parse_goes_on_to_the_next_attempt() -> None:
    """A stray quote once ended the loop on the kept first answer; now the third answer gets its turn."""
    broken = '[{"design": "a "quoted" word"}, {"id": "x"'
    model = Model(
        designed(skill("first")),
        json.dumps(SHORT),
        broken,
        designed(skill("third")),
        json.dumps(REVIEW),
    )
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert [m.id for m in proposal.mutations] == ["third"]
    assert proposal.notes["attempts"] == 3
    # The reason names where the JSON broke, so the retry prompt tells the model what to close or escape.
    assert proposal.notes["dropped_attempts"] == [
        "answer 2: the reply's JSON does not parse (Expecting ',' delimiter at line 1 column 17)"
    ]
    # A design that says no entry can deliver the request is an answer, not a slip: it is not asked again.
    model = Model(designed())
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes == {
        "design": "The user wants the tests run before every answer. Trigger: every task; "
        "no state. Nothing to set up.",
        "failure": "the reply holds no usable entry",
    }
    assert model.calls == 2


def test_a_retry_after_an_unusable_answer_keeps_the_last_reviews_findings_after_a_blank_line() -> None:
    """The review of answer 1 found a gap and answer 2 could not be used: answer 3 is asked with both, the review's
    findings first, and the retry starts after a blank line instead of running on from the prompt."""
    broken = '[{"design": "a "quoted" word"}, {"id": "x"'
    model = Model(designed(skill("first")), json.dumps(SHORT), broken, designed(skill("third")), json.dumps(REVIEW))
    evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    (third,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert "The review found:\n- no retry\n" in third
    assert third.index("The review found:") < third.index("could not be used")
    assert "\n\nAn earlier answer to this request was reviewed and fell short." in third


def test_a_review_reply_a_stray_quote_broke_is_asked_once_more() -> None:
    broken = '{"result": "complete", "covered": ["the "tests" run first"], "uncovered": []}'
    model = Model(designed(skill("run-tests")), broken, json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes["review"] == REVIEW and "review_failure" not in proposal.notes
    assert model.calls == 4 and "Your previous reply held no JSON object" in model.prompts[3]


def test_the_review_reads_the_whole_design_in_its_own_script_and_the_record_keeps_how_to_use() -> None:
    how_to_use = "How to use: type /chat to enter chat mode; /chat off leaves it."
    chat = chr(0x804A) + chr(0x5929)  # two CJK characters, which the review prompt must keep as they are
    design = f"{chat} " + "x" * 5000 + "\n\n" + how_to_use
    model = Model(designed(skill("run-tests", f"# {chat}\n\nchat"), design=design), json.dumps(REVIEW))
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    kept = proposal.notes["design"]
    assert len(kept) <= 4000 and kept.endswith(how_to_use) and "[...]" in kept
    review_prompt = model.prompts[2]
    assert design in review_prompt  # the whole design, not the record's cut
    assert f"# {chat}" in review_prompt and "\\u804a" not in review_prompt


#: A claude answer as a model writes it: the design, a command, rules and the web permission.
CLAUDE_ANSWER = [
    {"design": DESIGN},
    {
        "id": "chat",
        "name": "agent_command",
        "config": {"name": "chat", "text": "---\nallowed-tools: [WebSearch]\n---\nChat."},
    },
    {"id": "chat-rules", "name": "rules", "config": {"text": "While chat mode is on, only search the web."}},
    {
        "id": "chat-permissions",
        "name": "config",
        "config": {"target": "primary", "data": {"permissions": {"allow": ["WebSearch", "WebFetch"]}}},
    },
]


def test_a_reply_whose_outer_json_broke_is_written_again_not_read_by_a_list_nested_in_it() -> None:
    """The recorded claude answer lacked two closing braces: its nested allow list decoded on its own and the step
    ended with no entry and no retry. A value that decodes inside a broken outer one is a fragment, so the reply is
    a slip and the next answer is asked for, the reason naming where the JSON broke."""
    whole = json.dumps(CLAUDE_ANSWER)
    broken = whole[: whole.rindex("]}}}}]")] + "]}}]"
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(broken, whole, review)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="claude")
    assert [m.id for m in proposal.mutations] == ["chat", "chat-rules", "chat-permissions"]
    (dropped,) = proposal.notes["dropped_attempts"]
    assert dropped.startswith("answer 1: the reply's JSON does not parse (")
    assert "close every object and array" in model.prompts[2]
    # A stray bracket in prose before a whole answer is no slip: the text before it closes what it opened.
    prose = "Notes [draft] follow.\n" + whole
    assert evolution.slipped_json(prose) is None and evolution.slipped_json(broken) is not None
    assert evolution.slipped_json("no json here") is None


def test_a_design_that_says_no_entry_can_deliver_is_answered_with_no_change_and_its_limits() -> None:
    """Off pi, a design with no entry is an answer, not a failure: the review runs on it, so what the harness notes
    put out of reach reaches the pages, and the step records it under declined. A review that finds a point an
    entry could still deliver sends the request back."""
    limits_only = json.dumps(
        {"result": "complete", "delivers": False, "covered": [], "uncovered": [], "limits": ["a mode to enter"]}
    )
    model = Model(designed(), limits_only)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="terminus")
    assert proposal.mutations == () and "failure" not in proposal.notes
    assert proposal.notes["declined"] == evolution.DECLINED and proposal.notes["design"] == DESIGN
    assert proposal.notes["review"]["limits"] == ["a mode to enter"]
    assert model.answered == 2  # the answer and its review, no retry
    short = json.dumps(
        {"result": "partial", "delivers": False, "covered": [], "uncovered": ["a skill could search with curl"]}
    )
    complete = json.dumps({"result": "complete", "delivers": True, "covered": ["search"], "uncovered": []})
    model = Model(designed(), short, designed(skill("search")), complete)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="terminus")
    assert [m.id for m in proposal.mutations] == ["search"]
    assert "- a skill could search with curl" in model.prompts[3]


def test_an_answer_whose_entries_were_all_dropped_is_written_again_with_the_reasons() -> None:
    """A claude config entry that sets hooks is dropped by the parser, the answer's only entry: that answer is
    written again with the reason, never recorded as a design that declined."""
    hooks = {"id": "chat-hooks", "name": "config", "config": {"target": "primary", "data": {"hooks": {"x": []}}}}
    rules = {"id": "chat-rules", "name": "rules", "config": {"text": "While chat mode is on, only search the web."}}
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(designed(hooks), designed(rules), review)
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="claude")
    assert [m.id for m in proposal.mutations] == ["chat-rules"] and "declined" not in proposal.notes
    retry = [prompt for prompt in model.prompts if "An earlier answer to this request" in prompt]
    assert retry and "config 'chat-hooks' was dropped: it sets hooks" in retry[0]


def test_a_retry_for_a_refused_answer_carries_that_answers_design_and_entries() -> None:
    """opencode refused answer 1 for its agent's missing permission map alone; the retry shows that answer's design
    and entries, so what it got right (a correct leave path, say) is not lost when the model writes again."""
    agent = {"chat": {"mode": "primary", "prompt": "Leave with /agents, choosing build."}}
    unmapped = {"id": "chat-agent", "name": "config", "config": {"target": "primary", "data": {"agent": agent}}}
    command = {
        "id": "chat",
        "name": "agent_command",
        "config": {"name": "chat", "text": "---\nagent: chat\n---\nChat."},
    }
    mapped = {"chat": {**agent["chat"], "permission": {"*": "deny", "websearch": "allow"}}}
    fixed = {**unmapped, "config": {"target": "primary", "data": {"agent": mapped}}}
    review = json.dumps({"result": "complete", "delivers": True, "covered": ["chat"], "uncovered": []})
    model = Model(designed(unmapped, command), designed(fixed, command), review)
    evolution.propose(NODES, (), model, requests=(REQUEST,), adapter="opencode")
    (retry,) = [prompt for prompt in model.prompts if "could not be used" in prompt]
    assert "The refused answer's design and entries were" in retry
    assert "Leave with /agents, choosing build." in retry and DESIGN in retry


class _NotedBinding(ModelBinding):
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
    served = _NotedBinding(base_url="http://127.0.0.1:1", model="m")
    served.__dict__["replies"] = [designed(LOUD), designed(skill("run-tests")), json.dumps(REVIEW)]
    evolution.propose(NODES, (), ModelBindings(served=served), requests=(REQUEST,), entries=ENTRIES)
    ((kind, text, failed),) = served.__dict__["notes"]
    assert (kind, failed) == ("check", True)
    assert text.startswith("answer 1 written again: the harness refused the entries: ")
