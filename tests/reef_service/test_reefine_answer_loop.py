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
    ENTRIES,
    NODES,
    REQUEST,
    REVIEW,
    SHORT,
    Model,
    designed,
    extension,
    failure_of,
    skill,
)

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
    assert proposal.notes["dropped_attempts"] == ["answer 2: the reply holds no usable entry"]
    # A design that says no entry can deliver the request is an answer, not a slip: it is not asked again.
    model = Model(designed())
    proposal = evolution.propose(NODES, (), model, requests=(REQUEST,), entries=ENTRIES)
    assert proposal.notes == {
        "design": "The user wants the tests run before every answer. Trigger: every task; "
        "no state. Nothing to set up.",
        "failure": "the reply holds no usable entry",
    }
    assert model.calls == 2


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
