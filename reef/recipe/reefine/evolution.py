"""Compatibility references for Reefine; proposal execution lives in reef.train.reefine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.trajectory import final_assistant_text
from reef.recipe.reefine.prompts import INSTRUCTIONS
from reef.train.cordis_backend import Mutation, StepProposal
from reef.train.evaluation.reefine_health import evaluate, grade_text
from reef.train.reefine import proposer as implementation
from reef.train.types import TrajectoryItem

Proposal = implementation.Proposal
ANSWERS = {"[health]": "reef-ok"}
HEALTH_TASK_DIRECTORY = str(Path(__file__).with_name("health"))
_ENTRY_NAME = implementation.ENTRY_NAME
REQUEST_KINDS = implementation.REQUEST_KINDS
EXTENSION_ADAPTER = implementation.EXTENSION_ADAPTER
HARNESS_SECTION = INSTRUCTIONS.templates["HARNESS_SECTION"]
NO_EXTENSIONS_SECTION = INSTRUCTIONS.templates["NO_EXTENSIONS_SECTION"]
API_SKILL_NAME = implementation.API_SKILL_NAME
_PREVIEW_CHARS = implementation.PREVIEW_CHARS
_DESIGN_CHARS = implementation.DESIGN_CHARS
EARLIER_TEXT_CHARS = implementation.EARLIER_TEXT_CHARS
REVIEW_RESULTS = implementation.REVIEW_RESULTS
_REVIEW_ITEMS = implementation.REVIEW_ITEMS
_PROMPT_CHARS = implementation.PROMPT_CHARS
_VARIABLE_NAME = implementation.VARIABLE_NAME
_SHELL_VARIABLE = implementation.SHELL_VARIABLE
_ENV_READ = implementation.ENV_READ
_SESSION_ENV = implementation.SESSION_ENV
PLATFORMS_SENTENCE = implementation.PLATFORMS_SENTENCE
REQUEST_PROMPT = INSTRUCTIONS.templates["REQUEST_PROMPT"]
SETUP_SENTENCE = implementation.SETUP_SENTENCE
ENV_WORDS = implementation.ENV_WORDS
NO_SETUP_SENTENCE = implementation.NO_SETUP_SENTENCE
EXTENSIONS_SECTION = INSTRUCTIONS.templates["EXTENSIONS_SECTION"]
REVIEW_PROMPT = INSTRUCTIONS.templates["REVIEW_PROMPT"]
PI_REVIEW_COMMANDS = implementation.PI_REVIEW_COMMANDS
HARNESS_REVIEW_COMMANDS = implementation.HARNESS_REVIEW_COMMANDS
REVIEW_AGAIN = INSTRUCTIONS.templates["REVIEW_AGAIN"]
REQUEST_ATTEMPTS = implementation.REQUEST_ATTEMPTS
UNRESTRICTED_AGENT = implementation.UNRESTRICTED_AGENT
CLAUDE_PREAPPROVED = implementation.CLAUDE_PREAPPROVED
CLAUDE_FETCH_DOMAIN = implementation.CLAUDE_FETCH_DOMAIN
RETRY_SECTION = INSTRUCTIONS.templates["RETRY_SECTION"]
RETRY_UNUSABLE_SECTION = INSTRUCTIONS.templates["RETRY_UNUSABLE_SECTION"]
RETRY_EARLIER_ANSWER = INSTRUCTIONS.templates["RETRY_EARLIER_ANSWER"]
DECLINED = implementation.DECLINED
RETRY_UNDELIVERED = INSTRUCTIONS.templates["RETRY_UNDELIVERED"]
FAILURES_SECTION = INSTRUCTIONS.templates["FAILURES_SECTION"]
PLAN_PROMPT = INSTRUCTIONS.templates["PLAN_PROMPT"]
PLAN_SECTION = INSTRUCTIONS.templates["PLAN_SECTION"]
PLAN_NO_TOOL_SECTION = INSTRUCTIONS.templates["PLAN_NO_TOOL_SECTION"]
API_SECTION = INSTRUCTIONS.templates["API_SECTION"]
WrittenAnswer = implementation.WrittenAnswer
RequestAnswers = implementation.RequestAnswers
UnusableAnswer = implementation.UnusableAnswer
DeclinedAnswer = implementation.DeclinedAnswer
_NO_TEXT_HINT = implementation.NO_TEXT_HINT
request_kinds = implementation.request_kinds

request_config_keys = implementation.request_config_keys

file_name = implementation.file_name

kind_lines = implementation.kind_lines


def harness_section(adapter: str) -> str:
    return implementation.harness_section(adapter, instructions=INSTRUCTIONS)


review_commands = implementation.review_commands


def propose(
    nodes: Sequence[tuple[str, Any]],
    samples: Sequence[TrajectoryItem],
    models: ModelBindings,
    *,
    requests: Sequence[Mapping[str, Any]] = (),
    entries: Sequence[Mapping[str, Any]] = (),
    adapter: str = EXTENSION_ADAPTER,
    rejected: Sequence[Mapping[str, object]] = (),
) -> Mutation | StepProposal | None:
    return implementation.propose(
        nodes,
        samples,
        models,
        requests=requests,
        entries=entries,
        adapter=adapter,
        rejected=rejected,
        instructions=INSTRUCTIONS,
    )


def _answer_request(
    nodes: Sequence[tuple[str, Any]],
    request: Mapping[str, Any],
    samples: Sequence[TrajectoryItem],
    models: ModelBindings,
    entries: Sequence[Mapping[str, Any]],
    adapter: str = EXTENSION_ADAPTER,
) -> StepProposal | None:
    return implementation.answer_request(nodes, request, samples, models, entries, adapter, instructions=INSTRUCTIONS)


def _answer_once(
    prompt: str,
    request: Mapping[str, Any],
    models: ModelBindings,
    nodes: Sequence[tuple[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    own: Sequence[Mapping[str, Any]],
    kinds: Sequence[str] = tuple(REQUEST_KINDS),
    adapter: str = EXTENSION_ADAPTER,
) -> WrittenAnswer | StepProposal | UnusableAnswer | DeclinedAnswer:
    return implementation.answer_once(
        prompt, request, models, nodes, entries, own, kinds, adapter, instructions=INSTRUCTIONS
    )


def declined_answer(
    models: ModelBindings, request: Mapping[str, Any], design: str, own: Sequence[Mapping[str, Any]], adapter: str
) -> DeclinedAnswer:
    return implementation.declined_answer(models, request, design, own, adapter, instructions=INSTRUCTIONS)


entries_in_short = implementation.entries_in_short

config_agents = implementation.config_agents

unrestricted_agents = implementation.unrestricted_agents

widened_permissions = implementation.widened_permissions

_uncovered_count = implementation.uncovered_count

_nothing_to_apply = implementation.nothing_to_apply

client_text = implementation.client_text

reserved_sentence = implementation.reserved_sentence


def _request_prompt(
    nodes: Sequence[tuple[str, Any]],
    request: Mapping[str, Any],
    samples: Sequence[TrajectoryItem],
    models: ModelBindings,
    entries: Sequence[Mapping[str, Any]],
    adapter: str = EXTENSION_ADAPTER,
) -> str:
    return implementation.request_prompt(nodes, request, samples, models, entries, adapter, instructions=INSTRUCTIONS)


_request_mutations = implementation.request_mutations

_tree_ids = implementation.tree_ids


def _review(
    models: ModelBindings,
    request_text: str,
    design: str | None,
    mutations: Sequence[Mutation],
    requires: Sequence[Mapping[str, Any]],
    *,
    adapter: str = EXTENSION_ADAPTER,
) -> tuple[dict[str, Any] | None, str | None]:
    return implementation.review_answer(
        models, request_text, design, mutations, requires, adapter=adapter, instructions=INSTRUCTIONS
    )


_parse_review = implementation.parse_review

_strings_of = implementation.strings_of

design_text = implementation.design_text

kept_design = implementation.kept_design

_parse_design = implementation.parse_design

_undeclared_env = implementation.undeclared_env

_without_reefs_own = implementation.without_reefs_own


def _tool_steps(models: ModelBindings, request_text: str, entries_text: str, tools: str | None = None) -> list[str]:
    return implementation.plan_tool_steps(models, request_text, entries_text, tools, instructions=INSTRUCTIONS)


_timeout_s = implementation.timeout_s

_max_tokens = implementation.max_tokens

failures_text = implementation.failures_text

_ask = implementation.ask

provider_refusal = implementation.provider_refusal

_entry_view = implementation.entry_view

_parse_proposal = implementation.parse_proposal

_parse_requires = implementation.parse_proposed_requires

_screened_requires_item = implementation.screened_requires_item

_trimmed_prompt = implementation.trimmed_prompt

_named_env_check = implementation.named_env_check

_items_in = implementation.items_in

_json_in = implementation.json_in

json_found = implementation.json_found

slipped_json = implementation.slipped_json

open_brackets = implementation.open_brackets

_decoded_at = implementation.decoded_at

_rules_id = implementation.rules_id

parse_config_entry = implementation.parse_config_entry

_parse_entry = implementation.parse_entry

dropped_entry_reasons = implementation.dropped_entry_reasons

misnamed_entries = implementation.misnamed_entries

__all__ = [
    "INSTRUCTIONS",
    "ModelBinding",
    "ModelBindings",
    "Mutation",
    "StepProposal",
    "TrajectoryItem",
    "evaluate",
    "final_assistant_text",
    "grade_text",
    "implementation",
]
