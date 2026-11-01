"""ATIF trajectories as the token IDs that a training component trains: the conversation in the model's chat template.

The serving inference service returns no tokens and no log-probabilities. A training component that trains tokens
renders each ATIF trajectory sample itself with its model's chat template (``render_trajectory``), and computes the
log-probabilities that its objective needs with its own engine.

``ChatTemplate`` is the interface of a chat template, and ``TokenizerChatTemplate`` the one of a Hugging Face
tokenizer. The message helpers bring provider messages into the form that chat templates expect.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.job_data import JobData, Sample, TrajectorySample
from reef.core.trajectories import atif_messages


def flatten_content(content: Any) -> str:
    """The plain text of an OpenAI-style message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, Mapping) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def normalize_messages_for_template(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages as a chat template expects them: text content, chat roles, tool arguments as objects."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "developer":
            entry["role"] = "system"
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            entry["content"] = flatten_content(content)
        if entry.get("tool_calls"):
            entry["tool_calls"] = [normalize_tool_call(call) for call in entry["tool_calls"]]
        normalized.append(entry)
    return normalized


def normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """A tool call with its function arguments as an object, as chat templates render them."""
    normalized = dict(call)
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        normalized["function"] = function
    return normalized


class ChatTemplate(ABC):
    """A model's chat template: messages and tools in, token IDs out.

    Purpose: ``render_trajectory`` renders an ATIF trajectory with it; a training component makes one from its
    model's tokenizer.
    """

    @abstractmethod
    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None, *, prompt: bool
    ) -> list[int]:
        """The token IDs of ``messages`` with ``tools``; with ``prompt``, followed by the generation prompt of the
        next assistant turn. ``ValueError``: the template cannot render them."""

    @abstractmethod
    def with_arguments(self, arguments: Mapping[str, Any]) -> ChatTemplate:
        """This template with ``arguments`` over its own template arguments, key by key, as a request's
        ``chat_template_kwargs`` override the serving engine's."""


class TokenizerChatTemplate(ChatTemplate):
    """The chat template of a Hugging Face tokenizer, with fixed template arguments.

    Values: ``tokenizer`` has ``apply_chat_template``; ``template_kwargs`` are the arguments of every render, such as
    ``{"enable_thinking": False}``. They must equal the arguments that the serving engine rendered with.
    """

    def __init__(self, tokenizer: Any, template_kwargs: Mapping[str, Any] | None = None) -> None:
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})

    def with_arguments(self, arguments: Mapping[str, Any]) -> TokenizerChatTemplate:
        return TokenizerChatTemplate(self.tokenizer, {**self.template_kwargs, **arguments})

    def render(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None, *, prompt: bool
    ) -> list[int]:
        # tokenize=True and return_dict=False return a flat list of token IDs. A template raises what its Jinja code
        # raises, so every failure becomes the ValueError of a conversation that cannot render.
        try:
            ids = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools) if tools else None,
                tokenize=True,
                add_generation_prompt=prompt,
                return_dict=False,
                **self.template_kwargs,
            )
        except Exception as exc:
            raise ValueError(f"the chat template cannot render the conversation: {exc}") from exc
        return [int(token) for token in ids]


@dataclass(frozen=True)
class RenderedTrajectory:
    """The token IDs of a rendered trajectory and the loss mask of its trained turns.

    Values: ``tokens`` is a non-empty list of token IDs. ``loss_mask`` holds one 0 or 1 for each token after the
    generation prompt of the first trained turn, so ``0 < len(loss_mask) < len(tokens)``: 1 on the tokens of each
    trained turn, 0 on the tokens between them. The constructor checks the lengths.
    Purpose: ``render_trajectory`` makes it; a training component writes it into the sample's
    ``extra.reef.training`` as ``tokens`` and ``loss_mask``.
    """

    tokens: list[int]
    loss_mask: list[int]

    def __post_init__(self) -> None:
        if not 0 < len(self.loss_mask) < len(self.tokens):
            raise ValueError("a rendered trajectory has a prompt and one or more trained tokens")


def trajectory_tools(trajectory: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The tools of an ATIF trajectory, as a chat template takes them: its agent's ``tool_definitions``, or None."""
    agent = trajectory.get("agent")
    tools = agent.get("tool_definitions") if isinstance(agent, Mapping) else None
    return [dict(tool) for tool in tools] if isinstance(tools, list) and tools else None


def trained_turns(trajectory: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """The chat messages of an ATIF trajectory, normalized for a chat template, and the positions of its trained
    turns: the agent steps that are not copied context (``render_trajectory``).

    ``ValueError``: ``is_copied_context`` is not a boolean; the trajectory has no agent step that trains, or its first
    trained step has no step before it. A trajectory that passes may still fail to render, with a chat template that
    does not render each trained turn as a prefix of the whole conversation.
    """
    steps = trajectory.get("steps") or []
    conversation = normalize_messages_for_template(atif_messages(trajectory))
    # atif_messages turns each agent step into exactly one assistant message, in step order.
    copied = [step.get("is_copied_context", False) for step in steps if step["source"] == "agent"]
    if any(not isinstance(value, bool) for value in copied):
        raise ValueError("ATIF is_copied_context must be a boolean")
    agent_turns = [index for index, message in enumerate(conversation) if message["role"] == "assistant"]
    trained = [index for index, is_copied in zip(agent_turns, copied, strict=True) if not is_copied]
    if not trained:
        raise ValueError("a trajectory needs one or more agent steps to train; copied context steps train nothing")
    if trained[0] == 0:
        raise ValueError("a trajectory needs a system or user step before its first trained agent step")
    return conversation, trained


def render_trajectory(template: ChatTemplate, trajectory: Mapping[str, Any]) -> RenderedTrajectory:
    """Render an ATIF trajectory with ``template``, with every agent step trained that is not copied context.

    An agent step with ``is_copied_context: true`` (ATIF's mark for a step copied from earlier context) is context:
    it renders, and it trains nothing. Every other agent step trains. The tools are the agent's
    ``tool_definitions``. The tokens are the conversation up to its last trained turn; the steps after it train
    nothing and are left out. The loss mask covers the tokens after the first trained turn's generation prompt
    (``RenderedTrajectory``).

    ``ValueError``: ``is_copied_context`` is not a boolean; the trajectory has no agent step that trains, or its first
    trained step has no step before it; or the chat template does not render each trained turn as a prefix of the
    whole conversation (a template that drops earlier turns' reasoning does not).
    """
    conversation, trained = trained_turns(trajectory)
    tools = trajectory_tools(trajectory)
    conversation = conversation[: trained[-1] + 1]
    tokens = template.render(conversation, tools, prompt=False)
    spans: list[tuple[int, int]] = []
    for index in trained:
        # The turn's tokens are those between its generation prompt and its end, both prefixes of the whole.
        prompt_ids = template.render(conversation[:index], tools, prompt=True)
        turn_ids = template.render(conversation[: index + 1], tools, prompt=False)
        prefixes = tokens[: len(prompt_ids)] == prompt_ids and tokens[: len(turn_ids)] == turn_ids
        if not prefixes or len(turn_ids) <= len(prompt_ids):
            raise ValueError(
                "the chat template does not render each turn as a prefix of the whole conversation, "
                "so its agent turns cannot all train"
            )
        spans.append((len(prompt_ids), len(turn_ids)))
    prompt_length = spans[0][0]
    loss_mask = [0] * (len(tokens) - prompt_length)
    for start, end in spans:
        loss_mask[start - prompt_length : end - prompt_length] = [1] * (end - start)
    return RenderedTrajectory(tokens, loss_mask)


@dataclass(frozen=True)
class TeacherRequest:
    """What a distillation teacher reads before the student's reply, as a sample's ``extra.reef.teacher`` holds it.

    Values: ``messages`` is None, for a teacher that reads the student's own sequence, or a list of chat messages
    that a chat template takes; ``tools`` is None or a list of tools; ``max_prompt_tokens`` and ``max_tokens`` are
    non-negative ints, 0 for no limit. The constructor checks the types.
    Purpose: a distillation stage writes it into a sample (``as_json``); the training component reads it
    (``of_sample``) and renders the teacher sequence (``teacher_tokens``).
    """

    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    max_prompt_tokens: int = 0
    max_tokens: int = 0

    def __post_init__(self) -> None:
        if self.messages is not None and not isinstance(self.messages, list):
            raise ValueError("teacher messages must be a list or None")
        if self.tools is not None and not isinstance(self.tools, list):
            raise ValueError("teacher tools must be a list or None")
        for limit in (self.max_prompt_tokens, self.max_tokens):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("teacher token limits must be non-negative ints")

    def as_json(self) -> dict[str, Any]:
        """The JSON object of the request, as ``extra.reef.teacher``."""
        return {
            "messages": self.messages,
            "tools": self.tools,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def of_sample(cls, sample: TrajectorySample) -> TeacherRequest | None:
        """The teacher request of ``sample``, or None for a sample that distils from no teacher."""
        teacher = sample.metadata.get("teacher")
        if teacher is None:
            return None
        if not isinstance(teacher, Mapping):
            raise ValueError("extra.reef.teacher must be an object")
        return cls(
            teacher.get("messages"),
            teacher.get("tools"),
            teacher.get("max_prompt_tokens", 0),
            teacher.get("max_tokens", 0),
        )

    def teacher_tokens(self, template: ChatTemplate, student: RenderedTrajectory) -> list[int] | None:
        """The teacher sequence of the rendered student: the student's tokens for ``messages`` None, else
        ``messages`` rendered with the generation prompt, cut to ``max_prompt_tokens``, then the student's reply
        tokens verbatim, so a teacher pass scores the student's own tokens; None when it is longer than
        ``max_tokens``."""
        if self.messages is None:
            tokens = list(student.tokens)
        else:
            prompt = template.render(self.messages, self.tools, prompt=True)
            if self.max_prompt_tokens:
                prompt = prompt[: self.max_prompt_tokens]
            tokens = [*prompt, *student.tokens[-len(student.loss_mask) :]]
        return None if self.max_tokens and len(tokens) > self.max_tokens else tokens


def teacher_candidates(sample: TrajectorySample) -> tuple[tuple[str, TeacherRequest], ...]:
    """The teacher candidates of ``sample`` (``extra.reef.teacher_candidates``): each a hint and the request that a
    teacher reads with it, as a list of ``{"hint", **TeacherRequest.as_json()}`` objects; empty when it names none.

    A recipe whose teacher reads one of several prompts, such as OpenClaw-RL's hint candidates, writes them; the
    training component renders each (``rendered_job_data``).
    """
    candidates = sample.metadata.get("teacher_candidates")
    if candidates is None:
        return ()
    if not isinstance(candidates, list) or not all(isinstance(candidate, Mapping) for candidate in candidates):
        raise ValueError("extra.reef.teacher_candidates must be a list of objects")
    return tuple(
        (
            str(candidate.get("hint", "")),
            TeacherRequest(
                candidate.get("messages"),
                candidate.get("tools"),
                candidate.get("max_prompt_tokens", 0),
                candidate.get("max_tokens", 0),
            ),
        )
        for candidate in candidates
    )


def recorded_template_arguments(sample: TrajectorySample) -> Mapping[str, Any]:
    """The ``chat_template_kwargs`` of the request of the last call that ``sample`` recorded (``extra.reef.records``),
    or none: the arguments with which the serving engine rendered that call over its own."""
    records = sample.metadata.get("records")
    last = records[-1] if isinstance(records, list) and records else None
    payload = last.get("payload") if isinstance(last, Mapping) else None
    arguments = payload.get("chat_template_kwargs") if isinstance(payload, Mapping) else None
    return arguments if isinstance(arguments, Mapping) else {}


def needs_rendering(data: JobData) -> bool:
    """Whether ``data`` holds a trajectory sample without ``tokens``, which ``rendered_job_data`` renders."""
    return any(isinstance(sample, TrajectorySample) and not sample.training.get("tokens") for sample in data.samples)


def rendered_job_data(data: JobData, template: ChatTemplate) -> tuple[JobData, tuple[int, ...]]:
    """``data`` with each trajectory sample that has no ``tokens`` rendered with ``template`` (``render_trajectory``),
    and the positions in ``data.samples`` of the trajectories that cannot render, which the result leaves out.

    A rendered sample gets ``tokens`` and ``loss_mask`` in its ``extra.reef.training`` and no log-probabilities: the
    training component computes them with its own weights. A sample with a teacher request (``TeacherRequest``) also
    gets its ``teacher_tokens``; one whose teacher sequence is longer than the request allows is left out, as is one
    whose tokens are more than its ``extra.reef.max_tokens``, when it names a positive one. A sample with teacher
    candidates (``teacher_candidates``) gets ``teacher_candidates``: each candidate's hint and teacher sequence; a
    candidate whose request fails to render drops, and a sample whose candidates all drop is left out. A trajectory
    with tokens, such as a task rollout of the component, and a task sample stay as they are. A sample renders with
    ``template`` and the ``chat_template_kwargs`` of its last recorded request over the template's own arguments
    (``recorded_template_arguments``), as the serving engine rendered it.
    """
    samples: list[Sample] = []
    left_out: list[int] = []
    for index, sample in enumerate(data.samples):
        if not isinstance(sample, TrajectorySample) or sample.training.get("tokens"):
            samples.append(sample)
            continue
        arguments = recorded_template_arguments(sample)
        sample_template = template.with_arguments(arguments) if arguments else template
        try:
            rendered = render_trajectory(sample_template, sample.trajectory)
        except ValueError:
            left_out.append(index)
            continue
        max_tokens = sample.metadata.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0 and len(rendered.tokens) > max_tokens:
            left_out.append(index)
            continue
        teacher = TeacherRequest.of_sample(sample)
        fields: dict[str, Any] = {"tokens": rendered.tokens, "loss_mask": rendered.loss_mask}
        if teacher is not None:
            teacher_tokens = teacher.teacher_tokens(sample_template, rendered)
            if teacher_tokens is None:
                left_out.append(index)
                continue
            fields["teacher_tokens"] = teacher_tokens
        candidates = teacher_candidates(sample)
        if candidates:
            rendered_candidates = []
            for hint, request in candidates:
                # A candidate whose prompt the template cannot render drops, as a failed hint does.
                try:
                    candidate_tokens = request.teacher_tokens(sample_template, rendered)
                except ValueError:
                    continue
                if candidate_tokens is not None:
                    rendered_candidates.append({"hint": hint, "teacher_tokens": candidate_tokens})
            if not rendered_candidates:
                left_out.append(index)
                continue
            fields["teacher_candidates"] = rendered_candidates
        samples.append(sample.with_training(**fields))
    return JobData(data.data_id, tuple(samples), request=data.request), tuple(left_out)


__all__ = [
    "ChatTemplate",
    "RenderedTrajectory",
    "TeacherRequest",
    "TokenizerChatTemplate",
    "flatten_content",
    "needs_rendering",
    "normalize_messages_for_template",
    "normalize_tool_call",
    "recorded_template_arguments",
    "render_trajectory",
    "rendered_job_data",
    "teacher_candidates",
    "trained_turns",
    "trajectory_tools",
]
