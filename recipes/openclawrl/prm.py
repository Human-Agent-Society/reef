"""The OpenClaw-RL PRM client: judges, hindsight hints, directive teacher.

An upstream-verbatim port of the method's scoring stack
(``openclaw_api_server.py`` / ``openclaw_opd_api_server.py`` at the pinned
commit). Everything talks to one sglang server over its native ``/generate``
API — judge generations and teacher prefills alike, byte-identical request
shapes to the upstream servers. One deliberate deviation: every request has
a timeout, and failures degrade instead of blocking — a failed judge call is
a failed vote feeding the same tie rules, a failed prefill returns ``None``
and the turn trains RL-only.

:mod:`recipes.openclawrl.processor` owns when these are called; this
module owns what one call does.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import functools
import json
import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)

_HINT_JUDGE_SYSTEM_PROMPT = (
    "You are a process reward model used for hindsight hint extraction.\n"
    "You are given:\n"
    "1) The assistant response at turn t.\n"
    "2) The next state at turn t+1, along with its **role**.\n\n"
    "## Understanding the next state's role\n"
    "- role='user': A reply from the user (follow-up, correction, new request, etc.).\n"
    "- role='tool': The return value of a tool the assistant invoked. "
    "This content was NOT available before the assistant's action — "
    "it exists BECAUSE the assistant called the tool. "
    "A successful, non-error tool output generally means the assistant's "
    "action was appropriate; do NOT treat it as information the assistant "
    "should have already known.\n\n"
    "Your goal is to decide whether the next state reveals useful hindsight information\n"
    "that could have helped improve the assistant response at turn t.\n\n"
    "Output format rules (strict):\n"
    "- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n"
    "- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n"
    "  wrapped between [HINT_START] and [HINT_END].\n"
    "- If decision is \\boxed{-1}, do not provide a hint block.\n"
    "- Hint must be concrete and actionable for improving the previous response."
)

_JUDGE_SYSTEM_PROMPT = (
    "You are a process reward model (PRM) evaluating an AI assistant.\n"
    "You will see the assistant's output and the subsequent next state.\n"
    "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
    "at that step, using the next state as evidence.\n\n"
    "## Understanding the next state's role\n"
    "- role='user': A reply from the user.\n"
    "- role='tool': The return value of a tool the assistant invoked. "
    "This content was NOT available before the assistant's action — "
    "it exists BECAUSE the assistant called the tool. "
    "A successful, non-error tool output means the assistant's action worked correctly "
    "and should be scored positively.\n\n"
    "## Scoring rules\n"
    "- \\boxed{1} (good): The next state shows the task progressed as expected — "
    "e.g. the user moves on, says thanks, the environment confirms success, "
    "or a tool returns a successful, non-error result.\n"
    "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
    "incomplete, or unwanted. **Key negative signals include:**\n"
    "  * The user asks the assistant to **redo, retry, or repeat** the same action "
    '("do it again", "try again", "one more time").\n'
    "  * The user requests a **correction or modification** to what the assistant just did "
    '("change X to Y", "no, I meant …", "not that, …", "please fix …").\n'
    "  * The user **rephrases or restates** the same request, implying the assistant "
    "did not understand or execute it correctly.\n"
    "  * The environment returns an **error, failure, or unexpected result** caused "
    "by the assistant's action.\n"
    "- \\boxed{0} (neutral): The next state is ambiguous — e.g. the user gives an "
    "unrelated follow-up that neither confirms nor denies success, or there is "
    "insufficient information to judge.\n\n"
    "## Important\n"
    "A change request IS negative feedback — it means the previous output did not "
    "meet the user's need. Do NOT treat it as a neutral new instruction.\n\n"
    "Think step-by-step, then give your final score inside \\boxed{}."
)


def build_prm_judge_messages(
    response_text: str,
    next_state_text: str,
    next_state_role: str = "user",
) -> list[dict[str, str]]:
    """Upstream's judge prompt, verbatim."""
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_hint_judge_messages(
    response_text: str,
    next_state_text: str,
    next_state_role: str = "user",
) -> list[dict[str, str]]:
    """Upstream's hindsight hint-extraction prompt, verbatim."""
    user = (
        f"## Assistant response (turn t)\n{response_text}\n\n"
        f"## Next state (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n"
        "Now output your decision and (if positive) the hint in the required format."
    )
    return [
        {"role": "system", "content": _HINT_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_prm_score(text: str) -> int | None:
    """Extract the LAST \\boxed{±1|0} from a judge response, or None."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    value = int(matches[-1])
    return value if value in (1, -1, 0) else None


def parse_hint_judgment(text: str) -> tuple[int | None, str]:
    """Extract (\\boxed{±1}, hint) from a hint-judge response."""
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hints = _HINT_RE.findall(text)
    return score, hints[-1].strip() if hints else ""


def majority_vote(scores: list[int | None]) -> float:
    """Upstream's vote rule: plurality wins; a tied plurality or zero valid
    votes yields neutral 0.0."""
    valid = [score for score in scores if score is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


def collect_hint_candidates(votes: list[dict[str, Any]], max_cand: int = 3) -> list[str]:
    """Upstream select rule: ALL substantive accepted hints, deduped,
    shortest-first, capped at ``max_cand``."""
    hints: list[str] = []
    for vote in votes:
        if vote.get("score") == 1 and isinstance(vote.get("hint"), str):
            hint = vote["hint"].strip()
            if len(hint) > 10 and hint not in hints:
                hints.append(hint)
    hints.sort(key=len)
    return hints[:max_cand]


def flatten_content(content: Any) -> str:
    """The plain text of an OpenAI-style message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    """Upstream's hint injection: suffix the last user message."""
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[user's hint / instruction]\n{hint}"}]
    target_idx = None
    for index in range(len(cloned) - 1, -1, -1):
        if cloned[index].get("role") == "user":
            target_idx = index
            break
    if target_idx is None:
        target_idx = len(cloned) - 1
    content = cloned[target_idx].get("content")
    if not isinstance(content, str):
        content = flatten_content(content)
    cloned[target_idx]["content"] = (content + f"\n\n[user's hint / instruction]\n{hint.strip()}").strip()
    return cloned


def normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Upstream's template normalization: flatten content, fix roles/tool args."""
    normalized = []
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "developer":
            entry["role"] = "system"
        raw = entry.get("content")
        if not isinstance(raw, str) and raw is not None:
            entry["content"] = flatten_content(raw)
        if entry.get("tool_calls"):
            entry["tool_calls"] = [_normalize_tool_call(call) for call in entry["tool_calls"]]
        normalized.append(entry)
    return normalized


def _normalize_tool_call(call: dict) -> dict:
    call = dict(call)
    function = call.get("function")
    if isinstance(function, dict):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        call["function"] = function
    return call


def render_response_fallback(message: Any) -> str:
    """The judged assistant turn without a tokenizer: content plus actions."""
    text = flatten_content(message.get("content"))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        text += f"\n[tool_call: {function.get('name')}({function.get('arguments', '')})]"
    return text.strip()


_TOKENIZER_IMPORT_LOCK = threading.Lock()


@functools.lru_cache(maxsize=2)
def load_tokenizer(path: str):
    """The shared judge/teacher tokenizer.

    Imported inside the function because transformers must not tax
    ``import reef``; every load happens on the worker thread, off the
    trainer's.

    The import is serialized. ``transformers`` is a large lazy-import
    package, and concurrent first importers can observe it half-built and
    raise ``ImportError: cannot import name 'AutoTokenizer'`` — which the
    callers here turn into a failed vote or a fallback render, silently
    degrading the judge rather than failing. ``lru_cache`` hides how close
    that is: it happens only on the first burst, so it looks like noise.
    """
    with _TOKENIZER_IMPORT_LOCK:
        from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


async def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> Any | None:
    """POST and parse one sglang request; any failure is ``None``."""
    from aiohttp import ClientError, ClientSession, ClientTimeout

    try:
        async with (
            ClientSession(timeout=ClientTimeout(total=timeout_s)) as session,
            session.post(url, json=payload) as response,
        ):
            if response.status >= 400:
                return None
            return await response.json()
    except (asyncio.TimeoutError, ClientError, OSError, ValueError):
        return None


@dataclass
class PRMJudge:
    """Run m independent PRM judgments per turn.

    ``generate_url`` is the PRM sglang server's native ``/generate``
    endpoint; requests are byte-identical to the upstream API server's. A
    failed or timed-out generation is a failed vote flowing into the same
    tie rules, so a flaky PRM degrades to neutral instead of blocking.
    """

    generate_url: str
    tokenizer_path: str
    m: int = 3
    temperature: float = 0.6
    max_tokens: int = 8192
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    async def evaluate(
        self,
        response_text: str,
        next_state_text: str,
        next_state_role: str = "user",
    ) -> float:
        """The evaluative judgment: the majority of m votes, ties neutral."""
        messages = build_prm_judge_messages(response_text, next_state_text, next_state_role)
        texts = await asyncio.gather(*(self._generate(messages) for _ in range(self.m)))
        return majority_vote([parse_prm_score(text) for text in texts])

    async def evaluate_hint_votes(
        self,
        response_text: str,
        next_state_text: str,
        next_state_role: str = "user",
    ) -> list[dict[str, Any]]:
        """m hindsight hint votes, unselected (the caller applies upstream's
        candidate rule, :func:`collect_hint_candidates`)."""
        messages = build_hint_judge_messages(response_text, next_state_text, next_state_role)
        texts = await asyncio.gather(*(self._generate(messages) for _ in range(self.m)))
        votes = []
        for text in texts:
            score, hint = parse_hint_judgment(text)
            votes.append({"score": score, "hint": hint})
        return votes

    async def _generate(self, messages: list[dict[str, str]]) -> str:
        """One judge generation (upstream's payload, verbatim); "" on failure."""
        try:
            tokenizer = await asyncio.to_thread(load_tokenizer, self.tokenizer_path)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            logger.exception("PRM judge templating failed; the vote fails")
            return ""
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": self.temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": self.max_tokens,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": False,
        }
        data = await _post_json(self.generate_url, payload, self.timeout_s)
        if not isinstance(data, dict):
            return ""
        return str(data.get("text", ""))


class ResponseRenderer:
    """Upstream's response_text: the chat-template diff of the full turn.

    Renders request messages plus the complete assistant message (content,
    reasoning_content, tool_calls) with the shared tokenizer and returns the
    text past ``add_generation_prompt`` — the same string upstream feeds to
    both judges and the directive teacher, covering the tokens the student
    trains on (reasoning included).
    """

    def __init__(self, tokenizer_path: str) -> None:
        self.tokenizer_path = tokenizer_path

    def render(self, request_messages: list[dict], response_message: dict, tools: Any = None) -> str | None:
        try:
            tokenizer = load_tokenizer(self.tokenizer_path)
            response = dict(response_message)
            if response.get("content") is None:
                response["content"] = ""
            norm_msgs = normalize_messages_for_template(list(request_messages))
            norm_resp = normalize_messages_for_template([response])[0]
            prompt_text = tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                [*norm_msgs, norm_resp], tools=tools, tokenize=False, add_generation_prompt=False
            )
            if full_text.startswith(prompt_text):
                return full_text[len(prompt_text) :]
            return full_text
        except Exception:
            logger.exception("response render failed; falling back to visible text")
            return None


@dataclass
class DirectiveTeacher:
    """Candidate builder for the Megatron teacher pass.

    Upstream's topk-select loss requires the Megatron PRM teacher — its
    launcher forces ``OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`` with the
    comment that the inference-side teacher path "does not produce per-cand
    top-K". So the judge ships candidate TOKEN SEQUENCES, never numbers:
    upstream's ``sample.teacher_tokens_candidates``, here one
    ``teacher_tokens`` list per accepted hint (the hint-enhanced prompt's
    re-tokenization plus the native response ids, verbatim) and the
    un-enhanced native ids for the RL-only anchor. The train actor runs the
    K-loop against the frozen-base weights and gathers exact full-vocab
    log-probs at S^q — no prefill window, no synthetic floor.
    """

    tokenizer_path: str

    async def candidate_tokens(
        self,
        enhanced_messages: list[dict],
        *,
        tools: Any = None,
        native_tokens: list[int],
        response_token_count: int,
    ) -> list[int] | None:
        """One hint candidate's token sequence, or ``None`` on templating
        failure (the candidate silently drops; the anchor never fails)."""
        if not (0 < response_token_count < len(native_tokens)):
            return None
        try:
            tokenizer = await asyncio.to_thread(load_tokenizer, self.tokenizer_path)
            normalized = normalize_messages_for_template(enhanced_messages)
            prompt_text = tokenizer.apply_chat_template(
                normalized, tools=tools, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        except Exception:
            logger.exception("teacher templating failed; the candidate drops")
            return None
        return [int(v) for v in prompt_ids] + [int(v) for v in native_tokens[-response_token_count:]]


def build_clients(config: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """The judge's PRM clients as the recipe config describes them; the
    all-``None`` triple (no ``prm_url``) is correlate-only mode.

    Defaults mirror ``OpenClawRLRecipe``'s config fields; the recipe always sends
    every value; the fallbacks serve direct construction in tests.
    """
    url = str(config.get("prm_url", "") or "").strip()
    if not url:
        return None, None, None
    tokenizer_path = str(config.get("prm_tokenizer_path", "") or "").strip()
    if not tokenizer_path:
        raise ValueError("OpenClaw-RL judging requires prm_tokenizer_path next to prm_url")
    generate_url = url.rstrip("/") + "/generate"
    prm = PRMJudge(
        generate_url=generate_url,
        tokenizer_path=tokenizer_path,
        m=int(config.get("prm_m", 3)),
        temperature=float(config.get("prm_temperature", 0.6)),
        max_tokens=int(config.get("prm_max_tokens", 8192)),
        timeout_s=float(config.get("prm_timeout_s", 120.0)),
    )
    return prm, DirectiveTeacher(tokenizer_path=tokenizer_path), ResponseRenderer(tokenizer_path)
