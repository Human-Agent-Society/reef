"""Checks, expected scores, wire log and report of the native inference qualification.

``tests.smoke.native_inference`` runs the engines and the Reef apps; this module checks what they answered and stored.
It imports no engine, model or tensor library at import time: ``LogprobOracle`` imports Torch and Transformers when it
is made. A failed check raises ``AssertionError`` that names the request. A malformed answer can instead raise
the error of the read that it breaks (``Exchange``); the run fails in both cases.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import json
import math
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession

from reef.core.records_types import AgentRecord
from reef.core.values import JsonValue
from reef.service.inference_service import BUCKET_COUNT, RECORD_HEADER

#: The largest accepted difference between an engine's log probability and the Hugging Face float32 score.
LOGPROB_TOLERANCE = 1e-3
#: The alternatives that each scored request asks for with ``top_logprobs``.
TOP_ALTERNATIVES = 2
#: Keys of Reef's native capture: SGLang's raw per-token log probabilities. No answer or record of Reef's native
#: facade may hold one with a value.
PRIVATE_FIELDS = frozenset({"output_token_logprobs", "input_token_logprobs"})
#: A stream chunk with generated text, as opposed to the opening chunk, which only names the role.
GENERATED_CONTENT = re.compile(rb'"content":\s*"[^"]')


@dataclass(frozen=True)
class Checkpoint:
    """One synthetic checkpoint whose language-model head alone sets each next-token distribution.

    Values: ``name`` is a non-empty label. ``directory`` holds the saved Hugging Face model and the tokenizer that
    every checkpoint of the run shares. ``words`` are three distinct non-empty words; the text ``" " + word`` of
    each is one vocabulary token. Construction checks the name and the words.
    Purpose: ``native_inference.write_checkpoints`` makes the checkpoints of a run, with different names and
    disjoint words, so generated text names the checkpoint that produced it. The checks read ``words``, and
    ``LogprobOracle`` loads ``directory``.
    """

    name: str
    directory: Path
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or len(self.words) != 3 or len(set(self.words)) != 3 or not all(self.words):
            raise ValueError("a checkpoint has a name and three distinct non-empty words")

    def tokens(self) -> tuple[str, ...]:
        """The decoded text of each word's token, in ``words`` order."""
        return tuple(" " + word for word in self.words)


@dataclass(frozen=True)
class ExpectedToken:
    """The Hugging Face scores of one generated position.

    Values: ``token_id`` is the generated token, a non-negative int, and ``log_probability`` is its float32 log
    probability, finite and at most 0. ``alternatives`` holds the ``TOP_ALTERNATIVES`` most likely tokens at that
    position as (token ID, log probability) pairs: distinct non-negative token IDs, scores finite and at most 0, in
    descending score order, with tied scores in any order. When the generated token is one of them, its pair's
    score is ``log_probability``. Otherwise ``log_probability`` is at most the last pair's score.
    The constructor checks only these local conditions. That the scores and tokens are the saved model's own is up
    to the maker.
    Purpose: ``LogprobOracle.score`` makes it, from one float32 row of the saved model's log probabilities; the
    logprob checks compare engine answers with it.
    """

    token_id: int
    log_probability: float
    alternatives: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        scores = [self.log_probability, *(score for token_id, score in self.alternatives)]
        if not all(math.isfinite(score) and score <= 0 for score in scores):
            raise ValueError(f"log probabilities are finite and at most 0, not {scores}")
        token_ids = [self.token_id, *(token_id for token_id, score in self.alternatives)]
        if not all(type(token_id) is int and token_id >= 0 for token_id in token_ids):
            raise ValueError(f"token IDs are non-negative integers, not {token_ids}")
        listed = dict(self.alternatives)
        if len(self.alternatives) != TOP_ALTERNATIVES or len(listed) != TOP_ALTERNATIVES:
            raise ValueError(f"a position has {TOP_ALTERNATIVES} distinct most likely tokens: {self.alternatives}")
        if any(earlier[1] < later[1] for earlier, later in itertools.pairwise(self.alternatives)):
            raise ValueError(f"the most likely token comes first: {self.alternatives}")
        if self.token_id in listed:
            consistent = self.log_probability == listed[self.token_id]
        else:
            consistent = self.log_probability <= self.alternatives[-1][1]
        if not consistent:
            raise ValueError(
                f"the generated token {self.token_id} scored {self.log_probability} disagrees with the most likely "
                f"tokens {self.alternatives}"
            )

    def known(self) -> dict[int, float]:
        """The tokens an engine may list as alternatives here, with their scores: the most likely ones and the
        generated one."""
        return {**dict(self.alternatives), self.token_id: self.log_probability}


class LogprobOracle:
    """Hugging Face float32 forward passes of the saved checkpoints on the CPU: the reference for engine scores.

    Values: ``tokenizer`` is the tokenizer that the checkpoints share; ``models`` maps each checkpoint's name to its
    float32 model in evaluation mode; ``word_ids`` maps the decoded text of each checkpoint word's token to its token
    ID, and ``symbols`` maps the same tokens' vocabulary symbols to their IDs.
    Purpose: a native qualification run makes one oracle after it writes its checkpoints. ``token_id`` reads a
    token that an engine reported; ``score`` gives the expected scores of a generated token sequence that follows
    the chat template's rendering of ``messages``.
    """

    def __init__(self, checkpoints: Sequence[Checkpoint]) -> None:
        import torch
        from transformers import AutoTokenizer, Qwen2ForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoints[0].directory)
        self.models = {
            checkpoint.name: Qwen2ForCausalLM.from_pretrained(checkpoint.directory, dtype=torch.float32).eval()
            for checkpoint in checkpoints
        }
        decoded: dict[str, list[int]] = {}
        for token_id in range(len(self.tokenizer)):
            decoded.setdefault(self.tokenizer.decode([token_id]), []).append(token_id)
        self.word_ids: dict[str, int] = {}
        self.symbols: dict[str, int] = {}
        for checkpoint in checkpoints:
            for token in checkpoint.tokens():
                matches = decoded.get(token, [])
                if len(matches) != 1:
                    raise RuntimeError(f"{token!r} must be exactly one vocabulary token, not {len(matches)}")
                self.word_ids[token] = matches[0]
                self.symbols[self.tokenizer.convert_ids_to_tokens(matches[0])] = matches[0]

    def token_id(self, token: str) -> int:
        """The ID of a checkpoint word's token, given as its decoded text or as its vocabulary symbol."""
        found = self.word_ids.get(token, self.symbols.get(token))
        assert found is not None, f"the engine reported the token {token!r}, which is no checkpoint word"
        return found

    def text(self, token_id: int) -> str:
        return str(self.tokenizer.decode([token_id]))

    def prompt_ids(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True)
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        return [int(token_id) for token_id in rendered]

    def score(
        self, checkpoint: Checkpoint, messages: Sequence[Mapping[str, Any]], token_ids: Sequence[int]
    ) -> tuple[ExpectedToken, ...]:
        """The scores of ``token_ids`` generated after ``messages``, one for each token, in order."""
        import torch

        prompt = self.prompt_ids(messages)
        with torch.no_grad():
            logits = self.models[checkpoint.name](torch.tensor([prompt + list(token_ids)])).logits[0]
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        expected = []
        for index, token_id in enumerate(token_ids):
            # The logits at position p predict the token at position p + 1.
            row = log_probabilities[len(prompt) + index - 1]
            values, ids = torch.topk(row, TOP_ALTERNATIVES)
            expected.append(
                ExpectedToken(
                    int(token_id),
                    float(row[token_id]),
                    tuple((int(i), float(v)) for i, v in zip(ids.tolist(), values.tolist(), strict=True)),
                )
            )
        return tuple(expected)


def parse_events(content: bytes, label: str) -> list[object]:
    """The data of each SSE event in order: a decoded JSON value, or the literal ``"[DONE]"``."""
    frames, separator, remainder = content.rpartition(b"\n\n")
    assert separator and not remainder.strip(), f"{label}: the stream ended inside an event: {remainder[-200:]!r}"
    values: list[object] = []
    for frame in frames.split(b"\n\n"):
        lines = [line for line in frame.split(b"\n") if line and not line.startswith(b":")]
        data = b"\n".join(line[5:].removeprefix(b" ") for line in lines if line.startswith(b"data:"))
        assert data or not lines, f"{label}: an event has no data: {frame!r}"
        if data:
            values.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    return values


def first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = chunk.get("choices") or [{}]
    return choices[0]


def chunk_content(chunk: Mapping[str, Any]) -> str:
    content = (first_choice(chunk).get("delta") or {}).get("content")
    return content if isinstance(content, str) else ""


def chunk_logprobs(chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    logprobs = first_choice(chunk).get("logprobs")
    return list(logprobs.get("content") or []) if isinstance(logprobs, Mapping) else []


@dataclass(frozen=True)
class Exchange:
    """One HTTP request to a Reef app and its complete answer, without credentials.

    Values: ``label`` names the request in checks and in the wire log; ``path`` is the request path; ``body`` the
    JSON request body; ``headers`` the request headers other than ``Authorization``; ``status`` the HTTP status;
    ``response_headers`` the answer's headers with lowercase names; ``content`` the answer's bytes. ``sent`` and
    ``ended`` are ``time.monotonic`` readings when the request was sent and when its answer ended.
    Purpose: ``ReefClient.post`` makes it from its request and the ``HttpAnswer`` of ``send``, whose construction
    checked these header names and readings. The checks read the value, and the wire log keeps it.

    The accessors below read a successful chat-completion answer, buffered or streamed. Each raises
    ``AssertionError`` with the label when the answer is of the other kind, has no single receipt or ``meta_info``,
    or is a stream without its final receipt and ``[DONE]``. An answer that is no JSON, or that lacks a field that an
    accessor reads, raises the error of that read instead, such as ``ValueError``, ``KeyError`` or ``IndexError``.
    """

    label: str
    path: str
    body: Mapping[str, Any]
    headers: Mapping[str, str]
    status: int
    response_headers: Mapping[str, str]
    content: bytes
    sent: float
    ended: float

    @property
    def streamed(self) -> bool:
        return "text/event-stream" in self.response_headers.get("content-type", "")

    def require_success(self) -> None:
        assert self.status == 200, f"{self.label}: HTTP {self.status}: {self.content[:400]!r}"

    def answer(self) -> dict[str, Any]:
        """The JSON object of a buffered answer."""
        assert not self.streamed, f"{self.label}: expected a buffered answer"
        value = json.loads(self.content)
        assert isinstance(value, dict), f"{self.label}: the answer is not a JSON object"
        return value

    def chunks(self) -> list[dict[str, Any]]:
        """The data events of a stream before Reef's receipt, which comes right before the one ``[DONE]``."""
        values = parse_events(self.content, self.label)
        assert values.count("[DONE]") == 1 and values[-1] == "[DONE]", f"{self.label}: no single final [DONE]"
        receipt = values[-2] if len(values) >= 2 else None
        assert isinstance(receipt, dict) and receipt.get("choices") == [], f"{self.label}: no receipt before [DONE]"
        chunks = values[:-2]
        assert all(isinstance(value, dict) and "reef" not in value for value in chunks), f"{self.label}: bad event"
        return [value for value in chunks if isinstance(value, dict)]

    def receipt(self) -> str:
        """The record ID that the answer names: the record header, or the stream's receipt event."""
        if self.streamed:
            receipt = parse_events(self.content, self.label)[-2]
            assert isinstance(receipt, dict), f"{self.label}: no receipt event"
            record_id = receipt.get("reef", {}).get("agent_record_id")
        else:
            record_id = self.response_headers.get(RECORD_HEADER)
        assert isinstance(record_id, str) and record_id, f"{self.label}: no record receipt"
        return record_id

    def text(self) -> str:
        """The generated text: the message content, or the stream's content deltas joined in order."""
        if not self.streamed:
            return str(self.answer()["choices"][0]["message"]["content"])
        return "".join(chunk_content(chunk) for chunk in self.chunks())

    def logprob_entries(self) -> list[dict[str, Any]]:
        """The OpenAI log-probability entries of the first choice, in token order."""
        if not self.streamed:
            return list(self.answer()["choices"][0]["logprobs"]["content"])
        return [entry for chunk in self.chunks() for entry in chunk_logprobs(chunk)]

    def meta_info(self) -> dict[str, Any]:
        """The first choice's ``meta_info``: Reef's native answer and the final chunk of its stream carry it."""
        if not self.streamed:
            return dict(self.answer()["choices"][0]["meta_info"])
        carried = [first_choice(chunk)["meta_info"] for chunk in self.chunks() if "meta_info" in first_choice(chunk)]
        assert len(carried) == 1, f"{self.label}: expected one stream chunk with meta_info"
        return dict(carried[0])


@dataclass(frozen=True)
class HttpAnswer:
    """The complete answer to one HTTP request, with the clock readings of that request.

    Values: ``status`` is the HTTP status; ``headers`` maps each header name of the answer, in lowercase, to its
    value; ``content`` is all the bytes of its body. ``sent`` and ``ended`` are ``time.monotonic`` readings, from
    before the request went out and from after the body ended, so ``ended`` is not before ``sent``. Construction
    checks the header names and the order of the readings.
    Purpose: ``ReefClient.send`` returns one for each request. ``post`` puts its values into an ``Exchange`` with
    the request, and ``control`` checks the status and reads the content.
    """

    status: int
    headers: dict[str, str]
    content: bytes
    sent: float
    ended: float

    def __post_init__(self) -> None:
        if any(name != name.lower() for name in self.headers):
            raise ValueError(f"the header names of an answer are lowercase, not {sorted(self.headers)}")
        if self.ended < self.sent:
            raise ValueError(f"an answer ends before its request was sent: sent {self.sent}, ended {self.ended}")


class ReefClient:
    """Requests to one Reef app with its service token, each appended to a JSON-lines wire log.

    Values: ``session`` is an open aiohttp session without default headers; ``base_url`` is the app's
    ``http://host:port``; ``token`` is the app's service token, which goes only into the ``Authorization`` header,
    never into the log; ``wire_log`` is a file in an existing directory. It receives one JSON object per request:
    the label, method, path, other headers and JSON body of the request, the status, headers and complete content of
    the answer, SSE bytes included, and ``sent`` and ``ended``, ``time.monotonic`` readings that order the requests.
    Purpose: each qualification profile makes one client for its app. ``post`` sends one inference request and
    returns its ``Exchange``; ``control`` sends one request of the public control API and returns its JSON answer.
    """

    def __init__(self, session: ClientSession, base_url: str, token: str, wire_log: Path) -> None:
        self.session = session
        self.base_url = base_url
        self.token = token
        self.wire_log = wire_log

    async def send(
        self,
        label: str,
        method: str,
        path: str,
        body: Mapping[str, JsonValue] | None,
        headers: Mapping[str, str],
        first_content: asyncio.Future[float] | None = None,
    ) -> HttpAnswer:
        """Send one request, append it with its complete answer to the wire log, and return the answer.

        ``body`` is the JSON body. None sends no body, and the log keeps that apart from an empty object.
        ``headers`` are the request headers other than ``Authorization``. ``first_content``, a future that is not
        done yet, is set to the moment the first generated text arrives.
        """
        sent = time.monotonic()
        async with self.session.request(
            method, self.base_url + path, json=body, headers={**headers, "Authorization": f"Bearer {self.token}"}
        ) as response:
            content = bytearray()
            async for chunk in response.content.iter_any():
                content.extend(chunk)
                if first_content is not None and not first_content.done() and GENERATED_CONTENT.search(chunk):
                    first_content.set_result(time.monotonic())
            answer = HttpAnswer(
                response.status,
                {name.lower(): value for name, value in response.headers.items()},
                bytes(content),
                sent,
                time.monotonic(),
            )
        try:
            text: dict[str, str] = {"response_text": answer.content.decode()}
        except UnicodeDecodeError:
            text = {"response_base64": base64.b64encode(answer.content).decode()}
        line = {
            "label": label,
            "method": method,
            "path": path,
            "request_headers": dict(headers),
            "request_body": body,
            "status": answer.status,
            "response_headers": answer.headers,
            **text,
            "sent": sent,
            "ended": answer.ended,
        }
        with self.wire_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, sort_keys=True) + "\n")
        return answer

    async def post(
        self,
        label: str,
        path: str,
        body: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        first_content: asyncio.Future[float] | None = None,
    ) -> Exchange:
        """Send one inference request, with ``first_content`` as ``send`` takes it; return its ``Exchange``."""
        sent_body, extra = dict(body), dict(headers or {})
        answer = await self.send(label, "POST", path, sent_body, extra, first_content)
        return Exchange(
            label, path, sent_body, extra, answer.status, answer.headers, answer.content, answer.sent, answer.ended
        )

    async def control(
        self, label: str, method: str, path: str, body: dict[str, JsonValue] | None = None, *, status: int = 200
    ) -> dict[str, JsonValue]:
        """Send one request of the public control API and return its answer, which must have the HTTP ``status``.

        An answer with a ``status`` below 400 must be a JSON object. A refusal that the caller expects is plain
        text: the result is then ``{"error": <the answer's text>}``.
        """
        answer = await self.send(label, method, path, body, {})
        text = answer.content.decode(errors="replace")
        assert answer.status == status, f"{label}: HTTP {answer.status}, expected {status}: {text[:400]!r}"
        if status >= 400:
            return {"error": text}
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        assert isinstance(value, dict), f"{label}: the answer is not a JSON object: {text[:400]!r}"
        return value


def check_words(checkpoint: Checkpoint, text: str, word_count: int, label: str) -> tuple[str, ...]:
    """Check that ``text`` is ``word_count`` words of ``checkpoint``; return them."""
    words = tuple(text.split())
    assert len(words) == word_count, f"{label}: {len(words)} words, expected {word_count}: {text[:120]!r}"
    assert set(words) <= set(checkpoint.words), f"{label}: words of another checkpoint: {text[:120]!r}"
    return words


def check_alternatives(
    oracle: LogprobOracle, listed: Sequence[tuple[str, float]], position: ExpectedToken, where: str
) -> None:
    """Check one position's listed alternatives (token text, log probability).

    There must be ``TOP_ALTERNATIVES`` distinct tokens, among them the most likely one. Each must be one of the most
    likely tokens or the sampled token, which an engine may list even when it ranks lower. Each score must be its own
    token's, within ``LOGPROB_TOLERANCE``.
    """
    ids = [oracle.token_id(token) for token, value in listed]
    assert len(ids) == TOP_ALTERNATIVES and len(set(ids)) == len(ids), f"{where}: alternatives {listed!r}"
    known = position.known()
    for token_id, (token, value) in zip(ids, listed, strict=True):
        assert token_id in known, f"{where}: {token!r} is neither a most likely token nor the sampled one"
        assert abs(value - known[token_id]) <= LOGPROB_TOLERANCE, f"{where}: {token!r} scored {value}, not {known}"
    assert position.alternatives[0][0] in ids, f"{where}: the most likely token is not listed: {listed!r}"


def check_logprobs(
    oracle: LogprobOracle, checkpoint: Checkpoint, exchange: Exchange, *, native_facade: bool
) -> tuple[int, ...]:
    """Check each sampled token's log-probability entry against the Hugging Face scores; return the sampled token IDs.

    The entries must name the generated tokens in order: one entry per token, whose texts join to the generated
    text. Each log probability is finite, at most 0 and within ``LOGPROB_TOLERANCE`` of the oracle's score of that
    token. Reef's native facade (``native_facade``) writes each token's decoded text and UTF-8 bytes and lists only
    the sampled token as its alternative, as documented. A provider's entries may name a token by its decoded text or
    its vocabulary symbol; its bytes, when present, are those of the text it gives, and its alternatives must pass
    ``check_alternatives``.
    """
    entries = exchange.logprob_entries()
    token_ids = [oracle.token_id(entry["token"]) for entry in entries]
    decoded = "".join(oracle.text(token_id) for token_id in token_ids)
    if native_facade:
        assert [entry["token"] for entry in entries] == [oracle.text(i) for i in token_ids], exchange.label
        assert decoded == exchange.text(), f"{exchange.label}: the entries spell {decoded!r}"
    else:
        assert decoded.strip() == exchange.text().strip(), f"{exchange.label}: the entries spell {decoded!r}"
    expected = oracle.score(checkpoint, exchange.body["messages"], token_ids)
    for index, (entry, position) in enumerate(zip(entries, expected, strict=True)):
        where = f"{exchange.label} token {index}"
        if native_facade or entry.get("bytes") is not None:
            assert entry["bytes"] == list(entry["token"].encode()), f"{where}: bytes"
        value = entry["logprob"]
        assert math.isfinite(value) and value <= 0, f"{where}: log probability {value}"
        assert abs(value - position.log_probability) <= LOGPROB_TOLERANCE, f"{where}: {value} != {position}"
        alternatives = entry["top_logprobs"]
        if native_facade:
            sampled = {key: entry[key] for key in ("token", "bytes", "logprob")}
            assert alternatives == [sampled], f"{where}: the native facade listed {alternatives!r}"
        else:
            for item in alternatives:
                if item.get("bytes") is not None:
                    assert item["bytes"] == list(item["token"].encode()), f"{where}: alternative bytes"
            listed = [(item["token"], item["logprob"]) for item in alternatives]
            check_alternatives(oracle, listed, position, where)
    return tuple(token_ids)


def check_native_alternatives(
    oracle: LogprobOracle, checkpoint: Checkpoint, exchange: Exchange, token_ids: Sequence[int]
) -> None:
    """Check SGLang's own alternatives, which Reef's native answer passes on in ``meta_info.output_top_logprobs``."""
    rows = exchange.meta_info()["output_top_logprobs"]
    expected = oracle.score(checkpoint, exchange.body["messages"], token_ids)
    assert len(rows) == len(expected), f"{exchange.label}: {len(rows)} rows of SGLang alternatives"
    for index, (row, position) in enumerate(zip(rows, expected, strict=True)):
        # SGLang writes each alternative as [log probability, token ID, text or null].
        listed = [(oracle.text(int(item[1])), float(item[0])) for item in row]
        check_alternatives(oracle, listed, position, f"{exchange.label} SGLang alternatives {index}")


def private_fields_in(value: object) -> list[str]:
    """The ``PRIVATE_FIELDS`` that are keys with a value other than null anywhere in the JSON value ``value``."""
    found: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in PRIVATE_FIELDS and child is not None:
                    found.add(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def check_private_fields_absent(value: object, label: str) -> None:
    """Check that ``value``, the JSON of an answer or record of Reef's native facade, holds no ``PRIVATE_FIELDS``."""
    leaked = private_fields_in(value)
    assert not leaked, f"{label}: private native fields {leaked} reached a client answer or record"


def answer_values(exchange: Exchange) -> list[object]:
    """The JSON values of a successful answer: its object, or the data of each event of its stream."""
    return parse_events(exchange.content, exchange.label) if exchange.streamed else [exchange.answer()]


def artifact_of_bucket(bucket: int, weights: Mapping[str, int]) -> str:
    """The artifact that owns ``bucket`` under ``weights``: artifact IDs in sorted order own consecutive buckets.

    This is the documented split rule, written out here so that the checks do not read Reef's own mapping.
    """
    end = 0
    for artifact_id in sorted(weights):
        end += weights[artifact_id]
        if bucket < end:
            return artifact_id
    raise AssertionError(f"bucket {bucket} is outside the weights {dict(weights)!r}")


@dataclass(frozen=True)
class ExpectedTraffic:
    """The traffic split that one recorded request must store.

    Values: ``weights`` are the weights in force when the request selected its artifact. They are a split as Reef
    stores one: a non-empty map from non-empty artifact IDs to integer weights of at least 0, which sum to
    ``BUCKET_COUNT``. ``bucket`` is the bucket that the run computed for the request's conversation, an integer from
    0 to ``BUCKET_COUNT`` - 1, or None when the run did not compute one. Construction checks both fields.
    Purpose: a request that a split routed is recorded with one; ``check_traffic`` compares it with the record.
    """

    weights: Mapping[str, int]
    bucket: int | None = None

    def __post_init__(self) -> None:
        shares = self.weights.values()
        if not all(self.weights) or not all(type(share) is int and share >= 0 for share in shares):
            raise ValueError(f"a split maps artifact IDs to integer weights of at least 0: {dict(self.weights)!r}")
        if sum(shares) != BUCKET_COUNT:
            raise ValueError(f"the weights of a split sum to {BUCKET_COUNT}: {dict(self.weights)!r}")
        if self.bucket is not None and (type(self.bucket) is not int or not 0 <= self.bucket < BUCKET_COUNT):
            raise ValueError(f"a bucket is an integer from 0 to {BUCKET_COUNT - 1}, not {self.bucket!r}")


def check_traffic(stored: object, expected: ExpectedTraffic | None, artifact_id: str, label: str) -> None:
    """Check a record's ``metadata.traffic`` against the split that routed it, or its absence without a split."""
    if expected is None:
        assert stored is None, f"{label}: a request outside the split stored traffic metadata {stored!r}"
        return
    assert isinstance(stored, Mapping) and set(stored) == {"bucket", "weights"}, f"{label}: traffic {stored!r}"
    assert stored["weights"] == dict(expected.weights), f"{label}: stored weights {stored['weights']!r}"
    bucket = stored["bucket"]
    assert type(bucket) is int and 0 <= bucket < BUCKET_COUNT, f"{label}: bucket {bucket!r}"
    assert expected.bucket is None or bucket == expected.bucket, f"{label}: bucket {bucket}, not {expected.bucket}"
    assert artifact_of_bucket(bucket, expected.weights) == artifact_id, f"{label}: bucket {bucket} owner differs"


def check_stored_response(response: Mapping[str, Any], exchange: Exchange) -> None:
    """Check that a record keeps the answer that the client received: the same JSON for a buffered answer, and
    for a stream the same text and log-probability entries, and a complete delivery.

    A captured stream (Reef's native facade) keeps the canonical response; an opaque provider stream keeps the
    provider's SSE body, whose events are read again.
    """
    label = exchange.label
    if not exchange.streamed:
        assert response == exchange.answer(), f"{label}: the record's response differs from the answer"
        return
    delivery = response.get("stream_delivery", response)
    assert isinstance(delivery, Mapping) and delivery.get("complete") is True, f"{label}: incomplete delivery"
    if "body" in response:
        events = [value for value in parse_events(str(response["body"]).encode(), label) if isinstance(value, dict)]
        text = "".join(chunk_content(event) for event in events)
        entries = [entry for event in events for entry in chunk_logprobs(event)]
    else:
        choice = response["choices"][0]
        text, entries = str(choice["message"]["content"]), list(choice["logprobs"]["content"])
    assert text == exchange.text(), f"{label}: the record's text differs"
    assert entries == exchange.logprob_entries(), f"{label}: the record's logprobs differ"


def replay_records(directory: Path, scenario: str) -> list[AgentRecord]:
    """The records of ``scenario`` in a closed SQLite scenario store under ``directory``, in order, read through
    the store's replay, as training reads them.

    Opening a store writes its settings, so the replay reads a copy and leaves the run's files unchanged.
    """
    from reef.storage.sqlite import SQLiteScenarioStorage

    with tempfile.TemporaryDirectory(prefix="reef-native-records-") as scratch:
        copy = Path(scratch) / "records"
        shutil.copytree(directory, copy)
        storage = SQLiteScenarioStorage(copy)
        try:
            store = storage.open(scenario)
            try:
                return list(store.records.replay(scenario))
            finally:
                store.close()
        finally:
            storage.close()


def write_records(path: Path, records: Iterable[AgentRecord]) -> None:
    """Write replayed records as JSON lines, for replay without the store."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            line = {
                "agent_record_id": record.agent_record_id,
                "scenario": record.scenario,
                "artifact_id": record.artifact_id,
                "payload": record.payload,
            }
            stream.write(json.dumps(line, sort_keys=True, default=json_value) + "\n")


def json_value(value: object) -> object:
    """A read-only mapping or a tuple of a record as JSON writes it."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_secrets_absent(directory: Path, secrets: Iterable[str]) -> int:
    """Check that no file under ``directory`` contains a secret; return the number of files read."""
    encoded = [secret.encode() for secret in secrets]
    files = [path for path in sorted(directory.rglob("*")) if path.is_file() and not path.is_symlink()]
    for path in files:
        content = path.read_bytes()
        assert not any(secret in content for secret in encoded), f"{path}: a credential reached an output file"
    return len(files)


class QualificationReport:
    """What one native qualification run measured, and which of its checks passed.

    Values: ``facts`` maps names to JSON values that identify the run: profile, engines, versions, options, digests
    and hashes. ``checks`` maps the name of each check that passed to its JSON details, in the order they passed; a
    check that failed or did not run is absent. ``error`` is None until the run fails, then the failure's type and
    message.
    Purpose: a native qualification run fills it while it runs; ``document`` gives the JSON object that the run
    writes once its processes have stopped. A run passed exactly when ``error`` is None.
    """

    def __init__(self) -> None:
        self.facts: dict[str, JsonValue] = {}
        self.checks: dict[str, JsonValue] = {}
        self.error: str | None = None

    def passed(self, name: str, **details: JsonValue) -> None:
        if name in self.checks:
            raise ValueError(f"check {name!r} passed twice")
        self.checks[name] = details
        print(f"passed: {name}", flush=True)

    def document(self) -> dict[str, JsonValue]:
        return {
            "passed": self.error is None,
            "error": self.error,
            "facts": self.facts,
            "checks": self.checks,
            "measurement": "functional qualification with tiny synthetic checkpoints; not a performance result",
        }
