"""The OpenClaw-RL processor: judge every main turn by its next state.

A computed-feedback recipe on the shared engine
(:class:`~reef.train.processors.computed.ComputedFeedbackProcessor` owns the record
lifecycle, the judging worker, and the batch cycle). This file is the
method: ``ingest`` correlates main turns into sessions and hands each
completed turn over for judgment, ``judge`` is upstream's combine
dispatch, ``make_sample`` validates the tensors, ``make_batch`` shapes
the batch. One turn, one judgment, one sample — rewards pass through raw
(upstream ``--n-samples-per-prompt 1``; no grouping, no normalization).

No proxy and no judgment on the wire: the method derives every verdict
from the traffic the agent already produces. Correlation prefers a
harness-stamped ``x-reef-tag-session`` and falls back to trace matching,
which holds only for agents that resend their transcript each turn. The
machinery — session matching, the PRM clients — lives in
:mod:`recipes.openclawrl`.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import pathlib
import time
from dataclasses import replace
from typing import Any

from recipes.openclawrl.prm import (
    append_hint_to_messages,
    build_clients,
    collect_hint_candidates,
    render_response_fallback,
)
from recipes.openclawrl.sessions import SessionIndex
from recipes.openclawrl.turns import (
    TurnJob,
    TurnJudgment,
    main_turn_message,
    turn_job,
    turn_request_messages,
    validate_teacher_cands,
)
from reef.core.records_types import AgentRecord, RequestType
from reef.train.processors.common import make_policy_sample
from reef.train.processors.computed import ComputedFeedbackProcessor, JudgingWorker
from reef.train.types import PolicyBatch, PolicySample, ProcessorContext, policy_row_violation

logger = logging.getLogger(__name__)


def _session_tag(payload: Any) -> str | None:
    """The harness-declared conversation id (``x-reef-tag-session``), if any."""
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    value = tags.get("session") if isinstance(tags, dict) else None
    return str(value).strip() or None if value is not None else None


class OpenClawRLProcessor(ComputedFeedbackProcessor):
    """Turns judged by their next state; terminal exclusions mirror upstream.

    Side calls, retried turns, session-final turns,
    judge-declined turns, and samples whose tensors the top-K select loss
    cannot train on (the capture must cover every response token) all
    retire terminal instead of ever entering a batch.
    """

    output_schema = PolicyBatch
    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    def __init__(
        self,
        context: ProcessorContext,
        *,
        worker: JudgingWorker | None = None,
        clients: tuple[Any, Any, Any] | None = None,
    ) -> None:
        config = context.config
        # The judge's clients (PRM judge, directive teacher, renderer) and
        # its config fields; ``clients`` overrides for tests.
        self._prm, self._teacher, self._renderer = clients if clients is not None else build_clients(config)
        self._max_hint_candidates = max(1, int(config.get("prm_max_hint_candidates", 3)))
        self._index = SessionIndex(float(config.get("session_ttl_s", 900.0)))
        # What the judge actually saw. Upstream persists this per turn to a
        # PRM record file; without it a batch mean is the only visible number
        # and every question about the judged population — the eval mix, how
        # often a hint was accepted, which next states are being judged —
        # needs a bespoke replay to answer. A file, not a log line: reef
        # configures no logging, so anything below WARNING is discarded.
        self._judged: collections.Counter[str] = collections.Counter()
        record_file = str(config.get("prm_record_file", "") or "").strip()
        self._record_path = pathlib.Path(record_file) if record_file else None
        if worker is None and self._prm is not None:
            worker = JudgingWorker(self.judge, concurrency=int(config.get("prm_concurrency", 8)))
        super().__init__(context, worker=worker)

    # ------------------------------------------------ the recipe's data path
    #  ingest (correlate sessions) → judge (async, on the worker)
    #  → make_sample → make_batch; lifecycle and the cycle are the engine's.

    def ingest(self, item: AgentRecord) -> None:
        # 1. Catch up on derived state. Expiry runs before observing, so an
        #    expired session can never capture this request as its successor.
        now = time.monotonic()
        self.catch_up(now)
        # 2. Reports never train here; owning them keeps the store
        #    compactable (nothing else would ever release them).
        if item.request_type is not RequestType.INFERENCE:
            self.retire(item.agent_record_id)
            return
        # 3. Side calls (titles, compaction — no tools) never train.
        payload = item.payload
        if main_turn_message(payload) is None:
            self.retire(item.agent_record_id)
            return
        # 4. Correlate into a session: a retry is terminal; the request that
        #    follows a turn delivers its next state — send that turn to the
        #    judging worker. A harness-stamped ``session`` tag names the
        #    conversation outright; without one this falls back to trace
        #    matching, which only holds for agents that resend their whole
        #    transcript every turn. Every main turn is judged against
        #    whatever arrived next — a tool result is a next state, and its
        #    role is an input to the judge rather than a filter.
        observation = self._index.observe(
            item.agent_record_id,
            turn_request_messages(payload),
            now,
            session_tag=_session_tag(payload),
        )
        if observation.duplicate:
            self.retire(item.agent_record_id)
            return
        if observation.binding is not None:
            record = self.tracked_record(observation.binding.receipt)
            if record is not None:
                self.dispatch(turn_job(record, observation.binding))
        # 5. This turn now awaits its own next state.
        self.track(item)

    async def judge(self, job: TurnJob) -> TurnJudgment:
        """Upstream combine dispatch: evaluative and directive judges per turn.

        ┌────────────────┬──────────────┬──────────────────────────────────┐
        │ hint accepted? │ eval = ±1?   │ judgment                         │
        ├────────────────┼──────────────┼──────────────────────────────────┤
        │ yes            │ yes          │ score=eval + teacher signal      │
        │ yes            │ no           │ score=0.0 + teacher signal       │
        │ no             │ yes          │ score=eval + un-enhanced anchor  │
        │ no             │ no           │ terminal                         │
        └────────────────┴──────────────┴──────────────────────────────────┘

        Runs on the judging worker's thread, never the trainer's. The
        un-enhanced anchor is upstream's RL-only convention
        (``teacher_tokens_candidates = [un_enhanced]``): every RL sample
        carries the frozen PRM's log-probs under the PLAIN prompt as a
        single OPD candidate, so the distillation term keeps pulling the
        policy toward the frozen base distribution on its top-K.
        """
        # 1. The judges and teacher see the same text upstream does: the
        #    chat-template re-render of the FULL assistant turn (reasoning +
        #    content + native tool calls); the visible render is the fallback.
        response_text = None
        if self._renderer is not None:
            response_text = await asyncio.to_thread(
                self._renderer.render,
                list(job.request_messages),
                dict(job.response_message),
                tools=job.request_tools,
            )
        response_text = response_text or render_response_fallback(job.response_message)
        # 2. Both PRM judges vote.
        eval_score, hint_votes = await asyncio.gather(
            self._prm.evaluate(response_text, job.next_state_text, job.next_state_role),
            self._prm.evaluate_hint_votes(response_text, job.next_state_text, job.next_state_role),
        )
        # 3. The accepted hindsight hints, shortest first, capped.
        candidates = collect_hint_candidates(hint_votes, max_cand=self._max_hint_candidates)
        # 4. Every candidate ships as a TOKEN SEQUENCE — upstream's
        #    ``teacher_tokens_candidates`` — for the train actor's Megatron
        #    teacher pass; an RL-only turn ships the un-enhanced native ids
        #    (the anchor, whose construction cannot fail). Without the
        #    generation-time capture the turn cannot train regardless, so no
        #    candidate is built for it.
        messages = list(job.request_messages)
        native_ok = bool(
            job.native_tokens
            and job.response_token_count
            and 0 < job.response_token_count < len(job.native_tokens)
            and job.topk_indices
            and len(job.topk_indices) == job.response_token_count
        )
        teacher_cands: list[dict[str, Any]] = []
        if candidates and messages and native_ok:
            if job.native_tokens is None or job.response_token_count is None:
                raise RuntimeError("native OpenClaw job lost its token capture")
            for candidate in candidates:
                enhanced = append_hint_to_messages(messages, candidate)
                ids = await self._teacher.candidate_tokens(
                    enhanced,
                    tools=job.request_tools,
                    native_tokens=job.native_tokens,
                    response_token_count=job.response_token_count,
                )
                if ids is not None:
                    teacher_cands.append({"hint": candidate, "teacher_tokens": ids})
        if not teacher_cands and eval_score in (1.0, -1.0) and native_ok:
            if job.native_tokens is None:
                raise RuntimeError("native OpenClaw job lost its token capture")
            teacher_cands.append({"hint": "", "teacher_tokens": [int(v) for v in job.native_tokens]})
        # 5. The dispatch table decides.
        has_rl = eval_score in (1.0, -1.0)
        self._judged[f"next_state:{job.next_state_role}"] += 1
        self._judged[f"eval:{eval_score:+.0f}" if has_rl else "eval:0"] += 1
        self._judged["hint_accepted" if candidates else "hint_declined"] += 1
        self._judged["anchor_only" if (not candidates and teacher_cands) else "teacher_cands"] += 1
        if not has_rl and not teacher_cands:
            self._judged["declined"] += 1
            return TurnJudgment(job.receipt)
        return TurnJudgment(job.receipt, score=eval_score if has_rl else 0.0, teacher_cands=tuple(teacher_cands))

    def make_sample(self, record: AgentRecord, judgment: TurnJudgment) -> PolicySample | None:
        # 1. A declined judgment never trains. (Narrow on the score itself:
        #    it is what step 2 needs, and a property would not narrow it.)
        if judgment.score is None:
            return None
        # 2. The policy tensors must satisfy the bridge contract, and the
        #    top-K capture must cover every response token.
        sample = make_policy_sample(record, judgment.score)
        if policy_row_violation(sample.tokens, sample.loss_mask, sample.rollout_log_probs) is not None:
            return None
        if not sample.topk_indices or len(sample.topk_indices) != len(sample.loss_mask):
            return None
        # 3. Every trained sample carries at least one teacher candidate —
        #    upstream constructs one per sample by design (RL-only turns get
        #    the anchor), so a candidate-less judgment here means candidate
        #    construction failed and the turn retires instead of training a
        #    shape the reference never produces.
        if not judgment.teacher_cands:
            return None
        validated = validate_teacher_cands(judgment.teacher_cands, sample)
        if validated is None:
            return None
        return replace(sample, extras={**sample.extras, "teacher_cands": validated})

    def make_batch(self, samples: tuple[PolicySample, ...], batch_number: int) -> PolicyBatch:
        self._record_batch(samples, batch_number)
        return PolicyBatch(f"{self.scenario}:openclawrl:{batch_number}", samples)

    def _record_batch(self, samples: tuple[PolicySample, ...], batch_number: int) -> None:
        """Append this batch's judged population to the PRM record file.

        One line per batch: what the judges returned since the previous one,
        and the reward mix that reached the trainer. A batch mean alone
        cannot distinguish "the judges disagree" from "the judges never
        ran", and neither can it show which next states are being judged.
        """
        record = {
            "batch": batch_number,
            "scenario": self.scenario,
            "samples": len(samples),
            "rewards": dict(sorted(collections.Counter(f"{s.reward:+.0f}" for s in samples).items())),
            "judged": dict(sorted(self._judged.items())),
        }
        self._judged.clear()
        if self._record_path is None:
            return
        try:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            with self._record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            # Diagnostics must never take a training step down with them.
            logger.warning("openclawrl: could not append to %s", self._record_path, exc_info=True)

    def expire(self, now: float) -> tuple[str, ...]:
        """Sessions idle past the TTL are over: drop them from the index and
        retire the turn each was still holding.

        Only a session's newest turn is ever unbound — every earlier one was
        dispatched the moment its successor arrived — so what comes back is
        one turn per dead session, a turn whose next state will never come.
        Without this the index would grow for the life of the process and
        those records would stay protected from compaction forever.
        """
        return self._index.expire(now)
