"""OpenClaw-RL's record-to-record stage: every main turn is judged by its next state.

:class:`OpenClawRLRecord2Record` is a :class:`~reef.record2record.JudgingRecord2Record`
that runs :class:`OpenClawRLJudging` on a PRM worker. ``observe`` correlates
main turns into sessions and hands each completed turn over for judgment;
``judge`` is upstream's combine dispatch. The deployment configures the stage
in ``pipeline.record2record.config`` (:class:`OpenClawRLJudgingSettings`).

No proxy and no judgment on the wire: the method derives every verdict
from the traffic the agent already produces. Correlation prefers a
harness-stamped ``x-reef-tag-session`` and falls back to trace matching,
which holds only for agents that resend their transcript each turn.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from recipes.openclawrl.prm import (
    DirectiveTeacher,
    PRMJudge,
    ResponseRenderer,
    append_hint_to_messages,
    collect_hint_candidates,
    render_response_fallback,
)
from recipes.openclawrl.sessions import DEFAULT_MAX_SESSIONS, SessionIndex
from recipes.openclawrl.turns import TurnJob, TurnJudgment, main_turn_message, turn_job, turn_request_messages
from reef.background_jobs import BackgroundJobWorker
from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.records_types import AgentRecord, RequestType
from reef.pipeline_context import PipelineContext
from reef.record2record import JudgingRecord2Record, RecordJudging

logger = logging.getLogger(__name__)

# How often a run may repeat the "nothing is binding" warning, and how many
# dead-on-arrival sessions it takes to raise it at all. A conversation whose
# last turn never got a next state is ordinary; a run where those outnumber
# the turns that did bind is a correlation failure.
UNBOUND_WARN_INTERVAL_S = 300.0
UNBOUND_WARN_FLOOR = 8


def session_tag(payload: Any) -> str | None:
    """The harness-declared conversation id (``x-reef-tag-session``), if any."""
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    tags = metadata.get("tags") if isinstance(metadata, dict) else None
    value = tags.get("session") if isinstance(tags, dict) else None
    return str(value).strip() or None if value is not None else None


@dataclass(frozen=True)
class OpenClawRLJudgingSettings(PluginSettings):
    """The config of ``OpenClawRLRecord2Record``: session correlation and the PRM judge.

    An empty ``prm_url`` is correlate-only mode: sessions resolve, nothing is
    judged, and nothing trains.
    """

    session_ttl_s: float = config_option(900.0, help="The idle time in seconds before a session expires.")
    # The trace-matching fallback opens one session for each request that it
    # cannot match, so without a ceiling an unmatched client grows the table
    # with the request rate for a whole TTL.
    max_open_sessions: int = config_option(
        DEFAULT_MAX_SESSIONS, help="The live sessions, at most; the least recently active are evicted above it."
    )
    prm_url: str = config_option("", help="The PRM's SGLang server; empty is correlate-only mode.")
    prm_tokenizer_path: str = config_option("", help="The PRM's tokenizer. Required when prm_url is set.")
    prm_m: int = config_option(3, help="The judge votes for each turn.")
    prm_temperature: float = config_option(0.6, help="The judge sampling temperature.")
    prm_max_tokens: int = config_option(8192, help="The judge generation budget in tokens.")
    prm_timeout_s: float = config_option(120.0, help="The timeout of one judge call in seconds.")
    prm_concurrency: int = config_option(8, help="The turns judged at the same time.")
    prm_max_hint_candidates: int = config_option(3, help="The hint candidates collected for each turn, at most.")

    def __post_init__(self) -> None:
        if self.session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        if self.max_open_sessions <= 0:
            raise ValueError("max_open_sessions must be positive")
        if self.prm_url.strip() and not self.prm_tokenizer_path.strip():
            raise ValueError("OpenClaw-RL judging requires prm_tokenizer_path next to prm_url")
        if self.prm_m <= 0:
            raise ValueError("prm_m must be positive")
        if self.prm_temperature < 0:
            raise ValueError("prm_temperature must be non-negative")
        if self.prm_max_tokens <= 0:
            raise ValueError("prm_max_tokens must be positive")
        if self.prm_timeout_s <= 0:
            raise ValueError("prm_timeout_s must be positive")
        if self.prm_concurrency <= 0:
            raise ValueError("prm_concurrency must be positive")
        if self.prm_max_hint_candidates <= 0:
            raise ValueError("prm_max_hint_candidates must be positive")


def prm_clients(settings: OpenClawRLJudgingSettings) -> tuple[Any, Any, Any]:
    """The PRM judge, the directive teacher, and the response renderer that ``settings`` describe.

    Without ``prm_url`` all three are ``None``: correlate-only mode.
    """
    url = settings.prm_url.strip()
    if not url:
        return None, None, None
    tokenizer_path = settings.prm_tokenizer_path.strip()
    prm = PRMJudge(
        generate_url=url.rstrip("/") + "/generate",
        tokenizer_path=tokenizer_path,
        m=settings.prm_m,
        temperature=settings.prm_temperature,
        max_tokens=settings.prm_max_tokens,
        timeout_s=settings.prm_timeout_s,
    )
    return prm, DirectiveTeacher(tokenizer_path=tokenizer_path), ResponseRenderer(tokenizer_path)


class OpenClawRLJudging(RecordJudging):
    """Turns judged by their next state; terminal exclusions mirror upstream.

    Side calls, retried turns, session-final turns, and reports retire
    terminal instead of ever being judged. ``clients`` (PRM judge, directive
    teacher, renderer) overrides the configured clients for tests.
    """

    def __init__(self, settings: OpenClawRLJudgingSettings, clients: tuple[Any, Any, Any] | None = None) -> None:
        self.prm, self.teacher, self.renderer = clients if clients is not None else prm_clients(settings)
        self.max_hint_candidates = settings.prm_max_hint_candidates
        self.index = SessionIndex(settings.session_ttl_s, max_sessions=settings.max_open_sessions)
        # Correlation reports itself through the log, not the record file:
        # the failure worth catching is "nothing ever bound", and a run in
        # that state never produces the batch that would write a line.
        self.warned_untagged = False
        self.warned_evicted = False
        self.last_unbound_warning = -UNBOUND_WARN_INTERVAL_S
        # What the judge actually saw, since the processor last wrote the PRM record file.
        self.judged: collections.Counter[str] = collections.Counter()

    def observe(self, item: AgentRecord, now: float, stage: JudgingRecord2Record) -> None:
        # 1. The stage has caught up: expiry ran before this read, so an
        #    expired session can never capture this request as its successor.
        # 2. Reports never train here; owning them keeps the store
        #    compactable (nothing else would ever release them).
        if item.request_type is not RequestType.INFERENCE:
            stage.retire(item.agent_record_id)
            return
        # 3. Side calls (titles, compaction — no tools) never train.
        payload = item.payload
        if main_turn_message(payload) is None:
            stage.retire(item.agent_record_id)
            return
        # 4. Correlate into a session: a retry is terminal; the request that
        #    follows a turn delivers its next state — send that turn to the
        #    judging worker. A harness-stamped ``session`` tag names the
        #    conversation outright; without one this falls back to trace
        #    matching, which only holds for agents that resend their whole
        #    transcript every turn. Every main turn is judged against
        #    whatever arrived next — a tool result is a next state, and its
        #    role is an input to the judge rather than a filter.
        observation = self.index.observe(
            item.agent_record_id,
            turn_request_messages(payload),
            now,
            session_tag=session_tag(payload),
        )
        if observation.duplicate:
            stage.retire(item.agent_record_id)
            return
        if observation.binding is not None:
            record = stage.tracked_record(observation.binding.receipt)
            if record is not None:
                stage.dispatch(turn_job(record, observation.binding))
        # 5. This turn now awaits its own next state.
        stage.track(item)

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
        if self.renderer is not None:
            response_text = await asyncio.to_thread(
                self.renderer.render,
                list(job.request_messages),
                dict(job.response_message),
                tools=job.request_tools,
            )
        response_text = response_text or render_response_fallback(job.response_message)
        # 2. Both PRM judges vote.
        eval_score, hint_votes = await asyncio.gather(
            self.prm.evaluate(response_text, job.next_state_text, job.next_state_role),
            self.prm.evaluate_hint_votes(response_text, job.next_state_text, job.next_state_role),
        )
        # 3. The accepted hindsight hints, shortest first, capped.
        candidates = collect_hint_candidates(hint_votes, max_cand=self.max_hint_candidates)
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
                ids = await self.teacher.candidate_tokens(
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
        self.judged[f"next_state:{job.next_state_role}"] += 1
        self.judged[f"eval:{eval_score:+.0f}" if has_rl else "eval:0"] += 1
        self.judged["hint_accepted" if candidates else "hint_declined"] += 1
        self.judged["anchor_only" if (not candidates and teacher_cands) else "teacher_cands"] += 1
        if not has_rl and not teacher_cands:
            self.judged["declined"] += 1
            return TurnJudgment(job.receipt)
        return TurnJudgment(job.receipt, score=eval_score if has_rl else 0.0, teacher_cands=tuple(teacher_cands))

    def expire(self, now: float) -> tuple[str, ...]:
        """Sessions idle past the TTL are over: drop them from the index and
        retire the turn each was still holding.

        Only a session's newest turn is ever unbound — every earlier one was
        dispatched the moment its successor arrived — so what comes back is
        one turn per dead session, a turn whose next state will never come.
        Without this the index would grow for the life of the process and
        those records would stay protected from compaction forever.
        """
        retired = self.index.expire(now)
        self.warn_on_correlation(now)
        return retired

    def warn_on_correlation(self, now: float) -> None:
        """Say out loud when correlation is producing nothing.

        A turn that never binds is never judged and never trains, and it does
        so without an error: the only symptom is a training set that stays
        empty, which is invisible without reading this file. WARNING because
        reef configures no logging and anything below it is discarded; rate
        limited because this is reached once per record.
        """
        stats = self.index.stats
        if stats.untagged and not self.warned_untagged:
            self.warned_untagged = True
            logger.warning(
                "openclawrl: correlating by trace matching — this traffic carries no "
                "x-reef-tag-session tag. Trace matching binds only agents that resend their "
                "whole transcript on every request; an agent that keeps or rewrites its "
                "history locally yields no trainable turns. Stamp x-reef-tag-session, stable "
                "and unique per conversation, to correlate reliably."
            )
        if stats.evicted and not self.warned_evicted:
            self.warned_evicted = True
            logger.warning(
                "openclawrl: the session table hit max_open_sessions and is evicting the least "
                "recently active sessions (%d so far); their pending turns retire untrained. "
                "Unmatched requests each open a session, so this usually means correlation is "
                "failing rather than that the traffic has that many live conversations.",
                stats.evicted,
            )
        if (
            stats.dropped_unbound >= UNBOUND_WARN_FLOOR
            and stats.dropped_unbound > stats.bound
            and now - self.last_unbound_warning >= UNBOUND_WARN_INTERVAL_S
        ):
            self.last_unbound_warning = now
            logger.warning(
                "openclawrl: %d session(s) expired without ever binding a turn, against %d bound "
                "turn(s). Unbound turns are never judged and never train — check that every "
                "inference in a conversation carries the same x-reef-tag-session value.",
                stats.dropped_unbound,
                stats.bound,
            )


class OpenClawRLRecord2Record(JudgingRecord2Record):
    """OpenClaw-RL's record stage: sessions correlated, and turns judged on a PRM worker.

    Config: :class:`OpenClawRLJudgingSettings`. Without a ``worker``, the
    stage starts one when the config names a PRM judge. ``worker`` and
    ``clients`` (PRM judge, directive teacher, renderer) replace the
    configured ones in tests.
    """

    def __init__(
        self,
        context: PipelineContext,
        settings: OpenClawRLJudgingSettings,
        *,
        worker: BackgroundJobWorker | None = None,
        clients: tuple[Any, Any, Any] | None = None,
    ) -> None:
        #: The judging that the batches read for the PRM record file.
        self.openclawrl_judging = OpenClawRLJudging(settings, clients)
        if worker is None and self.openclawrl_judging.prm is not None:
            worker = BackgroundJobWorker(
                self.openclawrl_judging, thread_name="openclawrl-judging", concurrency=settings.prm_concurrency
            )
        super().__init__(context, self.openclawrl_judging, worker)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, OpenClawRLJudgingSettings.parse(config, plugin_class=cls))
