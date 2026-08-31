"""SkillClaw as a pool method on harness evolution: the method module skillclaw.yaml references.

``propose`` runs the sealed night flow (benchmarks/skill_claw at commit
0519eefb) over the day's batched traffic: rebuild each task's session
digest, summarize every session, judge the unscored ones, group by
referenced skill, then one decision per skill group plus the no-skill
bucket - ``improve_skill``, ``optimize_description``, ``create_skill``, or
``skip``, exactly the sealed action set. The decisions land on a scratch
pool (night.night_step, merge and registry semantics included) and the
pool diff maps to one composite mutation sequence: one ``create`` or
``update`` per changed skill, applied under one snapshot and settled under
one gate verdict. The sealed night never removes a skill, so no ``remove``
mutation is ever proposed. A night that applies nothing returns ``None``
and the pool version stands.

``evaluate`` grades the probe episodes by exact final answer. With
``selection: always`` in skillclaw.yaml those scores are recorded, never
gating: SkillClaw publishes every non-skip night and the next day's
traffic judges the pool, so the ungated mode is the method's own regime.

The day's task metadata (score, success, error, breakdown) travels through
the report files the driver writes per task (``round-<n>/reports/``); a
sample without one falls back to what the trace itself carries, with the
-1.0 sentinel score mapping back to unscored.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from . import evolver, night, prompts
from .config import RUN, SUCCESS_THRESHOLD, UNSCORED_SENTINEL, WORKDIR
from .sessions import annotate_score, build_aggregate_session, parse_recorded, read_skill_names


class EpisodeResultLike(Protocol):
    trajectory: tuple[dict[str, Any], ...]


class TraceSampleLike(Protocol):
    score: float
    payload: dict[str, Any]
    source_agent_record_id: str


#: Expected final answers of the probe tasks, keyed by the stable prefix
#: each task starts with (the tasks live in skillclaw.yaml's evolution section).
ANSWERS = {
    "[sieve]": "9592",
    "[fib]": "2880067194370816120",
    "[csv]": "30",
}


def _report_index() -> dict[str, dict[str, Any]]:
    """The day ledger: report metadata keyed by the referenced record id."""
    index: dict[str, dict[str, Any]] = {}
    for path in sorted((WORKDIR / RUN).glob("round-*/reports/*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        reference = str(meta.get("reference") or "")
        if reference:
            index[reference] = meta
    return index


def _fallback_meta(sample: TraceSampleLike) -> dict[str, Any]:
    """Task metadata derived from the trace alone, for a sample without a
    ledger entry. The -1.0 sentinel is the unscored report, never a grade.

    The wire prompt is the composed one (preamble, optional hint, task
    text); the digest wants the task text, so the fixed preamble is
    stripped back off. Success cannot check the no-error rule here: the
    error channel only exists in the ledger this sample is missing."""
    score: float | None = None if sample.score == UNSCORED_SENTINEL else float(sample.score)
    prompt = ""
    for message in sample.payload.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                prompt = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                prompt = str(content)
            break
    preamble_tail = prompts.AGENT_PREAMBLE.rsplit("{timeout_seconds}", 1)[-1]
    if preamble_tail and preamble_tail in prompt:
        prompt = prompt.split(preamble_tail, 1)[1]
    round_match = re.search(r"-r(\d+)-", sample.source_agent_record_id or "")
    return {
        "task_id": sample.source_agent_record_id,
        "prompt": prompt,
        "round": int(round_match.group(1)) if round_match else 0,
        "score": score,
        "success": score is not None and score >= SUCCESS_THRESHOLD,
        "error": "",
        "breakdown": {},
    }


def digest(sample: TraceSampleLike, meta: dict[str, Any]) -> dict[str, Any]:
    """One task's aggregated session from its recorded traffic and verdict."""
    task_id = str(meta.get("task_id") or sample.source_agent_record_id)
    source = parse_recorded(dict(sample.payload), task_id=task_id)
    score = meta.get("score")
    scored = isinstance(score, (int, float))
    if scored:
        annotate_score(source, float(score))
    record = {
        "device_id": "reef",
        "score": float(score) if scored else None,
        "success": bool(meta.get("success")),
        "error": str(meta.get("error") or ""),
        "breakdown": meta.get("breakdown") if isinstance(meta.get("breakdown"), dict) else {},
        "response_text": source.get("response_text", ""),
        "used_skills": read_skill_names(source),
    }
    return build_aggregate_session(
        task_id=task_id,
        prompt=str(meta.get("prompt") or ""),
        record=record,
        source=source,
        round_index=int(meta.get("round") or 0),
    )


def _materialize_incumbent(nodes: tuple[tuple[str, Any], ...], root: Path) -> None:
    """The skill nodes as the on-disk pool the sealed night reads."""
    for kind, options in nodes:
        if kind != "skill" or not isinstance(options, dict):
            continue
        name = str(options.get("name") or "").strip()
        if not name:
            continue
        skill_md = root / "skills" / name / "SKILL.md"
        skill_md.parent.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(str(options.get("text") or ""), encoding="utf-8")


def _pool_mutations(incumbent: dict[str, str], evolved: dict[str, str]) -> Sequence[object]:
    """The night's pool diff as node mutations, name order.

    The sealed night only adds or rewrites ``skills/<name>/SKILL.md``
    entries, so the diff is creates and updates; there is no remove.
    """
    from reef.train.harness_backend import Mutation  # lazy: this module imports without reef installed

    mutations = []
    for name in sorted(evolved):
        text = evolved[name]
        if name in incumbent and incumbent[name] == text:
            continue
        op = "update" if name in incumbent else "create"
        mutations.append(Mutation(op, name, {"name": "skill", "config": {"name": name, "text": text}}))
    return tuple(mutations)


def propose(
    nodes: tuple[tuple[str, Any], ...],
    samples: tuple[TraceSampleLike, ...],
    models: evolver.ModelsLike,
) -> Sequence[object] | None:
    """One night step over a day of traffic: incumbent pool in, mutations out.

    Rebuilds the digests the sealed experiment feeds the evolver and runs
    the unchanged night pipeline (summarize, judge backfill, group, one
    decision per skill group plus the no-skill bucket, with every change selected).
    """
    if not samples:
        return None
    index = _report_index()
    metas = [index.get(sample.source_agent_record_id) or _fallback_meta(sample) for sample in samples]
    round_index = max((int(meta.get("round") or 0) for meta in metas), default=0)
    # The sealed backend drains session files in name order.
    digests = sorted(
        (digest(sample, meta) for sample, meta in zip(samples, metas, strict=True)),
        key=lambda session: str(session.get("session_id")),
    )
    work_dir = WORKDIR / RUN / f"round-{round_index}" / "night"
    incumbent = Path(tempfile.mkdtemp(prefix="skillclaw-incumbent-"))
    try:
        _materialize_incumbent(nodes, incumbent)
        # The evolver is the model under test: reef's served binding, wrapped
        # in their chat loop. Resolved at call time so a test can swap it.
        evolved = night.night_step(
            sessions=digests,
            incumbent=incumbent,
            work_dir=work_dir,
            frozen=False,
            llm=evolver.chat_client(models.served),
        )
        (work_dir / "audit.json").parent.mkdir(parents=True, exist_ok=True)
        (work_dir / "audit.json").write_text(
            json.dumps({"advanced": evolved["advanced"], "audit": evolved["audit"]}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        if not evolved["advanced"]:
            return None
        mutations = _pool_mutations(night.read_pool(incumbent), night.read_pool(Path(evolved["pool"])))
        return mutations or None
    finally:
        shutil.rmtree(incumbent, ignore_errors=True)


def evaluate(task: str, result: EpisodeResultLike) -> float:
    """Grade the last line of the probe episode's final assistant text, 1.0 exact."""
    return grade_probe(task, _final_assistant_text(result.trajectory))


def grade_probe(task: str, text: str | None) -> float:
    """The probe grader: 1.0 when the last non-empty line is the expected
    answer for the task's prefix, else 0.0."""
    expected = next((answer for prefix, answer in ANSWERS.items() if task.startswith(prefix)), None)
    if expected is None or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == expected else 0.0


def _final_assistant_text(trajectory: tuple[dict[str, Any], ...]) -> str | None:
    """The final assistant text in a session log, tolerant of both flat
    role/content events and pi's wrapped message events with text parts."""
    for event in reversed(trajectory):
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, dict) else event
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            texts = [part for part in parts if isinstance(part, str)]
            if texts:
                return "\n".join(texts)
    return None
