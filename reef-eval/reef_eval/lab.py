"""The Lab: reef-eval's one public surface.

A Lab is a directory. Inside it live the results database (``results.sqlite``)
and, when the Harbor executor is used, the Harbor trial directories
(``trials/``). Everything a Lab ever does is:

- ``run()``: execute one episode (a Harbor task under an agent), store the
  trusted reward plus the judge's submission log, and skip work that
  already has a stored result.
- ``df()``: everything as a pandas DataFrame; metrics are queries.

Persistence lives in the data: there is no daemon, and re-running a crashed
script resumes it because completed keys are skipped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from reef_eval.budget import Budget
from reef_eval.executors import Executor, HarborExecutor
from reef_eval.types import EpisodeSpec, Row, Tags

logger = logging.getLogger("reef_eval")


class Lab:
    def __init__(
        self,
        root: Path | str,
        *,
        executor: Executor | None = None,
        concurrency: int = 4,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        from reef_eval.store import Store

        self.store = Store(self.root / "results.sqlite")
        self.executor: Executor = executor or HarborExecutor(self.root / "trials")
        self._semaphore = asyncio.Semaphore(concurrency)
        # In-flight episode deduplication: the first caller for a key runs the
        # executor; concurrent callers with the same key await its result
        # instead of racing on a second execution and a duplicate insert.
        # One asyncio task == one thread, so the claim check-and-set below is
        # atomic as long as it stays await-free; the lock documents that.
        self._inflight: dict[str, asyncio.Future[Row]] = {}
        self._inflight_lock = asyncio.Lock()
        self._warned_no_judge = False

    # ------------------------------------------------------------- episodes

    async def run(
        self,
        task: str,
        agent: dict[str, Any],
        *,
        tags: Tags | None = None,
        budget: Budget | dict[str, Any] | float | int | str | None = None,
        key: str | None = None,
        **overrides: Any,
    ) -> Row:
        """Run one episode and store its trusted result.

        ``key`` is the episode's stable ID; when omitted it is derived
        from (task, agent, tags, overrides), so identical calls are one
        episode. If the key already has a stored row, that row is
        returned and nothing runs.
        ``overrides`` pass through to the executor (for Harbor: TrialConfig
        fields such as ``verifier=...`` or ``timeout_multiplier=...``).

        ``budget`` bounds the run across any of three dimensions (time,
        submissions, tokens); see :class:`reef_eval.budget.Budget`.
        A bare number is hours. It sets the timeout, hands the agent
        ``REEF_EVAL_*`` budget-signal env vars, and tags the episode with its
        budget so runs group and pivot by it. What was actually spent comes
        back as ``used_*`` columns.
        """
        tags = dict(tags or {})
        agent = dict(agent)
        overrides = dict(overrides)
        budget = Budget.coerce(budget)
        if budget is not None and not budget.is_empty:
            self._warn_if_no_judge(task, budget)
            tags = {**budget.to_tags(), **tags}
            if budget.timeout_sec() is not None:
                agent.setdefault("override_timeout_sec", budget.timeout_sec())
            env = budget.to_env()
            if env:
                agent["env"] = {**env, **agent.get("env", {})}  # local + exec delivery
                env_cfg = dict(overrides.get("environment") or {})
                env_cfg["env"] = {
                    **env,
                    **(env_cfg.get("env") or {}),
                }  # container startup
                overrides["environment"] = env_cfg
        spec = EpisodeSpec(task=task, agent=agent, overrides=overrides)
        key = key or self._default_key(spec, tags)

        existing = self.store.get(key)
        if existing is not None:
            logger.info("skip %s (already recorded)", key)
            return existing

        # Claim leadership for this key. Only the leader executes; identical
        # concurrent callers find the in-flight future and await it, so the
        # executor runs exactly once and every caller gets the same Row.
        async with self._inflight_lock:
            in_flight = self._inflight.get(key)
            if in_flight is None:
                leader = True
                in_flight = asyncio.get_running_loop().create_future()
                self._inflight[key] = in_flight
            else:
                leader = False

        if not leader:
            return await in_flight

        try:
            async with self._semaphore:
                # A retried episode may have left trace rows behind; clear them so
                # the trace is never double-recorded.
                self.store.delete_prefix(f"{key}#")
                logger.info("run %s", key)
                result = await self.executor.execute(spec)

            for i, point in enumerate(result.trace):
                self.store.put(
                    Row(
                        key=f"{key}#t{i}",
                        kind="trace",
                        task=task,
                        # The log's own fields go first: the episode's tags
                        # and t identify the row and must not be shadowed by
                        # whatever a task recorded alongside a submission.
                        tags={**point.data, **tags, "t": point.t},
                        rewards={"score": point.score},
                        uri=result.uri,
                    )
                )

            used = {f"used_{k}": v for k, v in result.usage.items()}
            row = Row(
                key=key,
                kind="episode",
                task=task,
                tags={
                    **tags,
                    **used,
                    **({"error": result.error} if result.error else {}),
                    **(
                        {"agent_exit_code": result.agent_exit_code}
                        if result.agent_exit_code
                        else {}
                    ),
                },
                rewards=dict(result.rewards),
                uri=result.uri,
            )
            self.store.put(row)
            logger.info("done %s: %s", key, row.rewards or result.error or "no reward")
            in_flight.set_result(row)
            return row
        except BaseException as exc:
            # Executor failure or cancellation: surface the same outcome to
            # every waiter and release the claim so a later call can retry.
            # Nothing has been stored, so the store stays duplicate-free.
            if not in_flight.done():
                in_flight.set_exception(exc)
            raise
        finally:
            # Single await-free dict op: atomic in one asyncio task, and
            # safe to run even while unwinding a cancellation.
            self._inflight.pop(key, None)

    async def run_many(self, calls: list[dict[str, Any]]) -> list[Row]:
        """Run many episodes concurrently (bounded by the Lab's concurrency).

        Each entry is a kwargs dict for :meth:`run`. Order of results matches
        the order of calls.
        """
        return list(await asyncio.gather(*(self.run(**call) for call in calls)))

    # -------------------------------------------------------------- queries

    def df(self, kind: str | None = None) -> pd.DataFrame:
        return self.store.df(kind)

    # -------------------------------------------------------------- helpers

    def _warn_if_no_judge(self, task: str, budget: Budget) -> None:
        """Warn when a submission cap is set on a task that has no judge.

        ``max_submissions`` is enforced by the judge sidecar, so a task
        graded only by its verifier ignores it. The budget is still tagged
        on the row, which makes the no-op easy to miss. Warns once per Lab,
        and only for local task directories (a registry id is not readable
        from here).
        """
        if budget.max_submissions is None or self._warned_no_judge:
            return
        task_dir = Path(task)
        if not task_dir.is_dir():
            return
        if (task_dir / "environment" / "judge_config.json").is_file():
            return
        self._warned_no_judge = True
        logger.warning(
            "%s has no judge (environment/judge_config.json), so its "
            "max_submissions=%s does nothing: the task is graded by its "
            "verifier after the episode and has nothing to submit to. "
            "Bound this run with time or tokens instead.",
            task,
            budget.max_submissions,
        )

    @staticmethod
    def _digest(obj: Any) -> str:
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def _default_key(cls, spec: EpisodeSpec, tags: Tags) -> str:
        # Normalize a filesystem-path task to its canonical absolute form so
        # that "tasks/x", "tasks/x/", and an absolute path all key to one
        # episode (otherwise resume silently re-runs). Harbor registry ids
        # (non-paths) are hashed verbatim.
        task_path = Path(spec.task)
        task_id = str(task_path.resolve()) if task_path.exists() else spec.task
        digest = cls._digest(
            {
                "task": task_id,
                "agent": spec.agent,
                "tags": tags,
                "overrides": spec.overrides,
            }
        )
        short_task = spec.task.rstrip("/").split("/")[-1]
        return f"{short_task}:{digest}"
