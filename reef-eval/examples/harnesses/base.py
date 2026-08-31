"""The one base class for benchmark harness adapters.

Every harness here follows the same standard operating procedure,
encoded as the ``run`` template:

1. ``setup``: install the pinned tool into the container (Harbor's hook;
   native agents provide their own).
2. ``_prepare``: upload configs, seeds, and graders. Runs before the
   ``try``, so a preparation failure fails the trial.
3. ``_launch``: the long-horizon run itself. A timeout here is a normal
   ending, not a failure.
4. ``_finalize``: make sure the verifier has something to grade by
   submitting the run's final artifact if the judge saw zero
   submissions. Does nothing by default.
5. ``_collect_usage``: meter tokens into Harbor's ``AgentContext``.
   Does nothing by default.

``run`` is the template and is not overridden. Phases 4 and 5 run in a
``finally`` because a stopped run still left artifacts and spent tokens,
and each is best-effort so that metering or fallback trouble never masks
the run's own outcome.

Custom frameworks (OpenEvolve, CORAL) subclass ``TideHarnessBase``
directly and use its command-pipeline utilities (``_checked``,
``_populate_usage``, ...). Adapters for Harbor-native agents list it
first and delegate ``_launch`` to Harbor's own agent; see
``codex.CodexHarness``.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent

logger = logging.getLogger(__name__)

HARNESS_FINALIZE = Path(__file__).parent / "finalize.py"
REMOTE_FINALIZE = "/tmp/reef_eval_finalize.py"


class TideHarnessBase(BaseAgent):
    """The procedure template plus the helpers every harness runs on."""

    remote_root = Path("/opt/reef-eval-harness")

    # ------------------------------------------------------------- SOP

    async def run(self, instruction, environment, context) -> None:
        prepared = await self._prepare(instruction, environment)
        try:
            await self._launch(prepared, instruction, environment, context)
        finally:
            await self._best_effort("finalize", self._finalize(environment))
            await self._best_effort(
                "collect usage", self._collect_usage(environment, context)
            )

    async def _prepare(self, instruction, environment) -> Any:
        """Bundle and upload everything the launch needs. May raise."""

    async def _launch(self, prepared, instruction, environment, context) -> None:
        """Run the framework against the task. Timeout is a normal ending."""
        raise NotImplementedError

    async def _finalize(self, environment) -> None:
        """Guarantee the verifier a result (e.g. final-artifact fallback)."""

    async def _collect_usage(self, environment, context) -> None:
        """Record measured tokens onto Harbor's AgentContext."""

    async def _best_effort(self, phase: str, coro) -> None:
        try:
            await coro
        except Exception:
            logger.warning("harness %s phase failed; continuing", phase, exc_info=True)

    async def _submit_final_artifact(self, environment, artifact: str) -> None:
        """Submit ``artifact`` to the judge iff it saw zero submissions."""
        await environment.upload_file(HARNESS_FINALIZE, REMOTE_FINALIZE)
        await environment.exec(
            command=f"python3 {REMOTE_FINALIZE} {shlex.quote(artifact)}"
        )

    # ------------------------------------ command-pipeline utilities

    async def _checked(
        self,
        environment,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ):
        """Execute a command and raise with a useful tail when it fails."""
        result = await environment.exec(command=command, env=env)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "")[-2_000:]
            raise RuntimeError(
                f"harness command failed ({result.return_code}): {detail}"
            )
        return result

    def _model_name(self) -> str:
        """Normalize Harbor's optional provider-qualified model name."""
        if not self.model_name:
            raise ValueError("model_name is required for this harness")
        return self.model_name.split("/", 1)[-1]

    @staticmethod
    def _require_api_key(env: dict[str, str]) -> None:
        """Require the API key supplied through Harbor's environment mapping."""
        if not env.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY must be passed through the Harbor agent's env field"
            )

    def _populate_usage(self, context, value: str) -> None:
        """Populate Harbor's standard token fields from JSONL output."""
        records = _parse_usage_jsonl(value)
        if not records:
            return
        _populate_usage_context(context, records)


def _parse_usage_jsonl(value: str) -> list[dict[str, Any]]:
    """Parse normalized token-usage records from mixed command output."""
    records: list[dict[str, Any]] = []
    for line in value.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and all(
            isinstance(record.get(field), int) and record[field] >= 0
            for field in ("input_tokens", "cached_input_tokens", "output_tokens")
        ):
            records.append(record)
    return records


def _populate_usage_context(context, records: list[dict[str, Any]]) -> None:
    """Populate Harbor's standard token fields from usage records."""
    if not records:
        return

    context.n_input_tokens = sum(record["input_tokens"] for record in records)
    context.n_cache_tokens = sum(record["cached_input_tokens"] for record in records)
    context.n_output_tokens = sum(record["output_tokens"] for record in records)

    metadata = dict(context.metadata or {})
    metadata["usage_records"] = len(records)
    context.metadata = metadata
