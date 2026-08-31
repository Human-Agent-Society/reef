"""Version-pinned Harbor Codex adapter."""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.codex import Codex

from examples.harnesses.base import TideHarnessBase

CODEX_VERSION = "0.147.0"


class CodexHarness(TideHarnessBase, Codex):
    """Use Harbor's standard non-interactive Codex agent at a fixed version.

    ``TideHarnessBase`` comes first so the SOP's ``run`` template wins over
    ``Codex.run``; the ``_launch`` phase then delegates to Harbor's own
    non-interactive agent. Installation, auth, and trajectory-based usage
    collection are all Harbor's, so ``_collect_usage`` needs nothing
    (Harbor's ``populate_context_post_run`` meters usage from the native
    trajectory).

    ``final_artifact`` (default ``"solution.json"``, relative to the task
    image's workdir; pass ``None`` to disable) is the workspace file the
    task's judge scores. The ``_finalize`` phase submits it to the judge
    *only if the run never submitted anything*: the verifier grades the
    judge's submission log, so without this fallback a run that never
    called ``$JUDGE_URL/submit`` scores 0.0 regardless of what it left in
    the workspace. When the agent did submit, best-of semantics already
    cover the run and the fallback stays out of the way.
    """

    def __init__(
        self,
        *args: Any,
        version: str = CODEX_VERSION,
        final_artifact: str | None = "solution.json",
        **kwargs: Any,
    ) -> None:
        if version != CODEX_VERSION:
            raise ValueError(f"CodexHarness requires Codex {CODEX_VERSION}")
        super().__init__(*args, version=version, **kwargs)
        self.final_artifact = final_artifact

    async def _launch(self, prepared, instruction, environment, context) -> None:
        # Harbor's own non-interactive run: auth + `codex exec --json`.
        await Codex.run(self, instruction, environment, context)

    async def _finalize(self, environment) -> None:
        if self.final_artifact is None:
            return
        workdir = ((await environment.exec(command="pwd")).stdout or "").strip()
        artifact = self.final_artifact
        if not artifact.startswith("/"):
            artifact = f"{workdir.rstrip('/')}/{artifact}"
        await self._submit_final_artifact(environment, artifact)
