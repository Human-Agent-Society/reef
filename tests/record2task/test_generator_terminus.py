"""The generator's Terminus 2: an episode that fills its context or fails a send-keys still reaches the verifier."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

harbor_terminus = pytest.importorskip("harbor.agents.terminus_2", reason="harbor is not installed")
tmux_session = pytest.importorskip("harbor.agents.terminus_2.tmux_session")
harbor_environments = pytest.importorskip("harbor.environments.base")
harbor_llms = pytest.importorskip("harbor.llms.base")
agent_context = pytest.importorskip("harbor.models.agent.context")
generator_terminus = pytest.importorskip("reef.record2task.generator.terminus")


def terminal_lost() -> RuntimeError:
    """The error Terminus 2 raises when its tmux server is gone, built by harbor itself."""
    session = SimpleNamespace(environment=SimpleNamespace(session_id="task__abc__env"))
    result = harbor_environments.ExecResult(
        return_code=1, stdout="", stderr="no server running on /tmp/tmux-0/default\n"
    )
    return tmux_session.TmuxSession._send_keys_error(
        session, "non-blocking", "tmux send-keys -t terminus-2 -- 'ls\\n'", result
    )


def run_raising(monkeypatch: pytest.MonkeyPatch, error: BaseException | None) -> agent_context.AgentContext:
    async def parent_run(self, instruction, environment, context) -> None:
        context.metadata = {"n_episodes": 3}
        if error is not None:
            raise error

    monkeypatch.setattr(harbor_terminus.Terminus2, "run", parent_run)
    agent = object.__new__(generator_terminus.VerifiedTerminus2)
    agent.logger = logging.getLogger("test")
    context = agent_context.AgentContext()
    asyncio.run(agent.run("task", SimpleNamespace(), context))
    return context


@pytest.mark.parametrize(
    ("error", "reason"),
    [(harbor_llms.ContextLengthExceededError(), "context_length_exceeded"), (terminal_lost(), "terminal_lost")],
)
def test_a_full_context_or_a_lost_terminal_ends_the_run_for_verification(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, reason: str
) -> None:
    context = run_raising(monkeypatch, error)
    assert context.metadata == {"n_episodes": 3, generator_terminus.EPISODE_END_KEY: reason}


def test_a_finished_run_is_unmarked_and_other_errors_still_fail_the_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_raising(monkeypatch, None).metadata == {"n_episodes": 3}
    with pytest.raises(RuntimeError, match="model endpoint"):
        run_raising(monkeypatch, RuntimeError("model endpoint unreachable"))
