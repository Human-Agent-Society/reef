"""Terminus 2 for the generator's plays: an episode the agent itself ends badly is still verified.

Harbor runs a task's verifier after the agent only when the agent finished, timed out or exited nonzero; any
other agent error skips it, so the play has no reward and the task player files no report.
:class:`VerifiedTerminus2` ends the run instead on two such errors, so the verifier scores the workspace as the
agent left it: the agent filled its context window, or a send-keys to its tmux pane failed. Every other error
still fails the trial.

A deployment selects it with ``import_path: reef.record2task.generator.terminus:VerifiedTerminus2`` in place of
``name: terminus-2``. Nothing else in Reef imports this module, so Reef never needs harbor to load.
"""

from __future__ import annotations

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext

#: The agent context metadata key that names why a verified episode ended early.
EPISODE_END_KEY = "reef_episode_end"


class VerifiedTerminus2(Terminus2):
    """Terminus 2 whose run ends, instead of failing, when its context is full or a send-keys to its pane fails."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        try:
            await super().run(instruction, environment, context)
            return
        except ContextLengthExceededError:
            reason = "context_length_exceeded"
        except RuntimeError as error:
            # Harbor raises a plain RuntimeError for any failed send-keys (TmuxSession._send_keys_error), such as
            # one to a tmux server that went away; its message is the only mark.
            message = str(error)
            if ": failed to send " not in message or " keys: command=" not in message:
                raise
            reason = "terminal_lost"
        self.logger.warning("ending the episode for verification: %s", reason)
        context.metadata = {**(context.metadata or {}), EPISODE_END_KEY: reason}
