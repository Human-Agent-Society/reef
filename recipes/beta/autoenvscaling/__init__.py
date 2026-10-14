"""AutoEnvScaling (preprint forthcoming): a coding agent rebuilds recorded sessions as Harbor tasks, a task generator.

The author is a coding agent: it writes a Harbor task in a workspace from one recorded coding-agent
session and the repository the session changed, runs the task's oracle and nop check itself through a
tool, and revises until the check passes. The generator service's one-reply prompt Designer is the
baseline author on the same sessions. A pass-rate band marks the tasks the solver can reach. The built-in
generated-task recipe plays the tasks and trains on them.

- ``sources``: the sources file (session records and their repositories) and the session renderer.
- ``author``: the author contract and its two implementations: the agent, pi in an E2B sandbox behind a
  gateway that records its model calls on Reef, and the one-reply prompt Designer.
- ``checks``: the task checks: Harbor's oracle and nop agents, and the test review.
- ``generator``: ``AutoEnvScalingGenerator``: the sessions walked with a cursor, admission and the history
  step, the agent's check and the task checks through the generator service, and the band.
"""

from recipes.beta.autoenvscaling.author import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    SourceError,
    TaskAuthor,
    workspace_reply,
)
from recipes.beta.autoenvscaling.checks import CheckResult, OracleCheck, ReviewCheck, ReviewFinding, TaskCheck
from recipes.beta.autoenvscaling.generator import AutoEnvScalingGenerator, GeneratorChecks, admitted_task
from recipes.beta.autoenvscaling.sources import Repository, SessionSource, load_sources, render_session

__all__ = [
    "AgentAuthor",
    "AuthorError",
    "AuthoredTask",
    "AutoEnvScalingGenerator",
    "CandidateChecks",
    "CheckResult",
    "GeneratorChecks",
    "OracleCheck",
    "PromptAuthor",
    "Repository",
    "ReviewCheck",
    "ReviewFinding",
    "SessionSource",
    "SourceError",
    "TaskAuthor",
    "TaskCheck",
    "admitted_task",
    "load_sources",
    "render_session",
    "workspace_reply",
]
