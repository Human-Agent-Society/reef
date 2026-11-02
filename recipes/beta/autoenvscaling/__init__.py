"""AutoEnvScaling (preprint forthcoming): a coding agent rebuilds recorded sessions as Harbor tasks, a task generator.

The author is a coding agent: it writes a Harbor task in a workspace from one recorded coding-agent
session and the repository the session changed, runs the task's oracle and nop check itself through a
tool, and revises until the check passes. The generator service's one-reply prompt Designer is the
baseline author on the same sessions. A pass-rate band marks the tasks the solver can reach, and an optional
calibration with the solver's rollouts keeps tasks outside it out. The built-in generated-task recipe runs the
tasks and trains on them.

- ``sources``: the sources file (session records and their repositories) and the session renderer.
- ``author``: the author contract and its two implementations: the agent, pi in a cloud sandbox (E2B or Modal) behind a
  gateway that records its model calls on Reef, and the one-reply prompt Designer.
- ``claude_author``: Claude Code as the agent, in the same sandbox, signed in with its owner's OAuth token; it calls
  Anthropic itself and checks its task with the ``task_check`` command.
- ``checks``: the task checks: Harbor's oracle and nop agents, the contamination check, the test review, the
  solver calibration, and the seed check.
- ``generator``: ``AutoEnvScalingGenerator``: the sessions walked with a cursor, admission and the history
  step, the agent's check and the task checks through the generator service, the band, the parents and the seeds.
- ``tmax``: the command that converts extracted TMax tasks into seed tasks.
"""

from recipes.beta.autoenvscaling.author import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    SeedVariantAuthor,
    SourceError,
    TaskAuthor,
    TaskVariant,
    VariantAuthor,
    workspace_reply,
)
from recipes.beta.autoenvscaling.checks import (
    CalibrationCheck,
    CheckResult,
    ContaminationCheck,
    OracleCheck,
    ReviewCheck,
    ReviewFinding,
    SeedCheck,
    TaskCheck,
)
from recipes.beta.autoenvscaling.claude_author import ClaudeCodeAuthor
from recipes.beta.autoenvscaling.generator import AutoEnvScalingGenerator, GeneratorChecks, admitted_task
from recipes.beta.autoenvscaling.sources import Repository, SessionSource, load_sources, render_session

__all__ = [
    "AgentAuthor",
    "AuthorError",
    "AuthoredTask",
    "AutoEnvScalingGenerator",
    "CalibrationCheck",
    "CandidateChecks",
    "CheckResult",
    "ClaudeCodeAuthor",
    "ContaminationCheck",
    "GeneratorChecks",
    "OracleCheck",
    "PromptAuthor",
    "Repository",
    "ReviewCheck",
    "ReviewFinding",
    "SeedCheck",
    "SeedVariantAuthor",
    "SessionSource",
    "SourceError",
    "TaskAuthor",
    "TaskCheck",
    "TaskVariant",
    "VariantAuthor",
    "admitted_task",
    "load_sources",
    "render_session",
    "workspace_reply",
]
