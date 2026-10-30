"""AutoEnvScaling (preprint forthcoming): a coding agent rebuilds recorded sessions as Harbor tasks, a task generator.

The Designer is a coding agent: it writes a Harbor task in a workspace from one recorded coding-agent
session and the repository the session changed, runs the task's oracle and nop check itself through a
tool, and revises until the check passes. The generator service's one-reply prompt Designer is the
baseline Designer on the same sessions. A pass-rate band marks the tasks the solver can reach, and an optional
calibration with the solver's rollouts keeps tasks outside it out. The built-in generated-task recipe runs the
tasks and trains on them.

- ``sources``: the sources file (session records and their repositories) and the session renderer.
- ``designer``: the Designer contract and its two implementations: the agent, pi in a cloud sandbox (E2B or Modal) behind a
  gateway that records its model calls on Reef, and the one-reply prompt Designer.
- ``claude_designer``: Claude Code as the agent, in the same sandbox, signed in with its owner's OAuth token; it calls
  Anthropic itself and checks its task with the ``task_check`` command.
- ``checks``: the task checks: Harbor's oracle and nop agents, the contamination check, the test review, the
  solver calibration, and the seed check.
- ``generator``: ``AutoEnvScalingGenerator``: the sessions walked with a cursor, admission and the history
  step, the agent's check and the task checks through the generator service, the band, the parents and the seeds.
- ``tmax``: the command that converts extracted TMax tasks into seed tasks.
"""

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
from recipes.beta.autoenvscaling.claude_designer import ClaudeCodeDesigner
from recipes.beta.autoenvscaling.designer import (
    AgentDesigner,
    CandidateChecks,
    DesignedTask,
    DesignerOutage,
    PromptDesigner,
    SeedVariantDesigner,
    SourceError,
    TaskDesigner,
    TaskVariant,
    VariantDesigner,
    workspace_reply,
)
from recipes.beta.autoenvscaling.generator import AutoEnvScalingGenerator, GeneratorChecks, admitted_task
from recipes.beta.autoenvscaling.sources import Repository, SessionSource, load_sources, render_session

__all__ = [
    "AgentDesigner",
    "AutoEnvScalingGenerator",
    "CalibrationCheck",
    "CandidateChecks",
    "CheckResult",
    "ClaudeCodeDesigner",
    "ContaminationCheck",
    "DesignedTask",
    "DesignerOutage",
    "GeneratorChecks",
    "OracleCheck",
    "PromptDesigner",
    "Repository",
    "ReviewCheck",
    "ReviewFinding",
    "SeedCheck",
    "SeedVariantDesigner",
    "SessionSource",
    "SourceError",
    "TaskCheck",
    "TaskDesigner",
    "TaskVariant",
    "VariantDesigner",
    "admitted_task",
    "load_sources",
    "render_session",
    "workspace_reply",
]
