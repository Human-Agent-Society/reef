"""Credential checks and redaction for Claude Code author runs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from recipes.beta.autoenvscaling.author import AuthorError, CandidateChecks, workspace_reply

REDACTED = "[redacted]"
#: Retain rendered configuration and transcripts, but remove account state.
KEPT_CONFIG = frozenset({"settings.json", "projects"})
TOKEN_REFUSAL = (
    "the check refused the task: a file in ./task/ holds this run's credential. Remove it from the task; a task "
    "that holds it is never checked or admitted."
)


class CredentialHeld(AuthorError):
    """The agent's task held the run's Claude Code token, so it is not admitted.

    Values: the message names no token. Purpose: :meth:`ClaudeCodeAuthor.author` raises it once the run's directory
    is redacted; the generator ends the generation with stage ``credential``, since the token may have left the
    sandbox.
    """


def scrubbed(text: str, token: str) -> str:
    """``text`` with every occurrence of ``token`` replaced. Cut a text only after this, so no part of the token
    stays."""
    return text.replace(token, REDACTED)


def holding_files(directory: Path, token: str) -> list[Path]:
    """The regular files under ``directory`` whose bytes hold ``token``, symbolic links not followed."""
    needle = token.encode()
    held = []
    for folder, _, names in os.walk(directory):
        for name in names:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink() and needle in path.read_bytes():
                held.append(path)
    return held


def rename_holding(directory: Path, token: str) -> None:
    """Rename every file and folder under ``directory`` whose name holds ``token``, with the token replaced; one
    that cannot take the new name, because another of a different kind holds it, is deleted."""
    for folder, folders, names in os.walk(directory, topdown=False):
        for name in (*names, *folders):
            if token in name:
                path = Path(folder) / name
                try:
                    path.rename(path.with_name(scrubbed(name, token)))
                except OSError:
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)


def holds_token(task_dir: Path, token: str) -> bool:
    """Whether the task in ``task_dir`` holds ``token``: in a regular file, or in the reply that a check and the
    admission send, which follows symbolic links."""
    return bool(holding_files(task_dir, token)) or token in workspace_reply(task_dir)


class CredentialChecks(CandidateChecks):
    """Refuse credential-bearing tasks before delegating candidate checks.

    Values: ``checks`` implements CandidateChecks; ``token`` is a non-empty string.
    The wrapper copies the delegate's check limit and review mode at construction.
    Purpose: ClaudeCodeAuthor supplies this wrapper to AgentTools. A contaminated
    task returns ``ran: false`` without consuming the delegate's check budget.
    Other tasks return the delegate's result unchanged; delegate errors propagate.
    """

    def __init__(self, checks: CandidateChecks, *, token: str) -> None:
        if not token:
            raise ValueError("the checks need the run's token to refuse a task that holds it")
        super().__init__(checks.max_checks, is_reviewing=checks.is_reviewing)
        self.checks = checks
        self.token = token

    def check(self, task_dir: Path) -> dict[str, object]:
        """Refuse a task holding the token; otherwise return its candidate check result."""
        if holds_token(task_dir, self.token):
            return {"ran": False, "passed": False, "error": TOKEN_REFUSAL}
        return self.checks.check(task_dir)


def clean_run(work_dir: Path, token: str) -> bool:
    """Remove temporary state and redact retained files; report whether the task held the token.

    Inspect the task before redaction, including files reached through its symbolic links.
    The caller must refuse admission when the returned value is true.
    """
    shutil.rmtree(work_dir / "inputs" / "repo", ignore_errors=True)
    shutil.rmtree(work_dir / "home", ignore_errors=True)
    config_dir = work_dir / "claude"
    for path in config_dir.iterdir() if config_dir.is_dir() else ():
        if path.name in KEPT_CONFIG:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    is_held = holds_token(work_dir / "workspace" / "task", token)
    for path in holding_files(work_dir, token):
        path.write_bytes(path.read_bytes().replace(token.encode(), REDACTED.encode()))
    rename_holding(work_dir, token)
    return is_held
