"""Reef's own git state for team stages with ``workspace: own``: one repository outside the main worktree.

The git directory and one linked worktree per member live under ``team_path``; every command on the main worktree
names that directory with ``--git-dir`` and ``--work-tree``, so the main worktree never gets a ``.git`` and a task's
own ``.git`` there is never read or changed. ``.reef/`` (tool output, and the team state itself on the host) is
excluded, so it never merges. Commands go through a ``CommandRunner``: a subprocess on the host here, a command in
the task environment where the tools run there.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

#: The longest one git command of a team stage may take; a large task tree is staged and merged whole.
GIT_TIMEOUT_SECONDS = 600.0
#: Git reads no system or user configuration: a signing key, a hook or an exclude list of whoever runs the episode
#: would change what a stage commits and merges.
GIT_ENV = ("env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_TERMINAL_PROMPT=0")
GIT_CONFIG = (
    *("-c", "user.name=Reef", "-c", "user.email=reef@localhost", "-c", "safe.directory=*"),
    *("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"),
)
#: What the git directory never tracks, in the main worktree and in every member's.
EXCLUDED_PATH = ".reef/"


class TeamWorkspaceError(Exception):
    """A git command of a team stage failed; the stage ends ``gave_up`` and its message names the command."""


@dataclass(frozen=True)
class CommandOutcome:
    return_code: int
    stdout: str
    stderr: str


class CommandRunner(ABC):
    """Runs one command where the tools run; a command that cannot start or finish in time is a nonzero outcome."""

    @abstractmethod
    def run(self, argv: Sequence[str], *, cwd: str, timeout_seconds: float) -> CommandOutcome: ...


class HostCommandRunner(CommandRunner):
    """A subprocess on this host: 127 when the command cannot start, 124 when it runs out of time."""

    def run(self, argv: Sequence[str], *, cwd: str, timeout_seconds: float) -> CommandOutcome:
        try:
            done = subprocess.run(
                list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds, check=False
            )
        except OSError as exc:
            return CommandOutcome(127, "", f"{argv[0]}: {exc}")
        except subprocess.TimeoutExpired:
            return CommandOutcome(124, "", f"{argv[0]} did not finish in {timeout_seconds:g} s")
        return CommandOutcome(done.returncode, done.stdout, done.stderr)


@dataclass(frozen=True)
class MergeResult:
    """One member's branch merged into the main worktree: ``merged``, ``conflict`` (with the files) or ``empty``."""

    agent: str
    branch: str
    result: str
    files: tuple[str, ...] = ()


class TeamWorkspaces:
    """One git directory under ``team_path`` for every ``own`` stage run of an episode, and a worktree per member."""

    def __init__(self, runner: CommandRunner, *, main_path: PurePosixPath, team_path: PurePosixPath) -> None:
        self.runner = runner
        self.main_path = main_path
        self.team_path = team_path
        self.git_path = team_path / "git"
        self.is_started = False

    def git(self, *args: str, cwd: PurePosixPath, is_main: bool = True, is_checked: bool = True) -> CommandOutcome:
        """One git command in ``cwd``; on the main worktree it names Reef's git directory, in a member's worktree
        the worktree's own ``.git`` file does."""
        repository = (f"--git-dir={self.git_path}", f"--work-tree={self.main_path}") if is_main else ()
        outcome = self.runner.run(
            [*GIT_ENV, "git", *GIT_CONFIG, *repository, *args], cwd=str(cwd), timeout_seconds=GIT_TIMEOUT_SECONDS
        )
        if is_checked and outcome.return_code != 0:
            detail = (outcome.stderr or outcome.stdout).strip()[-600:]
            raise TeamWorkspaceError(f"git {' '.join(args[:2])} exited {outcome.return_code}: {detail}")
        return outcome

    def is_git_available(self) -> bool:
        return self.git("--version", cwd=self.main_path, is_main=False, is_checked=False).return_code == 0

    def member_path(self, stage_run: int, instance: str) -> PurePosixPath:
        return self.team_path / f"s{stage_run}-{instance}"

    @staticmethod
    def branch(stage_run: int, instance: str) -> str:
        return f"reef/s{stage_run}/{instance}"

    def start(self, stage_run: int) -> str:
        """Commit the main worktree as it stands and return that commit, the base every member branches from."""
        if not self.is_started:
            self.git("init", "-q", "--bare", str(self.git_path), cwd=self.main_path, is_main=False)
            script = 'mkdir -p "$1/info" && printf "%s\\n" "$2" >> "$1/info/exclude"'
            excluded = self.runner.run(
                ["sh", "-c", script, "sh", str(self.git_path), EXCLUDED_PATH],
                cwd=str(self.main_path),
                timeout_seconds=GIT_TIMEOUT_SECONDS,
            )
            if excluded.return_code != 0:
                raise TeamWorkspaceError(
                    f"the git exclude file could not be written: {excluded.stderr.strip()[-600:]}"
                )
            self.is_started = True
        self.git("add", "-A", cwd=self.main_path)
        self.git("commit", "-q", "--allow-empty", "-m", f"reef: before team stage run {stage_run}", cwd=self.main_path)
        return self.git("rev-parse", "HEAD", cwd=self.main_path).stdout.strip()

    def add_member(self, stage_run: int, instance: str, base: str) -> PurePosixPath:
        """A linked worktree for one member, on its own branch from ``base``; its path is the member's workdir."""
        path = self.member_path(stage_run, instance)
        self.git("worktree", "add", "-q", "-b", self.branch(stage_run, instance), str(path), base, cwd=self.main_path)
        return path

    def merge(self, stage_run: int, instances: Sequence[str]) -> list[MergeResult]:
        """Commit each member's worktree, then merge its branch into the main worktree in ``instances`` order.

        A conflict is aborted, so the main worktree holds only clean merges; the conflicting member's worktree stays
        until ``close`` so its changes can still be read. A merged or empty member's worktree is removed."""
        results = []
        for instance in instances:
            path, branch = self.member_path(stage_run, instance), self.branch(stage_run, instance)
            self.git("add", "-A", cwd=path, is_main=False)
            staged = self.git("diff", "--cached", "--quiet", cwd=path, is_main=False, is_checked=False)
            if staged.return_code not in (0, 1):
                raise TeamWorkspaceError(f"git diff exited {staged.return_code}: {staged.stderr.strip()[-600:]}")
            if staged.return_code == 1:
                self.git("commit", "-q", "-m", f"reef: {instance}, stage run {stage_run}", cwd=path, is_main=False)
            # Counted from the main worktree's head: a member that changed nothing has no commit it lacks.
            ahead = int(self.git("rev-list", "--count", f"HEAD..{branch}", cwd=self.main_path).stdout.strip())
            if ahead == 0:
                merged = None
            else:
                merged = self.git("merge", "--no-ff", "--no-edit", "-q", branch, cwd=self.main_path, is_checked=False)
            if merged is not None and merged.return_code != 0:
                conflicts = self.git("diff", "--name-only", "--diff-filter=U", cwd=self.main_path).stdout
                self.git("merge", "--abort", cwd=self.main_path)
                results.append(MergeResult(instance, branch, "conflict", tuple(conflicts.splitlines())))
                continue
            self.git("worktree", "remove", "--force", str(path), cwd=self.main_path)
            results.append(MergeResult(instance, branch, "empty" if merged is None else "merged"))
        return results

    def close(self) -> None:
        """Remove the git directory and every worktree left; nothing is run when no stage used the directory."""
        if self.is_started:
            self.runner.run(
                ["rm", "-rf", str(self.team_path)], cwd=str(self.main_path), timeout_seconds=GIT_TIMEOUT_SECONDS
            )
            self.is_started = False
