"""Reef's own git state for team stages with ``workspace: own``: one repository outside the main worktree.

The git directory and one clone per member live under ``team_path``; every command on the main worktree names that
directory with ``--git-dir`` and ``--work-tree``, so the main worktree never gets a ``.git`` and a task's own ``.git``
there is never read or changed. A member's clone keeps its own refs, so one member's stash or branch is never
another's. Every file is tracked, ignored by the task's ``.gitignore`` or not, except ``.reef/`` (tool output, and
the team state itself on the host), which never merges. A nested git repository would be tracked as a bare commit id
and empty directories not at all, so a stage refuses the first and drops the second. Commands go through a
``CommandRunner``: a subprocess on the host here, a command in the task environment where the tools run there.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

#: The longest one git command of a team stage may take; a large task tree is staged and merged whole.
GIT_TIMEOUT_SECONDS = 600.0
#: Git reads no system or user configuration: a signing key, a hook or an exclude list of whoever runs the episode
#: would change what a stage commits and merges.
GIT_ENV = ("env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_TERMINAL_PROMPT=0")
GIT_CONFIG = (
    *("-c", "user.name=Reef", "-c", "user.email=reef@localhost", "-c", "safe.directory=*"),
    *("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "-c", "gc.auto=0"),
)
#: What the git directory never tracks, in the main worktree and in every member's.
EXCLUDED_PATH = ".reef/"
#: Every path under the worktree but ``.reef/``, whatever the task's ``.gitignore`` says.
STAGED_PATHSPEC = ("-A", "-f", "--", ".", ":(exclude).reef")
#: Prints every ``.git`` under a worktree but its own and those under ``.reef/``: the nested repositories.
NESTED_GIT_FIND = (
    *("find", ".", "(", "-path", "./.git", "-o", "-path", "./.reef", ")", "-prune"),
    *("-o", "-name", ".git", "-print", "-prune"),
)
#: Which repository a git command names: Reef's for the main worktree, the clone in ``cwd``, or none yet.
Repository = Literal["main", "clone", "none"]


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
    """One member's branch merged into the main worktree: ``merged``, ``empty``, ``conflict`` (with the files), or
    ``failed`` (with the error) when git could not commit, fetch or start the merge."""

    agent: str
    branch: str
    result: str
    files: tuple[str, ...] = ()
    error: str = ""


class TeamWorkspaces:
    """One git directory under ``team_path`` for every ``own`` stage run of an episode, and a clone per member."""

    def __init__(self, runner: CommandRunner, *, main_path: PurePosixPath, team_path: PurePosixPath) -> None:
        self.runner = runner
        self.main_path = main_path
        self.team_path = team_path
        self.git_path = team_path / "git"
        self.is_started = False

    def git(
        self, *args: str, cwd: PurePosixPath, repository: Repository = "main", is_checked: bool = True
    ) -> CommandOutcome:
        """One git command in ``cwd``, naming its repository: a member whose clone lost its ``.git`` never reaches
        a repository above it, such as a task's own."""
        if repository == "main":
            named: tuple[str, ...] = (f"--git-dir={self.git_path}", f"--work-tree={self.main_path}")
        elif repository == "clone":
            named = (f"--git-dir={cwd / '.git'}", f"--work-tree={cwd}")
        else:
            named = ()
        outcome = self.runner.run(
            [*GIT_ENV, "git", *GIT_CONFIG, *named, *args], cwd=str(cwd), timeout_seconds=GIT_TIMEOUT_SECONDS
        )
        if is_checked and outcome.return_code != 0:
            detail = (outcome.stderr or outcome.stdout).strip()[-600:]
            raise TeamWorkspaceError(f"git {' '.join(args[:2])} exited {outcome.return_code}: {detail}")
        return outcome

    def is_git_available(self) -> bool:
        return self.git("--version", cwd=self.main_path, repository="none", is_checked=False).return_code == 0

    def member_path(self, stage_run: int, instance: str) -> PurePosixPath:
        return self.team_path / f"s{stage_run}-{instance}"

    @staticmethod
    def branch(stage_run: int, instance: str) -> str:
        return f"reef/s{stage_run}/{instance}"

    def exclude(self, git_path: PurePosixPath) -> None:
        """Add ``.reef/`` to the exclude file of the git directory ``git_path``, for git status and plain adds."""
        script = 'mkdir -p "$1/info" && printf "%s\\n" "$2" >> "$1/info/exclude"'
        excluded = self.runner.run(
            ["sh", "-c", script, "sh", str(git_path), EXCLUDED_PATH],
            cwd=str(self.main_path),
            timeout_seconds=GIT_TIMEOUT_SECONDS,
        )
        if excluded.return_code != 0:
            raise TeamWorkspaceError(f"the git exclude file could not be written: {excluded.stderr.strip()[-600:]}")

    def commit_all(self, cwd: PurePosixPath, repository: Repository, message: str, *, is_empty_allowed: bool) -> None:
        """Stage every file under ``cwd`` but ``.reef/`` and commit it, with no commit when nothing changed unless
        ``is_empty_allowed``.

        A nested git repository is refused first: git would keep only its commit id, so its files would be lost."""
        found = self.runner.run(NESTED_GIT_FIND, cwd=str(cwd), timeout_seconds=GIT_TIMEOUT_SECONDS)
        if found.return_code != 0:
            raise TeamWorkspaceError(f"find exited {found.return_code}: {found.stderr.strip()[-600:]}")
        nested = [line.removeprefix("./").removesuffix("/.git") for line in found.stdout.splitlines() if line]
        if nested:
            raise TeamWorkspaceError(
                f"{', '.join(nested[:5])} in {cwd} is a git repository of its own, which workspace: own cannot copy; "
                "use workspace: shared"
            )
        self.git("add", *STAGED_PATHSPEC, cwd=cwd, repository=repository)
        if not is_empty_allowed:
            staged = self.git("diff", "--cached", "--quiet", cwd=cwd, repository=repository, is_checked=False)
            if staged.return_code not in (0, 1):
                raise TeamWorkspaceError(f"git diff exited {staged.return_code}: {staged.stderr.strip()[-600:]}")
            if staged.return_code == 0:
                return
        self.git("commit", "-q", "--allow-empty", "-m", message, cwd=cwd, repository=repository)

    def start(self, stage_run: int) -> str:
        """Commit the main worktree as it stands and return that commit, the base every member branches from."""
        is_team_in_workdir = self.team_path.is_relative_to(self.main_path)
        if is_team_in_workdir and not self.team_path.is_relative_to(self.main_path / EXCLUDED_PATH):
            raise TeamWorkspaceError(
                f"the workdir {self.main_path} holds the team directory {self.team_path}; workspace: own needs a "
                "workdir that does not"
            )
        if not self.is_started:
            self.git("init", "-q", "--bare", str(self.git_path), cwd=self.main_path, repository="none")
            self.exclude(self.git_path)
            self.is_started = True
        self.commit_all(self.main_path, "main", f"reef: before team stage run {stage_run}", is_empty_allowed=True)
        return self.git("rev-parse", "HEAD", cwd=self.main_path).stdout.strip()

    def add_member(self, stage_run: int, instance: str, base: str) -> PurePosixPath:
        """A clone for one member, on its own branch from ``base`` and with no remote; its path is the member's
        workdir. The clone borrows the objects of Reef's git directory, so it costs only its checkout."""
        path = self.member_path(stage_run, instance)
        source, target = str(self.git_path), str(path)
        self.git("clone", "-q", "--shared", "--no-checkout", source, target, cwd=self.main_path, repository="none")
        self.git("checkout", "-q", "-b", self.branch(stage_run, instance), base, cwd=path, repository="clone")
        self.git("remote", "remove", "origin", cwd=path, repository="clone")
        self.exclude(path / ".git")
        return path

    def merge(self, stage_run: int, instances: Sequence[str]) -> list[MergeResult]:
        """Commit the main worktree and each member's clone, then merge each member's branch into the main worktree
        in ``instances`` order; one result per member, whatever happened to the others.

        Changes made in the main worktree during the stage are committed first, so they merge like a member's. A
        conflict is aborted, so the main worktree holds only clean merges. A conflicting or failed member's clone
        stays until ``close`` so its changes can still be read; a merged or empty member's clone is removed."""
        try:
            self.commit_all(
                self.main_path, "main", f"reef: workdir after team stage run {stage_run}", is_empty_allowed=False
            )
        except TeamWorkspaceError as exc:
            return [MergeResult(name, self.branch(stage_run, name), "failed", error=str(exc)) for name in instances]
        results: list[MergeResult] = []
        for instance in instances:
            branch = self.branch(stage_run, instance)
            try:
                results.append(self.merge_member(stage_run, instance))
            except TeamWorkspaceError as exc:
                results.append(MergeResult(instance, branch, "failed", error=str(exc)))
        return results

    def merge_member(self, stage_run: int, instance: str) -> MergeResult:
        """Commit one member's clone, bring its head into Reef's git directory as its branch, and merge that."""
        path, branch = self.member_path(stage_run, instance), self.branch(stage_run, instance)
        self.commit_all(path, "clone", f"reef: {instance}, stage run {stage_run}", is_empty_allowed=False)
        self.git("fetch", "-q", str(path), f"+HEAD:refs/heads/{branch}", cwd=self.main_path)
        # Counted from the main worktree's head: a member that changed nothing has no commit it lacks.
        ahead = int(self.git("rev-list", "--count", f"HEAD..{branch}", cwd=self.main_path).stdout.strip())
        if ahead:
            merged = self.git("merge", "--no-ff", "--no-edit", "-q", branch, cwd=self.main_path, is_checked=False)
            if merged.return_code != 0:
                merge_head = self.git(
                    "rev-parse", "-q", "--verify", "MERGE_HEAD", cwd=self.main_path, is_checked=False
                )
                if merge_head.return_code != 0:
                    # Git refused to start the merge, so there is nothing to abort; its message says why.
                    detail = (merged.stderr or merged.stdout).strip()[-600:]
                    return MergeResult(
                        instance, branch, "failed", error=f"git merge exited {merged.return_code}: {detail}"
                    )
                conflicts = self.git("diff", "--name-only", "--diff-filter=U", cwd=self.main_path).stdout
                self.git("merge", "--abort", cwd=self.main_path)
                return MergeResult(instance, branch, "conflict", tuple(conflicts.splitlines()))
        # A clone left behind is removed with the team directory at ``close``.
        self.runner.run(["rm", "-rf", str(path)], cwd=str(self.main_path), timeout_seconds=GIT_TIMEOUT_SECONDS)
        return MergeResult(instance, branch, "merged" if ahead else "empty")

    def close(self) -> None:
        """Remove the git directory and every clone left; nothing is run when no stage used the directory."""
        if self.is_started:
            self.runner.run(
                ["rm", "-rf", str(self.team_path)], cwd=str(self.main_path), timeout_seconds=GIT_TIMEOUT_SECONDS
            )
            self.is_started = False
