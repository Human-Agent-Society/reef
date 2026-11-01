"""Small object boundary around git subprocess execution."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from reef.artifact.artifact import ArtifactPublicationError, ArtifactSourceError


class GitClient:
    """Execute git commands and translate process failures to artifact errors.

    ``config`` maps Git config keys to values that every command reads, through Git's environment
    (``GIT_CONFIG_COUNT``), so a secret value, such as an authorization header, appears in no command line,
    config file or error message.
    """

    def __init__(self, clone_dir: Path, config: Mapping[str, str] | None = None) -> None:
        self.clone_dir = Path(clone_dir)
        pairs = tuple((config or {}).items())
        self.config_environment = {"GIT_CONFIG_COUNT": str(len(pairs))} if pairs else {}
        for index, (key, value) in enumerate(pairs):
            self.config_environment[f"GIT_CONFIG_KEY_{index}"] = key
            self.config_environment[f"GIT_CONFIG_VALUE_{index}"] = value

    def git(self, *args: str) -> str:
        return self.run(("git", *args), cwd=self.clone_dir)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        source_error: bool = False,
        input_text: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        try:
            return subprocess.run(
                list(command),
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                input=input_text,
                env=(
                    None
                    if environment is None and not self.config_environment
                    else {**os.environ, **self.config_environment, **(environment or {})}
                ),
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            error = ArtifactSourceError if source_error else ArtifactPublicationError
            raise error(f"command failed: {' '.join(command)}: {stderr.strip()}") from exc
