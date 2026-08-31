"""Small object boundary around git subprocess execution."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from reef.artifact.artifact import ArtifactPublicationError, ArtifactSourceError


class GitClient:
    """Execute git commands and translate process failures to artifact errors."""

    def __init__(self, clone_dir: Path) -> None:
        self.clone_dir = Path(clone_dir)

    def git(self, *args: str) -> str:
        return self.run(("git", *args), cwd=self.clone_dir)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        source_error: bool = False,
    ) -> str:
        try:
            return subprocess.run(
                list(command),
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", "") or str(exc)
            error = ArtifactSourceError if source_error else ArtifactPublicationError
            raise error(f"command failed: {' '.join(command)}: {stderr.strip()}") from exc
