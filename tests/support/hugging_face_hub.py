"""A Hugging Face Hub in memory: one repo at one commit, for reference imports (RFC #995)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

COMMIT = "0123456789abcdef0123456789abcdef01234567"
#: Files from this size up are LFS files: the hub lists their SHA-256, as Hugging Face does above 10 MB.
LFS_SIZE = 8


class MemoryHub(HfApi):
    """``repo`` at ``COMMIT``, its ``main`` branch, holds ``files``; ``downloads`` lists each file a caller downloaded."""

    def __init__(self, cache: Path, repo: str, files: dict[str, bytes]) -> None:
        super().__init__()
        self.cache, self.repo, self.files = cache, repo, files
        self.downloads: list[str] = []

    def model_info(self, repo_id: str, *, revision: str | None = None, **kwargs: Any) -> Any:
        # The default branch, ``main``, is at ``COMMIT``.
        if repo_id != self.repo or revision not in (None, "main", COMMIT):
            missing = httpx2.Response(404, request=httpx2.Request("GET", "https://huggingface.co"))
            raise RevisionNotFoundError(f"{repo_id}@{revision} is not on the hub", response=missing)
        return SimpleNamespace(
            sha=COMMIT,
            siblings=[
                SimpleNamespace(
                    rfilename=path,
                    lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if len(data) >= LFS_SIZE else None,
                )
                for path, data in self.files.items()
            ],
        )

    def hf_hub_download(self, repo_id: str, filename: str, *, revision: str | None = None, **kwargs: Any) -> str:
        self.downloads.append(filename)
        target = self.cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)
