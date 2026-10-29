"""Export the weights of a Spindle full-parameter checkpoint to a Hugging Face repo, on Modal (RFC #996).

Spindle's ``save_state`` writes a full-parameter checkpoint's Hugging Face weights next to its training state, in
the deployment's checkpoint Volume, at ``<name>/<model_id>/`` for the path ``tinker://<model_id>/weights/<name>``.
Spindle serves no download of them. So a function on Modal mounts the Volume, adds the base's tokenizer and chat
template, and commits the files to the release repo; the release then refers to that commit, which any serving
runtime loads.
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from reef.artifact.sources import HuggingFaceRevision
from reef.train.tinker_backend.config import SpindleFullConfig

#: Where the export function mounts the checkpoint Volume; Spindle mounts it at the same path.
CHECKPOINT_MOUNT = "/checkpoints"
#: The files of a checkpoint directory that are its Hugging Face weights; the rest is training state.
WEIGHT_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json")
#: The base's files that a model needs beside its weights: its tokenizer, chat template and processor.
BASE_FILES = (
    "tokenizer*",
    "*.jinja",
    "chat_template*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab*",
    "merges.txt",
    "processor_config.json",
    "*preprocessor_config.json",
)


def checkpoint_directory(state_path: str) -> str:
    """The directory in the checkpoint Volume of the Spindle checkpoint ``state_path``; ``ValueError`` otherwise."""
    parts = state_path.removeprefix("tinker://").split("/")
    if not state_path.startswith("tinker://") or len(parts) != 3 or parts[1] != "weights" or not all(parts):
        raise ValueError(f"{state_path} is no Spindle checkpoint path tinker://<model_id>/weights/<name>")
    return f"{parts[2]}/{parts[0]}"


class CheckpointExporter(ABC):
    """Puts the Hugging Face weights of a full-parameter checkpoint into a repo on the Hub.

    Values: ``export(state_path, base)`` returns a commit of the exporter's release repo whose files are exactly the
    weights of the checkpoint at ``state_path`` (its safetensors, ``config.json`` and ``generation_config.json``) and
    the tokenizer, chat template and processor files of ``base``. It raises ``ValueError`` for a path that names no
    checkpoint with weights, and ``RuntimeError`` when the export fails. Purpose: the Spindle full-parameter
    component exports each saved checkpoint, so that its release refers to the weights on the Hub.
    """

    @abstractmethod
    def export(self, state_path: str, base: HuggingFaceRevision) -> HuggingFaceRevision: ...


class ModalCheckpointExporter(CheckpointExporter):
    """Export with a function on Modal, in the Spindle deployment's environment, which mounts its checkpoint Volume.

    It needs Modal credentials (``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, or a profile) and the Modal Secret
    ``config.hf_secret``, whose ``HF_TOKEN`` writes ``config.release_repo``. Each export starts an ephemeral app.
    """

    def __init__(self, config: SpindleFullConfig, *, timeout_s: float = 3 * 3600) -> None:
        self.repo = config.release_repo
        self.volume = config.checkpoint_volume
        self.environment = config.modal_environment
        self.secret = config.hf_secret
        self.timeout_s = timeout_s

    def export(self, state_path: str, base: HuggingFaceRevision) -> HuggingFaceRevision:
        directory = checkpoint_directory(state_path)
        try:
            modal: Any = importlib.import_module("modal")
        except ImportError as exc:
            raise RuntimeError(
                "exporting Spindle checkpoints needs Modal: uv pip install 'reef-infra[modal]'"
            ) from exc
        app = modal.App("reef-spindle-export")
        # A serialized function runs under the Python version that pickled it.
        python = f"{sys.version_info.major}.{sys.version_info.minor}"
        upload = app.function(
            image=modal.Image.debian_slim(python_version=python).pip_install("huggingface_hub[hf_xet]>=1.0"),
            volumes={CHECKPOINT_MOUNT: modal.Volume.from_name(self.volume, environment_name=self.environment)},
            secrets=[modal.Secret.from_name(self.secret, environment_name=self.environment)],
            env={"HF_XET_HIGH_PERFORMANCE": "1"},
            cpu=4,
            memory=16384,
            timeout=int(self.timeout_s),
            serialized=True,
        )(checkpoint_upload())
        try:
            with app.run(environment_name=self.environment):
                commit = upload.remote(CHECKPOINT_MOUNT, directory, self.repo, base.repo, base.commit, state_path)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        except modal.exception.Error as exc:
            raise RuntimeError(f"exporting {state_path} to {self.repo} failed: {exc}") from exc
        return HuggingFaceRevision(self.repo, commit)


def checkpoint_upload() -> Callable[[str, str, str, str, str, str], str]:
    """The function that commits a checkpoint's weights and the base's other files to a repo, on Modal.

    ``upload(mount, directory, repo, base_repo, base_commit, state_path)`` reads ``directory`` of the checkpoint
    Volume at ``mount``, and commits its weights and the base's ``BASE_FILES`` to ``repo``: exactly those files,
    since it deletes what the repo's previous commit held. It returns the commit SHA, and raises
    ``FileNotFoundError`` when the directory holds no safetensors. It is a nested function so that Modal pickles it
    by value: the container has huggingface_hub, and no Reef.
    """

    def upload(mount: str, directory: str, repo: str, base_repo: str, base_commit: str, state_path: str) -> str:
        import shutil
        import tempfile
        from pathlib import Path

        from huggingface_hub import HfApi, snapshot_download

        source = Path(mount) / directory
        weights = [
            path
            for path in (sorted(source.iterdir()) if source.is_dir() else ())
            if path.suffix == ".safetensors" or path.name in WEIGHT_FILES
        ]
        if not any(path.suffix == ".safetensors" for path in weights):
            raise FileNotFoundError(f"the Spindle checkpoint {state_path} holds no Hugging Face weights in {source}")
        staging = Path(tempfile.mkdtemp())
        base = Path(snapshot_download(base_repo, revision=base_commit, allow_patterns=list(BASE_FILES)))
        for path in base.iterdir():
            if path.is_file():
                shutil.copy2(path, staging / path.name)
        for path in weights:
            # The checkpoint's own config and weights replace the base's; a link uploads without a copy.
            (staging / path.name).unlink(missing_ok=True)
            (staging / path.name).symlink_to(path)
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo, folder_path=staging, commit_message=f"Spindle checkpoint {state_path}", delete_patterns=["*"]
        )
        return str(commit.oid)

    return upload
