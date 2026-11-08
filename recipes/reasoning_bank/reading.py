"""Configuration and tree files for task-dependent memory reads on pi and native."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.harness.tree import HarnessTree

RUNTIME_DIR = Path(__file__).parent / "runtime"
RUNTIME_PATHS = (
    "extensions/reasoning_bank.ts",
    "native/hooks/reasoning_bank.py",
    "native/tools/memory_search.py",
)


@dataclass(frozen=True)
class ReadSettings(PluginSettings):
    """How a published bank is read.

    Values: reader is skills, all, or top_k; inject is system, message, or tool. top_k is a positive integer.
    top_k reads need an HTTP(S) embedding_url without credentials, query or fragment, and a nonempty model.
    Other readers require both embedding fields to be empty. skills requires inject=system, the unused default.
    timeout_seconds is finite and positive. The runtime reads an optional bearer key from
    REEF_MEMORY_EMBEDDING_API_KEY; no key is stored in a tree. skills keeps the existing Agent Skills delivery.
    Purpose: ReasoningBank parses these settings and embeds them with notes in its runtime files.
    """

    reader: str = config_option("skills", help="Read through agent skills, all notes, or embedding top_k.")
    inject: str = config_option("system", help="Put selected notes in system, message, or a memory_search tool.")
    top_k: int = config_option(1, help="The most notes an embedding search returns.")
    embedding_url: str = config_option("", help="The complete OpenAI-compatible embeddings endpoint URL.")
    embedding_model: str = config_option("", help="The embedding model, identical during evaluation and use.")
    timeout_seconds: float = config_option(20.0, help="The timeout of one embedding request.")

    def __post_init__(self) -> None:
        if self.reader not in ("skills", "all", "top_k") or self.inject not in ("system", "message", "tool"):
            raise ValueError("reader must be skills, all or top_k; inject must be system, message or tool")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if not isinstance(self.embedding_url, str) or not isinstance(self.embedding_model, str):
            raise ValueError("embedding_url and embedding_model must be strings")
        if self.reader == "top_k":
            endpoint = urlsplit(self.embedding_url)
            if (
                endpoint.scheme not in ("http", "https")
                or not endpoint.hostname
                or endpoint.username is not None
                or endpoint.password is not None
                or endpoint.port == 0
                or endpoint.query
                or endpoint.fragment
                or not self.embedding_model.strip()
            ):
                raise ValueError(
                    "top_k needs embedding_url without credentials, query or fragment and embedding_model"
                )
        elif self.embedding_url or self.embedding_model:
            raise ValueError("embedding_url and embedding_model apply only to reader: top_k")
        if self.reader == "skills" and self.inject != "system":
            raise ValueError("inject applies to all or top_k; skills keeps the agent's own delivery")


@dataclass(frozen=True)
class ReadNote:
    """One model-visible note exported by a method.

    Values: id, title, description, and content are nonempty UTF-8 strings. id is unique within one exported bank.
    Purpose: ReasoningBank makes notes from its items; runtime_files serializes them into a published reader.
    """

    id: str
    title: str
    description: str
    content: str

    def __post_init__(self) -> None:
        for text in (self.id, self.title, self.description, self.content):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("memory note fields must be nonempty strings")
            text.encode("utf-8")


def without_memory(tree: HarnessTree) -> HarnessTree:
    """Remove only ReasoningBank's owned skills and runtime files, preserving all other harness files."""
    return HarnessTree(
        {
            path: text
            for path, text in tree.files.items()
            if not path.startswith("skills/rb-") and path not in RUNTIME_PATHS
        }
    )


def runtime_files(notes: Sequence[ReadNote], settings: ReadSettings, adapter: str) -> dict[str, str]:
    """The pi extension or native hook/tool that reads notes with settings.

    Refuse duplicate IDs, skills mode, or an adapter without a supported runtime. Native and pi receive the
    same data and selection settings. Runtime files contain no credentials. An empty bank writes no file.
    """
    if adapter not in ("pi", "native"):
        raise ValueError("all and top_k memory readers currently support pi and native")
    if settings.reader == "skills":
        raise ValueError("skills delivery does not build a runtime reader")
    if len({note.id for note in notes}) != len(notes):
        raise ValueError("memory note IDs must be unique")
    if not notes:
        return {}
    payload = json.dumps({"settings": asdict(settings), "notes": [asdict(note) for note in notes]}, ensure_ascii=True)
    if adapter == "pi":
        source = (RUNTIME_DIR / "pi.mjs").read_text(encoding="utf-8")
        return {RUNTIME_PATHS[0]: source.replace("const bank = {};", f"const bank = {payload};", 1)}
    source = (RUNTIME_DIR / "native.py").read_text(encoding="utf-8")
    source = source.replace('BANK_JSON = "{}"', f"BANK_JSON = {payload!r}", 1)
    if settings.inject == "tool":
        source = source.replace('NAME = "reasoning_bank"', 'NAME = "memory_search"', 1)
        if settings.reader == "all":
            source = source.replace('CAPABILITIES = ["network"]', "CAPABILITIES = []", 1)
        return {RUNTIME_PATHS[2]: source}
    return {RUNTIME_PATHS[1]: source}
