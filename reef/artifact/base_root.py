"""The base root: an artifact of a weight component that is the deployment's base model and holds no weight files.

A base root holds one file, ``BASE_FILE``, which names the default start by its inputs (``BaseRoot``). A reader
that finds the file serves or loads the base model; it never guesses the base from an empty artifact. A reference
artifact (``reef.artifact.sources.REFERENCE_FILE``, RFC #995) is another form: it names a Hugging Face commit whose
files are the weights. A backend whose default start is such a commit keeps the reference and writes no base file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from reef.artifact.artifact import REPOSITORY_FILES
from reef.artifact.files import ArtifactFiles

#: The one file of a base root, at the root of the weight component.
BASE_FILE = "base.json"
#: The format marker of ``BASE_FILE``. A reader refuses a file with another marker.
BASE_FORMAT = "reef-base-root-1"
#: The input that names the base model. Every base root has it.
MODEL_INPUT = "model"


@dataclass(frozen=True)
class BaseRoot:
    """The content of ``BASE_FILE``: the inputs of the default start whose weights the artifact stands for.

    Values: ``inputs`` is a mapping from non-empty names to non-empty strings that holds ``MODEL_INPUT``, the base
    model. The other inputs are the settings that decide the untrained weights, as the training backend states them
    in ``TrainingBackendFacts.default_start_inputs``: for a LoRA its rank (``lora_rank``), for Slime also
    ``lora_alpha`` and ``lora_target_modules``, and for Tinker the service's base URL (``service``) when the
    deployment names one. The constructor raises ``ValueError`` for another value and keeps a read-only copy.

    Meaning: the weights of the artifact are the weights of ``model`` with no adapter. An untrained LoRA adapter
    changes no output of its base model, so the value also stands for the untrained adapter of those inputs.

    Operations: ``to_dict`` gives the JSON object ``{"format": BASE_FORMAT, "inputs": inputs}`` and ``from_dict``
    reads it back to an equal value. ``to_text`` is that object as the text of ``BASE_FILE``.
    ``check_inputs(configured)`` returns when ``inputs`` equal ``configured`` and raises ``ValueError`` otherwise.

    Purpose: ``read_base_root`` and ``base_root_of`` make the value from an artifact. The serving runtimes read it to
    serve the base model, and a training component's layout check compares it with its own default start.
    """

    inputs: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, Mapping):
            raise ValueError("a base root's inputs are a mapping of names to strings")
        for name, value in self.inputs.items():
            if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
                raise ValueError(f"base root input {name!r} must map a non-empty name to a non-empty string")
        if MODEL_INPUT not in self.inputs:
            raise ValueError(f"a base root names its base model in the input {MODEL_INPUT!r}")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    @property
    def model(self) -> str:
        """The base model, as the training backend's configuration names it."""
        return self.inputs[MODEL_INPUT]

    def to_dict(self) -> dict[str, object]:
        return {"format": BASE_FORMAT, "inputs": dict(self.inputs)}

    @classmethod
    def from_dict(cls, value: object) -> BaseRoot:
        """Read a ``to_dict`` value; ``ValueError`` for another shape or another format marker."""
        if not isinstance(value, Mapping) or set(value) != {"format", "inputs"}:
            raise ValueError(f"{BASE_FILE} is the object {{format, inputs}}")
        if value["format"] != BASE_FORMAT:
            raise ValueError(f"{BASE_FILE} has the format {value['format']!r}; this Reef reads {BASE_FORMAT!r}")
        return cls(value["inputs"])

    @classmethod
    def from_bytes(cls, data: bytes) -> BaseRoot:
        """Read the bytes of ``BASE_FILE``; ``ValueError`` when they are no ``to_dict`` value."""
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{BASE_FILE} is no JSON object: {exc}") from exc
        return cls.from_dict(value)

    def to_text(self) -> str:
        """The text of ``BASE_FILE`` for this value, which ``from_bytes`` reads back."""
        return json.dumps(self.to_dict(), sort_keys=True) + "\n"

    def check_inputs(self, configured: Mapping[str, str]) -> None:
        """Raise ``ValueError`` unless ``inputs`` equal ``configured``, the inputs of a deployment's default start.

        The message names each input that differs, with both values.
        """
        differing = sorted(
            name for name in set(self.inputs) | set(configured) if self.inputs.get(name) != configured.get(name)
        )
        if differing:
            details = ", ".join(
                f"{name}: {self.inputs.get(name)!r}, not {configured.get(name)!r}" for name in differing
            )
            raise ValueError(
                f"the base root names another default start than this deployment's ({details}); it stands for the "
                "base model of the deployment that made it"
            )


def read_base_root(directory: Path) -> BaseRoot | None:
    """The base root that the artifact in ``directory`` is; ``None`` for an artifact without ``BASE_FILE``.

    An artifact is a base root exactly when ``BASE_FILE`` is at its root. Raise ``ValueError`` when the file is no
    ``BaseRoot.to_dict`` value. The other files of the directory are not read: ``base_root_of`` checks them when
    the artifact is admitted.
    """
    if not (directory / BASE_FILE).is_file():
        return None
    return BaseRoot.from_bytes((directory / BASE_FILE).read_bytes())


def base_root_of(files: ArtifactFiles) -> BaseRoot | None:
    """The base root that ``files`` are; ``None`` for files without ``BASE_FILE``.

    Raise ``ValueError`` when the file is no ``BaseRoot.to_dict`` value, and when ``files`` hold another file beside
    it that a repository did not add (``REPOSITORY_FILES``): a base root holds no weights of its own.
    """
    paths = files.paths()
    if BASE_FILE not in paths:
        return None
    others = sorted(path for path in paths if path != BASE_FILE and path.split("/")[0] not in REPOSITORY_FILES)
    if others:
        raise ValueError(f"a base root holds {BASE_FILE} alone, but the artifact also holds {others[:5]}")
    return BaseRoot.from_bytes(files.read_bytes(BASE_FILE))


__all__ = ["BASE_FILE", "BASE_FORMAT", "MODEL_INPUT", "BaseRoot", "base_root_of", "read_base_root"]
