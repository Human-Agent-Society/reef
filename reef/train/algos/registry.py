"""Lazy references from a method package to its code for each training backend.

A method package holds its backend code in modules that only that backend imports. It records here where they are, so
a backend finds them by name, in each process, and the package's public half imports no backend. Reef reads neither
table: each backend reads its own (``reef.train.slime_backend``, ``reef.train.tinker_backend``).
"""

from __future__ import annotations

#: The backend of a reference that names none.
_DEFAULT_BACKEND = "slime"
#: ``{backend: {loss_family: reference}}``.
_loss_family_refs: dict[str, dict[str, str]] = {}
#: ``{backend: {objective_name: reference}}``.
_objective_refs: dict[str, dict[str, str]] = {}


def _register_ref(table: dict[str, dict[str, str]], kind: str, name: str, reference: str, backend: str) -> None:
    if ":" not in reference:
        raise ValueError(f"{kind} reference {reference!r} must be 'package.module:ATTRIBUTE'")
    if not backend:
        raise ValueError(f"{kind} references need a non-empty backend name")
    references = table.setdefault(backend, {})
    existing = references.get(name)
    if existing is not None and existing != reference:
        raise ValueError(f"{kind} {name!r} is already registered to {existing!r} for {backend}")
    references[name] = reference


def register_loss_family_ref(loss_family: str, reference: str, *, backend: str = _DEFAULT_BACKEND) -> None:
    """Record that ``backend`` implements ``loss_family`` at the dotted ``reference``.

    A method registers one reference per backend it supports; the same family
    name selects each backend's implementation.
    """
    _register_ref(_loss_family_refs, "loss family", loss_family, reference, backend)


def loss_family_refs(backend: str = _DEFAULT_BACKEND) -> dict[str, str]:
    """The registered ``{loss_family: reference}`` table of one backend (a copy)."""
    return dict(_loss_family_refs.get(backend, {}))


def _unregister_loss_family_ref(loss_family: str, *, backend: str = _DEFAULT_BACKEND) -> bool:
    """Remove a registered family reference, returning whether it existed."""
    return _loss_family_refs.get(backend, {}).pop(loss_family, None) is not None


def register_objective_ref(name: str, reference: str, *, backend: str = _DEFAULT_BACKEND) -> None:
    """Record that ``backend`` implements the objective ``name`` at the dotted ``reference``.

    A method registers one reference per backend it supports, so a recipe's default ``objective`` selects each
    backend's implementation by one name.
    """
    _register_ref(_objective_refs, "objective", name, reference, backend)


def objective_refs(backend: str = _DEFAULT_BACKEND) -> dict[str, str]:
    """The registered ``{objective_name: reference}`` table of one backend (a copy)."""
    return dict(_objective_refs.get(backend, {}))
