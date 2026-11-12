"""Slime's loss families: their registration, and their resolution in the service, the driver and the workers.

A loss family is a :class:`~reef.train.slime_backend.algorithm.SlimeAlgorithm`. A family registers with
:func:`register_loss_family`: the decorator of its class, in the module that defines it. A method package defines
its family beside its objective, so the import of the objective's module registers the family too; the package's
public half registers only the objective, by reference (``reef.train.slime_backend.objectives``). The built-in
families are known by reference from the start. A family is also named as a dotted ``"package.module:SPEC"``
reference wherever a loss family is named, where ``SPEC`` is a family class or instance.

A worker process starts without the driver's registrations. The driver therefore stamps the family's reference on
``args.loss_family_ref`` (:func:`loss_family_reference`), and the worker imports it
(:func:`resolve_args_loss_family`). This module never imports a method package by itself, and it imports no torch,
Ray or Slime.
"""

from __future__ import annotations

import importlib
from argparse import Namespace
from typing import overload

from reef.train.slime_backend.algorithm import WORKER_HOOKS, SlimeAlgorithm


class UnknownLossFamilyError(RuntimeError):
    """Raised when the driver is asked to boot an unsupported loss family."""


#: The loss families that this process imported, by name. ``register_loss_family`` and ``resolve_loss_family``
#: fill it.
LOSS_FAMILIES: dict[str, SlimeAlgorithm] = {}
#: Where a process imports a family from: ``{name: "package.module:SPEC"}``. It holds the built-in families from
#: the start, so that only the driver and the workers import their modules, and every family that registered as a
#: class or resolved from a reference. The driver sends a family's reference to the workers.
LOSS_FAMILY_REFERENCES: dict[str, str] = {
    "sft": "reef.train.slime_backend.sft:SftAlgorithm",
    "importance_sampling": "reef.train.slime_backend.importance_sampling:ImportanceSamplingAlgorithm",
    "slime": "reef.train.slime_backend.native:SlimeNativeAlgorithm",
}


@overload
def register_loss_family(family: type[SlimeAlgorithm]) -> type[SlimeAlgorithm]: ...


@overload
def register_loss_family(family: SlimeAlgorithm) -> SlimeAlgorithm: ...


def register_loss_family(family: SlimeAlgorithm | type[SlimeAlgorithm]) -> SlimeAlgorithm | type[SlimeAlgorithm]:
    """Register a family under its ``loss_family`` name: decorate its zero-argument class, or pass an instance.

    A class at module level also registers its reference, which a worker process imports. An instance is known in
    this process only. Registering the same instance twice changes nothing. ``TypeError``: the value is no
    ``SlimeAlgorithm``. ``ValueError``: the family declares no name, or its name holds another family.
    """
    if isinstance(family, type):
        if not issubclass(family, SlimeAlgorithm):
            raise TypeError(f"a loss family registers a SlimeAlgorithm, got {family.__name__}")
        spec = family()
        # Only a class at module level can be imported by its reference.
        reference = None if "<locals>" in family.__qualname__ else f"{family.__module__}:{family.__qualname__}"
    elif isinstance(family, SlimeAlgorithm):
        spec = family
        reference = None
    else:
        raise TypeError(f"a loss family registers a SlimeAlgorithm, got {type(family).__name__}")
    name = spec.loss_family
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a loss family must declare a non-empty loss_family")
    if LOSS_FAMILIES.get(name) is spec:
        return family
    if name in LOSS_FAMILIES or LOSS_FAMILY_REFERENCES.get(name, reference) != reference:
        raise ValueError(
            f"loss family {name!r} is already registered; registered families: {', '.join(loss_family_names())}"
        )
    LOSS_FAMILIES[name] = spec
    if reference is not None:
        LOSS_FAMILY_REFERENCES[name] = reference
    return family


def unregister_loss_family(loss_family: str) -> None:
    """Remove a loss family and its reference, for extension teardown. ``KeyError``: no family has the name."""
    spec = LOSS_FAMILIES.pop(loss_family, None)
    reference = LOSS_FAMILY_REFERENCES.pop(loss_family, None)
    if spec is None and reference is None:
        registered = ", ".join(loss_family_names()) or "none"
        raise KeyError(f"loss family {loss_family!r} is not registered; registered families: {registered}")


def loss_family_names() -> tuple[str, ...]:
    """The names that ``resolve_loss_family`` resolves in this process, sorted."""
    return tuple(sorted({*LOSS_FAMILIES, *LOSS_FAMILY_REFERENCES}))


def loss_family_reference(loss_family: str) -> str | None:
    """The ``package.module:SPEC`` reference that another process imports the family from, or None without one."""
    return LOSS_FAMILY_REFERENCES.get(loss_family)


def resolve_loss_family(loss_family: str) -> SlimeAlgorithm:
    """The family of a registered name or of a dotted ``"package.module:SPEC"`` reference.

    A reference is imported at its first resolution, and its family registers under its ``loss_family`` name with
    that reference. ``UnknownLossFamilyError``: the name is unknown, the reference is malformed or does not
    import, it names no ``SlimeAlgorithm``, or its family name holds another family.
    """
    spec = LOSS_FAMILIES.get(loss_family)
    if spec is not None:
        return spec
    reference = LOSS_FAMILY_REFERENCES.get(loss_family)
    if reference is None:
        if ":" not in loss_family:
            raise UnknownLossFamilyError(
                f"unsupported Slime loss family: {loss_family!r}; available families: "
                f"{', '.join(loss_family_names())}. A family registers with register_loss_family() in the module "
                f"that its objective names, or is named as a dotted 'package.module:SPEC' reference."
            )
        reference = loss_family
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise UnknownLossFamilyError(f"dotted loss family {reference!r} must be 'package.module:SPEC'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise UnknownLossFamilyError(f"cannot import loss family {reference!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, SlimeAlgorithm):
        # The import of a decorated class already registered its instance. Use that one.
        registered = LOSS_FAMILIES.get(candidate.loss_family)
        candidate = registered if isinstance(registered, candidate) else candidate()
    if not isinstance(candidate, SlimeAlgorithm):
        raise UnknownLossFamilyError(
            f"loss family {reference!r} is not a SlimeAlgorithm (got {type(candidate).__name__})"
        )
    name = candidate.loss_family
    if LOSS_FAMILIES.get(name) is not candidate:
        if name in LOSS_FAMILIES or LOSS_FAMILY_REFERENCES.get(name, reference) != reference:
            raise UnknownLossFamilyError(
                f"dotted loss family {reference!r} names family {name!r}, which is already registered to a "
                f"different spec; pick a family name not among: {', '.join(loss_family_names())}"
            )
        LOSS_FAMILIES[name] = candidate
    LOSS_FAMILY_REFERENCES.setdefault(name, reference)
    return candidate


def resolve_args_loss_family(args: Namespace) -> SlimeAlgorithm:
    """Resolve the loss family a worker's ``args`` names.

    The driver stamps the canonical ``loss_family``, and the family's
    reference on ``loss_family_ref`` when it has one: the driver's
    registrations do not travel with the Namespace, so a fresh worker process
    has to import the module itself. The reference wins when present.
    """
    return resolve_loss_family(getattr(args, "loss_family_ref", None) or args.loss_family)


def resolve_objective_paths(args: Namespace) -> None:
    """Resolve custom objective function paths onto ``args`` (worker-side).

    Called during worker init.  Imports the algorithm's ``objective`` module
    (triggering ``@worker_hook`` decorators) and projects each registered
    dotted path onto ``args``.  A family without an ``objective`` module is
    skipped when it declares no hooks.  Import failures raised from
    inside an objective module propagate so a broken worker dependency cannot
    silently disable the training objective.
    """
    loss_family = getattr(args, "loss_family", None)
    if not loss_family:
        return
    spec = resolve_args_loss_family(args)
    required_hooks = set(spec.required_objective_hooks)
    if spec.loss_type == "custom_loss":
        required_hooks.add("custom_loss_function_path")
    if spec.uses_pg_loss_primitive:
        required_hooks.add("custom_pg_loss_function_path")

    module_path = f"{spec.__class__.__module__}.objective"
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name == module_path:
            if not required_hooks:
                return
            hooks = ", ".join(sorted(required_hooks))
            raise RuntimeError(
                f"loss family {loss_family!r} requires objective hooks {hooks}, but {module_path} is missing"
            ) from error
        raise

    registered = WORKER_HOOKS.get(module_path, {})
    missing_hooks = required_hooks.difference(registered)
    if missing_hooks:
        hooks = ", ".join(sorted(missing_hooks))
        raise RuntimeError(f"loss family {loss_family!r} did not register required objective hooks: {hooks}")
    for arg_name, fn_name in registered.items():
        setattr(args, arg_name, f"{module_path}.{fn_name}")


def reset_objective_paths(args: Namespace) -> None:
    """Clear every objective hook path a family registered in this worker, before another family resolves its own.

    A job of another loss family re-points the worker (see
    ``worker_hooks.switch_loss_family``); a hook the previous family
    registered and the next one does not must not stay on ``args``.
    """
    for arg_name in {name for hooks in WORKER_HOOKS.values() for name in hooks}:
        setattr(args, arg_name, None)
