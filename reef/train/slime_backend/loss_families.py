"""Registry for Reef loss families supported by the Slime training backend.

A bundled family registers a dotted reference to its spec in
:mod:`reef.train.algos.registry` from its package ``__init__``
(imported by ``reef``); this registry imports the reference the first time
the name is resolved, and the ``@register_loss_family`` class decorator in
the spec module registers the instance. This module never imports a method
package. The same lazy path keeps the table full
in the driver and in every worker.  The table is open for external families the
same way recipe kinds and step preparers are: an external family registers its
:class:`SlimeAlgorithm` with :func:`register_loss_family` from a module imported
at boot, or is named as a dotted ``"package.module:SPEC"`` reference wherever a
loss family is named (``REEF_TRAINING_LOSS``, a recipe's ``loss_family``).
``SPEC`` is the ``@register_loss_family``-decorated class (an instance is
accepted too). Resolving a dotted reference imports the module and registers
the spec under its canonical ``loss_family`` name, so the wire payload's plain
family name resolves from then on — in that process. A fresh worker process
starts with an empty registry, which is why the driver also stamps the
reference itself on ``args.loss_family_ref`` (see
:func:`~reef.train.slime_backend.algorithm.resolve_args_loss_family`).

Renamed families keep their old name as an alias for one release
(``_DEPRECATED_ALIASES``); resolving through the old name warns.
"""

from __future__ import annotations

import importlib
import warnings

from reef.train.algos.registry import loss_family_refs
from reef.train.slime_backend.algorithm import SlimeAlgorithm, _loss_families


class UnknownLossFamilyError(RuntimeError):
    """Raised when the driver is asked to boot an unsupported loss family."""


_DEPRECATED_ALIASES = {"openclawrl-topk": "openclawrl"}


class LossFamilyRegistry:
    """Bundled families come from the name → reference table and are imported
    on first resolve; their specs land in ``_loss_families`` through the
    decorator. External families are tracked separately so ``unregister`` can
    refuse bundled ones while allowing plugin teardown.
    """

    def __init__(self) -> None:
        self._external: dict[str, SlimeAlgorithm] = {}

    @property
    def _bundled(self) -> frozenset[str]:
        return frozenset(loss_family_refs())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._bundled | set(_loss_families) | set(self._external)))

    def register(self, spec: SlimeAlgorithm) -> SlimeAlgorithm:
        """Register an external loss-family spec under its ``loss_family`` name."""
        if not isinstance(spec, SlimeAlgorithm):
            raise TypeError(f"a loss family registers a SlimeAlgorithm, got {type(spec).__name__}")
        existing = self._external.get(spec.loss_family)
        if existing is spec:
            return spec
        bundled = _loss_families.get(spec.loss_family)
        if bundled is spec:
            return spec
        if bundled is not None or existing is not None:
            raise ValueError(
                f"loss family {spec.loss_family!r} is already registered; registered families: {', '.join(self.names)}"
            )
        self._external[spec.loss_family] = spec
        return spec

    def unregister(self, loss_family: str) -> None:
        """Remove a previously registered external family (test/plugin teardown)."""
        if loss_family in self._bundled:
            raise ValueError(f"loss family {loss_family!r} is bundled and cannot be unregistered")
        if loss_family in self._external:
            self._external.pop(loss_family)
            return
        # A decorated class imported after boot lands in the decorator table
        # without being bundled; it is external all the same.
        if loss_family in _loss_families and loss_family not in self._bundled:
            _loss_families.pop(loss_family)
            return
        registered = ", ".join(sorted(self._external)) or "none"
        raise KeyError(f"loss family {loss_family!r} is not registered; registered external families: {registered}")

    def resolve(self, loss_family: str) -> SlimeAlgorithm:
        """Resolve a registered family name or a dotted ``"package.module:SPEC"`` reference."""
        canonical = _DEPRECATED_ALIASES.get(loss_family)
        if canonical is not None:
            warnings.warn(
                f"loss family {loss_family!r} is now {canonical!r}; the old name is accepted for one release",
                DeprecationWarning,
                stacklevel=2,
            )
            loss_family = canonical
        spec = _loss_families.get(loss_family) or self._external.get(loss_family)
        if spec is not None:
            return spec
        reference = loss_family_refs().get(loss_family)
        if reference is not None:
            return self._resolve_dotted(reference)
        if ":" in loss_family:
            return self._resolve_dotted(loss_family)
        available = ", ".join(self.names)
        raise UnknownLossFamilyError(
            f"unsupported Slime loss family: {loss_family!r}; available families: {available}. "
            f"An unbundled family can be registered with register_loss_family() from a module "
            f"imported at boot, or named as a dotted 'package.module:SPEC' reference."
        )

    def _resolve_dotted(self, reference: str) -> SlimeAlgorithm:
        """Import a dotted spec reference and register it under its family name."""
        module_name, _, attribute = reference.partition(":")
        if not module_name or not attribute:
            raise UnknownLossFamilyError(f"dotted loss family {reference!r} must be 'package.module:SPEC'")
        try:
            candidate = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise UnknownLossFamilyError(f"cannot import loss family {reference!r}: {exc}") from exc
        if isinstance(candidate, type) and issubclass(candidate, SlimeAlgorithm):
            # The decorated class: importing it already registered an
            # instance under its family name. Use that one.
            registered = _loss_families.get(candidate.loss_family)
            candidate = registered if isinstance(registered, candidate) else candidate()
        if not isinstance(candidate, SlimeAlgorithm):
            raise UnknownLossFamilyError(
                f"loss family {reference!r} is not a SlimeAlgorithm (got {type(candidate).__name__})"
            )
        bundled_ref = loss_family_refs().get(candidate.loss_family)
        if bundled_ref is not None and bundled_ref != reference and candidate.loss_family not in _loss_families:
            # A bundled family is imported on first resolve; load it before an
            # external reference can claim its name.
            self._resolve_dotted(bundled_ref)
        bundled = _loss_families.get(candidate.loss_family)
        if bundled is candidate:
            return candidate
        if bundled is not None:
            raise UnknownLossFamilyError(
                f"dotted loss family {reference!r} names family {candidate.loss_family!r}, which is "
                f"already registered to a different spec; pick a family name not among: "
                f"{', '.join(self.names)}"
            )
        existing = self._external.get(candidate.loss_family)
        if existing is candidate:
            return candidate
        if existing is not None:
            raise UnknownLossFamilyError(
                f"dotted loss family {reference!r} names family {candidate.loss_family!r}, which is "
                f"already registered to a different spec; pick a family name not among: "
                f"{', '.join(self.names)}"
            )
        self._external[candidate.loss_family] = candidate
        return candidate


LOSS_FAMILIES = LossFamilyRegistry()


def register_loss_family(spec: SlimeAlgorithm) -> SlimeAlgorithm:
    """Register an external loss-family spec (module-level convenience)."""
    return LOSS_FAMILIES.register(spec)


def unregister_loss_family(loss_family: str) -> None:
    """Remove an external loss family registered by :func:`register_loss_family`."""
    LOSS_FAMILIES.unregister(loss_family)


def resolve_loss_family(loss_family: str) -> SlimeAlgorithm:
    return LOSS_FAMILIES.resolve(loss_family)
