"""The report contract of a scenario that several components share."""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from reef.core.reports.base import ReportBase, ReportValidationError

__all__ = ["AnyComponentReport", "any_component_report", "shared_report_type"]


class AnyComponentReport(ReportBase):
    """A report contract that a payload meets when any of ``report_types`` parses it.

    Values: the class is a contract only, and no instance of it is made. ``report_types`` holds the contracts of
    the components, in order and without a repeat; ``any_component_report`` makes each subclass with its own.

    ``from_dict(payload)`` returns the report of the first of ``report_types`` that parses ``payload``. It raises
    ``ReportValidationError`` when none does; the message names each contract with its refusal.

    Purpose: the scenario factory binds it as the ingress contract of a scenario whose components declare
    different contracts (``reef.scenario.factory.scenario_report_type``). Each trainer parses with its own
    component's contract, so ingress refuses only a report that no component takes.
    """

    report_types: ClassVar[tuple[type[ReportBase], ...]] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReportBase:
        refusals = []
        for report_type in cls.report_types:
            try:
                return report_type.from_dict(payload)
            except ReportValidationError as exc:
                refusals.append(f"{report_type.__name__}: {exc}")
        raise ReportValidationError("no component accepts this report: " + "; ".join(refusals))


@functools.cache
def any_component_report(report_types: tuple[type[ReportBase], ...]) -> type[ReportBase]:
    """The contract a payload meets when any of ``report_types`` parses it, named after them. The same
    ``report_types`` give the same class at every call."""
    name = "AnyOf" + "".join(item.__name__ for item in report_types)
    return type(name, (AnyComponentReport,), {"report_types": report_types})


def shared_report_type(report_types: Iterable[type[ReportBase] | None]) -> type[ReportBase] | None:
    """The one report contract of a scenario whose components declare ``report_types``, in order.

    ``None`` entries declare no contract and are left out, and so is a repeat. The result is ``None`` when no
    contract is left, which keeps ingress open. It is the contract itself when one is left. Else it is the
    ``any_component_report`` of the contracts that are left.
    """
    declared: list[type[ReportBase]] = []
    for report_type in report_types:
        if report_type is not None and report_type not in declared:
            declared.append(report_type)
    if not declared:
        return None
    if len(declared) == 1:
        return declared[0]
    return any_component_report(tuple(declared))
