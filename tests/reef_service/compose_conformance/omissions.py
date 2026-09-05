"""Reference cases the port deliberately does not mirror.

Every entry names a permanent omission recorded in
``reef/train/cordis_backend/compose/UPSTREAM.md``. Deleting an entry without
adding the matching mirror -- or adding one without a reason that appears in
UPSTREAM.md -- is what ``test_conformance_coverage`` is here to catch.
"""

from __future__ import annotations

TRACEABLE = "traceable/shadow machinery (proxy-based rebinding of service.ctx)"
ASSOCIATE = f"the association machinery: Service.tracker and dotted service names resolve through {TRACEABLE}"
CALLABLE = "callable services (a Python class defines __call__)"
DECORATOR = "the Inject decorator's method form"
LOGGER = "LoggerService (stdlib logging replaces it)"
HOST = "a JS host artifact with a native Python counterpart"
MODULES = "module resolution, file watching and HMR (node module-system machinery)"

OMITTED: dict[tuple[str, str], str] = {
    ("core/tests/associate.spec.ts", "service injection"): ASSOCIATE,
    ("core/tests/associate.spec.ts", "property injection"): ASSOCIATE,
    ("core/tests/associate.spec.ts", "associated type - service injection"): ASSOCIATE,
    ("core/tests/associate.spec.ts", "associated type - accessor injection"): ASSOCIATE,
    ("core/tests/associate.spec.ts", "inspect"): ASSOCIATE,
    ("core/tests/associate.spec.ts", "associated access follows the service fiber"): ASSOCIATE,
    ("core/tests/decorator.spec.ts", "@Inject on class method"): DECORATOR,
    ("core/tests/invoke.spec.ts", "functional service"): CALLABLE,
    ("core/tests/invoke.spec.ts", "uses the service shadow for callable extensions"): CALLABLE,
    ("core/tests/logger.spec.ts", "keeps the bounded buffer in place and chronological"): LOGGER,
    ("core/tests/logger.spec.ts", "disposes the exporter that registered the disposer"): LOGGER,
    ("core/tests/logger.spec.ts", "uses fiber name when called from outside any service"): LOGGER,
    ("core/tests/logger.spec.ts", "honours explicit name argument"): LOGGER,
    ("core/tests/logger.spec.ts", "honours intercept name"): LOGGER,
    (
        "core/tests/logger.spec.ts",
        "uses service name when called from inside a Service method (regression)",
    ): LOGGER,
    (
        "core/tests/logger.spec.ts",
        "still lets outer caller intercept override the service-derived name",
    ): LOGGER,
    ("core/tests/logger.spec.ts", "uses the innermost service name and restores the outer service"): LOGGER,
    (
        "core/tests/logger.spec.ts",
        "uses service name when called from inside [Service.init] (unchanged behaviour)",
    ): LOGGER,
    ("core/tests/shadow.spec.ts", "keeps caller metadata separate from the service shadow"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "exposes the caller without preserving shadow for noShadow services"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "exposes the caller to callable services"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "strips service shadow before creating plugins"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "resolves dependencies through a captured shadow"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "applies the service inject when entered from root"): TRACEABLE,
    ("core/tests/shadow.spec.ts", "keeps unchecked access for services provided on the root ctx"): TRACEABLE,
    ("core/tests/reflect.spec.ts", "Context.is()"): (f"{HOST}: cross-realm nominal typing, which isinstance covers"),
    ("core/tests/service.spec.ts", "traceable effect (with inject)"): TRACEABLE,
    ("core/tests/service.spec.ts", "traceable effect (without inject)"): TRACEABLE,
    (
        "loader/tests/internal.spec.ts",
        "tags the running loader with the resolver signature it accepts",
    ): MODULES,
}
