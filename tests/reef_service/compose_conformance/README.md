# The cordis spec suite, mirrored

Every test here corresponds to one `it()` of the reference test suite, at the
pin recorded in `reef/train/cordis_backend/compose/UPSTREAM.md`. UPSTREAM.md
says in prose where the port conforms; this suite executes those claims
against the reference's own cases.

Scope is the two packages the port covers: `packages/core/tests` and
`packages/loader/tests`. At the current pin that is 127 cases -- 98 mirrored
here, 29 recorded in `omissions.py` as belonging to a permanent omission.

## Correspondence rules

- **One mirror per `it()`.** `@upstream(spec, name)` carries the reference's
  file and its `it()` string verbatim. `test_conformance_coverage` reads
  those back, so an upstream case that nothing accounts for fails the suite
  by name.
- **File and order follow the reference.** One mirror file per spec file,
  cases in upstream order; a `describe` that shares state across its cases
  becomes a class over the `harness` fixture, which keeps that order.
- **Translate, don't transliterate.** The port is sync-first and has no
  proxies, so `await root.plugin(p)` becomes `root.plugin(p)` and
  `ctx[Context.intercept]` becomes a chain walk. What must not change is
  what the case asserts.
- **A divergence is asserted, not deleted.** Where the port's recorded
  divergence makes the reference's expectation impossible, the mirror
  asserts the port's behaviour and its docstring says which divergence and
  why. `test_update_config_while_injected_service_reloads` is the worked
  example: it also runs the reference's precondition, showing conformance
  returns with it.
- **Assertions inside a plugin go through `apply()`.** The port parks a load
  failure on the fiber instead of raising, so a bare `ctx.plugin()` would
  swallow a failed assertion and pass vacuously.

## Two host facilities the reference gets for free

`utils.py` supplies both, alongside `maybe_await` for the reference's
`await next()` on a value that may not be a coroutine -- JS awaits a
non-thenable, Python does not. `Mock` is `node:test`'s `mock.fn`; a class plugin
needs `mock_class` instead, because the port routes the init protocol on
`inspect.isclass`. `VirtualClock` is `vi.useFakeTimers`: an event loop whose
`time()` the test advances by hand, so the inertia and async-effect cases are
deterministic rather than racing a real clock.

## Upgrading the pin

1. `git submodule update --init third_party/cordis`, then bump it.
2. Run this suite. `test_conformance_coverage` names every case added or
   renamed upstream; the mirrors that fail name every behaviour that moved.
3. Re-port each hunk deliberately, as UPSTREAM.md's upgrade procedure says,
   and mirror the new cases. A case that lands in a permanently omitted area
   goes in `omissions.py` with its reason.
4. Update the pin line and the case counts above.

The suite skips itself when the submodule is absent, so a CI job that does
not check out submodules reports nothing here rather than failing.
