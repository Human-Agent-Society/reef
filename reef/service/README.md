# Service

`service/` is the request plane's front door: the aiohttp application that
accepts a request, answers it through the scenario's runtime, and hands the
exchange over as a record unless the caller selects serve-only inference with
`x-reef-capture: false`. Everything HTTP lives here — and only the HTTP parts
live in the HTTP layer: request handling itself is a transport-free object that
the routes adapt.

- [`request_service.py`](request_service.py) — `RequestService`, the
  transport-independent core: parses `x-reef-*` headers, normalizes typed
  report payloads, freezes the artifact version before every provider
  call (so concurrent publication cannot change what gets served or recorded),
  the scenario surface's inference `prepare_request`/`verify_response` hooks and
  the capture decision, and applies the configured inference retry policy.
- [`routes/`](routes) — thin aiohttp adapters over `RequestService`:
  [`inference.py`](routes/inference.py) (`/v1/chat/completions`,
  `/v1/messages`, including streaming, and `/v1/messages/count_tokens`),
  [`records.py`](routes/records.py)
  (`/reef/report`, with client-supplied `agent_record_id`
  retry dedup), [`scenarios.py`](routes/scenarios.py) (list, create,
  versions, rollback), [`system.py`](routes/system.py) (`/healthz`,
  `/reef/harness` with its `version` query, `/reef/harness/versions`,
  `/reef/status`).
- [`errors.py`](errors.py) — `ERROR_STATUS_TABLE`, the single
  Reef-error-to-HTTP-status table, ordered most-specific first and applied
  as the `translate_errors` middleware. Routes raise domain errors and never
  carry their own try/except tables.
- [`auth.py`](auth.py) — bearer-token middleware over the accepted token
  set (several may be accepted at once for rotation); constant-time digest
  comparison. `/healthz` stays reachable without credentials so liveness
  probes can run.
- [`streaming.py`](streaming.py) — SSE capture for recorded streams:
  `stream_record`, `aggregate_sse_text` (OpenAI and Anthropic shapes).
- [`app.py`](app.py) — `create_app`: builds the `RequestService`, installs
  the two middlewares, registers the routes, owns dispatcher cleanup.
- [`assembly.py`](assembly.py) — composition logic: a `ServiceSettings` in,
  a dispatcher and app out (`build_dispatcher`, `build_app`). It knows
  nothing about the deployment config format.
- [`deploy/`](deploy) — `reef serve`: [`config.py`](deploy/config.py)
  (YAML loading with `${VAR}` and `${dotted.path}` interpolation),
  [`settings.py`](deploy/settings.py) (the frozen `ServiceSettings`
  dataclass and the internal HTTP child's `run_service` entrypoint),
  [`orchestrator.py`](deploy/orchestrator.py) (start every declared process
  in dependency order, probe readiness, tear down in reverse). `deploy/` is
  a subpackage, not a sibling: its only job is starting this service.

**Adding a route**: write a `register_*` function in a [`routes/`](routes)
module and wire it into `register_routes` in
[`routes/__init__.py`](routes/__init__.py). The handler raises domain errors
and lets [`errors.py`](errors.py) translate them; if a new error type needs a
status other than 400, it gets a row in `ERROR_STATUS_TABLE`, not a local
try/except. The doc-contract script derives the route list from `routes/`
sources, so the route reference in
[Design](https://reefinfra.ai/docs/design/)
must name every route.

Boundaries this package holds:

- `RequestService` is transport-free; `routes/` is the only place aiohttp
  request/response types appear on the request path.
- `ServiceSettings` is frozen, and recipe-specific config fields are not fields on
  it: they ride in `recipe_settings` and each recipe extracts its own via
  `WeightTrainingRecipe.service_config`, so defaults live with the recipe
  ([`test_training_server.py`](../../tests/reef_service/test_training_server.py)
  pins this).
- Nothing here imports `reef.train.slime` at module scope
  ([`test_dependency_boundaries.py`](../../tests/reef_service/test_dependency_boundaries.py)).
- The service exposes no control route beyond the ones above
  ([`test_reef_contracts.py`](../../tests/reef_service/test_reef_contracts.py)).

The wire-level route reference is
[Design](https://reefinfra.ai/docs/design/).
