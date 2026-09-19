"""Service entrypoints, process assembly, and HTTP transport over the dispatcher.

``training_driver`` owns startup and shutdown of model deployment components.
The selected backend supplies a plan; the driver imports no concrete framework.
``slime_driver`` preserves the legacy explicit-process CLI.

Everything HTTP lives here, and only the HTTP parts live in the HTTP layer.
``RequestService`` is the transport-free core — it parses ``x-reef-*``
headers, freezes the release before every provider call so
concurrent publication cannot change what gets recorded, and applies the
surface's inference hooks; ``routes/`` are thin aiohttp adapters over it and
the only place aiohttp request/response types appear on the request path.

``ServiceConfig`` is frozen, and recipe-specific config fields are not
fields on it: they ride in ``recipe_settings`` and each recipe extracts its
own, so defaults live with the recipe. HTTP modules never import a concrete
training backend; backend definitions load their optional execution adapters.

Adding a route: write a ``register_*`` function in a ``routes/`` module and
wire it into ``register_routes``. The handler raises domain errors and lets
``errors.ERROR_STATUS_TABLE`` translate them; a new error type that needs a
status other than 400 gets a row there, not a local try/except. The docs
contract check derives the route list from ``routes/`` sources, so the HTTP
API reference must name every route.
"""
