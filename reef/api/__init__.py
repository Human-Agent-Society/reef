"""The HTTP process of each service: process assembly, and HTTP transport over the dispatcher.

``reef.deploy`` starts and stops the deployment's processes. ``assembly`` builds
the dispatcher and the app. It loads the recipe, which is configuration only,
and connects the selected integration's training component and inference
runtime. The process holds those objects in one
``reef.control.scenario.live_objects.LiveObjects`` value, which assembly gives to the
dispatcher; a recipe has no runtime and no training component. Assembly
imports no concrete training framework.

Everything HTTP lives here, and only the HTTP parts live in the HTTP layer.
``RequestService`` is the transport-free core — it parses ``x-reef-*``
headers, freezes the snapshot before every provider call so
concurrent publication cannot change what gets recorded, and stores each
exchange as a record; ``routes/`` are thin aiohttp adapters over it and
the only place aiohttp request/response types appear on the request path.
A route parses the ``x-reef-*`` headers and calls an operation of
``reef.control.operations``, which takes values and no headers.

Assembly wraps the scenario storage with ``reef.storage.scenario.task_import``, which
imports a deployment's Harbor tasks into each scenario when the scenario opens.

``ServiceSettings`` is frozen, and recipe-specific config fields are not
fields on it: they ride in ``recipe_settings`` and each recipe extracts its
own, so defaults live with the recipe. HTTP modules never import a concrete
training backend; backend definitions load their optional execution adapters.

Adding a route: write a ``register_*`` function in a ``routes/`` module and
wire it into ``register_routes``. The handler raises domain errors and lets
``errors.ERROR_STATUS_TABLE`` translate them; a new error type that needs a
status other than 400 gets a row there, not a local try/except. A failure
that the request did not cause derives from ``reef.core.errors.InternalError``
and answers 500. The docs
contract check derives the route list from ``routes/`` sources, so the HTTP
API reference must name every route.
"""
