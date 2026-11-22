"""The operations that a client can ask of Reef, without HTTP.

An operation reads and changes scenarios through the dispatcher and the stores. A route in ``reef.api.routes``
reads the request, calls one operation, and writes the response.

- ``data`` creates, reads, queries, updates and deletes the data items of one owner. ``collections`` handles its
  collections and a scenario's benchmarks, and ``tasks`` its task archives and a scenario's task listing.
- ``data_transfer`` imports and exports items, and ``data_service`` holds the worker of the data service.
- ``artifact_evaluations`` handles the evaluations of an artifact that a client asks for.
- ``artifact_imports`` imports an artifact into a scenario.
- ``scenario_files`` reads a scenario's artifact files.
- ``samples`` and ``sample_processing`` read a scenario's training samples and the work that makes them.

Nothing here imports ``reef.api``, ``reef.deploy`` or aiohttp. The dispatcher, the scenario package and the workers do not
import this package.

Import each module directly; this package exports nothing.
"""
