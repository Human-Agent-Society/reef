Codebase structure
==================

This page says which package should own a change.


Choose a destination
--------------------

``reef/`` holds every shared mechanism, including the harness evolution engine
at ``reef/train/cordis_backend/``. Paper-backed methods live in separate
packages under ``recipes/`` (``sao``, ``tttd``, ``openclawrl``, ``skillclaw``)
with that method's recipe, processor, step preparer, and, for weight methods,
the ``slime/`` subpackage only the training plane imports. Nothing under
``reef/`` imports a method package.

Reef is organized around an application kernel and three capability domains,
not a strict stack of top-level packages. The arrows below show the primary
composition and use paths; they are not an exhaustive Python import graph:

.. code:: mermaid

   flowchart TD
       accTitle: Reef application kernel, capability domains, and adapters

       Methods("<b>Method plug-ins</b><br/><code>recipes/*</code>")
       Entry("<b>Entrypoints</b><br/>HTTP · CLI")
       Policy("<b>Policy</b><br/><code>reef/recipe</code>")
       Service("<b>Delivery &amp; composition</b><br/><code>reef/service</code>")
       Kernel(["<b>Application kernel</b><br/><code>reef/dispatcher</code> · <code>reef/scenario</code>"])

       subgraph Domains["Capability domains"]
           direction LR
           Serving("<b>Serving</b><br/><code>runtime</code> · <code>surface</code>")
           Evolution("<b>Evolution</b><br/><code>train</code> · <code>harness</code>")
           State("<b>State</b><br/><code>artifact</code> · <code>records</code>")
       end

       subgraph Adapters["Concrete adapters"]
           direction LR
           ServingAdapters("runtime adapters<br/>surface implementations")
           EvolutionAdapters("training backends<br/>harness adapters")
           StateAdapters("Git/LFS repositories<br/>SQLite")
       end

       Observability(["<b>Cross-cutting</b><br/><code>reef/observability</code>"])
       Core(["<b>Shared kernel</b><br/><code>reef/core</code>"])

       Methods --> Policy --> Kernel
       Entry --> Service --> Kernel
       Kernel --> Serving & Evolution & State
       Serving --> ServingAdapters
       Evolution --> EvolutionAdapters
       State --> StateAdapters
       Kernel -. telemetry .-> Observability
       ServingAdapters & EvolutionAdapters & StateAdapters --> Core

Method packages provide policy through ``reef/recipe`` and may bind
``reef/train`` machinery directly. HTTP and CLI entrypoints compose Reef
through ``reef/service``; the transport-free dispatcher and scenario aggregate
coordinate serving, evolution, and state. ``reef/service`` also imports
``reef/artifact`` directly to stream artifact bytes.

Concrete integrations may depend on shared contracts; shared contracts never
import a concrete integration.

+----------------------+----------------------------------------------------------+--------------------------------------------+
| Package              | Owns                                                     | Does not own                               |
+======================+==========================================================+============================================+
| ``reef/core/``       | shared value types, wire shapes, artifact                | storage, I/O, runtime behavior             |
|                      | identity, root errors                                    |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/service/``    | HTTP routes, auth, streaming, process                    | training methods or domain logic           |
|                      | lifecycle                                                | tied to aiohttp                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/scenario/``   | scenario binding, commit ordering,                       | training algorithms, repository            |
|                      | recovery, checkpoint policy                              | implementations                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/recipe/``     | the contract a method implements, dotted                 | any particular method                      |
|                      | class resolution, and runtime instance binding           |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/train/``      | the trainer loop, processor engines, batch               | HTTP endpoints, deployment                 |
|                      | types, backend integrations                              | configuration parsing                      |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/runtime/``    | backend-neutral inference and training                   | a concrete training stack                  |
|                      | contracts                                                | integration                                |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/surface/``   | delivering a published artifact to the                   | proposing, evaluating, or                  |
|                      | process or client that uses it                           | selecting updates                          |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/artifact/``  | artifact bytes, repositories,                            | commit policy or delivery                  |
|                      | materialization, release heads                           | behavior                                   |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/harness/``    | harness descriptors, tree rendering,                     | recipe policy, the release chain           |
|                      | episodes, trajectories                                   |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``recipes/``         | one method per package: recipe, processor, preparer,     | shared machinery, or another method        |
|                      | and its runnable examples                                |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``tests/``           | repository-level tests grouped by responsibility         | tests hidden inside an integration subtree |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``docker/``          | container and GPU environment setup                      | Python dependency declarations             |
+----------------------+----------------------------------------------------------+--------------------------------------------+

The extension points those packages expose are in `Python API
<../reference/python-api.rst>`__.


- Is it a value or error needed by unrelated layers without behavior attached?
  Put it in ``reef/core/``.
- Does it own scenario state, commit ordering, recovery, or rollback? Put it in
  ``reef/scenario/``.
- Does it persist or materialize versioned bytes? Put it in
  ``reef/artifact/``. If it decides how consumers activate those bytes, put
  that behavior in ``reef/surface/`` instead.
- Does it define a backend-neutral model-service contract? Put it in
  ``reef/runtime/``. Put implementation tied to a concrete training stack in
  its own ``reef/train/<integration>/`` subtree.
  ``reef/train/cordis_backend/`` is the general harness evolution engine;
  its composition core derives from cordis 4.0.0-rc.8 with the conformance
  map in its ``compose/UPSTREAM.md``. ``reef/train/slime_backend/`` is the
  weights counterpart.
- Does it turn records and feedback into a batch or step signal? Put it in
  ``reef/train/processors/`` or ``reef/train/algos/``. A recipe selects and
  binds that machinery; it should not reimplement it.
- Is it HTTP-specific? Keep the aiohttp adapter in ``reef/service/routes/``
  and put transport-independent behavior in a service or domain object.
- Does it orchestrate a benchmark, task, grader, or external environment?
  Keep it under ``recipes/<name>/examples/`` or in the external harness.

Repository-level homes
----------------------

Code for a concrete training integration lives together under
``reef/train/<integration>/``, but its surrounding files remain at repository
level:

- internal integration tests in ``tests/<integration>/``;
- import and packaging contracts in ``tests/plugin_contracts/``;
- service-facing contracts in ``tests/reef_service/``;
- the learn-nothing deployment stacks, and the smallest example around them, in
  ``recipes/basic/``;
- configuration for one runnable deployment under ``recipes/<name>/examples/``;
- container and environment setup under ``docker/``; and
- Python dependencies, package data, and plugin entry points in
  ``pyproject.toml``.

Do not copy third-party source into an integration subtree. Pin or declare the
dependency in ``pyproject.toml`` and keep Reef-owned adapters local to the
integration.

Where the detailed rules live
-----------------------------

Each package's ``__init__`` docstring states the boundaries it holds and
how to extend it; there are no READMEs under ``reef/``. Design pages for the
two packages that need more than a docstring are `Surfaces
<../developer-guide/surface.rst>`__ and `Processors
<../developer-guide/processors.rst>`__; the extension points every package
exposes are in `Python API <../reference/python-api.rst>`__, and the public
harness wire contract is `HTTP API <../reference/http-api.rst>`__. The
`top-level README <../../README.md>`__ shows how the cookbook methods sit
beside ``reef/``.

``reef/train/cordis_backend/`` is the general harness evolution engine; its
composition core derives from cordis 4.0.0-rc.8 with the conformance map in
its ``compose/UPSTREAM.md``. ``reef/train/slime_backend/`` is the weights
counterpart.

Adding a new subpackage under ``reef/`` or a new method under ``recipes/``
requires an RFC that states which layer owns the behavior.
