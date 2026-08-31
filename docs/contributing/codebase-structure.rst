Codebase structure
==================

This page says which package should own a change.


Choose a destination
--------------------

``reef/`` holds every shared mechanism. Cookbook methods live in separate
packages under ``recipes/`` (``sao``, ``tttd``, ``openclawrl``, or
``harness_evolve``) with that method's recipe, processor, step
preparer, and, for weight methods, the ``slime/`` subpackage only the training
plane imports. Nothing under ``reef/`` imports a method package.

The packages form layers. A layer imports only the layers beneath it, so
a change that needs to reach upward is in the wrong package:

.. code:: mermaid

   flowchart TD
       accTitle: Dependency direction between package layers
       subgraph Methods["Methods in recipes/"]
           direction LR
           M["sao, tttd, openclawrl, harness_evolve<br/>recipe, processor, preparer"]
       end
       subgraph Orchestration["Orchestration"]
           direction LR
           Service["reef/service<br/>HTTP, assembly, lifecycle"]
           Scenario["reef/scenario<br/>commit order, recovery"]
       end
       subgraph Contracts["Contracts"]
           direction LR
           Recipe["reef/recipe<br/>what a method implements"]
           Runtime["reef/runtime<br/>inference and training"]
           Harness["reef/harness<br/>descriptors, episodes"]
       end
       subgraph Engines["Engines and storage"]
           direction LR
           Train["reef/train<br/>trainer, processors, backends"]
           Surfaces["reef/surface<br/>delivery of an artifact"]
           Artifacts["reef/artifact<br/>bytes, repositories, versions"]
       end
       Core["reef/core: value types, wire shapes, errors"]
       Methods --> Orchestration
       Orchestration --> Contracts
       Contracts --> Engines
       Engines --> Core

Two edges skip a layer and are allowed: a method package binds ``reef/train``
machinery directly, and ``reef/service`` imports ``reef/artifact`` to stream
artifact bytes. Everything else follows the arrows.

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
|                      | materialization, version heads                           | behavior                                   |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/harness/``    | harness descriptors, tree rendering,                     | recipe policy, the version chain           |
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

Adding a new subpackage under ``reef/`` or a new method under ``recipes/``
requires an RFC that states which layer owns the behavior.
