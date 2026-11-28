HTTP API: benchmarks, collections and trajectory sets
=====================================================

Group an owner's `data items <http-api-data-items.rst>`__ and assign them to scenarios.

Benchmarks and collections
--------------------------

A task's ``benchmarks`` metadata lists distinct nonempty names, such as
``{"benchmarks": ["terminal-bench"]}``. Set it at creation or through a metadata patch.
Tasks can belong to several benchmarks; benchmarks have no separate objects.

Collections name groups of benchmarks for assignment or `evaluation <http-api-evaluation.rst>`__:

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/collections/coding \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"benchmarks": ["terminal-bench", "swe-bench-lite"]}'

Collection routes start at ``/reef/namespaces/{owner}/collections``:

- ``PUT /{name}`` creates or replaces ``{benchmarks, collections, average}``.
  Both lists default to empty. Nested collections must share the owner;
  unknown names and cycles return 400.
  ``average`` selects evaluation weighting: ``task`` (default) or ``benchmark``.
  The response adds ``owner`` and ``name``.
- ``GET`` returns ``{owner, collections}``, sorted by name.
  ``?counts=true`` adds unique ``benchmark_count`` and ``task_count``, including nested collections.
- ``GET /{name}`` returns the collection, total ``task_count`` and
  ``benchmark_task_counts`` entries with ``benchmark`` and ``task_count``.
  Shared tasks count once in the total.
- ``DELETE /{name}`` returns ``{owner, name, deleted: true}`` without changing tasks.
  References from another collection prevent deletion with 400.
- ``POST /{name}/scenarios`` takes ``{scenario, split}`` and assigns current benchmarks;
  later benchmark additions are excluded. Empty collections return 400.
  The response is ``{owner, name, scenario, split, holds, task_count, synced}``;
  ``holds`` lists assignments and ``task_count`` counts shared tasks once.

Unknown collections return 404.

The sets that a scenario holds
------------------------------

Holding a benchmark or trajectory set uses its items without rewriting them.
A trajectory's ``trajectory_sets`` metadata lists distinct nonempty set names,
such as ``{"trajectory_sets": ["week-12"]}``.
Generated training trajectories belong to ``samples:<scenario>``.

Routes start at ``/reef/namespaces/{owner}/scenarios/{scenario}``, followed by
``/benchmarks`` or ``/trajectory-sets``:

- ``PUT /{name}`` holds a benchmark with ``{split: train}`` or ``{split: eval}``;
  separate calls can hold both. Trajectory sets take ``{}`` and always train.
  The response is ``{owner, scenario, kind, name, split, item_count, synced}``.
  ``kind`` is ``benchmark`` or ``trajectory_set``.
  Train benchmarks require imported-task support; trajectory sets require uploaded-trajectory support.
  Unsupported holds return 400.
- ``DELETE /{name}?split=<split>`` releases a hold.
  Benchmarks require ``split``; trajectory sets do not.
  The response replaces ``item_count`` with ``released``, false for an absent hold.
- ``GET`` returns ``{owner, scenario, kind, holds}``, sorted by name.
  Each hold includes ``split`` and ``item_count``.

Membership follows item creation and metadata changes.
Held sets and direct ``scenarios`` assignments can overlap without duplicating
job inputs or train counts.
Tasks assigned to both train and eval can appear in both.
Use disjoint splits when evaluation must exclude trained tasks.
