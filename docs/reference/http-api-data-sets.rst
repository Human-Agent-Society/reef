HTTP API: benchmarks, collections and trajectory sets
=====================================================

Name sets of an owner's `data items <http-api-data-items.rst>`__, and make a scenario hold a set so that it uses the set's items.

Benchmarks and collections
--------------------------

A benchmark is a name for a set of the owner's tasks. Reef does not store it: a task is in a benchmark when the list ``benchmarks`` of its metadata has the name. Set the list when you create the tasks, or with a metadata patch: ``metadata={"benchmarks": ["terminal-bench"]}``. It holds distinct non-empty strings, so a task can be in several benchmarks. Only a task has it.

A collection is a stored list of benchmarks under one name. Use it to make a scenario hold several benchmarks in one call, and to evaluate an artifact on them (`Evaluate an artifact <http-api-evaluation.rst>`__).

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/collections/coding \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"benchmarks": ["terminal-bench", "swe-bench-lite"]}'

- ``PUT /reef/namespaces/{owner}/collections/{name}`` creates the collection, or replaces the one with the name. The body takes ``benchmarks``, ``collections`` and ``average``. ``collections`` names other collections of the owner: the collection also has their benchmarks. Each list is empty by default, so ``{}`` creates an empty collection. ``average`` is ``task`` (the default) or ``benchmark``: how an evaluation on the collection computes its ``mean_score``. The response is ``{"owner", "name", "benchmarks", "collections", "average"}``. Reef returns 400 for a name in ``collections`` that the owner does not store, and for collections that name each other in a cycle.
- ``GET /reef/namespaces/{owner}/collections`` lists the owner's collections by name: ``{"owner", "collections"}``.
- ``GET /reef/namespaces/{owner}/collections/{name}`` returns one collection with ``benchmark_task_counts``, a list of ``{"benchmark", "task_count"}`` for each of its benchmarks, and ``task_count``. ``task_count`` counts a task that is in two benchmarks once.
- ``DELETE /reef/namespaces/{owner}/collections/{name}`` deletes the collection and returns ``{"owner", "name", "deleted": true}``. No task changes. Reef returns 400 while another collection names it.
- ``POST /reef/namespaces/{owner}/collections/{name}/scenarios`` takes ``{"scenario", "split"}``. The scenario then holds each benchmark that the collection has now, in that split (`The sets that a scenario holds`_). A benchmark that joins the collection later is not held. The response is ``{"owner", "name", "scenario", "split", "holds", "task_count", "synced"}``: ``holds`` lists the holds, and ``task_count`` counts a task of two benchmarks once. A collection without a benchmark returns 400.

An unknown collection returns 404.

The sets that a scenario holds
------------------------------

A scenario can hold benchmarks, which are sets of tasks, and trajectory sets. It then uses the items of each held set. A hold writes no item, so it takes the same time for a set of ten items and of ten thousand.

A trajectory set is a name for a set of the owner's trajectories, as a benchmark is for tasks: a trajectory is in the set when the list ``trajectory_sets`` of its metadata has the name. Set the list when you upload the trajectories, or with a metadata patch: ``metadata={"trajectory_sets": ["week-12"]}``. It holds distinct non-empty strings, so a trajectory can be in several sets. Only a trajectory has it. The trajectories that a scenario's trainer makes from its records are in the set ``samples:<scenario>``.

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/scenarios/my-agent/benchmarks/terminal-bench \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"split": "eval"}'

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/scenarios/my-agent/trajectory-sets/week-12 \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{}'

The routes of benchmarks are under ``/reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``, and those of trajectory sets under ``…/trajectory-sets``.

- ``PUT …/{name}`` makes the scenario hold the set. For a benchmark the body is ``{"split"}``, ``train`` or ``eval``. A scenario can hold one benchmark in both splits, with one call for each: the two holds are apart. For a trajectory set the body is ``{}``: a trajectory always trains. The response is ``{"owner", "scenario", "kind", "name", "split", "item_count", "synced"}``; ``kind`` is ``benchmark`` or ``trajectory_set``. A ``train`` hold of a benchmark needs a recipe that trains imported tasks, and a hold of a trajectory set one that takes uploaded trajectories, as a mark does; else Reef returns 400.
- ``DELETE …/{name}?split=<split>`` ends the hold in that split: ``{"owner", "scenario", "kind", "name", "split", "released", "synced"}``. A benchmark needs ``split``; a trajectory set does not. ``released`` is false when the scenario did not hold the set in that split.
- ``GET`` of the two base paths returns ``{"owner", "scenario", "kind", "holds"}``: each hold with its ``split`` and its ``item_count``, in the order of the names.

Membership is live: an item that joins a held set, by a create or a metadata patch, joins the scenario, and an item that leaves the set leaves the scenario.

A scenario uses an item once for each held set that the item is in, and once more when the item's own ``scenarios`` metadata names the scenario. Reef compares no item lists and refuses no overlap:

- A task in two held ``train`` benchmarks has two uses. A trainer reads the train uses as a set of items, so a job holds the task once, and the task has one train count. The same holds for a trajectory with a mark and a held set.
- A task in a ``train`` benchmark and in an ``eval`` benchmark is trained, and an evaluation runs it. The same holds for each task of a benchmark that the scenario holds in both splits. Hold a benchmark in one split when the evaluation must not see trained tasks.
