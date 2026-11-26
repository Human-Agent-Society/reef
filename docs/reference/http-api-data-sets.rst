HTTP API: benchmarks, collections and trajectory sets
=====================================================

Group an owner's `data items <http-api-data-items.rst>`__ and assign groups to scenarios.

Benchmarks and collections
--------------------------

A benchmark groups tasks by name in their ``benchmarks`` metadata. Set it at creation or through a metadata patch: ``metadata={"benchmarks": ["terminal-bench"]}``. The list accepts distinct nonempty strings and applies only to tasks. Tasks can belong to multiple benchmarks; there is no separate benchmark object.

A collection stores a named list of benchmarks. Assign it to a scenario in one call or use it for `artifact evaluation <http-api-evaluation.rst>`__.

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/collections/coding \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"benchmarks": ["terminal-bench", "swe-bench-lite"]}'

- ``PUT /reef/namespaces/{owner}/collections/{name}`` creates or replaces a collection. The body accepts ``benchmarks``, ``collections`` and ``average``. Nested ``collections`` add their benchmarks and must belong to the same owner. Both lists default to empty. ``average`` selects ``task`` (default) or ``benchmark`` weighting for evaluation's ``mean_score``. The response is ``{"owner", "name", "benchmarks", "collections", "average"}``. Unknown nested collections and cycles return 400.
- ``GET /reef/namespaces/{owner}/collections`` lists the owner's collections by name: ``{"owner", "collections"}``. Use ``?counts=true`` to add ``benchmark_count`` and ``task_count`` to each collection. The counts include benchmarks of nested collections. Reef counts each benchmark and each current task item once per collection.
- ``GET /reef/namespaces/{owner}/collections/{name}`` returns one collection with ``benchmark_task_counts``, a list of ``{"benchmark", "task_count"}`` for each of its benchmarks, and ``task_count``. ``task_count`` counts a task that is in two benchmarks once.
- ``DELETE /reef/namespaces/{owner}/collections/{name}`` deletes the collection and returns ``{"owner", "name", "deleted": true}``. No task changes. Reef returns 400 while another collection names it.
- ``POST /reef/namespaces/{owner}/collections/{name}/scenarios`` takes ``{"scenario", "split"}`` and assigns the collection's current benchmarks (`The sets that a scenario holds`_). Later benchmark additions are not assigned. The response is ``{"owner", "name", "scenario", "split", "holds", "task_count", "synced"}``. ``holds`` lists assignments; ``task_count`` counts shared tasks once. Empty collections return 400.

An unknown collection returns 404.

The sets that a scenario holds
------------------------------

A scenario uses all items in its held benchmarks and trajectory sets. Holding a set does not rewrite its items.

A trajectory's ``trajectory_sets`` metadata lists its sets: ``metadata={"trajectory_sets": ["week-12"]}``. Set it at upload or through a metadata patch. It accepts distinct nonempty strings and applies only to trajectories. Generated training trajectories belong to ``samples:<scenario>``.

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

- ``PUT …/{name}`` holds a set. For benchmarks, send ``{"split": "train"}`` or ``{"split": "eval"}``; separate calls can hold both splits independently. For trajectory sets, send ``{}``; trajectories always train. The response is ``{"owner", "scenario", "kind", "name", "split", "item_count", "synced"}``, with ``kind`` set to ``benchmark`` or ``trajectory_set``. Train benchmarks require imported-task support; trajectory sets require uploaded-trajectory support. Unsupported holds return 400.
- ``DELETE …/{name}?split=<split>`` ends the hold in that split: ``{"owner", "scenario", "kind", "name", "split", "released", "synced"}``. A benchmark needs ``split``; a trajectory set does not. ``released`` is false when the scenario did not hold the set in that split.
- ``GET`` of the two base paths returns ``{"owner", "scenario", "kind", "holds"}``: each hold with its ``split`` and its ``item_count``, in the order of the names.

Membership is live: creating items or patching metadata adds or removes scenario assignments as items join or leave held sets.

A scenario can use an item through several held sets and its ``scenarios`` metadata. Overlaps are allowed:

- A task in two held ``train`` benchmarks has two uses. A trainer reads the train uses as a set of items, so a job holds the task once, and the task has one train count. The same holds for a trajectory with a mark and a held set.
- A task in a ``train`` benchmark and in an ``eval`` benchmark is trained, and an evaluation runs it. The same holds for each task of a benchmark that the scenario holds in both splits. Hold a benchmark in one split when the evaluation must not see trained tasks.
