HTTP API: data items
====================

Read, change, delete and download the data items that scenarios train on, and learn which items Reef makes itself. To add items, see `Create and transfer data items <http-api-data-upload.rst>`__. To group them, see `Benchmarks, collections and trajectory sets <http-api-data-sets.rst>`__.

A deployment keeps its data items in one data store. An item is content and metadata:

- The content is an ATIF trajectory, a Harbor task directory, a record, or a reference to the files of an artifact. It never changes. To change data, create a new item.
- The metadata is a JSON object. You can add keys. Reef reads these:

  - ``kind`` (``trajectory``, ``task``, ``record`` or ``artifact``) and ``owner``.
  - ``scenarios``: maps each scenario that uses the item by this mark to its split, ``train`` or ``eval``. A trajectory is always ``train``. An artifact item has no ``scenarios``: a scenario imports it instead (`Artifact items`_).
  - ``origin``: the scenario that made a record or an artifact. A record's ``origin.source`` is ``inference`` or ``evaluation``; a record without it is ``inference``.
  - ``benchmarks``, on a task: the names of its benchmarks (`Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__).
  - ``trajectory_sets``, on a trajectory: the names of its trajectory sets (`The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__).

  ``kind``, ``owner``, ``origin`` and ``evaluation_id`` never change.

Each item belongs to one owner, which the route paths name: ``/reef/namespaces/{owner}/data``. A scenario's owner is the scenario's name, or the ``owner`` that ``POST /reef/scenarios`` gave when it created the scenario. A scenario uses only the items of its owner, and Reef refuses an item that names a scenario of another owner. A scenario can also use a task through a benchmark that it holds, and a trajectory through a trajectory set, with no mark on the item (`The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__).

The data service answers the data routes. A one-process ``reef serve`` includes it. When the services run in separate processes, send the data routes to the data service: control does not answer them (`Run the services in separate processes <../user-guide/split-services.rst>`__).

Training reads the items that a scenario uses from the data store:

- Each trajectory is one sample toward ``job_samples`` and ``buffer_size``. Job data takes the samples in the order they joined the scenario.
- A train task trains like a trajectory. In ``sft`` it trains only in ``manual`` mode, for a request that gives the job's parameters (`Manual training <http-api-training.rst#manual-training>`__).
- An item that leaves the scenario before a job takes it does not train. It keeps its train counts there, so ``auto`` does not train it again if it comes back.
- Archiving a scenario deletes none of its items. It removes the scenario from their ``scenarios``, except from the record items that the scenario made.

For the recipe side, see `Uploaded trajectories <../developer-guide/recipe-control-service.rst#uploaded-trajectories>`__.

After a write adds items to a scenario or removes them, the data service asks control to sync each scenario that the write touched, and waits up to 30 seconds. Each create, update and delete response has ``synced``: true when every sync ended, false when one still waits. A waiting sync runs when control takes it, or when the scenario loads.

An unknown item ID returns 404. A deployment without a data store returns 503.

Record items
------------

Reef makes a record item from each inference call and report of a scenario. A training request gets no item. The item belongs to the scenario's owner:

- Its ID is ``<scenario>/<agent_record_id>``.
- Its content is ``{"agent_record_id", "request_type", "payload", "created_at", "references", "artifact_id"}``: the record as the scenario stores it.
- Its metadata is ``{"kind": "record", "scenarios": {"<scenario>": "train"}, "origin": {"scenario", "artifact_id"}}``. ``artifact_id`` names the artifact that answered the call, or is null when the record names none.

An item can arrive a short time after its record. When a scenario with older records first loads after an upgrade, the data service makes their items in pages while the scenario serves.

The scenario that made a record trains on its own record store, and always uses its item: a patch that removes it returns 400. To train another scenario of the same owner on a record, add that scenario to the item's ``scenarios``:

- Reef reads the record from the data store for the other scenario, with the item ID as its record ID, and stores no copy.
- Each reference ``x`` of a report reads as ``<origin scenario>/x``, so add a report together with the inferences that it references.
- The other scenario's recipe decides whether it trains on the record. For example, ``sft`` trains on it, and an on-policy recipe skips an inference that another runtime load served.

When an item leaves a scenario, new reads of that scenario stop giving it. A training set that a process already holds in memory can still train until the process restarts. A record store that an older Reef wrote can hold a copy of an item, and that copy stays.

``POST /reef/namespaces/{owner}/data`` also creates record items. The content must have the keys above, ``request_type`` must be ``inference`` or ``report``, and the ID must be ``<origin scenario>/<agent_record_id>``. Otherwise Reef returns 400.

Evaluation calls
----------------

Reef also keeps each model call of an evaluation rollout as a record item of the scenario's owner (`Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__). Such an item differs from a scenario's own record item:

- Its ID is ``<scenario>/<agent_record_id>``. The record ID is ``<evaluation_id>:<task_index>:<repeat>:<attempt>:<index>`` for a call of the agent, and ``<evaluation_id>:<task_index>:<repeat>:<attempt>:verifier:<index>`` for a call that the verifier wrote (`Calls that a verifier writes <http-api-evaluation-calls.rst#calls-that-a-verifier-writes>`__).
- Its content is an inference record. The payload is the request body plus ``response``, the response body or null. ``created_at`` is the start of the call.
- Its metadata is ``{"kind": "record", "scenarios": {}, "evaluation_id", "origin": {"scenario", "artifact_id", "source": "evaluation", "evaluation"}}``. ``artifact_id`` names the evaluated artifact. ``evaluation`` holds the ``evaluation_id`` and the call's summary: ``task_index``, ``repeat``, ``attempt``, ``index``, ``path``, ``status``, ``started_at``, ``duration_s``, ``prompt_tokens``, ``completion_tokens``, ``finish_reason``, ``tool_calls``, ``caller`` and ``label`` (`Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__). A call without ``caller`` is a call of the agent, and has no ``label``.

An evaluation call is the data of no scenario:

- A patch that adds a scenario to its ``scenarios`` returns 400.
- The scenario's records route does not list it. To read the calls of one evaluation, query the data with ``kind=record``, ``group_key=evaluation_id`` and the evaluation's ID as ``group_value``.
- An evaluation call links only to the earlier calls of the same attempt, so its trajectory node keeps only the messages after its parent's. A verifier call links only to the earlier verifier calls of the same attempt. A scenario's own call never links to an evaluation call.

A trajectory that a verifier wrote is a trajectory item of the scenario's owner. Its ID is ``<scenario>/<evaluation_id>:<task_index>:<repeat>:<attempt>:verifier:trajectory:<index>``, and its content is the ATIF document. Its metadata is ``{"kind": "trajectory", "scenarios": {}, "evaluation_id", "evaluation": {"evaluation_id", "task_index", "repeat", "attempt", "index", "name"}}``, where ``index`` counts the trajectory files of the attempt in name order and ``name`` is the file's name. It has no ``origin``. A patch that adds a scenario to it returns 400, so no scenario trains on it.

Artifact items
--------------

An owner's artifact items are its artifact library: the artifacts of all of its scenarios, including scenarios in other deployments that share the data store, and artifacts whose files you stored yourself. An artifact item is a reference: the files stay where they are. Its content is ``{source_uri, kind, component}``:

- ``source_uri``: the directory of the files, as an import reads it (`Import an artifact <http-api-artifacts.rst#import-an-artifact>`__), or null when there is none.
- ``kind``: the artifact's kind, such as ``model`` or ``model_adapter``, or null.
- ``component``: the component of the scenario that made the artifact, or null for your own item.

The item's ``origin.artifact_id`` identifies a Reef artifact. Older items may include ``content_id``: readers accept it but do not use it as an artifact identity, and new writes discard it. A retry of an older item preserves its original content and digest when every other content field matches.

The data service makes an item for each artifact that a scenario lists (``GET /reef/scenarios/{scenario}/artifacts``), after each commit. Its ID is ``<scenario>/<artifact_id>``, and its ``origin`` is ``{scenario, artifact_id}``. Its ``source_uri`` comes from the artifact backend. A backend that keeps snapshots where other deployments can read them, such as an object store, gives the URI of the snapshot's directory. The built-in ``git_lfs`` backend gives none, so its items have a null ``source_uri``.

To add files that you stored, create an item with ``POST /reef/namespaces/{owner}/data``, ``metadata`` ``{"kind": "artifact"}`` and no ``origin``. Its content needs a ``source_uri`` and a null ``component``.

To use an artifact item in a scenario, import it: ``POST /reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}``. The scenario must belong to the item's owner.

- Reef reads the item's ``source_uri`` and imports the files into the scenario. The import's ``source`` names the item. An item without a ``source_uri`` returns 400.
- The artifact keeps its ID, so the ID names one artifact in each scenario that uses it, and the library keeps one item of it. A second import of the item returns that artifact and reads nothing.
- An upload gets its ID at its first use, in ``metadata.artifact_id``, which never changes after that.

A new scenario can also start from an artifact item (`Create a scenario <http-api-scenarios.rst#create-a-scenario>`__).

Read, query, update and delete items
------------------------------------

**List items.** ``GET /reef/namespaces/{owner}/data`` lists the owner's items in the order they arrived: ``{"owner", "items", "next"}``. The query parameters are:

- ``kind`` and ``scenario`` filter the items. ``benchmark`` keeps only the tasks of one benchmark, ``trajectory_set`` only the trajectories of one trajectory set, and ``id_prefix`` only the items whose ID starts with it.
- ``group_key`` keeps only the items of one group of the groups route: those with the string ``group_value`` under ``group_key`` in their metadata, or with no string there when ``group_value`` is not given. ``group_value`` without ``group_key`` returns 400.
- ``after`` is the ``sequence`` of the last item of the previous page, 0 by default. ``limit`` is 1 to 1000, 1000 by default. ``next`` is the ``after`` of the next page, or null after the last item.
- ``before`` pages backward: the items come newest first, from the one just before the sequence ``before``, and ``next`` is the ``before`` of the next page. For example, ``trajectory_set=week-12``, ``before`` set to a trajectory's sequence and ``limit=1`` give the trajectory before it in the set.
- ``content=1`` adds each item's ``content``, as a read of the item gives it; the default ``content=0`` leaves it out. Use it with ``kind=artifact`` to read the artifact library's kinds and sources in one page. A page of trajectories with content can be large.

**Group items.** ``GET /reef/namespaces/{owner}/data/groups/{key}?kind=&scenario=&before=&limit=`` groups the owner's items of ``kind`` by the string under ``key`` in their metadata, for example the evaluation calls by ``evaluation_id``: ``{"owner", "groups", "next"}``.

- Each group is ``{"value", "item_count", "first_sequence", "last_sequence", "metadata"}``. ``value`` is null for the items without a string under ``key``. ``metadata`` is that of the group's first item.
- The groups come newest first, by ``first_sequence``.
- ``kind`` is required. ``key`` is 1 to 64 letters, digits and underscores.
- ``scenario`` groups only the items that the scenario uses, and the counts and sequences cover only those items.
- ``before`` returns only the groups whose ``first_sequence`` is lower. ``limit`` is 1 to 1000, 1000 by default. ``next`` is the ``before`` of the next page, or null after the last group.
- To read the items of one group, list the items with ``group_key`` and its ``value``, and page with ``after`` from its ``first_sequence``. Give the same ``scenario`` to that query.

**List sets.** ``GET /reef/namespaces/{owner}/data/sets/{kind}?scenario=&before=&limit=`` lists the owner's sets of ``kind``, ``benchmark`` or ``trajectory_set``, that have at least one item: ``{"owner", "sets", "next"}``.

- Each set is ``{"name", "item_count", "first_sequence", "last_sequence", "metadata"}``. ``metadata`` is that of the set's first item.
- The sets come newest first, by ``first_sequence``. ``before`` and ``limit`` page them as in the groups route.
- ``scenario`` lists only the sets with an item that the scenario uses, and counts only those items.
- The listing takes the same time for sets of ten items and of ten thousand.
- To read the items of one set, list the items with ``benchmark`` or ``trajectory_set``.

**Read one item.** ``GET /reef/namespaces/{owner}/data/{id}`` returns one item with its ``content``. A task's ``content`` is null.

**Patch metadata.** ``POST /reef/namespaces/{owner}/data/metadata`` takes ``{"ids": [...], "patch": {...}}``. Reef applies the JSON merge patch (RFC 7386) to the metadata of each item, all or none. A null value removes a key. For example, ``{"scenarios": {"my-agent": null}}`` removes the items from ``my-agent``, and ``{"scenarios": {"my-agent": "eval"}}`` moves tasks to the ``eval`` split. Reef checks the new uses as for a create call. A patch that changes ``kind``, ``owner`` or ``origin`` returns 400. The response is ``{"owner", "items", "synced"}``, without content.

**Delete one item.** ``DELETE /reef/namespaces/{owner}/data/{id}`` deletes one item and returns ``{"owner", "id", "deleted": true, "synced"}``.

- The scenarios that used a trajectory or a task lose its copy. A record stays in the record store of the scenario that made it.
- The item of an artifact that a scenario made returns 400 while the scenario exists.
- You can restore the item for 30 days (`Restore deleted items`_).

**List a scenario's tasks.** ``GET /reef/scenarios/{scenario}/tasks`` returns ``{"scenario", "tasks": [{"name", "split", "benchmark", "files_digest", "trained"}]}``: the uses of tasks by the scenario, in the order that they began. ``benchmark`` is the held benchmark that gives the use (`The sets that a scenario holds <http-api-data-sets.rst#the-sets-that-a-scenario-holds>`__), or ``null`` for the task's own mark. A task that the scenario uses more than once is listed once for each use. ``trained`` is true once a committed task job of any trainer of the scenario trained the task.

**Reserved IDs.** An item ID can hold ``/``, and two GET routes come before the item routes: ``.../data/<id>/archive`` downloads the item ``<id>``, and ``.../data/imports/<x>`` and ``.../data/exports/<x>`` read a transfer. So GET cannot read an item whose ID ends in ``/archive``, or whose ID is ``imports/<x>`` or ``exports/<x>`` with ``<x>`` one path segment.

Restore deleted items
---------------------

Reef keeps a deleted item of its owner for 30 days, and then purges it. A deleted item is in no listing, set, scenario or read of the routes above. A new item can take its ID.

- ``GET /reef/namespaces/{owner}/deleted-items?kind=&before=&limit=`` lists the owner's deleted items, newest item first by ``sequence``: ``{"owner", "items", "next"}``. Each item is ``{"id", "sequence", "digest", "metadata", "deleted_at", "purge_at"}``, without content. ``deleted_at`` and ``purge_at`` are seconds since the epoch. ``kind`` keeps one kind of item. ``before`` and ``limit`` page the items as in the item list; ``next`` is the ``before`` of the next page, or null after the last item.
- ``POST /reef/namespaces/{owner}/deleted-items/restore`` takes ``{"ids": [...]}``, 1 to 1000 IDs, and restores the items, all or none. An item comes back with its ``sequence``, metadata and content, and in its sets. Reef checks its ``scenarios`` as for a create call. Each scenario that uses it again reads it again, also through a held set. The response is ``{"owner", "items", "synced"}``, without content. An ID without a deleted item returns 404. An ID that the owner uses for an item again returns 409.
- ``DELETE /reef/namespaces/{owner}/deleted-items/{id}`` purges one deleted item for good and returns ``{"owner", "id", "purged": true}``. An ID without a deleted item returns 404.

The archive of a scenario removes it from the ``scenarios`` of the deleted items too. A reader that pages items with ``after`` and is past the ``sequence`` of a restored item does not see it again; a scenario reads it again through its uses.

Download an item
----------------

``GET /reef/namespaces/{owner}/data/{id}/archive`` downloads one item as an attachment, named after the last segment of its ID:

- A task downloads as ``<name>.tar.gz`` (``application/gzip``), with its files in the directory ``<name>``.
- Another item downloads as ``<name>.json`` (``application/json``): its content.

A browser or another service can download the item without the token through a `signed download URL <#signed-download-urls>`__.

Signed download URLs
--------------------

Add ``?signed=1`` to ``GET /reef/namespaces/{owner}/data/{id}/archive`` or to ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` (`Artifacts <http-api-artifacts.rst>`__) to get a URL that opens the download without the token. The response is ``{"url", "expires_at"}``:

- ``url`` is the path and query of the download on the same service. It carries ``expires`` and a ``signature``, which Reef computes from the token that the request presented.
- ``expires_at`` is the time, in Unix seconds, after which the URL returns 401.

``expires_in`` sets the life of the URL in seconds: 1 to 86400, 3600 by default. Another value returns 400. Send the URL with GET and without an Authorization header.

- The URL keeps the request's ``path`` and ``component`` queries, and it opens only that download: another path, or another ``path`` or ``component`` query, returns 401.
- When the service no longer accepts the token that signed it, the URL returns 401.
- A service without tokens returns the URL without a signature.
