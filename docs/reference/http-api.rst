HTTP API: inference, feedback, and releases
===========================================

Reef serves the provider's own inference routes: OpenAI at
``/v1/chat/completions`` and Anthropic at ``/v1/messages``. It forwards each
request to the runtime unchanged. It adds a small set of ``/reef/*`` routes for
feedback, scenarios, artifacts, and status.

For a complete request, receipt, and feedback example, start with the
`inference and feedback quickstart <../getting-started/quickstart.rst>`__.
To put the release routes into practice, follow the `agent harness tutorial
<../user-guide/evolve-your-harness.rst>`__ or the `model weight training guide
<../user-guide/evolve-your-model.rst>`__.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Routes
------

+-------------------------------------------------+---------------------------------------------------+
| Route                                           | Response                                          |
+=================================================+===================================================+
| ``GET /healthz``                                | readiness; the only unauthenticated route         |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/chat/completions``                   | OpenAI-format inference                           |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages``                           | Anthropic-format inference                        |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages/count_tokens``              | count request tokens; recorded like any inference |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/report``                           | submit feedback about one or more receipts        |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/train``                            | enqueue one training instruction                  |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios``                         | every known scenario and current release          |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                        | create a scenario explicitly                      |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/update``      | update the scenario training mode                 |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract``     | what this scenario accepts                        |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/releases``     | ``{scenario, releases}``, newest first            |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/rollback``    | republish an earlier release as the head          |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/promote``     | serve a release held for review                   |
+-------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``           | remove a scenario; its state is archived          |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                           | the served harness tree                           |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases``                  | the harness release catalog, oldest first         |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases/{step}/page``      | one HTML page per catalog step: why, what         |
|                                                 | changed, verdict, setup, chain                    |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/harness/proposals``                | an agent's proposed tree change, admitted or not  |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/install``                   | a shell script that installs the tree             |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/adapters``                  | every harness adapter this process resolves       |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/status``                            | training, serving, and storage state              |
+-------------------------------------------------+---------------------------------------------------+

Headers
-------

+-----------------------------------+---------------------------------------------------------+
| Header                            | Required for                                            |
+===================================+=========================================================+
| ``x-reef-scenario``               | inference, report, harness manifest, releases and       |
|                                   | proposals; optional on harness install. Names the       |
|                                   | workload a record belongs to.                           |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz``, when auth is       |
|                                   | configured.                                             |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-release-id``             | optional: bind a new scenario to this starting release; |
|                                   | on an existing scenario it must name the bound starting |
|                                   | release, or the request is HTTP 409. Inference always   |
|                                   | answers from the scenario's current release; to pull a  |
|                                   | specific release, use ``?release_id=`` on the harness   |
|                                   | manifest or install route.                              |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: opaque key/value context stored  |
|                                   | on the record under ``metadata.tags``, for a processor  |
|                                   | to correlate on. Reef never reads a value.              |
+-----------------------------------+---------------------------------------------------------+

Scenario model settings
-----------------------

``POST /reef/scenarios`` accepts an optional ``model`` object with ``url``,
``model``, ``api`` and ``api_key``. It applies only when creating a new
scenario. Existing scenarios retain their settings on repeated creation.
``POST /reef/scenarios/{scenario}/update`` accepts ``model`` alongside the
existing ``training_mode`` field. Send ``model: null`` to restore deployment
defaults. Responses redact the API key and report ``has_api_key`` instead.

See `Scenario model configuration <../user-guide/scenario-models.rst>`__ for
protocols, persistence and model bindings used throughout evolution.

Manual training
---------------

``POST /reef/train`` queues one training instruction for a scenario in
``data.training_mode: manual`` or ``hybrid`` (harness evolution with a
proposer that accepts ``requests``). It takes the user's ``text``,
originating ``session`` and ``release_id``. These fields identify the session
and release the request came from. The backend operates on the current
committed state; it does not restore the originating release. The API
requires no inference receipts or score.

A request may also carry ``requires``: what the change needs from the
person's machine, at most 8 ``{name, kind, check}`` items, default none.
``kind`` is ``permission`` (an OS permission the person grants), ``env``
(a variable the person sets; the extension reads it from the environment,
and its value never enters a request or the tree) or ``service`` (an
account or endpoint the person connects). ``name`` matches the entry name
pattern and is what a check off is recorded under. ``check`` is optional:
the variable name for ``env``; for ``permission`` and ``service`` a shell
command whose exit status zero means satisfied. For ``env`` the variable
named (the check, else the name) is a shell identifier,
``^[A-Za-z_][A-Za-z0-9_]*$``. The credential and directive screens run
over every ``name`` and ``check`` as they run over ``text``, with the same
HTTP 400 and a reason that names the rule; a malformed list is HTTP 400
naming the first bad item. The method's ``propose`` may add items of its
own to the mapping it received (an extension that reads a variable, say);
the backend merges them by name into the commit's
``training_request.requires`` after the same screens (a bad item of the
method's is dropped alone, named in the service log), the person's items
first and the list capped at 8 with the dropped items named in the service
log, so the releases row and the manifest carry what the person named and
what the change added. Nothing on the service runs a check:
``reef-<adapter> setup`` runs one after the person read it and confirmed,
the install script only reads the check offs, and the update notice prints
the setup list instead of the update while an item is unmet.

The three modes differ in what starts a step. ``auto``, the default,
batches on traffic and refuses an instruction with HTTP 400. ``manual``
runs instructions only and never batches on traffic; harness evolution
runs an instruction alone, without samples. ``hybrid`` batches on traffic and
runs instructions: a queued instruction goes first, oldest first, one per
step, and the units an automatic batch would take next, up to
``batch_size`` and possibly none, ride beside it as the batch's samples
(failing traces in the score window, or records under
``data.batch_policy: records``), so the proposer reads the request next
to them; with none queued the step batches as ``auto`` does.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "release_id":"release-1"}'

The response is ``{agent_record_id, scenario, request_type: "train"}``.
HTTP 200 acknowledges durable acceptance, not successful training. Requests
are executed one at a time by the normal training worker; later requests
do not change a step already in flight. A step that fails with an
instruction (a proposer error, for one) is not retried: the next step
consumes the instruction with a committed row whose ``skipped`` reads
``instruction failed`` and whose ``error`` carries the failure, and the
queue moves on. Send the instruction again to run it again. That row
consumes the instruction alone: in ``hybrid`` the units that rode beside it
stay held for the next batch, so no failing trace is consumed unread. The
``evolution.max_steps`` and ``evolution.max_failure_streak`` budgets count
every step, instruction steps included, and stop automatic steps only; an
instruction still runs past them. The existing evaluation and publication
rules still determine whether the result becomes served.
``GET /reef/status`` reports ``training_mode``. Processors using the shared
instruction queue also report ``buffered_requests`` (requests already
read into the processor; later records may still wait in storage) and
``pending_instructions`` (accepted instructions not yet consumed: the
buffered ones plus those still unread in storage).
Committed step metrics include ``training_request`` with the instruction
id, text, session, release id and ``requires``; the releases row of a step
that answered a request carries, for example:

.. code:: json

   {"training_request": {"id": "change-001", "text": "Text me when the run is blocked",
                         "session": "session-1", "release_id": "release-1",
                         "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}]}}

Supply ``agent_record_id`` to retry safely: an identical request is accepted
without another step, including after record compaction; reusing the id with
different content returns HTTP 409. Without it, each submission gets a fresh
id. Empty text, text longer than 4000 characters, missing or non-string
session/release fields, or a request to an ``auto`` scenario returns HTTP 400.
Text that carries a credential shaped literal or an instruction override
phrasing is refused with HTTP 400 and a reason that names the rule, never
the text; nothing is stored. The scenario must exist: an unknown scenario
answers HTTP 404 and creates nothing, whatever the implicit-scenario-creation
setting says. The normal bearer authentication applies.

Scenarios
---------

A scenario isolates the records, trainer, and release chain for one workload.
The first inference request or report carrying a new ``x-reef-scenario`` creates
the scenario using the deployment's configured recipe. Requests never select a
recipe.

.. code:: bash

   curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: hello-reef" \
     -H "Content-Type: application/json" \
     -d '{"model": "m", "messages": [{"role": "user", "content": "fix it"}]}'

The bindings never change; a request naming a different recipe with the same
scenario returns HTTP 409. This means the surface, runtime, inference backend,
and optional report schema chosen when the recipe is constructed are fixed with
it.

If the deployment sets ``reef.allow_implicit_scenario_creation: false``, an
unknown scenario returns HTTP 404 and you create it first:

+---------------------------------------------+---------------------------------------------+
| Route                                       | Body and response                           |
+=============================================+=============================================+
| ``POST /reef/scenarios``                    | ``{"name", "release_id"?}``                 |
|                                             | → ``{scenario, release_id,                  |
|                                             | content_id}``; 201 created, 200 already     |
|                                             | existed                                     |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios``                     | every known scenario and its current        |
|                                             | release once loaded                         |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | ``{scenario, processor,                     |
|                                             | required_request_types}``                   |
+---------------------------------------------+---------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``       | → ``{scenario, archived}``; 404 unknown     |
+---------------------------------------------+---------------------------------------------+

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` removes the scenario from the running
service and frees its name. Nothing is destroyed: the record store and commit
log move under ``<agent_record_dir>/archived/``, the recipe's own directories
(a proposal inbox, step records) move under an ``archived/`` sibling, and the
scenario's ref in the artifact repository is renamed into
``refs/reef/archived/``, so every release it published stays reachable. The
base artifact, the shared head and releases other scenarios may fork from
stay where they are. A step in flight for the scenario ends without a commit.
Only a local artifact repository can be archived; a remote one answers 501.

For a scenario that trains weights the deletion is Reef-side: the training
backend is told to retire the scenario, and the Slime backend does not yet
act on it, so the scenario's adapter stays resident in the serving engine
until it is evicted or the training group restarts, and the training job's
per-scenario history keeps its entry until then.

Scenario updates
~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/update`` updates an existing scenario.
Currently, only ``training_mode`` is supported; unknown fields are rejected.
To change the data processor's mode:

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.
The values are ``auto``, ``manual`` and ``hybrid``: ``"auto"`` resumes recipe
batching alone, ``"hybrid"`` keeps it and takes instructions too. The change
selects subsequent batches; a batch already reserved or running completes
in its original mode. The modes share buffered data, so auto and hybrid can
batch traffic collected while manual was selected. Accepted instructions
wait for a mode that takes them, including instructions not yet read when
the selector changes to auto.

A Reef process runs at most one scenario that trains full weights, on a single
thread, so preparation, remote execution, and commit never interleave. It may
run any number of scenarios that produce no updates or that update text
artifacts in process. Each one grows and commits on its own background thread,
so record acceptance never waits for artifact evolution.

Unknown scenarios return ``404`` without implicit creation; invalid payloads
return ``400``. A processor that does not support the requested mode returns
``501`` without changing its state. Harness ``manual`` and ``hybrid`` also
require a proposer that explicitly accepts ``requests``.

``GET /reef/status`` reports the selected ``training_mode``. This selector
is runtime state, not persisted scenario configuration: a service restart
uses the recipe's configured mode again, while a scenario reload after a
failed step keeps the selected mode.

Inference
---------

Request
~~~~~~~

Send the same body you would send to the provider. Reef never touches your
sampling parameters. 

Before calling the model, Reef reads the scenario's current artifact ref and
builds the request against that release. The stored exchange uses the same ref,
so an update completing mid-request does not change what the receipt records.

On a weight-serving deployment it adds engine
bookkeeping keys: ``lora_path`` to address the served adapter and
``return_meta_info`` so the record proves which weights answered; a body
naming a different ``lora_path`` is refused. Set ``"stream": true`` and read
the SSE response for streaming.

The receipt identifies the stored record:

+---------------+-----------------------------------------------------------+
| Response kind | Where the receipt is                                      |
+===============+===========================================================+
| non-streaming | the ``x-reef-agent-record-id`` response header            |
+---------------+-----------------------------------------------------------+
| OpenAI SSE    | ``reef.agent_record_id`` in a final empty-``choices``     |
|               | chunk, immediately before ``data: [DONE]``                |
+---------------+-----------------------------------------------------------+
| Anthropic SSE | the same field on ``message_stop``                        |
+---------------+-----------------------------------------------------------+

Streams carry it only after the record is stored.

On a scenario that serves harness files, every inference response also
carries ``x-reef-release-id``: the release ``GET /reef/harness`` serves when
the response is written, so a head that moves during the call shows on the
next one. A resident ``reef-native serve`` process compares it with the
release it mounted and learns of a new head on its next model call, with no
extra request. A weight serving scenario sends no such header.

Response
~~~~~~~~

Reef validates a response before recording it. ``prepare_request`` transforms
the outgoing payload, and Reef forwards *and records* the transformed payload.
``verify_response`` checks the provider's answer against the frozen release. On
failure Reef records nothing and returns the error.

Rejection if out of release window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For live weights, Reef asks the engine to report the ``runtime_load_id`` for
each generated token span. A response may cover several runtime loads if an
update lands mid-generation; Reef accepts it only when the span information
accounts for
every generated token and is consistent with the frozen release. Missing or
inconsistent spans are a backend contract error and return HTTP 409.

Pass-through streaming cannot do that check. Reef leaves ``return_meta_info``
disabled when ``stream`` is true and records a plain SSE exchange. The training
backend buffers its stream instead and validates the complete response against
the frozen release before recording.

Report
------

Reef stores four fields and drops other top-level keys. Put harness-specific
data inside ``metadata`` or ``feedback``.

+----------------+------------------+----------+-------------------------------------------+
| Field          | Type             | Required | Notes                                     |
+================+==================+==========+===========================================+
| ``score``      | number           | no       | a bool is not a number and is rejected    |
+----------------+------------------+----------+-------------------------------------------+
| ``feedback``   | string or object | no       | opaque to Reef's core: a rubric, judge    |
|                |                  |          | output, plain text                        |
+----------------+------------------+----------+-------------------------------------------+
| ``references`` | list of strings  | no       | the receipts this report grades; several  |
|                |                  |          | batch as one trajectory sample, none is   |
|                |                  |          | accepted but never trains                 |
+----------------+------------------+----------+-------------------------------------------+
| ``metadata``   | object           | no       | opaque, except                            |
|                |                  |          | ``training.eligible`` (default ``true``)  |
+----------------+------------------+----------+-------------------------------------------+

It answers ``{agent_record_id, scenario, request_type}``.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

The optional top-level ``agent_record_id`` makes posting retry-safe: Reef uses
it as the report's own record id, so an identical resend returns the stored
record instead of reprocessing it, while the same id with different content is
HTTP 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema. `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ declares one. In that case, Reef
validates the declared ``score`` and ``metadata`` fields at ingress and answers
HTTP 400 on a violation; ``feedback`` and undeclared ``metadata`` keys pass
through unvalidated. To record a report but keep it out of training, send
``"metadata": {"training": {"eligible": false}}``.

Record
------

Reef stores every exchange of inference and every report as ``AgentRecord``.
These records are referenced via a record id. In ``/reef/train`` and
``/reef/report`` API calls, users can provide an optional ``agent_record_id``
field for retry-safety. Appending the same id with different content returns a
conflict rather than overwriting. Reef keeps track of consumed records so
retried reports and late reports whose references already trained are not
counted twice.

Training compaction retires records without deleting their original payloads.
Reef's explicit Python audit reads can inspect retained requests, responses,
references, and compaction timestamps; ordinary training reads exclude retired
records. There is no new HTTP record-query endpoint. Separate background
retention limits compacted bodies to 7 days and a shared 20 GiB by default;
see `Configuration <configuration.rst>`__ for scope and `Python API <python-api.rst>`__
for audit and purge methods. A compaction timestamp alone does not prove that a record was
used for learning: per-step ``consumed_ids`` in the commit log identifies that
relationship.

Receiving an update
-------------------

For weight-training scenarios there is nothing to do: keep calling the same
inference endpoint and it serves the latest published weights. The artifact
lives in the inference runtime, so requests reach the new release directly.

A scenario whose artifact is the harness tree works the other way: the client
pulls. The harness should fetch the currently served tree with
``GET /reef/harness``, or the install script built on it, and run the agent on
that release. So the harness it uses is the one whose receipts it will later
report against.

Harness artifacts
~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------------+
| Route                          | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | ``{release_id, content_id, parent_release_id, files, gate,    |
|                                | requires}``, plus an ``x-reef-release-id`` response header    |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/releases`` | ``{scenario, releases}``, oldest first, each training row     |
|                                | carrying the gate metrics of the step that published it       |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/install``  | a self-contained POSIX shell script that installs the vendor  |
|                                | binary, writes the tree, and writes the adapter's model       |
|                                | binding at the address the request reached (a gateway in      |
|                                | front names it in ``x-forwarded-host`` and                    |
|                                | ``x-forwarded-proto``), the token filled from ``REEF_TOKEN``  |
|                                | when the script runs                                          |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/adapters`` | ``{adapters}`` — every harness adapter this process resolves, |
|                                | each with ``name``, ``binary``, ``trajectory_format``,        |
|                                | ``model_bindings``, and the pinned ``install`` spec           |
+--------------------------------+---------------------------------------------------------------+

The first three are read-only and take ``x-reef-scenario``. Install also requires
``?adapter=``, whose value may be ``pi``, ``opencode``, ``claude``, ``codex``,
``dsh``, ``hermes``, or an external descriptor. Only an adapter whose descriptor
declares an install section can be named here: ``native`` and ``terminus`` ship
with reef and pin no vendor binary, so they answer HTTP 400 rather than a
script. If install omits ``x-reef-scenario``, Reef creates a scenario with a
generated ``harness-`` name and embeds that assignment in the wrapper script;
when exactly one configured recipe serves harness files, it selects that recipe
automatically.

``files`` is the rendered tree, path to text. An adapter whose descriptor
declares ``files.tree`` (``native`` does: ``native/tree.json``) adds one more
file: the release's entries list, the same ``{id, name, config}`` objects the
commit log persists, as one JSON array. A resident ``reef-native serve``
process mounts that list entry by entry; an older ``reef-native`` ignores the
file and reads the rendered files as before. ``pi`` declares none: a pi
release is its rendered files, and the entries stay in the commit log, where
the proposals route and the evolve step read them.

``requires`` is what the release needs from the person who installs it:
every ``training_request.requires`` item (`Manual training
<#manual-training>`__ gives the shape) over the release's chain, merged by
name with the newest definition winning, since a request's items are per
release and a later release whose request named nothing still carries the
extension an earlier one added. The chain follows ``parent_release_id``
through the catalog, a promote or rollback row continuing at the release it
copied; a releases row carries its own step's list alone. The install
script embeds the list (the union of a chain is not bounded by one
request's cap of 8) and refuses, before it installs the binary or makes a
directory, while an item is not checked off in the ``.reef-harness-release``
release metadata file on disk: it prints the setup list and the newest release in the
chain that requires nothing, the one that installs on a machine with
nothing set up (``?release_id=<id>``), and exits 1. ``reef-<adapter> setup``
records the check offs; ``--release <id>`` names a pending release so its
items are checked off before its promote. The release metadata file the script writes
carries ``requires`` (the list) and ``setup`` (the check offs, ``{name,
checked_at, check}``, carried over from the previous release metadata file by name; an
item whose check is not the recorded one counts as unmet); a release metadata file the
stdlib client pull wrote carries neither, which reads as nothing required.

Use ``?release_id=`` on the manifest or install route to request a specific
catalog release. An unknown or unrestorable release returns HTTP 404.

Catalog and manifest reads do not wait for an evolve step's proposer or
evaluation episodes: they continue serving the existing releases while a
candidate is being prepared. Reads still serialize with publication and
rollback so a manifest's artifact and gate metrics come from the same
release: reads do not interleave with head movement and its commit-log
update.

Proposals
~~~~~~~~~

An agent running on the served tree can propose a change to it. The
proposal enters the same gate as the method's own: nothing it says is served
until paired episodes settle it.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"mutations": [{"op": "create", "id": "check", "options": {"name": "rules", "config": {"text": "Run the tests before you answer."}}}],
          "reason": "three of five sessions answered before the tests ran",
          "session": "3f1c2a9d0b7e", "release_id": "rel-12"}' \
     "$REEF_URL/reef/harness/proposals"

+----------------+----------------------------------------------------------------------+
| Field          | Meaning                                                              |
+================+======================================================================+
| ``mutations``  | a non-empty list of ``{op, id, options}``: ``create`` and ``update`` |
|                | carry ``options`` (``{name, config, disabled?}``, the entry without  |
|                | its id), ``remove`` carries none                                     |
+----------------+----------------------------------------------------------------------+
| ``reason``     | the proposer's own account, stored with the proposal                 |
+----------------+----------------------------------------------------------------------+
| ``session``    | the session that proposed, named in the commit that settles it       |
+----------------+----------------------------------------------------------------------+
| ``release_id`` | the release the proposer was running                                 |
+----------------+----------------------------------------------------------------------+

The service admits the mutations against the head release's entries with the
rules every mutation meets (a create on an existing id, an update on a missing
id or one that changes the entry's kind, a remove on a missing id, a config the
kind's admission refuses, a kind the adapter does not render, a tree that does
not render, any op on one of reef's own entries: ``reef-version-check``,
``reef-requests`` and ``reef-pi-extension-api`` are reserved ids) and answers
``{proposal_id, admitted, reason, release_id}``:
``reason`` is the rule that refused, else ``null``; ``release_id`` is the head
the proposal was admitted against. An admitted proposal waits in the
scenario's inbox (``evolution.proposals_dir``) until the next evolve step takes
it, oldest first, before the method's own ``propose`` is asked; the step admits
it again against its own entries, since the head may have moved, and the gate
settles it like any mutation. The commit that settles it carries ``proposal:
{id, session, release_id, reason}`` in its metrics, and the releases row
carries that commit. When ``evolution.max_pending_proposals`` already
wait, the answer is ``admitted: false`` with reason ``inbox full``; on a
scenario in ``data.training_mode: manual`` it is ``admitted: false`` with
reason ``manual mode takes instructions only``, since no automatic step runs
there to take the inbox. A malformed
body is HTTP 400; a scenario whose recipe is not a harness evolution recipe is
HTTP 404 naming that. `Operate a deployment
<../user-guide/operate.rst#read-the-proposal-inbox>`__ describes the inbox
directories.

Harness requests
~~~~~~~~~~~~~~~~

``reef-<adapter> harness "<request>"`` and pi's ``/reef-harness <request>``
submit the user's instruction through ``POST /reef/train``, described under
`Manual training <#manual-training>`__. Set ``data.training_mode: hybrid``
(the deployment keeps learning from failures) or ``manual``, or switch an
existing scenario with ``POST /reef/scenarios/{scenario}/update``.
The commands send ``text``, ``session`` and the installed ``release_id``;
HTTP 200 acknowledges durable acceptance and returns ``agent_record_id``.
They require no inference receipts and leave captured receipts available for
feedback. A scenario in ``auto`` refuses the request; the commands surface
that error and leave mode switching to the caller.

The existing trainer delivers the request to a proposer that explicitly
accepts ``requests``, then evaluates and publishes under the same policy as
automatic evolution. Its commit metrics carry ``training_request:
{id, text, session, release_id, requires}``, visible through the release
catalog.

Rollback
~~~~~~~~

Pulling an older release changes only your local copy. To move the release Reef
*serves*, send ``POST /reef/scenarios/{scenario}/rollback`` with
``{"release_id": "…"}``; it answers the new head. Reef republishes that
checkpoint as a new commit rather than rewinding history, so step numbers stay
monotonic.

Choose a target from ``GET /reef/scenarios/{scenario}/releases``, which lists
**newest first**; ``GET /reef/harness/releases`` lists oldest first. Only
releases marked ``restorable`` can be rolled back.

Review before serving
~~~~~~~~~~~~~~~~~~~~~

With ``evolution.publish: review``, or when a gate win touches a node kind
listed in ``evolution.review_kinds``, the winning tree is committed to the
catalog but not served: its row carries ``pending: true``, the manifest and
install routes keep serving the previous head, and ``?release_id=`` can pull
the pending tree for a trial install. ``POST /reef/scenarios/{scenario}/promote``
with ``{"release_id": "..."}`` serves it by the same republish path as
rollback, so the promotion is itself a commit record with
``operation: promote`` and the promoted tree becomes a new release.

Version page
~~~~~~~~~~~~

``GET /reef/harness/releases/{step}/page`` answers one self contained HTML
page (``text/html``, no asset, its data inline) for one catalog row. ``step``
is the row's position in ``GET /reef/harness/releases`` oldest first, the
creation row being 0, which is the commit step: a rejected step publishes
nothing and its row carries the head's release id, so the step is what names
it. The page has five sections in this order: Why (the request the step read,
else the claimed proposal's reason, else a failure in the batch), What changed
(the step's mutations; an extension's file as text for a create, and for an
update a line diff against the release the candidate ran on when that release
is restorable, else the new text), Verdict (the verdict with ``selected``,
``wins``, ``losses``, ``ties``, ``current_score``, ``candidate_score``,
``episode_failures``, and the step record directory when
``evolution.step_record_dir`` is set), Setup (the request's ``requires`` with
name, kind and check, then the items the release carries from earlier steps
in its chain, the same union the install script and ``reef-<adapter> setup``
read; a rejected or skipped row lists only its own items, since its release
id is the head's; nothing when both are empty) and Chain (the parent
release, this release, and its children: the steps gated on it, won, lost or
pending, and a promote or rollback made on it; a rejected or skipped step
published nothing, so its Chain names the head it ran on and no children).
The line under the title marks the served head as ``current``: the newest
row that is neither pending nor a rejected or skipped step. A pending row
that a later ``promote`` row names in ``rollback_target_release_id`` reads
``promoted at step N``, where N is that later row's step. A step outside the
catalog is HTTP 404 naming the range; a step that is not a number, or longer
than nine digits, is HTTP 404 too. The row itself rides in a
``<script type="application/json">`` block at the end of the page, every
``<`` escaped.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/releases/3/page" > harness-step-3.html

On pi, ``/reef-versions`` in a ``reef-pi`` session lists the chain, and
``/reef-versions 3`` prints this URL with the promote command, the trial
install (which replaces the installed tree) and the head's reinstall beside
it when the release is pending; a promoted release gets none of them.

Status
------

Read ``GET /reef/status`` when inference is still serving an older release while
an update is being trained or published.

.. code:: json

   {
     "error": null,
     "last_drain_at": 1756400000.0,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "scenario_step": 3,
         "last_committed_step": {
           "step": 3,
           "recorded_at": 1756400000.0,
           "metrics": {"published": false, "selection": {"reason": "candidate lost"}}
         },
         "current_runtime_load_id": "7f2a:12",
         "checkpoint_storage": {"...": "..."},
         "batch_ready": false,
         "processor": {"...": "..."},
         "inference_admission": {"...": "..."}
       }
     },
     "serving": {"...": "..."}
   }

``error`` and ``preload_errors`` report asynchronous training and preload
failures. ``batch_ready`` says whether the processor has a batch waiting.
``last_committed_step`` reports the latest durable training step number,
commit time, and its recipe-owned metrics; it is ``null`` before the first
training commit. This distinguishes a step still in flight from a completed
step that skipped or rejected its candidate. A rollback advances
``scenario_step`` without replacing the latest training outcome. A deployment
without an agent-record directory has no historical commit log, so after a
restart from a rollback checkpoint this field is ``null`` until the next
training commit.
``serving`` is runtime-wide but recipe-shaped: a LoRA deployment reports the
engine's shared adapter residency there, keyed by recipe. Each scenario's
``adapter_runtime_load_id`` appears in its own ``scenarios`` block.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | malformed body, a missing or empty ``x-reef-scenario`` on a |
|        | scenario-scoped route, or a report violating the recipe's   |
|        | declared schema                                             |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token                               |
+--------+-------------------------------------------------------------+
| 403    | relayed from the upstream provider. Reef issues none of its |
|        | own: an unaccepted token is 401, and per-scenario           |
|        | authorization belongs to the gateway in front of Reef.      |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (with implicit creation off, or on         |
|        | ``POST /reef/train``), unknown release, unknown adapter, no |
|        | configured harness recipe, or a scenario that serves no     |
|        | files                                                       |
+--------+-------------------------------------------------------------+
| 409    | a base artifact conflicting with the scenario registration, |
|        | record id resent with different content, a rollback naming  |
|        | a release that is not restorable, or an engine that reports |
|        | no serving runtime load ID                                  |
+--------+-------------------------------------------------------------+
| 502    | the upstream provider failed on its own account             |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, or inference kept losing |
|        | the weight-update race until its deadline                   |
+--------+-------------------------------------------------------------+

Reef relays upstream 4xx failures with the provider's original message; the
common client statuses (400, 401, 403, 404, 408, 409, 422, 429) keep their
status code, and any other upstream 4xx comes back as 400.

Browser consoles
----------------

The service can opt in to direct browser access with ``reef.console_origins``
in the serve YAML. List each trusted console origin explicitly, with no path,
trailing slash, credentials, or wildcard:

.. code-block:: yaml

   reef:
     console_origins:
       - "https://api.reefinfra.ai"
       - "http://localhost:3000"

Restart the service after changing the configuration. Without this option, Reef
does not add CORS headers. With it, unlisted browser origins are rejected before
a route runs. CORS preflight requests from listed origins do not need a service
token; actual requests retain the configured Bearer authentication. Browser
requests may use GET, POST or DELETE with Authorization, Content-Type and
x-reef-scenario headers. Cookies are not enabled through CORS.

CORS headers also cover errors and streamed responses, exposing
``x-reef-agent-record-id``, ``x-reef-release-id`` and ``x-reef-artifact-version``
to the browser. Clients without an Origin header keep their existing behavior.
A browser may additionally require local network permission or HTTPS; allowing
an origin in Reef does not override browser policy.

A connected console acts with the service token's existing permissions. Its
requests operate directly on this runtime's scenarios, without creating a
cloud deployment or uploading history as part of the CORS connection.

Learning inspection
-------------------

``GET /reef/scenarios/{scenario}/learning`` reads retained record summaries,
including compacted records. ``after_sequence`` defaults to 0 and ``limit``
defaults to 50 (1–100). Records are oldest first; ``next_after_sequence`` is
null at the end. This is a page of retained bodies, not a complete historical
count. The policy object names the live processor, required request types,
training mode, and processor status. Historical eligibility decisions and
evaluation thresholds are not exposed.

Each summary includes ``agent_record_id``, ``sequence``, ``request_type``,
``created_at``, ``compacted_at``, ``references``, ``score``, ``served_by``,
``learning_state``, ``reason``, and ``learning_steps``. Only commit-log
``consumed_ids`` establish consumption. Active unconsumed records are
``awaiting``; compacted unconsumed records have an ``unknown`` decision.
Consumption does not mean an update was selected or promoted. Candidate
identifiers are returned only when recorded in selection metrics. A link's
``release_id`` is the committed artifact, which can be the unchanged current
release after rejection.

``GET /reef/scenarios/{scenario}/records/{record_id}`` returns the same summary
and its stored ``payload``. The payload can contain sensitive request and
response content. Both routes use service authentication and return
``Cache-Control: no-store``. The platform console must bind the scenario to
the authenticated workspace. Missing bodies return 404: a body may have
expired or never been retained. Reading never reactivates a record, changes
retention, or duplicates its payload into another store.
