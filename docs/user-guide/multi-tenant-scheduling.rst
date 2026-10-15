Share local harness services across tenants
===========================================

Use the tenant scheduler when several users share local harness evolution
services. Each tenant receives a fixed execution quota across all registered
services. Requests that temporarily exceed that tenant's quota wait in its
queue. Services without a ``scheduling`` configuration keep their existing
behavior.

Operators start the scheduler first, then start compatible harness services
that register with it. The `scheduling configuration
<../reference/configuration.rst#tenant-scheduling>`__ includes a first-start
YAML with no service entries and the connection settings for each service.
Once the scheduler verifies a registered service, tenants can request
scenarios using the workflow below. For the underlying
harness workflow and client prerequisites, read `Evolve your harness
<evolve-your-harness.rst>`__.

Choose a service and tenant
---------------------------

Obtain the scheduler URL, your configured tenant ID, and the target service's
``recipe`` and ``service_name`` from the operator. A recipe may have several
independently deployed services; choose the named service you want to use.
The scheduler does not deploy services or choose a different service for you.

The local tenant ID is a manually supplied positive integer for ownership and quota
accounting. It is not proof of identity. Service and scheduler Bearer tokens
remain separate authentication settings; this mode does not authenticate a
user-to-tenant mapping. Use it within a trusted local deployment.

The examples use a Cordis service named ``harness-dev``, a configured tenant
``11``, and a scenario named ``my-harness``. Use the exact recipe identifier
registered by your operator. The scheduler URL receives admission requests;
the returned service URL receives harness, inference, and feedback requests.

.. code:: bash

   export REEF_SCHEDULER_URL=http://127.0.0.1:8902
   export REEF_TENANT_ID=11
   export REEF_SCENARIO=my-harness

The commands below assume ``REEF_SCHEDULER_TOKEN`` and ``REEF_TOKEN`` contain
the respective scheduler and service tokens. Omit the corresponding
Authorization header if that endpoint has no token configured.

Request a scenario
------------------

Submit the service, recipe, and scenario name once:

.. code:: bash

   curl -sS -i "$REEF_SCHEDULER_URL/reef/scenarios" \
     -H "Authorization: Bearer $REEF_SCHEDULER_TOKEN" \
     -H "x-reef-tenant: $REEF_TENANT_ID" \
     -H 'Content-Type: application/json' \
     -d '{"tenant_id":11,"recipe":"reef.recipe.cordis:CordisRecipe","service_name":"harness-dev","scenario_name":"my-harness"}'

The response includes ``scenario_id``, ``state``, ``execution_units``, and
``service_url``. HTTP 201 with ``state: admitted`` means the scenario is ready.
HTTP 202 means admission is still pending. Save the numeric scenario ID and
poll it at the scheduler until it becomes ``admitted`` or reaches a terminal
state (``deleted`` or ``failed``):

.. code:: bash

   export REEF_SCENARIO_ID=1
   curl -fsS "$REEF_SCHEDULER_URL/reef/scenarios/$REEF_SCENARIO_ID" \
     -H "Authorization: Bearer $REEF_SCHEDULER_TOKEN" \
     -H "x-reef-tenant: $REEF_TENANT_ID"

Replace ``1`` with the returned ID. Polling reads admission state; it does not
keep an idle scenario alive. Repeating the same creation request returns the
existing lifecycle while it is active. Changing its creation settings produces
a conflict. Another tenant cannot reuse that active name on the same service;
the same scenario name can exist on a different service.

A queued scenario has no running scenario or evaluation workers. The scheduler
starts requests in each tenant's FIFO order when quota becomes available.
A temporarily unavailable target service can delay that tenant's queue.

Install and evolve
------------------

After admission, set ``REEF_URL`` to the response's ``service_url`` and install
the harness from that service:

.. code:: bash

   export REEF_URL=http://127.0.0.1:8900
   curl -fsS "$REEF_URL/reef/harness/install?adapter=pi" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-tenant: $REEF_TENANT_ID" \
     -H "x-reef-scenario: $REEF_SCENARIO" | bash

The install record saves the tenant ID alongside the existing service and
scenario binding. Subsequent wrapper commands, captured inference, feedback,
and the bundled Pi extensions send ``x-reef-tenant`` automatically. The
``REEF_TENANT_ID`` environment variable overrides the installed tenant ID;
normally leave it unset after installation or keep it equal to the installed
tenant. Changing the ID does not transfer scenario ownership.

For a service configured for manual or hybrid training, request a change with:

.. code:: bash

   reef-pi evolve "run the tests before you report a fix as done"

You do not need to repeat the recipe, service name, scenario name, or tenant
on each evolution request. These bindings stay outside provider request bodies
and the published harness tree.

Direct calls to a scheduling-enabled service must also include
``x-reef-tenant``. Its existing ``POST /reef/scenarios`` body remains
``{"name":"my-harness"}``; the service supplies its own recipe and service
identity to the scheduler. Implicit creation through install or eligible
inference/record routes follows the same admission rules. If such an admission
request is pending, it returns HTTP 409 with admission information. Calls to a
scenario already starting or awaiting cleanup also return HTTP 409; poll at
the scheduler before retrying. ``/reef/train`` requires an already admitted
scenario.

Inspect quota and release it
----------------------------

.. code:: bash

   curl -fsS "$REEF_SCHEDULER_URL/reef/scheduler/status" \
     -H "Authorization: Bearer $REEF_SCHEDULER_TOKEN" \
     -H "x-reef-tenant: $REEF_TENANT_ID"

``allocated`` is your tenant's fixed quota; ``used`` counts execution units
still allocated to its scenarios; ``available`` is their difference. Units
remain charged during startup and until resource release is confirmed.
``global_allocated`` counts the fixed quotas of admitted tenants. Merely
listing a tenant in the configuration does not allocate its quota: the first
valid scenario request does so if the global budget can accommodate it.

Cancel a queued request, or delete an admitted scenario and release its units:

.. code:: bash

   curl -sS -i "$REEF_SCHEDULER_URL/reef/scenarios/cancel" \
     -H "Authorization: Bearer $REEF_SCHEDULER_TOKEN" \
     -H "x-reef-tenant: $REEF_TENANT_ID" \
     -H 'Content-Type: application/json' \
     -d '{"service_name":"harness-dev","scenario_name":"my-harness"}'

Cancellation is immediate for a queued scenario that has never started.
A scenario that has run before may need service cleanup, even if its workers
have already been reclaimed. HTTP 202 with ``state: cleanup_pending`` means
that final deletion is still pending. Continue polling its scenario ID. Any
still-allocated units remain charged until cleanup is confirmed; a timeout
does not release them. Deleting an already reclaimed scenario does not charge
its units again. Once every lifecycle for a tenant is terminal, its fixed
quota is returned to the global budget.

Idle resource recycling
-----------------------

Idle recycling is disabled by default. With ``idle_ttl_seconds`` configured,
the service reports when an admitted scenario is waiting for input and has no
running or pending work. After that interval, the scheduler asks the service
to release its execution resources. The service checks again before stopping
workers; a new request or ongoing evaluation prevents recycling. Status reads
do not count as activity.

After release is confirmed, the scenario moves from ``reclaim_pending`` to
``sleep``. It retains its ID, name, records, training mode, model configuration
and published progress, but holds no execution units and stays outside the
waiting queue. It remains asleep until a new evolve instruction arrives,
even when execution quota is available. The tenant keeps its fixed global
allocation while that logical lifecycle exists.

Submit a new instruction to ``POST /reef/train`` to wake a sleeping scenario.
The service durably retains the instruction before requesting admission, then
appends the same scenario to its tenant's FIFO tail. HTTP 202 means the
instruction was accepted and is waiting for admission; use its returned
``agent_record_id`` and request page or ``evolve --wait`` to follow it. A
scheduler outage delays admission while the service keeps and retries the
accepted instruction. An admitted scenario accepts further instructions
without requeueing or allocating additional execution units.

Recycling requires durable scenario and admission storage in the service.
When idle TTL is enabled, services lacking this support remain unavailable
for new admission and report the configuration problem. Configure durable
storage before enabling the setting. New instructions during queued, starting
or reclaiming states receive HTTP 409; only a retry of the already accepted
instruction can return its original acceptance. Retry after admission.
Unknown scenarios receive HTTP 404. Queries and installs do not wake an
existing sleeping scenario.

Try a three-slot CPU experiment
------------------------------

Harness evolution can run on a CPU-only host with a remote model endpoint.
The local workers run harness episodes; running the model on that same host
is a separate resource choice. Keep a working harness deployment's recipe,
model binding, storage and scheduling connection. Set
``recipe.config.training-mode`` to ``manual`` and
``execution.evolution.workers`` to ``1``. With a CPU episode scorer, this makes
each Cordis scenario cost one execution unit. No scheduler code override is
needed to admit three scenarios.

For a fresh scheduler, use the following configuration. At the default
70 percent, the host needs at least five logical threads to accommodate the
three-unit tenant quota. The quota limits this tenant to three slots even
when the host's computed global budget is larger.

.. code:: yaml

   database: .reef/three-slot-scheduler.sqlite
   token_env: REEF_SCHEDULER_TOKEN
   max_logical_thread_percentage: 70
   tenant_quotas:
     11: 3
   idle_ttl_seconds: 10

Start one service, and request ``s1``, ``s2``, ``s3`` and ``s4`` for tenant
``11`` using the creation workflow above. While the first three retain their
allocations, ``s4`` waits without constructing a scenario. Submit one evolution
instruction to each admitted scenario, using its own installed harness binding.
Submit ``s4``'s instruction only after it becomes admitted; a training request
sent while it is queued is not retained for later execution.

The idle timer also applies before a scenario's first evolution step. Prepare
the client tools and automate submission before starting this short-TTL
experiment, then submit promptly after admission. Manually installing each
harness can take longer than ten seconds; a scenario may recycle during that
setup. A new evolve wakes that scenario; poll admission before retrying an install.

A completed step does not close its scenario. With one manual instruction and
no remaining work, it becomes idle while still admitted and charged. There is
no separate ``blocked`` scheduler state. More instructions, unfinished evaluation
or other pending work prevent idleness. Ten seconds of observed continuous
idleness makes it eligible for recycling; polling and cleanup add time before
the slot is actually released.

If ``s1`` is reclaimed first, ``s4`` takes the released slot and ``s1``
becomes asleep. Later reclaiming ``s2`` and ``s3`` leaves them asleep too.
Once ``s4`` has finished and been reclaimed, all four can remain asleep with
zero execution use. Send a new evolve to any of them to queue that scenario
again. Actual reclamation order depends on observed idleness and maintenance;
restoration does not replay a previously consumed instruction.

Observe queue changes
---------------------

The standalone scheduler prints INFO-level ``tenant_queue_changed`` records
to stderr whenever a committed tenant FIFO changes, including when it becomes
empty. A nonempty queue restored at startup is also printed. Each record contains
``tenant_id``, the previous scenario IDs in ``queue_before``, and the new ordered
``queue`` with each scenario's ID, service ID and name. ``allocated_scenario_ids``
shows which scenarios still hold execution resources; ``used_execution_units``
and ``allocated_execution_units`` show their usage and the tenant's fixed quota.

Observe the order of the queue entries rather than the numeric scenario IDs:
a woken scenario keeps its ID but joins the tail. Confirming sleep changes
execution use without changing FIFO contents, so it does not itself produce
a queue-change record. Waking and subsequent admission produce separate
records when they change the queue. State queries, duplicate
requests and unsuccessful transitions do not print duplicate queue snapshots.
These records contain scheduling metadata, not evolution instructions or model
credentials. When embedding the scheduler in another application, enable INFO
logging for ``reef.scenario.scheduling`` in that application's logging setup.

Limits and recovery
-------------------

Execution units are a static estimate of evaluation worker slots per scenario.
For Cordis, they come from its resolved per-scenario evaluation worker count,
including ``execution.evolution.workers``. They are not a service-wide worker
limit or a hard bound on CPU, memory, threads, or storage. Every service using
the same registered recipe must report the same estimate. A scenario costing
four units under a twelve-unit tenant quota allows three such scenarios across
all services, even when an admitted scenario is temporarily idle.

HTTP 422 means the tenant has no configured quota or one scenario cannot fit
its total quota. HTTP 429 means the global budget cannot admit a new tenant's
fixed quota. These requests are rejected rather than queued. See the `HTTP
reference <../reference/http-api.rst#tenant-scheduler-api>`__ for other errors.

The scheduler keeps its latest service directory in memory and saves it to
the same startup YAML on normal shutdown. SQLite separately persists tenant
allocations, queue order, numeric IDs, and incomplete lifecycle operations,
including idle recycling. Preserve both files and the service state when
restarting the single authoritative scheduler.

Services repeat registration while running. After a scheduler restart, they
provide their control tokens again; raw tokens are not saved in the YAML.
Restored directory entries remain unavailable until checked against the live
service. A crash or forced exit can lose directory changes since the last
normal shutdown, but SQLite records remain and wait for the corresponding
service to register again. Unconfirmed resource releases remain charged.
See `Saving and restoring the service directory
<../reference/configuration.rst#saving-and-restoring-the-service-directory>`__
for snapshot examples and recovery requirements.

Graceful scheduler shutdown stops new admission and registration and drains
its in-flight control operations before saving the directory. It does not
shut down remote scenarios. Already running service work can finish while
the scheduler is offline. Restarting the scheduler restores coordination;
automatic process restart and multiple scheduler authorities are outside
this feature.

This scheduler does not schedule model-weight training or individual inference
requests. It has no dynamic load balancing, per-service capacity setting, or
tenant storage quota. Existing record retention remains a separate deployment
setting.
