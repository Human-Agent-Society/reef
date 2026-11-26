Evolve your harness
===================

Evolve rules, skills, commands and tools while keeping model weights fixed.
Reef checks each proposed tree and publishes changes that pass your policy. A model endpoint is enough; local training GPUs are not required.

Use the `pi tutorial <../getting-started/harness-tutorial.rst>`__ for a guided
first run. This page covers deployment options and client synchronization.

Quickstart
----------

Install Reef from a `checkout
<../getting-started/installation.rst#laptop-no-gpu>`__ and install ``reef-client``.
You also need Node with npm and a model endpoint that supports tool calls.
Offline tasks need Docker.

The tutorial listens on ``127.0.0.1:8901`` without a token. For an authenticated
deployment, add ``-H "Authorization: Bearer $REEF_TOKEN"`` to each curl command.

Online: learn from your sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export REEF_UPSTREAM_URL=https://api.moonshot.ai   # without /v1
   export REEF_UPSTREAM_MODEL=kimi-k2.6
   export REEF_UPSTREAM_API_KEY=sk-...
   reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml

   # On your machine: connect once, then use pi as you always do.
   reef-client harness connect <scenario> --reef http://127.0.0.1:8901
   pi

After the session, `report feedback on its last model call
<../getting-started/harness-tutorial.rst#tell-reef-how-it-went>`__. A row
of ``GET /reef/harness/artifacts`` with ``"selected": true`` is a new
snapshot, and the next ``pi`` session uses the active selection:

.. code:: bash

   curl -s -H "x-reef-scenario: <scenario>" http://127.0.0.1:8901/reef/harness/artifacts

Offline: learn from uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Change the deployment's evaluation and add a ``pipeline`` section:

.. code:: yaml

   recipe:
     config:
       evolution:
         evaluation:
           imported_tasks: true
           scorer: reef.evaluation.harness:VerifierRewardScorer
           policy: compare
   pipeline:
     sample2job:
       module: reef.train.sample2job.traces:TrainTaskJobData
       config: {buffer_size: 4}

Upload Harbor directories through the `task API
<../reference/http-api-data-upload.rst#create-task-items>`__.
Tag each item with the scenario and its ``train`` or ``eval`` split.
Upload evaluation tasks first: the first train task starts the run.

Run the example
~~~~~~~~~~~~~~~

The example runs online learning with automatic feedback:

.. code:: bash

   export REEF_UPSTREAM_API_KEY=sk-...    # only if your endpoint needs one
   cd tutorials/evolve-your-harness && ./run.sh

``configs/serve.yaml`` holds the endpoint, the model and the token. Three
exact-answer tasks go through Reef, and each graded reply starts one step.
A tied or unchanged candidate keeps the current harness. See
`Troubleshooting <troubleshooting.rst>`__.

How it works
------------

.. figure:: ../assets/harness-architecture-light.svg
   :alt: Agent sessions supply traces; Reef edits and evaluates harness trees; clients sync selected trees.

   Agent sessions, evaluation and client synchronization.

Reef records model calls and joins them to scored reports. Once enough traces
wait, the method edits its training start: the newest selected or synced tree,
also called the head. Invalid or unchanged trees skip evaluation.

The publication policy selects candidates. Activation decides when a selected
tree serves. Clients sync the active tree at their next session and send back
the files they changed.

Two ways to evolve
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Training input
     - Evaluation input
   * - Online
     - Scored agent sessions.
     - Configured tasks, or none with ``policy: none``.
   * - Offline
     - Uploaded train tasks, run on the head.
     - Uploaded eval tasks.

Online plugins support ``claude``, ``pi``, ``opencode`` and ``codex``.
Offline tasks need an adapter with an npm, uv or git install.
Meta-Harness instead starts from manual training requests and fixed evaluation tasks.

The harness tree
----------------

Each snapshot contains text files: ``rules.md``, skills, commands, configuration
or code supported by its adapter. Unsupported paths and embedded credentials
are rejected. Reef supplies endpoint bindings and credentials at runtime.
See `render paths <../developer-guide/harness-adapters.rst#render-paths>`__
before adding a file.

How a step runs
---------------

Step data
~~~~~~~~~

Automatic steps need ``pipeline.sample2job.config.job_samples`` scored reports
(default 1). Each report supplies its referenced trajectories.
Manual steps use ``POST /reef/train`` with an instruction; see
`Manual training <../reference/http-api-training.rst#manual-training>`__.

For unscored inference traffic, replace the report stage:

.. code:: yaml

   pipeline:
     record2sample: {module: reef.samples.traces:InferenceSamples}

Offline runs on uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrainTaskJobData`` takes uploaded train tasks in order, once each.
``buffer_size`` caps tasks per step and defaults to ``job_samples``.
Recorded sessions still form separate steps. Evaluation uses imported eval
tasks, which must also be directories with a scorer configured.
See `Offline: learn from uploaded tasks`_ and `Task directories`_.

Evaluation: the publication policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure ``evolution.evaluation`` with tasks, a scorer and a policy:

.. list-table::
   :header-rows: 1

   * - Policy
     - Publication condition
   * - ``compare``
     - Candidate beats the current tree by more than ``min_margin``; ties keep the current tree.
   * - ``floor``
     - Every completed run reaches ``floor_score``, with a completed run for every task.
   * - ``none``
     - Valid candidates publish without evaluation; no tasks or scorer required.

``compare`` defaults to pairwise ``wins``; ``mean`` compares mean scores.
Completed runs without answers score 0. Infrastructure failures are unscored:
``wins`` drops the pair; ``mean`` excludes the failed run.
Cached tree/task/repeat results can be reused.

Use prompts, task directories, a manifest or imported tasks. Prompts must
contain the full problem because runs start in empty directories.
See `evaluation options
<../reference/configuration-harness.rst#harness-evolution-keys>`__ for task sources,
repeats and thresholds.

Running runs
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``evolution.executor``
     - Where a task runs
   * - ``local`` (default)
     - a plain subprocess, with no isolation; for development
   * - ``sandbox``
     - a local sandbox: read-only base, explicit credentials only, no
       network unless ``evolution.sandbox.egress_hosts`` is set. Linux uses
       a bubblewrap jail; macOS uses a Seatbelt profile
   * - ``e2b``, ``modal``
     - one cloud sandbox per run (``reef-infra[e2b]`` or
       ``reef-infra[modal]``), with outbound internet

``evolution.sandbox.env_from`` lists environment variables to pass into a run. See
`Sandbox environments <../developer-guide/sandboxes.rst>`__.

Task directories
^^^^^^^^^^^^^^^^

Harbor tasks run in containers with the pinned agent and rendered tree.
Use ``VerifierRewardScorer`` for test rewards. Images need glibc.

Task directories require the ``local`` executor, an adapter with npm/uv/git
installation, and ``collect_failures: false``. Configure
``evolution.evaluation.agent_host`` so containers reach Reef. Its default,
``host.docker.internal``, needs host-gateway configuration on Linux.

Collection: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``collect_failures: true`` to add trace prompts to later evaluations.
The limits are ``max_collected_per_client`` (default 5) and
``max_collected_tasks`` (default 50). See `Collect traffic prompts
<../developer-guide/write-a-harness-method.rst#collect-traffic-prompts>`__
for selection and screening.

History
~~~~~~~

Read ``GET /reef/harness/artifacts`` for patches, scores and summaries.
Each row has a `step page <../reference/http-api-harness.rst#version-page>`__.

``<evolution.state_dir>/<scenario>/harness/`` retains trees, runs, decisions
and model calls without pruning. It contains client traffic; protect it.

Use a published tree
--------------------

Connect the agent selected by ``evolution.adapter``:

.. code:: bash

   pip install reef-client
   reef-client harness connect --reef http://127.0.0.1:8901

``connect`` installs the active tree and any missing pinned agent, then makes
the agent's command run through Reef. It asks for a scenario and token when
needed. Files live under ``~/.local/share/reef-client/harness/<scenario>``.

.. list-table::
   :header-rows: 1

   * - Command
     - Action
   * - ``reef-client harness run``
     - Sync, run the agent, then upload changed files. Agent arguments follow ``--``.
   * - ``reef-client harness sync``
     - Merge changed files and install the active tree. ``--discard`` drops local changes.
   * - ``reef-client harness status``
     - Show installed and active artifacts and local changes.
   * - ``reef-client harness disconnect``
     - Remove the installation and command wrapper.

Plugins support ``claude``, ``pi``, ``opencode`` and ``codex``.
OpenCode's plugin entry remains after disconnect but loads nothing.
Evaluation excludes personal configuration; recorded calls may still contain it.

For conflicts and plugin setup, follow the `client tutorial
<../getting-started/harness-tutorial.rst#when-you-and-reef-change-the-same-file>`__.
Custom clients use the `harness HTTP API <../reference/http-api-harness.rst>`__.

Activate an inactive artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree code runs with your privileges. Configure top-level ``activation``:

- ``module: never`` holds every new artifact for review.
- ``module: always`` with ``config: {review_kinds: [extension]}`` holds changes
  to those kinds. Always-reviewed kinds, such as ``native_loop``, still wait.

Inactive artifacts show ``active: false`` and ``activated: false``.
They can become training starts before serving. Activate one with:

.. code:: bash

   curl -sS -X POST -H "Content-Type: application/json" \
     -d '{"served": {"<artifact_id>": 100}}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/update

See `activation settings
<../reference/configuration-recipe.rst#the-activation-section>`__ and
`train-from <operate.rst#train-from-an-artifact>`__ for the separate training start.

Choose or write a method
------------------------

The deployment names one method in ``evolution.method``:

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - Method
     - How it learns
     - Policy
   * - `GEPA <recipes/gepa.rst>`__
     - A reflection model rewrites one file from a minibatch's feedback.
     - ``compare``, mean
   * - `Meta-Harness <../../recipes/meta_harness/README.md>`__
     - On each training request, a model reads every earlier candidate and its
       scores, and writes a new tree.
     - ``compare``, mean
   * - `ACE <../../recipes/ace/README.md>`__
     - A reflector and a curator edit a playbook in ``rules.md``.
     - ``none``
   * - `ReasoningBank <../../recipes/reasoning_bank/README.md>`__
     - Judged successes and failures become memory skills.
     - ``none``
   * - `Reefine <../../recipes/beta/reefine/README.md>`__ (beta)
     - A coding agent answers plain-language requests.
     - ``floor``

`Write a harness method <../developer-guide/write-a-harness-method.rst>`__
shows how to write your own. `Harness adapters
<../developer-guide/harness-adapters.rst>`__ shows how to connect another
agent.
