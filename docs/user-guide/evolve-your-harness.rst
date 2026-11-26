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

Then upload Harbor task directories as task items of the scenario's owner,
which defaults to the scenario name. Metadata names the scenario and split.
Upload evaluation tasks first. The first train task starts the run:

.. code:: bash

   tar -czf eval.tar.gz -C <tasks-directory> <task-c>
   curl -f "http://127.0.0.1:8901/reef/namespaces/<scenario>/data/tasks" \
     --get --data-urlencode 'metadata={"scenarios": {"<scenario>": "eval"}}' -X POST \
     -H "Content-Type: application/gzip" --data-binary @eval.tar.gz
   tar -czf train.tar.gz -C <tasks-directory> <task-a> <task-b>
   curl -f "http://127.0.0.1:8901/reef/namespaces/<scenario>/data/tasks" \
     --get --data-urlencode 'metadata={"scenarios": {"<scenario>": "train"}}' -X POST \
     -H "Content-Type: application/gzip" --data-binary @train.tar.gz

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

A harness tree is a directory of text files. Each snapshot holds one tree.
Adapters use these paths for its parts:

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Path
     - Part
     - Holds
   * - ``rules.md``
     - ``rules``
     - the text every session reads
   * - ``skills/<name>/SKILL.md``
     - ``skill``
     - an Agent Skill
   * - ``commands/<name>.md``
     - ``command``
     - a prompt template a person types as a command
   * - ``extensions/<name>.<ext>``
     - ``extension``
     - code the agent loads
   * - ``config/<target>.json``
     - ``config``
     - keys merged into one of the agent's config files
   * - ``native/{tools,hooks,graphs,agents,loops}/<name>``
     - ``native_*``
     - the native loop's tools, hooks, control flow, subagents and loop code
   * - ``program/<name>.py``
     - ``program``
     - a module of the ``python`` adapter's program; ``main`` is the entry
       point

Adapters support different parts. See the `adapter guide
<../developer-guide/harness-adapters.rst#render-paths>`__ before adding a file.
Reef rejects paths the adapter does not support. It supplies the model endpoint
and credentials when it renders the tree; do not put credentials in tree files.

How a step runs
---------------

Step data
~~~~~~~~~

A step starts when ``pipeline.sample2job.config.job_samples`` traces wait
(default 1). One trace is one scored report, with every model call it names.
In ``manual`` mode (``recipe.config.training_mode: manual``), a training
request starts each step instead: ``POST /reef/train`` with a ``text`` that
the method reads as an instruction (`Manual training
<../reference/http-api-training.rst#manual-training>`__). The step needs no trace.
To learn from inference traffic without reports, name the inference stage;
each call is then one unscored trace:

.. code-block:: yaml

   pipeline:
     record2sample: {module: reef.samples.traces:InferenceSamples}

Offline runs on uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.train.sample2job.traces:TrainTaskJobData`` makes each step from the train
split of the scenario's imported tasks:

- Each uploaded train task trains once, in upload order. A step runs up to
  ``buffer_size`` of them (default ``job_samples``) on the head, and the
  method reads those runs.
- A step holds either traces or train tasks, so recorded sessions still
  start their own steps.
- The eval split is the evaluation's task source. The stage needs a scorer
  and task directories (see `Task directories`_).

Upload the tasks as in `Offline: learn from uploaded tasks`_ (`Create task
items <../reference/http-api-data-upload.rst#create-task-items>`__), or import them with
the deployment's ``task-import`` section.

Evaluation: the publication policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set tasks, a ``scorer`` and a ``policy`` under ``evolution.evaluation``.
Use ``tasks`` for prompts or directories, ``task_manifest`` with ``tasks_root``,
or ``imported_tasks``. A prompt must state the complete problem: its run starts
in an empty directory.

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Policy
     - Runs
     - Publishes the candidate when
   * - ``compare``
     - the candidate and the head, ``repeats`` times per task
     - ``statistic: wins`` (default): it wins more than ``min_margin``
       pairings beyond its losses; ``statistic: mean``: its mean beats the
       head's by more than ``min_margin``
   * - ``floor``
     - the candidate only
     - every run that completed scores at least ``floor_score``, and every task
       has such a run
   * - ``none``
     - nothing
     - always; it needs no tasks and no scorer

A completed run without an answer scores 0. Infrastructure failures do not
score: ``wins`` drops the pair; ``mean`` leaves out the failed run.
A tie publishes nothing. Reef reuses scored runs for the same tree, task and
repeat, so evaluation usually needs only new candidate runs.
`Harness evolution keys <../reference/configuration-harness.rst#harness-evolution-keys>`__
lists every key.

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

A Harbor task contains instructions, an environment and tests. Each task runs
in its own container with the pinned agent and rendered tree.
``VerifierRewardScorer`` uses the test reward. The image needs glibc.

The agent reaches Reef at ``evolution.evaluation.agent_host`` (default
``host.docker.internal``) in place of a loopback address. On Linux, set it
to an address the container reaches, such as the Docker bridge gateway.
Startup refuses task directories when the adapter has no npm, uv or git
``install``, when the executor is not ``local``, or with
``collect_failures``. An offline run's evaluation tasks must be directories
too.

Collection: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``collect_failures: true`` to add trace prompts to later evaluations.
The limits are ``max_collected_per_client`` (default 5) and
``max_collected_tasks`` (default 50). See `Collect traffic prompts
<../developer-guide/write-a-harness-method.rst#collect-traffic-prompts>`__
for selection and screening.

History
~~~~~~~

- ``GET /reef/harness/artifacts`` has a row per step: the patch, the scores,
  the method's summary and its cost. ``/reef/harness/artifacts/<step>/page``
  explains one step.
- ``<evolution.state_dir>/<scenario>/harness/`` keeps ``history/`` (every
  tree, run and decision; the score cache) and ``steps/<step>/`` (each
  workspace and its model calls). Nothing prunes it, and it holds client
  traffic: protect it.

Use a published tree
--------------------

Install the active tree as your agent's plugin. The deployment's
``evolution.adapter`` selects the agent; ``GET /reef/scenarios`` lists it.
Sessions started through Reef send their model calls to Reef.

.. code:: bash

   pip install reef-client
   reef-client harness connect --reef http://127.0.0.1:8901

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Command
     - What it does
   * - ``connect``
     - Installs the active tree under ``~/.local/share/reef-client/harness/<scenario>``
       and as the agent's plugin. It installs the pinned agent if missing.
       The agent's command then runs through Reef. It requests a token and
       harness selection when needed.
   * - ``run``
     - Syncs (waiting at most 2 s for Reef), starts the agent with its model
       calls on Reef, and sends the session's changes from the background when
       it exits. Arguments after ``--`` go to the agent.
   * - ``sync``
     - Sends changed files and installs the merged result. Each part's ``merge``
       selects line merges for text or file merges for code. The committed tree
       becomes active and supplies the next training start. ``--discard`` drops
       your files.
   * - ``status``
     - Shows the installed artifact, the active selection and the changed
       files.
   * - ``disconnect``
     - Removes the plugin, the install and the agent command.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Agent
     - The snapshot, as your agent's plugin
   * - ``claude``
     - The plugin ``reef``: skills and commands as ``reef:<name>``, the rules
       as its output style, the extensions as its hooks.
   * - ``pi``
     - A pi package: skills, prompts and extensions beside yours, the rules
       after your system prompt.
   * - ``opencode``
     - A global plugin: rules, skills, commands and config merged over yours.
       opencode has no command that removes a plugin, so its entry stays after
       ``disconnect`` and loads nothing.
   * - ``codex``
     - The plugin ``reef`` in ``~/.codex``: skills and commands as
       ``$reef:<name>``.

Reef evaluates trees without your personal configuration. Your recorded model
calls can still contain personal rules and skill names.

A client of your own reads a snapshot with ``GET /reef/harness`` and commits a
sync with ``POST /reef/harness/import`` (see `HTTP API
<../reference/http-api-harness.rst#importing-a-harness>`__).

Activate an inactive artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tree code runs with your privileges. To make a person approve a change, set
the deployment's top-level ``activation`` section (`Configuration
<../reference/configuration-recipe.rst#the-activation-section>`__):

- ``module: never`` leaves every win inactive.
- ``module: always`` with ``config: {review_kinds: [extension]}`` leaves
  inactive only the wins that touch those kinds. The other wins serve at
  once.
- A kind that is always reviewed, such as native's ``native_loop``, always
  waits.

An inactive artifact has ``active: false`` and ``activated: false`` in the
artifact list. Later steps can train from it before it serves.
To activate it, set its ``artifact_id`` to weight 100 in ``served``:


.. code:: bash

   curl -sS -X POST -H "Content-Type: application/json" -d '{"served": {"<artifact_id>": 100}}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/update

`Set what a scenario serves <operate.rst#set-what-a-scenario-serves>`__ describes
the request, and `Train from an artifact
<operate.rst#train-from-an-artifact>`__ describes how to make the steps train
from another snapshot.

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
