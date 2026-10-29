Evolve your harness
===================

A harness is everything around the model: the control loop, rules, prompt
templates, skills, tools, config, and extension code. Together, the model and
harness form an agent. Harness evolution improves the harness while the model
weights stay fixed. Reef needs no GPU for this. The model stays a fixed
endpoint, hosted or local, and the agent stays online throughout.

At each step, a method edits a copy of the current harness from the step's
traces. Reef evaluates the result and publishes it as a new release when the
deployment's publication policy selects it. Clients pull the published
harness. You choose three things in the deployment config:

- the agent, with ``evolution.adapter`` (see `The harness tree`_);
- the method, which decides what change to try, with ``evolution.method``
  (see `Choose or write a method`_);
- the evaluation, which decides whether a change is published, with
  ``evolution.evaluation`` (see `Evaluation: the publication policy`_).

Harness evolution fits when the problem is in the text around the model: a
prompt that mishandles a task family, a missing skill, or a config default
that is wrong for the deployment. It also fits a closed model endpoint with no
weight access. It does not fit when the model itself cannot do the task.

This page explains how harness evolution works, then runs it. Two developer
guides cover the rest: `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ says how a step changes
the tree, and `Harness adapters <../developer-guide/harness-adapters.rst>`__
says how to connect another agent.

How it works
------------

The figure shows what runs on a person's machine, what runs in Reef, and the
routes between them. The numbers follow one session into one training step.
The bottom row shows the two ways a step gets its traces.

.. diagram::
   :caption: The client installs the head as a layer over the person's own configuration and sends back what a session changed. Reef records the model calls and reports, runs a training step on a batch of reports or of uploaded train tasks, and publishes a selected candidate as the new head.

   <img class="fig-light" src="../assets/harness-architecture-light.svg" alt="Harness evolution: what runs on a person's machine, what runs in Reef, the routes between them, and the two ways a step gets its traces">
   <img class="fig-dark" src="../assets/harness-architecture-dark.svg" alt="Harness evolution: what runs on a person's machine, what runs in Reef, the routes between them, and the two ways a step gets its traces">

One session, step by step
~~~~~~~~~~~~~~~~~~~~~~~~~

Once per machine, ``reef-client harness connect`` installs the head, installs
the agent when the machine lacks it, and makes the agent's own command run
through Reef (see `Use a published tree`_). Then each session goes like
this:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Step
     - What happens
   * - before
     - The agent's command runs ``reef-client harness run``. It installs the
       head, rendered as a layer for this machine (``GET
       /reef/harness?form=client``).
   * - 1
     - The agent starts with the person's own configuration, then the layer.
   * - 2
     - Its model calls go to ``/reef/scenarios/<scenario-name>/v1/*``. Reef
       forwards each call to the provider as sent, and records it with its
       reply. The answer names the record in ``x-reef-agent-record-id``.
   * - after
     - When the agent exits, a background process sends the layer's files
       that the session changed (``POST /reef/harness/import``). Reef merges
       them onto the newest head.
   * - 3
     - The person, or a program that checks the work, scores the session
       with ``POST /reef/report``. The report and the calls it names are one
       raw trace.
   * - 4
     - When ``evolution.batch_size`` traces wait, a training step starts. Its
       workspace gets a copy of the head and the batch.
   * - 5
     - The method edits the copy (``step(context)``).
   * - 6
     - Reef checks the new tree. An unchanged or invalid tree skips the step.
   * - 7
     - The candidate and the head run the same tasks as episodes, and the
       policy compares their scores.
   * - 8
     - The release rule publishes a selected candidate as the new head, or
       holds it until a person promotes it.
   * - next
     - The next session's sync installs the new head.

Reef never reads, syncs or evolves the person's own configuration. `Use a
published tree`_ lists what that configuration holds for each agent.

Uploaded tasks, step by step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offline, the person who runs the deployment gives Reef Harbor tasks instead
of sessions. The deployment sets ``pipeline.sample2job`` to
``TrainTaskRollouts`` and ``evolution.evaluation.task_store`` to
``imported-tasks`` (see `Offline: learn from uploaded tasks`_). Then a run
goes like this:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Step
     - What happens
   * - upload
     - The person uploads task directories to the scenario, train and eval
       (``POST /reef/scenarios/<scenario-name>/tasks?split=train`` or
       ``?split=eval``). The first train task starts the run.
   * - 4
     - A step takes the next ``tasks_per_step`` train tasks (8 by default).
       Each one runs on the head in its own container: Harbor starts the
       container, Reef installs the agent there, the agent plays the task's
       instruction, and the task's tests score it. These episodes are the
       step's traces.
   * - 5, 6
     - The method edits a copy of the head, and Reef checks it, as for
       sessions.
   * - 7
     - The candidate and the head play each eval task, each in its own
       container, and the policy compares their scores.
   * - 8
     - The release rule publishes a selected candidate as the new head.
   * - after
     - ``GET /reef/scenarios/<scenario-name>/tasks`` shows which tasks have
       trained. After ``epochs`` passes over the train split (1 by default),
       the run stops. People use the head as after a session.

Two ways to evolve
~~~~~~~~~~~~~~~~~~

A step gets its traces in one of two ways. Both ways go into the same
training step (4 to 8 above).

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - What
     - Online: learn from sessions
     - Offline: learn from uploaded tasks
   * - Traces
     - The sessions that people run and score (1 to 3).
     - The episodes of the next train tasks, run on the head.
   * - Evaluation tasks
     - Prompts in ``evolution.evaluation.tasks``, or none with
       ``policy: none``.
     - The eval split of the uploaded tasks.
   * - Agents
     - The adapters with a layer: ``claude``, ``pi``, ``opencode``,
       ``codex``.
     - Every adapter with an npm ``install``.
   * - Data comes from
     - ``reef-client harness`` and ``POST /reef/report``.
     - ``POST /reef/scenarios/<scenario-name>/tasks``, with Harbor task
       directories.
   * - Recipes
     - ACE and ReasoningBank (``policy: none``); GEPA (``compare`` on eval
       prompts, as in its AIME example).
     - GEPA with ``TrainTaskRollouts``; Meta-Harness (``compare`` on task
       directories, as in its Terminal-Bench example).

`Choose or write a method`_ describes each recipe. A deployment can also mix
the two ways: recorded traffic still starts steps in an offline run, and it
goes first.

Quickstart
----------

Before you start
~~~~~~~~~~~~~~~~

- A Reef checkout (`Laptop, no GPU
  <../getting-started/installation.rst#laptop-no-gpu>`__) and
  ``pip install reef-client``.
- An endpoint that serves a model that can call tools, hosted or local.
- Node and ``npm``. At startup, the deployment installs the adapter's pinned
  agent under ``~/.local/share/reef-harness/<adapter>``.
  ``REEF_HARNESS_PREFIX`` moves that root, and ``evolution.binary`` names a
  binary of your own instead. A missing ``npm`` or a failed install refuses
  startup.
- For the offline way: Docker. Docker Desktop and colima both work.

The tutorial's deployment listens on ``127.0.0.1:8901`` without a token. When
a deployment has one, add ``-H "Authorization: Bearer $REEF_TOKEN"`` to each
``curl`` below.

Online: learn from your sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start the tutorial's deployment, which evolves pi's rules:

.. code:: bash

   export REEF_UPSTREAM_URL=https://api.moonshot.ai   # without /v1
   export REEF_UPSTREAM_MODEL=kimi-k2.6
   export REEF_UPSTREAM_API_KEY=sk-...
   reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml

Connect your machine, open a new terminal, and use pi as you always do:

.. code:: bash

   reef-client harness connect <scenario-name> --reef http://127.0.0.1:8901
   pi

Score the session with one report over its model calls, ``0`` for a failure
and ``1`` for a success:

.. code:: bash

   CALLS=$(curl -s "http://127.0.0.1:8901/reef/scenarios/<scenario-name>/records?request_type=inference&limit=100" \
     | jq -c '[.records[].agent_record_id]')
   curl -s -X POST http://127.0.0.1:8901/reef/report \
     -H "x-reef-scenario: <scenario-name>" -H "Content-Type: application/json" \
     -d "{\"references\": $CALLS, \"score\": 0, \"feedback\": \"it said the test passed without running it\"}"

This deployment runs a step for each report. List the steps and their
results:

.. code:: bash

   curl -s -H "x-reef-scenario: <scenario-name>" http://127.0.0.1:8901/reef/harness/releases

A row with ``selected: true`` is a new head, and your next ``pi`` session
starts on it. `Tutorial: evolve your agent's harness
<../getting-started/harness-tutorial.rst>`__ walks through the same steps in
detail.

Offline: learn from uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``tutorials/evolve-your-harness/configs/deployment.yaml`` and give it
tasks instead of users. In ``evolution.evaluation``, replace ``tasks`` and
``scorer``, and add the ``pipeline`` section:

.. code:: yaml

   recipe:
     config:
       evolution:
         evaluation:
           task_store: imported-tasks
           scorer: reef.evaluation.harness:VerifierRewardScorer
           policy: compare
   pipeline:
     sample2job:
       module: reef.sample2job.traces:TrainTaskRollouts
       config: {tasks_per_step: 4}

Start Reef with the new file. Then upload Harbor task directories, each with
its instruction, environment and tests. The train tasks are what the method
learns from, and the eval tasks decide each release:

.. code:: bash

   tar -czf train.tar.gz -C <tasks-directory> <task-a> <task-b>
   curl -f "http://127.0.0.1:8901/reef/scenarios/<scenario-name>/tasks?split=train" \
     -H "Content-Type: application/gzip" --data-binary @train.tar.gz
   tar -czf eval.tar.gz -C <tasks-directory> <task-c>
   curl -f "http://127.0.0.1:8901/reef/scenarios/<scenario-name>/tasks?split=eval" \
     -H "Content-Type: application/gzip" --data-binary @eval.tar.gz

The first train task starts the run. Each step runs the next train tasks on
the head in their own containers, and the method reads those episodes as its
traces. Watch the steps with the same ``GET /reef/harness/releases``. `Offline
runs on uploaded tasks`_ explains the settings and the network between a
task's container and Reef.

Run the example
~~~~~~~~~~~~~~~

The example's driver runs the online way without a person. From a Reef
checkout:

.. code:: bash

   export REEF_UPSTREAM_API_KEY=sk-...    # only if your endpoint needs one
   cd tutorials/evolve-your-harness
   ./run.sh

``configs/serve.yaml`` holds the endpoint (``http://127.0.0.1:8000``, with no
``/v1`` suffix), the model (``qwen3-8b``), and the service token. Edit them
to point at your own. The method, the episodes and the served traffic all use
that one model. `evolve-your-harness.ipynb
<../../tutorials/evolve-your-harness/evolve-your-harness.ipynb>`__ runs the
same pass as a notebook.

``run.sh`` starts the service on port 8900 and runs ``run.py``. Three
exact-answer coding tasks go through Reef, and ``run.py`` grades each reply
and reports it. ``evolution.batch_size: 1`` makes each report its own step.
The example's method, in ``tutorials/evolve-your-harness/harness/plugins.py``,
makes one model call that reads the step's failing traces and rewrites
``rules.md``. When the loop finishes, ``run.sh`` prints the published release,
its evaluation metrics, and the changed files, then stops the service. The
scenario is ``harness-evolve-demo``. ``configs/serve-native.yaml`` runs the
same loop on the native harness.

To follow it from a second terminal while ``run.sh`` runs:

.. code:: bash

   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness            # the seed tree until a step publishes
   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness/releases

The demo service listens on loopback only and runs without a token. On a
shared machine, export ``REEF_TOKEN`` before ``run.sh`` starts it.

The run has succeeded when a step's row shows ``selected: true`` and
``GET /reef/harness`` serves a release other than the seed. A step that runs
but does not win leaves the seed in place; that is a normal result. If no step
publishes after a few minutes, read the step's row. Both sides scoring
nothing usually means the episodes could not run: vLLM, for example, needs
``--enable-auto-tool-choice --tool-call-parser hermes``.
`Troubleshooting <troubleshooting.rst>`__ lists the other causes.

The harness tree
----------------

A harness tree is one directory of text files: the mutable, versioned part of
one harness. Each release holds one tree, and each step changes it with a
patch: paths mapped to their new text, or to ``null`` for a removal.

Each adapter declares the parts of the tree it takes in the ``evolve`` section
of its ``adapter.yaml`` (for example `reef/harness/adapters/pi/adapter.yaml
<../../reef/harness/adapters/pi/adapter.yaml>`__). A part names a tree path, the
format Reef checks the file as, and the path where the agent reads it. The
bundled adapters use the same tree path for the same part, so a tree means the
same thing to every adapter that takes its parts:

+-------------------------------+------------------+-----------------------------------------------+
| Path                          | Part             | Holds                                         |
+===============================+==================+===============================================+
| ``rules.md``                  | ``rules``        | the text every session reads                  |
+-------------------------------+------------------+-----------------------------------------------+
| ``config/<target>.json``      | ``config``       | a JSON object merged into one of the agent's  |
|                               |                  | config files                                  |
+-------------------------------+------------------+-----------------------------------------------+
| ``skills/<name>/SKILL.md``    | ``skill``        | an Agent Skill                                |
+-------------------------------+------------------+-----------------------------------------------+
| ``commands/<name>.md``        | ``command``      | a prompt template a person types as a command |
+-------------------------------+------------------+-----------------------------------------------+
| ``extensions/<name>.<ext>``   | ``extension``    | code the harness loads in its process         |
+-------------------------------+------------------+-----------------------------------------------+
| ``native/tools/<name>.py``    | ``native_tool``  | a tool of the native loop                     |
+-------------------------------+------------------+-----------------------------------------------+
| ``native/hooks/<name>.py``    | ``native_hook``  | a listener at one event of the native loop    |
+-------------------------------+------------------+-----------------------------------------------+
| ``native/graphs/<name>.json`` | ``native_graph`` | the native loop's control flow                |
+-------------------------------+------------------+-----------------------------------------------+
| ``native/agents/<name>.json`` | ``native_agent`` | one agent the native loop delegates to        |
+-------------------------------+------------------+-----------------------------------------------+
| ``native/loops/<name>.py``    | ``native_loop``  | the native loop itself as code                |
+-------------------------------+------------------+-----------------------------------------------+

Turning the tree into the agent's files is called rendering. Reef refuses a
tree file at a path that no part of the adapter takes, at validation and at
render. The bundled adapters take these parts:

+--------------+-----------------------------+---------------------------------------------+
| Adapter      | Agent                       | Parts it takes                              |
+==============+=============================+=============================================+
| ``pi``       | pi coding agent             | config, rules, command, skill, extension    |
+--------------+-----------------------------+---------------------------------------------+
| ``opencode`` | OpenCode                    | config, rules, command, skill, extension    |
+--------------+-----------------------------+---------------------------------------------+
| ``claude``   | Claude Code                 | config, rules, command, skill, extension    |
+--------------+-----------------------------+---------------------------------------------+
| ``codex``    | Codex CLI                   | config, rules, command, skill               |
+--------------+-----------------------------+---------------------------------------------+
| ``dsh``      | DeepSeek Harness            | config, rules, command, skill, extension    |
+--------------+-----------------------------+---------------------------------------------+
| ``hermes``   | Hermes Agent                | config, rules, command, skill, extension    |
+--------------+-----------------------------+---------------------------------------------+
| ``native``   | Reef's own loop             | config, rules, skill, native_tool,          |
|              |                             | native_hook, native_graph, native_agent,    |
|              |                             | native_loop                                 |
+--------------+-----------------------------+---------------------------------------------+

With ``native``, the method can change the agent's tools, its hooks, its
control flow, and the loop itself, not only the text around a vendor binary.
The adapter guide's `Tree parts
<../developer-guide/harness-adapters.rst#tree-parts>`__ describes the
``evolve`` section, and its `Render paths
<../developer-guide/harness-adapters.rst#render-paths>`__ gives where each
adapter writes each part.

A tree does not choose where model calls go. Reef writes the endpoint, the
key and the model when it renders the tree for an episode. With most
adapters, render refuses a tree that sets them itself. The adapter guide's
`Model binding checks <../developer-guide/harness-adapters.rst#model-binding-checks>`__
list the keys for each adapter.

The loop
--------

Harness evolution has five parts. Data and sync happen between steps, and
each step runs evolve, evaluation and release. Data, evaluation and release
work as they do for weight training; evolve and sync are the harness's own.
Each part is one relation between a few values. ``T`` is a harness tree
(`The harness tree`_), and ``A`` is the deployment's adapter.

.. list-table::
   :header-rows: 1
   :widths: 14 56 30

   * - Part
     - Definition
     - Details
   * - Data
     - Records and uploaded tasks give ``D``, one step's job data: traces
       or train tasks. Weight training uses the same pipeline.
     - `Step data`_
   * - Evolve
     - ``method(T, D, s) = (T', s')``: from the head tree ``T``, the data
       and the method's state ``s``, the candidate ``T'``. An unchanged or
       invalid ``T'`` skips the step.
     - `The method`_
   * - Evaluation
     - ``score(T, task, repeat)`` is one episode's score, cached by tree,
       task, repeat and evaluation context. ``policy`` compares the scores
       of ``T'`` with those of ``T``, and selects or rejects ``T'``.
     - `Evaluation: the publication policy`_
   * - Release
     - A selected ``T'`` becomes a git release and the head.
       ``publish: review`` holds it until a person promotes it. A rejected
       ``T'`` makes no release; the history keeps it.
     - `Promote a pending release`_
   * - Sync
     - ``render_A(T)`` is the files that ``A``'s agent reads.
       ``import_A(F)`` reads those files back, so
       ``import_A(render_A(T)) = T``. A client installs ``render_A`` of the
       head ``H`` as a layer over the person's own configuration. When its
       files ``F`` changed since the installed release ``B``,
       ``merge(B, H, import_A(F))`` takes each file from the side that
       changed it, and a file both changed by its part's ``merge``.
     - `Use a published tree`_

Step data
~~~~~~~~~

A step starts from recorded traffic, when
``pipeline.sample2job.config.job_samples`` traces wait. Without that key, it
is ``evolution.batch_size`` (default 1). By default, one trace is one valid
scored report, a passing report included. A report over several model calls
is one trace that holds every call in order. Reef keeps at most
``pipeline.sample2job.config.buffer_size`` waiting traces (default four times
``job_samples``) and drops the oldest ones beyond that.

To evolve on recorded inference traffic without reports, name the inference
stages. Every inference is then one unscored trace, for methods that judge
for themselves:

.. code-block:: yaml

   pipeline:
     record2record:
       module: reef.record2record:InferenceRecord2Record
     record2sample:
       module: reef.record2sample.traces:InferenceSamples

Offline runs on uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A deployment that has tasks instead of users runs offline. The stage
``reef.sample2job.traces:TrainTaskRollouts`` makes each step from the train
split of the scenario's ``imported-tasks`` store. Each step runs the next
``tasks_per_step`` train tasks (default 8) on the head tree, and the method
reads those episodes as its traces. The run goes through the train split
``epochs`` times (default 1), in a new random order each time, and then
stops. The eval split stays the evaluation's task source.

.. code-block:: yaml

   recipe:
     config:
       evolution:
         adapter: pi
         method: recipes.gepa:GEPA
         evaluation:
           task_store: imported-tasks
           scorer: reef.evaluation.harness:VerifierRewardScorer
           policy: compare
           statistic: mean
   pipeline:
     sample2job:
       module: reef.sample2job.traces:TrainTaskRollouts
       config: {tasks_per_step: 8, epochs: 2}

Upload the tasks while the service runs, with
``POST /reef/scenarios/{scenario}/tasks?split=train`` and ``?split=eval`` (see
`Upload Harbor tasks <../reference/http-api.rst#upload-harbor-tasks>`__), or
import them with the deployment's ``task-import`` section. The first train
task starts the run. Recorded traffic still starts steps, and it goes first.
After a restart, the run continues after the last committed step. A task
uploaded during an epoch joins the next epoch.

The stage needs a scorer, and the evaluation tasks must be directories too:
a ``task_store`` or a ``task_manifest``. Each task runs in its own container (see `Task
directories`_).

The method
~~~~~~~~~~

Reef runs the method once per step in a workspace. The workspace holds a
copy of the head tree, the step's traces, the evaluation tasks, and the
method's files from earlier steps. The
tree that the method leaves there is the candidate. Reef skips the step when
the tree is unchanged or fails validation for the adapter; there is no retry.
`Write a harness method <../developer-guide/write-a-harness-method.rst>`__
documents the workspace, the tools, and the method's state.

Evaluation: the publication policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evolution.evaluation`` names the evaluation tasks, the episode scorer, and
the publication policy. `Harness evolution keys
<../reference/configuration.rst#harness-evolution-keys>`__ lists every key
and its default.

The tasks come from one source:

- ``tasks``: a list of prompts, or of task directories;
- ``task_manifest`` with ``tasks_root``: the eval split of a task manifest;
- ``task_store``: the eval split of a task store, such as ``imported-tasks``.
  Upload eval tasks with ``POST /reef/scenarios/{scenario}/tasks?split=eval``.

A prompt episode runs in a fresh, empty root with only the rendered tree: no
repository and no files from your machine. A prompt task must therefore state
the whole problem. A task directory brings its own environment and tests
(see `Task directories`_).

``scorer`` names the ``EpisodeScorer`` that scores each episode. ``policy``
is one of three:

- ``compare`` (the default) runs the candidate and the head on the same tasks,
  ``repeats`` times each. With ``statistic: wins`` (the default), the
  candidate must win more than ``min_margin`` task pairings beyond its losses.
  With ``statistic: mean``, its mean score must exceed the head's by more
  than ``min_margin``.
- ``floor`` runs only the candidate, and publishes it when every episode
  scores at least ``floor_score``.
- ``none`` runs no episode, and publishes every valid candidate. It needs no
  tasks and no scorer.

Reef never publishes by accident. An episode that could not run ranks below
every score under ``wins``, counts as 0 under ``mean``, and misses the floor
under ``floor``. A tie publishes nothing. Under ``compare`` and ``floor``, a
step with no evaluation task publishes nothing. A rejected candidate leaves
the head as it was.

Most of a step's cost is the episodes. ``compare`` runs at most
``2 x tasks x repeats`` episodes, ``floor`` runs ``tasks x repeats``, and
``none`` runs none. Each episode is limited to
``evolution.evaluation.episode_timeout_s`` (600 s by default). Reef keeps
every scored episode and reuses it for the same tree, task and repeat, as long
as the model, adapter, binary, scorer, episode settings, and executor are
unchanged. The head's scores usually come from the step that published it,
so a step usually runs only the candidate's episodes. An episode that could
not run is never reused. A method that scores trees itself adds its own
episodes, at most ``evolution.max_episodes_per_step`` when that is set.

By default a selected candidate is served at once. To make a person approve
it first, see `Promote a pending release`_.

Running episodes
^^^^^^^^^^^^^^^^

``evolution.executor`` chooses where each episode runs:

- ``local`` (the default) runs the agent as a plain subprocess, with no
  isolation. Use it for development.
- ``sandbox`` runs each episode in a local bubblewrap jail: a read-only base
  filesystem, explicit credentials only, resource limits, and no network
  unless ``evolution.sandbox.egress_hosts`` is set. ``egress_hosts`` enables
  the network; it does not filter hostnames. A hosted service that evaluates
  model-written trees should use it.
- ``e2b`` or ``modal`` creates one cloud sandbox per episode. Use them on
  macOS or on a host without Linux namespaces. Install ``reef-infra[e2b]`` or
  ``reef-infra[modal]``. The cloud sandboxes have outbound internet access and
  reject ``egress_hosts`` and ``limits``. A Modal sandbox lives at most a day,
  so a longer episode timeout is refused.

.. code-block:: yaml

   evolution:
     adapter: pi
     executor: e2b
     sandbox:
       e2b_api_key_env: E2B_API_KEY
       # Only for a model that listens on the evaluation worker's localhost:
       forward_ports: [8000]

A task directory runs in its own container instead, under the ``local``
executor (see `Task directories`_).

``evolution.sandbox.env_from`` lists the environment variables to pass into
an episode; a missing variable fails configuration. No other variable
reaches the agent, and Reef keeps the E2B key and the Modal token to itself.
A public model endpoint needs no forwarded port. For an npm-installed
adapter, Reef builds the cloud image with the agent. Any other adapter needs
``e2b_template`` or ``modal_image`` with its binary and bubblewrap
installed, and ``evolution.binary`` then names a path inside that image.
Startup checks provider access, the image, the binary, and namespace
support. `Sandbox environments <../developer-guide/sandboxes.rst>`__
describes the shared execution layer. A method that runs a coding agent,
such as ``AgentMethod`` or the beta Reefine recipe, sets where the agent runs
with ``evolution.agent.sandbox``, apart from ``evolution.executor``.

Task directories
^^^^^^^^^^^^^^^^

A Harbor task directory holds an instruction, an environment and the tests
that score the work. Tasks from a ``task_store``, from a ``task_manifest``,
and the train tasks of an offline run are directories. Reef runs each task
in its own container:

#. Harbor starts the task's container with Docker.
#. Reef installs the adapter's pinned agent there with npm, copies in the
   rendered tree, and runs the agent with the task's instruction as its
   prompt, in the task's working directory.
#. The task's tests run after the agent. With
   ``scorer: reef.evaluation.harness:VerifierRewardScorer``, their reward is
   the episode's score.

The adapter needs nothing more than an npm ``install`` for this (see `How
Reef runs an agent
<../developer-guide/harness-adapters.rst#how-reef-runs-an-agent>`__). An
adapter whose own runner takes the task directory, such as the Meta-Harness
Terminal-Bench campaign's ``terminus``, plays it that way instead. The
task's image needs glibc, because Node comes from nvm. When the image has no
``curl``, Reef installs it with ``apt-get``.

The agent in the container calls the served model's route on Reef. When that
route's address is a loopback one, such as ``127.0.0.1``, the container
cannot reach it, so Reef puts ``evolution.evaluation.agent_host`` (default
``host.docker.internal``) in its place and keeps the port and the path.
Another address stays as it is. Docker Desktop and colima resolve
``host.docker.internal`` and reach a Reef that listens on ``127.0.0.1``. On
Linux, set ``agent_host`` to an address of the host that the container
reaches, such as the Docker bridge gateway (often ``172.17.0.1``), and make
Reef listen on it with ``reef.host``.

Startup refuses task directories when:

- the adapter has no npm ``install``;
- ``evolution.executor`` is not ``local``: the task's container is the
  boundary;
- ``collect_failures`` is on, because it adds prompts;
- an offline run's evaluation tasks are prompts in ``tasks``.

An adapter whose own runner takes the task directory skips the first, second
and last of these.

Collection: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``evolution.evaluation.collect_failures: true``, the prompts of each
step's traces become permanent evaluation tasks, so a later candidate cannot
win while it brings an earlier failure back. The scorer must then score any
prompt, and the evaluation tasks must be prompts, not task directories.

A collected prompt is client traffic, so Reef screens it first. It never
collects a prompt that holds a credential-shaped literal or an instruction
override, such as ``ignore the previous instructions``. One tagged client
adds at most ``max_collected_per_client`` tasks (default 5), and
``max_collected_tasks`` (default 50) caps the total. To choose which prompts
to collect, for example only those of failing traces, name a
``TaskCollector`` in ``evolution.evaluation.collector`` (see `Collect traffic
prompts <../developer-guide/write-a-harness-method.rst#collect-traffic-prompts>`__).

History
~~~~~~~

Every step adds a row to ``GET /reef/harness/releases``, whether its
candidate was published, rejected, or skipped. The row holds the patch, the
scores, the method's summary, and the method's cost in model calls and
tokens. Each step also has a page, ``GET /reef/harness/releases/<step>/page``,
that explains the step from its cause to its result (see `Version page
<../reference/http-api.rst#version-page>`__).

Reef keeps the full record under
``<evolution.state_dir>/<scenario>/<component>/``, where ``<component>`` is
``harness`` for ``HarnessEvolutionRecipe``:

- ``history/`` holds every tree, episode and decision. It is the score
  cache, and methods read it (see `Read the history
  <../developer-guide/write-a-harness-method.rst#read-the-history>`__).
- ``steps/<step>/`` holds each step's workspace, the method's model calls,
  and its result.

Nothing prunes these directories. They hold the method's raw model traffic,
the clients' traces, and every episode's full log, so protect
``evolution.state_dir`` as you protect the commit history.

Use a published tree
--------------------

``reef-client harness`` runs the agent on your machine with the scenario's
release as a layer over your own configuration. Your own settings, skills,
rules and sessions stay as they are; the release adds to them, and its model
calls go to Reef. The adapters ``claude``, ``pi``, ``opencode`` and ``codex``
support it. Install ``reef-client`` where the agent runs, and connect once:

.. code:: bash

   pip install reef-client
   reef-client harness connect --reef http://127.0.0.1:8900

The deployment decides the agent. Its ``evolution.adapter`` names the
agent, and every harness that one deployment serves runs that agent. To see
it before you connect, list the harnesses: ``curl
http://127.0.0.1:8900/reef/scenarios`` gives each one with its ``adapter``.

``connect`` asks at most two things: the token, when Reef refuses the call
without one and ``REEF_TOKEN`` is unset, and which harness, when Reef serves
more than one. The list it shows names each harness's agent. Everyone who
connects to one harness shares it. The Reef URL comes from ``--reef`` or
``REEF_URL``. Then it sets the machine up, and says one line when it is done:

- It installs the harness's head in
  ``~/.local/share/reef-client/harness/<scenario>``. It never reads or writes
  your own configuration. For Codex, which loads skills only from a plugin,
  it runs Codex's own ``codex plugin`` commands, which add the install as the
  plugin ``reef`` to your ``~/.codex``; ``disconnect`` removes it.
- When the agent is not installed, it installs the version that Reef's
  evaluation episodes run, with npm, under ``~/.local/share/reef-client/agents``.
- It makes the agent's own command run through Reef: a command with the
  agent's name in ``~/.local/share/reef-client/bin``, and that directory
  first on ``PATH`` in your shell's startup file (zsh, bash or fish). In a
  new terminal, the agent's command then runs as ``reef-client harness run``
  does. The last line that ``connect`` prints names it: ``Done. Open a new
  terminal and run <agent>.``

.. flow::
   :loop: each run starts on the newest release

   connect :: sets the machine up, once
   run :: syncs, starts the agent on the layer, and sends its changes when it exits
   Reef* :: records each model call; steps publish releases

- ``run`` syncs, then starts the agent with the layer. The agent's own
  command runs it, once ``connect`` added that command. The sync waits at
  most 2 seconds for each answer from Reef; without an answer, the agent
  starts on the installed files. When the agent exits, a background process
  sends what the session changed in the layer, so your terminal is back at
  once. The next ``run`` prints the result of that send. Arguments after
  ``--`` go to the agent. With one scenario
  installed, ``run``, ``sync``, ``status`` and ``disconnect`` need no
  scenario name.
- ``sync`` sends the layer's files that you or the agent changed since the
  installed release, then installs the head. Reef merges them onto its head
  as each part's ``merge`` says (`Tree parts
  <../developer-guide/harness-adapters.rst#tree-parts>`__): text such as
  rules, skills and memory merges by line, so only a line that a step and
  you both changed is refused, by the file's name; your files then stay as
  they are. The result serves at once, and Reef does not evaluate it: you
  are its author. While a step runs, your changes wait for the next ``run``
  or ``sync``. ``sync --discard`` drops them and installs the head.

  An install writes only the files that differ from the head. When the head
  is the files you just sent, no file changes. While an agent session runs
  on the install, a head that would change files waits for the next ``run``
  or ``sync``, so no file changes under the agent.
- ``status`` shows the installed release, the head and the changed files.
  ``disconnect`` removes the install, and the agent command and the ``PATH``
  line that ``connect`` added.

Your own configuration is the agent's own directory, and each repository's
own files, such as its ``AGENTS.md`` or ``.claude/``. How the layer goes over
it depends on the agent:

.. list-table::
   :header-rows: 1
   :widths: 12 38 50

   * - Adapter
     - Your own configuration
     - The release over it
   * - ``claude``
     - ``~/.claude``: ``CLAUDE.md``, ``settings.json``, ``skills/``,
       ``commands/``, ``agents/``, ``plugins/``, and ``projects/`` with your
       sessions and automatic memory. Your MCP servers are in
       ``~/.claude.json``.
     - The release's skills and commands load as a plugin, named
       ``reef:<name>`` beside your own. Its rules follow your ``CLAUDE.md``.
       Its settings bind the model, run the release's extensions as hooks,
       and win over yours. Claude Code's automatic memory goes to the
       release's ``memory`` part, for every working directory, so it syncs;
       memory written before ``connect`` stays under ``~/.claude/projects``.
   * - ``pi``
     - ``~/.pi/agent``: ``AGENTS.md``, ``settings.json``, ``models.json``,
       ``skills/``, ``prompts/``, ``extensions/`` and ``sessions/``.
     - The release's skills, prompts and extensions load beside yours; on a
       name both define, the release's wins. Its rules follow your
       ``AGENTS.md``; pi then skips your ``APPEND_SYSTEM.md``. The two
       settings a release can hold come as flags: its thinking level as
       ``--thinking``, and no built-in tools as ``--no-builtin-tools``.
       Changing the model in a session still writes your ``settings.json``.
   * - ``opencode``
     - ``~/.config/opencode``: ``opencode.json`` or ``.jsonc``, ``AGENTS.md``,
       ``agent/``, ``command/``, ``plugin/`` and ``skill/``. Your sessions are in
       ``~/.local/share/opencode``.
     - The release's config, skills, commands and plugins merge over yours;
       on a key or name both define, the release's wins. Your
       ``AGENTS.md`` is read from ``~/.config/opencode``.
   * - ``codex``
     - ``~/.codex``: ``config.toml`` with your models, providers and MCP
       servers, ``AGENTS.md``, ``auth.json``, ``skills/`` and ``sessions/``.
     - Arguments bind the model, add the release's rules after your
       ``AGENTS.md``, replacing your own ``developer_instructions``, and set
       the release's config keys over yours. The release's skills and
       commands load as the plugin ``reef``, which ``connect`` adds to your
       ``~/.codex`` with ``codex plugin`` and each ``run`` refreshes, so your
       other Codex sessions load them too. ``disconnect`` removes it.

Reef evolves only the layer. It never reads your own configuration, and it
never syncs or evolves it; only Codex's own ``plugin`` commands add an entry
to your ``~/.codex``. Two things follow:

- The agent puts parts of your configuration into its model calls, such as
  your rules and the names and descriptions of your skills. Reef records each
  call as the agent sent it, so the recorded calls hold that text. The calls
  use your Reef token, not your own provider key.
- A step learns from sessions that ran with your configuration, but the
  evaluation runs each release without it. A release that scores better in
  the evaluation was measured without your own rules, skills and settings.

Reef learns from your sessions through reports: a client that grades a run
sends ``POST /reef/report`` with a score over the receipts of its model calls.

A client of its own uses the routes under ``reef-client`` (see `HTTP API
<../reference/http-api.rst#harness-artifacts>`__):

- ``GET /reef/harness`` serves the head release as a JSON manifest:
  ``files`` by render path, ``release_id``, its parent, the ``evaluation``
  metrics of the step that published it.
  ``?release_id=<id>`` serves an older release. ``?form=source`` serves the
  source tree. ``?form=client&service_url=<url>&root=<directory>`` serves
  the release as a layer for an install at ``<directory>``, bound to
  ``<url>/reef/scenarios/<scenario>``, with the command that runs the agent.
- ``POST /reef/harness/import`` merges a person's harness files onto the
  head as a new release.

Promote a pending release
~~~~~~~~~~~~~~~~~~~~~~~~~

Code from the tree runs in the harness's process with your privileges. To
make a person approve such changes, hold them as pending releases:

- ``evolution.publish: review`` holds every win.
- ``evolution.review_parts`` holds only wins whose patch touches the listed
  parts of the adapter's tree. ``[extension]``, as in the tutorial's
  ``deployment.yaml``, publishes rules and config at once while code waits.
  Startup refuses a name that is not a part of the adapter's tree.
- A win that touches a part marked ``review: true`` in the adapter's
  ``evolve`` section always waits. Native's ``native_loop`` is one.

A pending release has ``pending: true`` in ``GET /reef/harness/releases`` and
is served to no client. The loop keeps evolving from it, so promoting the
latest pending release serves everything since the head. Read the change on
its page or with ``GET /reef/harness?release_id=<id>``, then promote it. The
answer is the new head, with a new release id:

.. code:: bash

   curl -sS -H "x-reef-scenario: <scenario>" \
     http://127.0.0.1:8901/reef/harness/releases    # the row with "pending": true
   curl -sS -X POST -H "Content-Type: application/json" \
     -d '{"release_id": "<the pending release id>"}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/promote

Serve the harness as a resident process
---------------------------------------

The native agent can also run as one resident process that follows a
scenario's head. A publish reaches the process between two steps of the open
turn, or at once when no turn is open, without a reinstall or a restart:

.. code:: bash

   mkdir -p reef-harness/native
   printf '{"api": "openai", "base_url": "http://127.0.0.1:8900", "api_key": "%s", "model": "qwen3-8b"}\n' \
     "${REEF_TOKEN:-reef-no-token}" > reef-harness/native/models.json
   reef-native serve --tree ./reef-harness --scenario harness-evolve-demo &
   reef-native turn --tree ./reef-harness -p "fix the failing test in auth.py"
   reef-native status --tree ./reef-harness

``native/models.json`` binds the model to Reef. ``turn`` prints each event of
the turn as a JSON line, and its last line, ``turn/result``, holds the
``receipts`` that a ``POST /reef/report`` references. ``--session <id>``
continues a session.

- ``--follow head`` (the default) mounts each new head. It polls every
  ``--poll-interval`` seconds (60). A tree that fails to load leaves the
  current one in place.
- ``--follow pinned`` logs each new head and waits. ``reef-native mount
  <release_id> --tree ./reef-harness`` applies a release by hand, an older one
  included.
- ``--self-tools`` lets the model inspect its harness and try a patch for
  the rest of a turn. A trial is never published.

The process runs tree code on your machine with your privileges, and the
evaluation's sandbox does not apply to it. Under ``--follow head``, anyone who
can publish to the scenario runs code on that machine. ``--follow pinned``
keeps a person in that loop. `Episode and serve modes
<../developer-guide/harness-adapters.rst#episode-and-serve-modes>`__
describes the process in detail.

Choose or write a method
------------------------

Reef ships no general method and no general episode scorer. The deployment
names one method in ``evolution.method``. The packages in this repository
learn in different ways:

- `GEPA <recipes/gepa.rst>`__ (``recipes/gepa/``): reflective prompt
  evolution. A reflection model rewrites one text file of the tree from the
  feedback of a minibatch of tasks, and the evaluation compares mean scores.
- `Meta-Harness <../../recipes/meta_harness/README.md>`__
  (``recipes/meta_harness/``): a model reads every earlier candidate with its
  scores, and writes a whole new tree.
- `ACE <../../recipes/ace/README.md>`__ (``recipes/ace/``): Agentic Context
  Engineering. A reflector and a curator turn each trace into edits of a
  playbook in ``rules.md``. It learns online, with ``policy: none``.
- `ReasoningBank <../../recipes/reasoning_bank/README.md>`__
  (``recipes/reasoning_bank/``): turns judged successes and failures into
  memory items, each an Agent Skill. It learns online, with ``policy: none``.
- `Reefine <../../recipes/beta/reefine/README.md>`__ (beta,
  ``recipes/beta/reefine/``): answers plain-language requests with skills,
  rules, commands or extensions. A ``floor`` evaluation checks the changed
  harness on a health task.
- The tutorial's own method (``tutorials/evolve-your-harness/harness/``): one
  model call that rewrites ``rules.md`` from failing traces.

To write your own method and scorer, read `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__.

Connect a different agent
-------------------------

An adapter is one file, ``adapter.yaml``. Its ``evolve`` section declares what the agent
takes and where each part is written; its other sections declare how the
binary is launched and how a model binds to it. To connect an agent
that has no adapter yet, read `Harness adapters
<../developer-guide/harness-adapters.rst>`__.

.. seealso::

   `Scenario model configuration <scenario-models.rst>`__ explains
   scenario-specific custom providers for the whole harness evolution model
   pipeline through Reef API Platform.
