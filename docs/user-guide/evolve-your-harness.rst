Evolve your harness
===================

A harness is everything around the model: rules, skills, commands, tools,
config and extension code. Harness evolution improves the harness while the
model weights stay fixed, so it needs no GPU and works with a closed model
endpoint. It fits when the problem is in the text around the model, not when
the model cannot do the task.

A deployment chooses three things:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Chooses
   * - ``evolution.adapter``
     - the agent (see `The harness tree`_)
   * - ``evolution.method``
     - how a step changes the tree (see `Choose or write a method`_)
   * - ``evolution.evaluation``
     - whether a change is published (see `Evaluation: the publication policy`_)

How it works
------------

.. diagram::
   :caption: The client installs the head as a layer over the person's own configuration and sends back what a session changed. Reef records the model calls and reports, runs a training step on a batch of reports or of uploaded train tasks, and publishes a selected candidate as the new head.

   <img class="fig-light" src="../assets/harness-architecture-light.svg" alt="Harness evolution: what runs on a person's machine, what runs in Reef, the routes between them, and the two ways a step gets its traces">
   <img class="fig-dark" src="../assets/harness-architecture-dark.svg" alt="Harness evolution: what runs on a person's machine, what runs in Reef, the routes between them, and the two ways a step gets its traces">

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - Step
     - What happens (the numbers of the figure)
   * - 1
     - ``reef-client harness run`` installs the head and starts the agent on
       the person's own configuration plus the release.
   * - 2
     - The agent's model calls go to ``/reef/scenarios/<scenario>/v1/*``.
       Reef forwards and records each one. When the agent exits, the client
       sends the files that the session changed.
   * - 3
     - The person, or a program that checks the work, scores the session with
       ``POST /reef/report``.
   * - 4
     - When ``evolution.batch_size`` traces wait, a training step starts.
   * - 5
     - The method edits a copy of the head (``step(context)``).
   * - 6
     - Reef checks the new tree. An unchanged or invalid tree skips the step.
   * - 7
     - The candidate and the head play the same tasks, and the policy
       compares their scores.
   * - 8
     - A selected candidate becomes the new head, or waits for a person.

Two ways to evolve
~~~~~~~~~~~~~~~~~~

Both ways feed the same training step (4 to 8).

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - What
     - Online: learn from sessions
     - Offline: learn from uploaded tasks
   * - Traces
     - sessions that people run and score
     - episodes of the next train tasks, run on the head
   * - Evaluation tasks
     - prompts in ``evolution.evaluation.tasks``, or none (``policy: none``)
     - the eval split of the uploaded tasks
   * - Agents
     - ``claude``, ``pi``, ``opencode``, ``codex`` (the adapters with a layer)
     - every adapter with an npm ``install``
   * - Recipes
     - ACE, ReasoningBank, GEPA (AIME example)
     - GEPA with ``TrainTaskRollouts``, Meta-Harness (Terminal-Bench example)

Quickstart
----------

You need a Reef checkout (`Laptop, no GPU
<../getting-started/installation.rst#laptop-no-gpu>`__), ``pip install
reef-client``, a model endpoint that can call tools, and Node with ``npm``.
The offline way also needs Docker. At startup, the deployment installs the
adapter's pinned agent under ``~/.local/share/reef-harness/<adapter>``, or
uses ``evolution.binary``. The tutorial's deployment listens on
``127.0.0.1:8901`` without a token; with one, add ``-H "Authorization:
Bearer $REEF_TOKEN"`` to each ``curl``.

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

   # Score the session over its model calls: 0 fails, 1 succeeds.
   CALLS=$(curl -s "http://127.0.0.1:8901/reef/scenarios/<scenario>/records?request_type=inference&limit=100" \
     | jq -c '[.records[].agent_record_id]')
   curl -s -X POST http://127.0.0.1:8901/reef/report \
     -H "x-reef-scenario: <scenario>" -H "Content-Type: application/json" \
     -d "{\"references\": $CALLS, \"score\": 0, \"feedback\": \"it said the test passed without running it\"}"

   # The steps and their results; a row with "selected": true is a new head.
   curl -s -H "x-reef-scenario: <scenario>" http://127.0.0.1:8901/reef/harness/releases

The next ``pi`` session starts on the new head. `Tutorial: evolve your
agent's harness <../getting-started/harness-tutorial.rst>`__ walks through
these steps.

Offline: learn from uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Change the deployment's evaluation and add a ``pipeline`` section:

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

Then upload Harbor task directories. The first train task starts the run:

.. code:: bash

   tar -czf train.tar.gz -C <tasks-directory> <task-a> <task-b>
   curl -f "http://127.0.0.1:8901/reef/scenarios/<scenario>/tasks?split=train" \
     -H "Content-Type: application/gzip" --data-binary @train.tar.gz
   tar -czf eval.tar.gz -C <tasks-directory> <task-c>
   curl -f "http://127.0.0.1:8901/reef/scenarios/<scenario>/tasks?split=eval" \
     -H "Content-Type: application/gzip" --data-binary @eval.tar.gz

Run the example
~~~~~~~~~~~~~~~

The example runs the online way without a person:

.. code:: bash

   export REEF_UPSTREAM_API_KEY=sk-...    # only if your endpoint needs one
   cd tutorials/evolve-your-harness && ./run.sh

``configs/serve.yaml`` holds the endpoint, the model and the token. Three
exact-answer tasks go through Reef, and each graded reply is one step. The
run succeeds when a row of ``/reef/harness/releases`` shows ``selected:
true``. When both sides score nothing, the episodes could not run: vLLM, for
example, needs ``--enable-auto-tool-choice --tool-call-parser hermes``. See
`Troubleshooting <troubleshooting.rst>`__.

The harness tree
----------------

A harness tree is one directory of text files. Each release holds one tree,
and each step changes it with a patch: paths mapped to their new text, or to
``null`` for a removal. Every adapter uses the same tree path for the same
part:

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

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Adapter
     - Agent
     - Parts
   * - ``pi``
     - pi coding agent
     - rules, skill, command, extension, config
   * - ``opencode``
     - OpenCode
     - rules, skill, command, extension, config
   * - ``claude``
     - Claude Code
     - rules, skill, command, extension, memory, config
   * - ``codex``
     - Codex CLI
     - rules, skill, command, config
   * - ``dsh``
     - DeepSeek Harness
     - rules, skill, command, extension, config
   * - ``hermes``
     - Hermes Agent
     - rules, skill, command, extension, config
   * - ``native``
     - Reef's own loop
     - rules, skill, config, and the five ``native_*`` parts

Reef refuses a file at a path that no part takes. A tree never chooses where
model calls go: Reef writes the endpoint, key and model at render time. The
adapter guide gives each adapter's `render paths
<../developer-guide/harness-adapters.rst#render-paths>`__ and `model binding
checks <../developer-guide/harness-adapters.rst#model-binding-checks>`__.

The loop
--------

``T`` is a harness tree, and ``A`` is the deployment's adapter.

.. list-table::
   :header-rows: 1
   :widths: 14 60 26

   * - Part
     - What it does
     - Details
   * - Data
     - Records and uploaded tasks give ``D``, one step's traces or train
       tasks.
     - `Step data`_
   * - Evolve
     - ``method(T, D, s) = (T', s')``: the candidate ``T'`` and the method's
       new state.
     - `Write a harness method <../developer-guide/write-a-harness-method.rst>`__
   * - Evaluation
     - Episodes of ``T'`` and ``T`` on the same tasks; the policy selects or
       rejects ``T'``.
     - `Evaluation: the publication policy`_
   * - Release
     - A selected ``T'`` becomes a git release and the head.
     - `Promote a pending release`_
   * - Sync
     - A client installs ``render_A(T)`` and sends back what a session
       changed; Reef merges it onto the head.
     - `Use a published tree`_

Step data
~~~~~~~~~

A step starts when ``evolution.batch_size`` traces wait (default 1), or
``pipeline.sample2job.config.job_samples`` when set. One trace is one scored
report, with every model call it names. To learn from inference traffic
without reports, name the inference stages; each call is then one unscored
trace:

.. code-block:: yaml

   pipeline:
     record2record: {module: reef.record2record:InferenceRecord2Record}
     record2sample: {module: reef.record2sample.traces:InferenceSamples}

Offline runs on uploaded tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.sample2job.traces:TrainTaskRollouts`` makes each step from the train
split of the ``imported-tasks`` store:

- Each step runs the next ``tasks_per_step`` train tasks (default 8) on the
  head, and the method reads those episodes.
- The run goes through the train split ``epochs`` times (default 1), in a new
  order each time, then stops. A restart continues after the last committed
  step.
- The eval split is the evaluation's task source. The stage needs a scorer
  and task directories (see `Task directories`_).

Upload tasks with ``POST /reef/scenarios/<scenario>/tasks?split=train`` or
``?split=eval`` (`Upload Harbor tasks
<../reference/http-api.rst#upload-harbor-tasks>`__), or import them with the
deployment's ``task-import`` section.

Evaluation: the publication policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evolution.evaluation`` names the tasks, the ``scorer`` (an
``EpisodeScorer``) and the ``policy``. The tasks come from ``tasks`` (prompts
or task directories), ``task_manifest`` with ``tasks_root``, or
``task_store``. A prompt episode starts in an empty root, so a prompt must
state the whole problem.

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
     - every episode scores at least ``floor_score``
   * - ``none``
     - nothing
     - always; it needs no tasks and no scorer

An episode that could not run never helps the candidate, and a tie publishes
nothing. Reef keeps every scored episode and reuses it for the same tree,
task and repeat, so a step usually runs only the candidate's episodes.
`Harness evolution keys <../reference/configuration.rst#harness-evolution-keys>`__
lists every key.

Running episodes
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``evolution.executor``
     - Where an episode runs
   * - ``local`` (default)
     - a plain subprocess, with no isolation; for development
   * - ``sandbox``
     - a bubblewrap jail: read-only base, explicit credentials only, no
       network unless ``evolution.sandbox.egress_hosts`` is set
   * - ``e2b``, ``modal``
     - one cloud sandbox per episode (``reef-infra[e2b]`` or
       ``reef-infra[modal]``), with outbound internet

``evolution.sandbox.env_from`` lists the only variables an episode gets. See
`Sandbox environments <../developer-guide/sandboxes.rst>`__.

Task directories
^^^^^^^^^^^^^^^^

A Harbor task directory holds an instruction, an environment and tests. Reef
runs each one in its own container: Harbor starts it, Reef installs the
adapter's pinned agent with npm and copies in the rendered tree, the agent
plays the instruction, and the tests' reward is the score
(``VerifierRewardScorer``). The image needs glibc.

The agent reaches Reef at ``evolution.evaluation.agent_host`` (default
``host.docker.internal``) in place of a loopback address. On Linux, set it
to an address the container reaches, such as the Docker bridge gateway.
Startup refuses task directories when the adapter has no npm ``install``,
when the executor is not ``local``, or with ``collect_failures``. An offline
run's evaluation tasks must be directories too.

Collection: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``collect_failures: true`` makes the prompts of each step's traces permanent
evaluation tasks, so a later candidate cannot bring an old failure back. Reef
screens each prompt, and caps them at ``max_collected_per_client`` (5) and
``max_collected_tasks`` (50). A ``TaskCollector`` chooses which prompts
(`Collect traffic prompts
<../developer-guide/write-a-harness-method.rst#collect-traffic-prompts>`__).

History
~~~~~~~

- ``GET /reef/harness/releases`` has a row per step: the patch, the scores,
  the method's summary and its cost. ``/reef/harness/releases/<step>/page``
  explains one step.
- ``<evolution.state_dir>/<scenario>/harness/`` keeps ``history/`` (every
  tree, episode and decision; the score cache) and ``steps/<step>/`` (each
  workspace and its model calls). Nothing prunes it, and it holds client
  traffic: protect it.

Use a published tree
--------------------

``reef-client harness`` runs the agent on your machine with the head as a
layer over your own configuration, and routes its model calls to Reef. The
deployment's ``evolution.adapter`` decides the agent; ``GET /reef/scenarios``
lists each harness with its ``adapter``.

.. code:: bash

   pip install reef-client
   reef-client harness connect --reef http://127.0.0.1:8900

.. flow::
   :loop: each run starts on the newest release

   connect :: sets the machine up, once
   run :: syncs, starts the agent on the layer, and sends its changes when it exits
   Reef* :: records each model call; steps publish releases

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Command
     - What it does
   * - ``connect``
     - Installs the head under ``~/.local/share/reef-client/harness/<scenario>``,
       installs the pinned agent when the machine lacks it, and makes the
       agent's own command run through Reef. It asks only for a token and a
       harness when it must.
   * - ``run``
     - Syncs (waiting at most 2 s for Reef), starts the agent on the layer,
       and sends the session's changes from the background when it exits.
       Arguments after ``--`` go to the agent.
   * - ``sync``
     - Sends your changed files, then installs the head. Reef merges them by
       each part's ``merge``: text by line, code by file. ``--discard`` drops
       them.
   * - ``status``
     - Shows the installed release, the head and the changed files.
   * - ``disconnect``
     - Removes the install and the agent command.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Agent
     - How the release goes over your own configuration
   * - ``claude``
     - Skills and commands as the plugin ``reef`` (``reef:<name>``); rules
       after your ``CLAUDE.md``; settings, extension hooks and automatic
       memory from the release.
   * - ``pi``
     - Skills, prompts and extensions beside yours; rules after your
       ``AGENTS.md``; settings as ``--thinking`` and ``--no-builtin-tools``.
   * - ``opencode``
     - Config, skills, commands and plugins merged over yours.
   * - ``codex``
     - Skills and commands as the plugin ``reef``, which ``connect`` adds to
       ``~/.codex``; rules and config keys as ``-c`` arguments.

Reef never reads your own configuration, and evaluates each release without
it. Your configuration still reaches the model calls that Reef records, such
as your rules and the names of your skills.

A client of its own uses ``GET /reef/harness`` (``?release_id=`` for an
older release, ``?form=source``, or ``?form=client&service_url=&root=`` for a
layer) and ``POST /reef/harness/import`` (see `HTTP API
<../reference/http-api.rst#harness-artifacts>`__).

Promote a pending release
~~~~~~~~~~~~~~~~~~~~~~~~~

Tree code runs with your privileges. To make a person approve a change:

- ``evolution.publish: review`` holds every win.
- ``evolution.review_parts: [extension]`` holds only wins that touch those
  parts.
- A part marked ``review: true``, such as native's ``native_loop``, always
  waits.

A pending release has ``pending: true`` and is served to no client.
``promote`` serves it; ``rollback`` serves an earlier release again. Each
answer is the new head, and clients install it at their next sync:

.. code:: bash

   curl -sS -X POST -H "Content-Type: application/json" -d '{"release_id": "<id>"}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/promote     # or .../rollback

Serve the harness as a resident process
---------------------------------------

The native agent can run as one process that follows the scenario's head and
mounts each new release between two steps:

.. code:: bash

   mkdir -p reef-harness/native
   printf '{"api": "openai", "base_url": "http://127.0.0.1:8900", "api_key": "%s", "model": "qwen3-8b"}\n' \
     "${REEF_TOKEN:-reef-no-token}" > reef-harness/native/models.json
   reef-native serve --tree ./reef-harness --scenario harness-evolve-demo &
   reef-native turn --tree ./reef-harness -p "fix the failing test in auth.py"

``--follow pinned`` waits for ``reef-native mount <release_id>`` instead of
mounting each head. The process runs tree code with your privileges, outside
the evaluation's sandbox. See `Episode and serve modes
<../developer-guide/harness-adapters.rst#episode-and-serve-modes>`__.

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
     - A model reads every earlier candidate and its scores, and writes a new
       tree.
     - ``compare``
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
