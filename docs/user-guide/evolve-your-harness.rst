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

The harness tree
----------------

A harness tree is one directory of text files: the mutable, versioned part of
one harness. Each release holds one tree, and each step changes it with a
patch: paths mapped to their new text, or to ``null`` for a removal. The
layout fixes where each kind of file lives, so the same tree means the same
thing to every agent:

+-------------------------------+--------------------+-----------------------------------------------+
| Path                          | Kind               | Holds                                         |
+===============================+====================+===============================================+
| ``rules.md``                  | ``rules``          | the text every session reads                  |
+-------------------------------+--------------------+-----------------------------------------------+
| ``config/<target>.json``      | ``config``         | a JSON object merged into one of the agent's  |
|                               |                    | config files                                  |
+-------------------------------+--------------------+-----------------------------------------------+
| ``skills/<name>/SKILL.md``    | ``skill``          | an Agent Skill                                |
+-------------------------------+--------------------+-----------------------------------------------+
| ``commands/<name>.md``        | ``agent_command``  | a prompt template a person types as a command |
+-------------------------------+--------------------+-----------------------------------------------+
| ``extensions/<name>.<ext>``   | ``code_extension`` | code the harness loads in its process         |
+-------------------------------+--------------------+-----------------------------------------------+
| ``native/tools/<name>.py``    | ``native_tool``    | a tool of the native loop                     |
+-------------------------------+--------------------+-----------------------------------------------+
| ``native/hooks/<name>.py``    | ``native_hook``    | a listener at one event of the native loop    |
+-------------------------------+--------------------+-----------------------------------------------+
| ``native/graphs/<name>.json`` | ``native_graph``   | the native loop's control flow                |
+-------------------------------+--------------------+-----------------------------------------------+
| ``native/agents/<name>.json`` | ``native_agent``   | one agent the native loop delegates to        |
+-------------------------------+--------------------+-----------------------------------------------+
| ``native/loops/<name>.py``    | ``native_loop``    | the native loop itself as code                |
+-------------------------------+--------------------+-----------------------------------------------+

An adapter maps each kind to the files of one agent. This is called
rendering. Each adapter accepts only some kinds, and Reef refuses a tree that
holds a path off the layout or a kind the adapter does not accept. The
bundled adapters accept these kinds:

+--------------+-----------------------------+---------------------------------------------+
| Adapter      | Agent                       | Kinds it renders                            |
+==============+=============================+=============================================+
| ``pi``       | pi coding agent             | config, rules, agent_command, skill,        |
|              |                             | code_extension                              |
+--------------+-----------------------------+---------------------------------------------+
| ``opencode`` | OpenCode                    | config, rules, agent_command, skill,        |
|              |                             | code_extension                              |
+--------------+-----------------------------+---------------------------------------------+
| ``claude``   | Claude Code                 | config, rules, agent_command, skill,        |
|              |                             | code_extension                              |
+--------------+-----------------------------+---------------------------------------------+
| ``codex``    | Codex CLI                   | config, rules, agent_command, skill         |
+--------------+-----------------------------+---------------------------------------------+
| ``dsh``      | DeepSeek Harness            | config, rules, agent_command, skill,        |
|              |                             | code_extension                              |
+--------------+-----------------------------+---------------------------------------------+
| ``hermes``   | Hermes Agent                | config, rules, agent_command, skill,        |
|              |                             | code_extension                              |
+--------------+-----------------------------+---------------------------------------------+
| ``terminus`` | Terminus 2 (Terminal-Bench) | config, rules, agent_command, skill,        |
|              |                             | code_extension (sandbox and remote tasks)   |
+--------------+-----------------------------+---------------------------------------------+
| ``native``   | Reef's own loop             | config, rules, skill, native_tool,          |
|              |                             | native_hook, native_graph, native_agent,    |
|              |                             | native_loop                                 |
+--------------+-----------------------------+---------------------------------------------+

With ``native``, the method can change the agent's tools, its hooks, its
control flow, and the loop itself, not only the text around a vendor binary.
`Harness adapters <../developer-guide/harness-adapters.rst>`__ gives the
rendered paths of each adapter and the rules of each native file.

A tree does not choose where model calls go. Reef writes the endpoint, the
key and the model when it renders the tree for an episode. With most
adapters, render refuses a tree that sets them itself. The adapter guide's
`Model binding checks <../developer-guide/harness-adapters.rst#model-binding-checks>`__
list the keys for each adapter.

The loop
--------

Harness evolution advances one step at a time:

.. flow::
   :loop: publish a selected candidate; the head stays otherwise

   Job data :: the traces of one step, or a queued request, or both
   Method :: edits a copy of the head tree in the step's workspace
   Evaluation* :: scores the candidate under the publication policy
   Result :: publish the candidate as a new release, or keep the head

A step starts from recorded traffic, from a training request
(``POST /reef/train``), or from both. ``recipe.config.training_mode``
chooses which:

- ``auto`` (the default): steps start from recorded traffic alone. Training
  requests are rejected.
- ``manual``: steps start from training requests alone. Reef still records
  traffic, but uses it only after a switch to ``auto`` or ``hybrid``.
- ``hybrid``: steps start from either. A queued request goes first, one per
  step, and that step also gets up to ``job_samples`` waiting traces.

``POST /reef/scenarios/{scenario}/update`` switches a running deployment
between the modes. `Manual training
<../reference/http-api.rst#manual-training>`__ documents the request fields,
the screens a request's text passes, and what happens when its step fails.

In ``auto`` and ``hybrid`` mode, a step starts when
``pipeline.sample2job.config.job_samples`` traces wait. Without that key, it
is ``evolution.batch_size`` (default 1). By default, one trace is one valid
scored report, a passing report included. A report over several model calls
is one trace that holds every call in order. Reef keeps at most
``pipeline.sample2job.config.buffer_size`` waiting traces (default four times
``job_samples``) and drops the oldest ones beyond that, in every mode.

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
         adapter: terminus
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
task starts the run. The stage needs an adapter that runs task directories
(``terminus``) and a scorer, and startup refuses it without them. It runs in
``auto`` and ``hybrid`` mode; a ``manual`` scenario refuses it. Recorded
traffic and training requests still start steps, and they go first. After a
restart, the run continues after the last committed step. A task uploaded
during an epoch joins the next epoch.

The method
~~~~~~~~~~

Reef runs the method once per step in a workspace. The workspace holds a
copy of the head tree, the step's traces, the request when the step answers
one, the evaluation tasks, and the method's files from earlier steps. The
tree that the method leaves there is the candidate. Reef skips the step when
the tree is unchanged or fails validation for the adapter; there is no retry.
`Write a harness method <../developer-guide/write-a-harness-method.rst>`__
documents the workspace, the tools, and the method's state.

Two kinds of step do not run the method: a recheck (see `Rechecking a
published tree`_), and a step that evaluates a proposal that a client sent
to ``POST /reef/harness/proposals``.

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

Task directories need an adapter that runs them, such as ``terminus``. An
episode runs in a fresh, empty root with only the rendered tree: no
repository and no files from your machine. A prompt task must therefore state
the whole problem.

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

``evolution.sandbox.env_from`` lists the environment variables to pass into
an episode; a missing variable fails configuration. No other variable
reaches the agent, and Reef keeps the E2B key and the Modal token to itself.
A public model endpoint needs no forwarded port. For an npm-installed
adapter, Reef builds the cloud image with the agent. Any other adapter needs
``e2b_template`` or ``modal_image`` with its binary and bubblewrap
installed, and ``evolution.binary`` then names a path inside that image.
Startup checks provider access, the image, the binary, and namespace
support. `Sandbox environments <../developer-guide/sandboxes.rst>`__
describes the shared execution layer. The beta Reefine recipe sets where its
own coding agent runs with ``evolution.agent.sandbox``, apart from
``evolution.executor``.

Collection: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``evolution.evaluation.collect_failures: true``, the prompts of each
step's traces become permanent evaluation tasks, so a later candidate cannot
win while it brings an earlier failure back. The scorer must then score any
prompt, and the adapter's task must be a prompt.

A collected prompt is client traffic, so Reef screens it first. It never
collects a prompt that holds a credential-shaped literal or an instruction
override, such as ``ignore the previous instructions``. One tagged client
adds at most ``max_collected_per_client`` tasks (default 5), and
``max_collected_tasks`` (default 50) caps the total. To choose which prompts
to collect, for example only those of failing traces, name a
``TaskCollector`` in ``evolution.evaluation.collector`` (see `Collect traffic
prompts <../developer-guide/write-a-harness-method.rst#collect-traffic-prompts>`__).

Rechecking a published tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A tree passed the evaluation as the suite stood when it was published. A
grown suite can later show it as a regression. With
``evolution.evaluation.recheck_every: N`` (``compare`` only; 0, off, by
default), every ``N`` steps the loop runs fresh episodes of the last published
tree against the tree it replaced, instead of running the method. It also
rechecks at once when the evaluation context changed since the publish. If
the older tree now wins, the loop publishes it again, which rolls the
deployment back. Only the most recent publish can be rolled back this way.

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

Before you start
----------------

- ``pip install reef-client``: the example's driver imports it.
- An OpenAI-compatible endpoint that serves the model, hosted or local. It
  must accept tool calls.
- Node and ``npm``. At startup, the deployment installs the adapter's pinned
  ``pi`` under ``~/.local/share/reef-harness/pi``. ``REEF_HARNESS_PREFIX``
  moves that root, and ``evolution.binary`` names a binary of your own
  instead. A missing ``npm`` or a failed install refuses startup.

Run the example
---------------

From a Reef checkout:

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
shared machine, export ``REEF_TOKEN`` before ``run.sh`` starts it. Every
``curl`` on this page then needs ``-H "Authorization: Bearer $REEF_TOKEN"``.

The run has succeeded when a step's row shows ``selected: true`` and
``GET /reef/harness`` serves a release other than the seed. A step that runs
but does not win leaves the seed in place; that is a normal result. If no step
publishes after a few minutes, read the step's row. Both sides scoring
nothing usually means the episodes could not run: vLLM, for example, needs
``--enable-auto-tool-choice --tool-call-parser hermes``.
`Troubleshooting <troubleshooting.rst>`__ lists the other causes.

To run a long-lived deployment without the example's driver, start
``configs/deployment.yaml`` from the repository root. It listens on
``127.0.0.1:8901``, runs in ``hybrid`` mode, and keeps state under
``tutorials/evolve-your-harness/work/deployment/``:

.. code:: bash

   export REEF_UPSTREAM_URL=http://127.0.0.1:11434 REEF_UPSTREAM_MODEL=gemma4:26b
   reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml

Use a published tree
--------------------

``GET /reef/harness`` serves the head release as a JSON manifest. ``files``
maps each path that the adapter renders to its text. Beside it are the
``release_id``, its parent, the ``evaluation`` metrics of the step that
published it, and ``requires``. A fresh scenario serves the recipe's seed, so
a client can pull before any step runs:

.. code:: bash

   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness | python3 -m json.tool | head

``?release_id=<id>`` serves an older release, which pins or rolls back one
client without moving the head. ``?form=source`` serves the source tree
instead, for a client that renders the tree itself or proposes changes to it.

The served files carry no endpoint and no credential. A client writes them
where its agent reads them and binds the agent's model to this Reef. Every
model call must carry the ``x-reef-scenario`` header and the token. Configure
the agent's provider to send them (pi sends ``providers.reef.headers`` from
``pi-agent/models.json``), or run a local proxy in front of Reef that adds
them. Reef answers each call with a receipt.

After a run, send ``POST /reef/report`` with a score and feedback over the
receipts of its model calls. A release's ``requires`` items name what the
machine must provide: an ``env`` variable, a ``permission``, a ``service``,
or a ``binary``, each with an optional ``check`` and a ``prompt`` sentence.
The client checks them before it runs the release. ``requires`` in the
manifest covers the release's whole chain of changes.
``recipes/beta/reefine/examples/run.py`` is a worked client.

Ask for a harness change
~~~~~~~~~~~~~~~~~~~~~~~~

You can ask for a change in plain words. A scenario in ``manual`` or
``hybrid`` mode takes training requests. Switch a scenario in ``auto`` mode
first:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"training_mode": "hybrid"}' \
     "$REEF_URL/reef/scenarios/code-repair/update"
   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"text": "run the tests before you report a fix as done", "session": "me", "release_id": ""}' \
     "$REEF_URL/reef/train"

The answer carries the request's id and ``page_path``, a page that a browser
opens as is. Acceptance does not mean that the change passed the evaluation.
The page follows the step while it runs, then shows the result, the method's
summary, the patch, and the next action.
``GET /reef/harness/requests/<id>/progress`` gives the same state as JSON.

Every method can read the request, but how well it answers one depends on the
method. The beta `Reefine <../../recipes/beta/reefine/README.md>`__ recipe is
made for requests. Its example, ``recipes/beta/reefine/examples/``, runs the
path from the request to a session on the new tree on one machine.

Promote a pending release
~~~~~~~~~~~~~~~~~~~~~~~~~

Code from the tree runs in the harness's process with your privileges. To
make a person approve such changes, hold them as pending releases:

- ``evolution.publish: review`` holds every win.
- ``evolution.review_kinds`` holds only wins whose patch touches the listed
  kinds. ``[code_extension]``, as in the tutorial's ``deployment.yaml``,
  publishes rules and config at once while code waits.
- A win that touches a ``native_loop`` always waits.

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
- ``--self-tools`` lets the model inspect its harness, try a patch for the
  rest of a turn, and propose a patch to Reef, which evaluates it like a
  method's candidate.

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
  evolution. A reflection model rewrites one part of the tree from the
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

An adapter is one descriptor file: where each kind of tree file is written,
how the binary is launched, and how a model binds to it. To connect an agent
that has no adapter yet, read `Harness adapters
<../developer-guide/harness-adapters.rst>`__.

.. seealso::

   `Scenario model configuration <scenario-models.rst>`__ explains
   scenario-specific custom providers for the whole harness evolution model
   pipeline through Reef API Platform.
