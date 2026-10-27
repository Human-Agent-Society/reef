Evolve your harness
===================

A harness is everything around the model: the control loop, rules, prompt
templates, skills, tools, config, and extension code. Together, the model and
harness form an agent. Harness evolution improves the harness tree while the
model weights stay fixed. Reef needs no GPU for this. The model stays a fixed
endpoint, hosted or local, and the agent stays online throughout.

Reef supplies the mechanism: it reads the head tree, asks for a patch, runs
the candidate and the head on the same tasks, and publishes a win as a new
release. You supply two Python classes: a ``Proposer`` (which change to try)
and an ``EpisodeScorer`` (how an episode scored). The deployment names them in ``evolution.propose`` and
``evolution.evaluate``. `Write a harness method <../developer-guide/write-a-harness-method.rst>`__ documents the
contract.

At a glance
-----------

+-------------------+--------------------------------------------------------------+
| What evolves      | Config, rules, prompt templates, skills, and extension code; |
|                   | on the native harness also its tools, its hook listeners,    |
|                   | the graph or the loop code that is its control loop, and the |
|                   | agents the loop delegates to.                                |
+-------------------+--------------------------------------------------------------+
| Which agents      | pi, opencode, Claude Code, Codex, DeepSeek Harness, Hermes   |
|                   | Agent, Terminus 2, and ``native``, Reef's own loop. One      |
|                   | adapter per agent maps the tree onto that agent's files.     |
+-------------------+--------------------------------------------------------------+
| What decides      | The candidate and the current tree run the same tasks in     |
|                   | fresh roots. The candidate is published only when it wins    |
|                   | more tasks than it loses, by more than a margin you set.     |
+-------------------+--------------------------------------------------------------+
| What holds it     | A rejected candidate publishes nothing. A failing            |
|                   | prompt joins the task suite only after a credential and an   |
|                   | instruction override screen, with a cap per client. Wins     |
|                   | that touch code can wait for a person before they are        |
|                   | served. A periodic recheck rolls back a publish that a       |
|                   | grown suite now scores as a regression. Episodes can run in  |
|                   | a sandbox with no host credentials and no network beyond the |
|                   | model endpoint.                                              |
+-------------------+--------------------------------------------------------------+
| Where to start    | `Run the example <#run-the-example>`__ evolves one skill on  |
|                   | ``pi`` with a local model and no GPU. The tutorial's         |
|                   | ``serve-native.yaml`` runs the same loop on the native       |
|                   | harness, where the model proposes tools, hooks, and graph    |
|                   | changes to its own loop.                                     |
+-------------------+--------------------------------------------------------------+

The harness tree
----------------

A harness tree is one directory of text files: the mutable, versioned part of
one harness. Each release in the scenario's git history holds one tree, and
each step changes it with a patch, a set of paths mapped to their new text, or
to ``null`` for a removal. The layout fixes where each kind of file lives, so
the same tree means the same thing to every adapter
(`reef/harness/tree/layout.py <../../reef/harness/tree/layout.py>`__):

+------------------------------+--------------------+-----------------------------------------------+
| Path                         | Kind               | Holds                                         |
+==============================+====================+===============================================+
| ``rules.md``                 | ``rules``          | the text every session reads                  |
+------------------------------+--------------------+-----------------------------------------------+
| ``config/<target>.json``     | ``config``         | a JSON object deep-merged into one of the     |
|                              |                    | agent's config files                          |
+------------------------------+--------------------+-----------------------------------------------+
| ``skills/<name>/SKILL.md``   | ``skill``          | an Agent Skill                                |
+------------------------------+--------------------+-----------------------------------------------+
| ``commands/<name>.md``       | ``agent_command``  | a prompt template a person types as a command |
+------------------------------+--------------------+-----------------------------------------------+
| ``extensions/<name>.<ext>``  | ``code_extension`` | code the harness loads in process; the        |
|                              |                    | adapter names the suffix                      |
+------------------------------+--------------------+-----------------------------------------------+
| ``native/tools/<name>.py``   | ``native_tool``    | a tool of the native loop                     |
+------------------------------+--------------------+-----------------------------------------------+
| ``native/hooks/<name>.py``   | ``native_hook``    | a listener at one event of the native loop    |
+------------------------------+--------------------+-----------------------------------------------+
| ``native/graphs/<name>.json``| ``native_graph``   | the native loop's control flow                |
+------------------------------+--------------------+-----------------------------------------------+
| ``native/agents/<name>.json``| ``native_agent``   | one agent the native loop delegates to        |
+------------------------------+--------------------+-----------------------------------------------+
| ``native/loops/<name>.py``   | ``native_loop``    | the native loop itself as code                |
+------------------------------+--------------------+-----------------------------------------------+

A path off this layout is refused. An adapter maps each kind to a concrete
file for one agent. Turning the tree into those files is called rendering,
and it is more than a copy: each config file merges over the adapter's
enforced defaults, and the adapter's quirks rewrite some files (a skill's
frontmatter, say). See the adapter guide's `Rendering
<../developer-guide/harness-adapters.rst#rendering>`__ section for the
details.

Reef bundles adapters for third-party coding agent CLIs (``pi``, ``opencode``,
``claude``, ``codex``, ``dsh`` (DeepSeek Harness), and ``hermes`` (Hermes
Agent)), for ``terminus``, Terminal-Bench's Terminus 2 run through a
Reef-owned Harbor runner, and for ``native``, Reef's own agent.

An adapter need not render every kind, and a file of a kind it leaves out is
refused at validation and at render rather than dropped, so the adapter decides
which files a tree may hold. Every bundled adapter renders ``rules``, ``config``
and ``skill``, so a change to one of those renders under any harness.

Every adapter except ``native`` renders ``agent_command``; a Codex command is a
skill invoked as ``$name``. ``native`` renders neither ``agent_command`` nor
``code_extension``, because the two presuppose a vendor binary: its loop reads
no prompt directory and loads no vendor plugin.

``code_extension`` is narrower still. Codex rejects it because lifecycle hooks
run outside its command sandbox. Terminus accepts one Python module defining
``Agent(Terminus2)``, and only when Reef's sandbox isolates the runner and
Harbor runs the tasks remotely (``e2b`` or ``modal``); see the adapter guide for the required
deployment settings.

Native files
~~~~~~~~~~~~

``native`` is a loop inside Reef rather than a vendor binary, so its five kinds
are the loop itself: its tools, the listeners at its loop events, its control
flow, the agents it delegates to, and the loop as code. The agent can
therefore evolve the tools it runs, how its loop reacts, the loop itself, and
who it delegates to, not only the text around a vendor binary. No other
adapter renders these kinds.

A tool file defines ``run(args, workdir) -> str`` and ends with the constants
``NAME``, ``DESCRIPTION``, ``PARAMETERS`` (a JSON schema) and ``CAPABILITIES``;
a hook file defines ``listen(payload, next)`` and ends with ``NAME`` and
``EVENT``. The loop reads those constants without running the module, and
``NAME`` must be the file name. A graph or agent file is a JSON object whose
``name`` is the file name, and validation refuses a graph or an agent that
names a tool, skill, graph or agent the tree does not hold, or agents that
delegate in a cycle.

A ``native_loop`` file goes one step past a graph: it defines
``run_turn(ctx)``, and that function runs the root turn in place of the graph
interpreter. ``ctx`` is the context API Reef owns: ``ctx.model()`` takes one
model step, ``ctx.run_tools()`` runs the last message's tool calls,
``ctx.agent(name)`` hands text to an agent and returns its answer,
``ctx.say(text)`` adds a user message, ``ctx.end(reason)`` ends the turn, and
``ctx.log(event, data)`` writes a ``loop/<event>`` line to the session. Hooks,
tools, budgets, the session log and the sandbox stay Reef's; a loop that
raises, or calls the context past its budget of transitions, ends the turn
with ``LOOP_ERROR`` and fails the checks, and the first end of a turn is final
whatever the loop code catches. A tree holds at most one loop. Loop code runs
inside the loop's own process, so a win that creates, changes or removes a
loop is always a pending release a person promotes, whether or not
``evolution.review_kinds`` names the kind, and ``harness_try`` refuses to
mount one on a serving process: the model proposes a loop, a person serves it.

Rendered paths
~~~~~~~~~~~~~~

With the ``pi`` adapter, ``GET /reef/harness`` serves:

.. code:: text

   pi-agent/
     settings.json             <- config/primary.json
     models.json               <- config/models.json
     AGENTS.md                 <- rules.md
     prompts/<name>.md         <- commands/<name>.md
     skills/<name>/SKILL.md    <- skills/<name>/SKILL.md
     extensions/<name>.ts      <- extensions/<name>.ts

``GET /reef/harness?form=source`` serves the tree itself. The adapter guide
gives the paths each other adapter renders to.

Model endpoints stay outside the tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A tree does not choose where model calls go. Reef's model binding writes the
endpoint, the key and the model when it renders an episode, and with the
``claude``, ``dsh``, ``hermes``, ``pi`` and ``terminus`` adapters, render
refuses a config file, a command or a skill that sets them or names another
provider, a transport, a proxy, a credential helper, a fallback or another
model, also inside a request body the tree passes to the endpoint. These
checks read the config the tree renders, not the requests a run sends, so a
request that a tool or a plugin builds itself can still name another model.
The adapter guide lists the keys for each adapter.

The loop
--------

Harness evolution advances one step at a time. A step is a single pass of
the cycle described above: Reef reads the head tree from git, asks
``propose`` for a patch, applies it to get the candidate tree, runs the
candidate and the head over the same tasks, and publishes the candidate as a
new release when it wins. The step is the unit the rest of this page counts
in.

.. flow::
   :loop: publish the winner; the head stays otherwise

   Job data :: the units of one step, or a queued request, or both
   ``propose`` :: one patch to the head tree, or ``None``
   Episodes* :: run the candidate and the head on the same tasks
   Result :: publish the candidate as a new release, or keep the head

A step begins either from job data of recorded traffic or from a training
request (i.e. ``POST /reef/train`` requests). ``data.training_mode`` in the recipe config
decides which of the two a deployment accepts.

- ``auto`` (the default): steps start from job data of recorded traffic
  alone. Training requests are ignored.
- ``manual``: steps start from training requests alone. Ordinary
  traffic never starts evolution.
- ``hybrid``: steps start from either, as the tutorial's ``deployment.yaml``
  sets. Training requests are queued and take precedence, and automatic steps continue
  between those requests.

``POST /reef/scenarios/{scenario}/update`` switches a running deployment
between the three.

Trigger of step
~~~~~~~~~~~~~~~

``auto`` mode
^^^^^^^^^^^^^

In ``auto`` mode, two settings decide when Reef starts a step:

- ``pipeline.task2job.config.job_samples``: when this number of traces is
  collected, Reef starts a step. Without the key, it is
  ``recipe.config.evolution.batch_size`` (default 1).
- The ``pipeline.record2record`` and ``pipeline.record2task`` stages decide
  what counts as one trace toward ``job_samples``.

The recipe's default stages count reports. Every valid scored report with
at least one unique inference reference counts as one trace, including
successful outcomes. A report over one receipt trains as that exchange. A
report over several receipts trains as one trajectory sample that carries
every referenced exchange in order, which is how a client reports a whole
run. Reports that reference inferences trained in earlier steps are
dropped.

To count recorded inference traffic instead, name the inference stages.
Every inference then counts as one trace. No report is necessary: recorded
traffic alone trains, unscored, for methods that judge for themselves.

.. code-block:: yaml

   recipe:
     config:
       evolution:
         batch_size: 4
   pipeline:
     record2record:
       module: reef.record2record:InferenceRecord2Record
     record2task:
       module: reef.record2task.traces:InferenceTasks

Reef holds the traces that wait for a step in a buffer. The buffer keeps at
most ``pipeline.task2job.config.buffer_size`` traces (default four times
``job_samples``). When more traces arrive, Reef releases the oldest ones. This
limit applies in every mode.

Under ``auto`` mode, training requests are rejected.

``manual`` mode
^^^^^^^^^^^^^^^

In ``manual`` mode, a step is triggered by every training request. These
requests are queued against the scenario, and the next step runs the oldest
queued request first, one per step. See `manual training API
<../reference/http-api.rst#manual-training>`__ for more details of this API,
e.g. what fields a request carries, the screens its text passes, and what
happens to a request whose step fails.

The ``pipeline`` stages trigger no evolution in this mode, but traces are
still collected while training requests run. However, these traces are never
used in evolution until ``data.training_mode`` is switched to ``auto`` or
``hybrid``. The buffer limit still applies: Reef releases the oldest traces
beyond ``buffer_size``.

``hybrid`` mode
^^^^^^^^^^^^^^^

In ``hybrid`` mode, either condition starts a step: ``job_samples`` traces
collected by the ``pipeline`` stages, or a training request received.

When both conditions are met, the training request takes precedence for the
next step. Reef builds one step that uses both, whether or not
``job_samples`` is met. The proposer then
receives the training request and the traces that automatic job data would
take next. It gets up to ``job_samples`` traces, or none, and uses both
to make a patch.

Propose: a patch to the head tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reef calls the proposer once per step, with the head tree, the step's trace
samples, the request when the step answers one, and the model bindings it may
call. It returns a patch, or ``None`` to skip the step. Reef validates the
patched tree for the adapter; a patch with problems gets one retry with the
problems in hand, and a second failure skips the step. The episodes and
comparison that follow are what test a valid patch.

For more detail, refer to `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__.

Evaluation: selection of the better tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reef runs episodes for the prompts from the evaluation suite (seeded from
``evolution.tasks``) on both the candidate tree (the head with the patch
applied) and the head, and asks the episode scorer to score the results. The
tree with the better result is selected to become the next head.

Running episodes
^^^^^^^^^^^^^^^^

Execution uses the shared ``reef.sandbox`` package and Harbor's environment
interface. Developers can use its command, transfer and lifecycle operations;
see `Sandbox environments <../developer-guide/sandboxes.rst>`__.

Each episode runs through an executor. The default ``local`` executor runs the
binary as a plain subprocess, which is right for development and the tests. A
hosted service that evaluates model-proposed trees sets ``evolution.executor:
sandbox`` so each episode runs in a bubblewrap jail (a fresh non-root
namespace, a read-only base filesystem, explicit credentials only, resource limits,
and no network unless ``sandbox.egress_hosts`` is configured); a deployment that
requires it refuses to start without the sandbox runtime. On the native
adapter the sandbox also runs each tool call in a nested jail that withholds
the network, workspace writes, or the shell and system binaries a tool did
not declare in its capabilities; it cannot withhold the tool's own
interpreter, or reads. The adapter guide states the full boundary. The
local executor enforces none of this.

``evolution.sandbox.env_from`` explicitly lists deployment environment variables
to forward, and missing variables fail configuration. This keeps remote sandbox
credentials out of candidate episodes. ``egress_hosts`` currently enables
network access; it does not enforce a hostname firewall.

On macOS or a host without Linux namespaces, select a cloud sandbox, ``e2b`` or
``modal``, to run ordinary evaluation episodes in disposable sandboxes. The
evaluation and a method's trial episodes use the same episode runner and
trajectory readers::

    evolution:
      adapter: pi
      executor: e2b
      episode_timeout_s: 600
      sandbox:
        e2b_api_key_env: E2B_API_KEY
        # Only needed for a model listening on this evaluation worker's localhost:
        forward_ports: [8000]

Install ``reef-infra[e2b]`` and supply the E2B key in the named environment
variable. ``e2b_api_key`` can also supply it explicitly. Only variables named
in ``env_from`` are forwarded; the E2B key is used by Reef, not passed to the
agent. Public model endpoints need no forwarded port. Each configured port
forwards the evaluation worker's ``127.0.0.1`` endpoint to the same port inside
the VM. With workers on other machines, localhost names that worker, not the
Reef HTTP host; use a reachable model URL or run the evaluation on the model's
host. Forwarding is deployment configuration, never inferred from candidate
files. The ordinary model binding still renders the upstream credential into
the episode; general credential isolation remains tracked in issue #204.

For an npm-installed adapter, Reef builds a pinned Node 22 template containing
the agent and bubblewrap. Its name ends in ``-episodes-v1`` so older templates
without the isolation dependency are not reused. ``e2b_template`` selects a
custom template; non-npm adapters require one with their binary and bubblewrap
already installed. ``evolution.binary`` names a binary in that template, not a
path on the Reef host. Startup checks provider access, the template, the
binary and namespace support using a short-lived sandbox. The host does not
install the remote agent. Terminus retains its own Harbor hosted configuration.

To use Modal instead, select ``executor: modal``::

    evolution:
      adapter: pi
      executor: modal
      episode_timeout_s: 600
      sandbox:
        modal_app: reef-sandboxes
        forward_ports: [8000]

Install ``reef-infra[modal]``. Modal's client takes its credentials from
``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, or from the profile that
``modal setup`` writes; Reef uses them and does not pass them to the agent. The
options are:

- ``modal_app``: the Modal app the sandboxes belong to (``reef-sandboxes``).
- ``modal_image``: a registry image to run. Without it, Reef builds Node 22
  with bubblewrap, Python 3 and the npm adapter, and Modal caches the image. A
  named image needs the binary, ``bwrap``, ``python3`` and coreutils'
  ``timeout``.
- ``modal_cpu`` and ``modal_memory_mb``: the sandbox's requests; Modal's
  defaults otherwise.

A Modal sandbox lives at most a day, so a run with a longer timeout is refused.
Nested bubblewrap runs in Modal's default gVisor runtime. As with E2B, startup
checks provider access, the image, the binary and namespace support with a
short-lived sandbox, and a restart stops the sandboxes a stopped run of the
same deployment left.

Every ordinary episode gets its own VM. The remote filesystem and rendered
inputs are read-only; the workspace and the adapter's declared state directories
are writable. Files are copied back before trajectory parsing and residue
checks. Copy-back or cleanup failure fails the episode. Timeouts terminate the
process and its children; the VM is destroyed when the episode finishes.
E2B has outbound internet access and resources set by its template. Nonempty
``egress_hosts`` or ``limits`` are rejected because these local-sandbox policies
are not implemented by this provider. Pi and Codex have opt-in real E2B tests
in ``tests/smoke/test_e2b_episode.py``; other adapters need their template and
runtime-state paths validated with their own agent binaries.

``evolution.proposer_agent.sandbox: e2b`` remains independent: it selects where
the proposer and its trials run, reusing the same E2B implementation while
keeping one VM open for that proposer run. Set ``evolution.executor: e2b`` to
select cloud execution for the regular evaluation episodes as well.

The throwaway root contains nothing except the rendered tree: a fresh working
directory and a fresh ``HOME``, with no repository and no files from your
machine. A task must therefore state the whole problem in its prompt. A task
that refers to files the episode cannot see fails on both sides, which ties
the comparison and publishes nothing.

Evaluating the result
^^^^^^^^^^^^^^^^^^^^^

Two settings shape the evaluation result:

- ``evolution.min_win_margin: M`` (0 by default) is a noise floor on the
  result: the candidate must win more than ``M`` task pairings beyond its
  losses, so on a stochastic episode a single lucky flip does not publish.
- ``evolution.max_rejected_history: N`` (25 by default, 0 off) keeps the
  last ``N`` rejected proposals in the scenario state, each with its step,
  its patch, and the result's reason. The proposer receives them as
  ``rejected`` and can stop re-proposing what the evaluation already refused.

By default a successful evaluation is served at once.
``evolution.publish: review`` holds every win as a pending release instead,
and ``evolution.review_kinds`` (a list of tree kinds, empty by default) holds
only the wins whose patch touches those kinds, so ``[code_extension]`` lets
rules and config auto publish while code waits for a person; a win that
touches a ``native_loop`` waits whether or not the list names it. A pending release sits in the catalog with its evaluation
metrics and is never served until ``POST /reef/scenarios/{scenario}/promote``
names it; the loop keeps evolving from it in the meantime, so promoting the
latest pending release serves everything accumulated since the head.

Similarly, for more detail, refer to `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__.

Promotion: growing evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The evaluation suite consists of the seed tasks specified in
``evolution.tasks`` by default. To grow it automatically from failed tasks,
configure ``evolution.promote_failures: true``, which adds failing traces'
prompts to the evaluation set permanently. This also means the method's
episode scorer must be able to score any arbitrary prompt. A
request step in ``hybrid`` promotes the failures it carries the same way.

A promoted prompt is real client traffic, so it is screened before it
becomes a task. A prompt carrying a key-shaped literal meets the tree's own
credential tripwire: it is never promoted, never persisted, and never re-run
as a task, and the step goes on without it. A prompt shaped like an
instruction override (``ignore the previous instructions``, a forged system
message, a chat-template control token) is screened the same way. One tagged
client holds at most ``evolution.max_promoted_per_client`` promoted tasks,
so a single sender cannot fill the suite.

Which prompts are promoted is the method's call. An optional
``evolution.promote`` names a ``Promoter`` class, whose
``promote(samples, *, manifest=None)`` receives the step's trace samples
and failure manifest and returns the prompts to promote. Without it, every
failing trace's user prompt is promoted. Reef dedupes, screens and caps
whatever it returns either way.

Rechecking published tree with grown evaluation suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A publish passes the evaluation as the suite stood at the time, so a suite that
keeps growing can later expose a published tree as a regression on a task the
evaluation had not seen yet. ``evolution.recheck_every: N`` (0, off, by default)
closes that gap: every ``N`` steps, and at once when the served model or the
adapter version has changed since the publish, the loop re-evaluates the last
published tree against the tree it replaced on the current suite instead of
proposing. If
the older tree now wins, the loop publishes it, which rolls the deployment
back; if the published tree still wins, nothing changes. Only the tree from
the most recent publish is kept as a rollback target, and a rollback consumes
it, so the recheck reverts one bad publish rather than walking the whole
history back.

Cost
^^^^

Most of a step's cost is the evaluation. Every task runs on both trees,
``episode_repeats`` times each (once by default), which makes
``2 x len(tasks) x episode_repeats`` headless episodes, interleaved so both
sides of a pairing see the same upstream conditions. Each episode renders one
side into a throwaway root, runs the agent binary with the task as its prompt
under the ``episode_timeout_s`` limit (600 s by default), reads the
trajectory back, and deletes the root.

Bookkeeping of the steps
~~~~~~~~~~~~~~~~~~~~~~~~

Commit history
^^^^^^^^^^^^^^

Every result is recorded in the scenario's commit history, whether the candidate
was published or rejected. A commit carries:

- its patch: each changed path with its new text, or ``null`` for a removal,
  so a rejected rewrite is readable too;
- both score vectors;
- what the proposer cost: ``proposer_calls``, ``proposer_seconds``,
  ``proposer_input_tokens`` and ``proposer_output_tokens`` (the tokens are
  recorded, never charged);
- the path each episode took, per side and task. On the native harness this
  is the stage names the loop exited in order and the reason its turn ended
  (``candidate_paths`` and ``current_paths``, one ``{stages, reason}`` per
  episode, with ``error`` and ``errored_agent`` when a turn ended on an
  error, beside ``candidate_agents``).

Step record
^^^^^^^^^^^

The commit history holds the result; the step record holds what decided it.
Recording of step records is off by default and exists for debugging and audit:
set ``evolution.step_record_dir`` to a directory, made absolute at build, and
each scenario's steps write their raw material under it.

A step writes three things:

- ``<scenario>/<step>/proposer.json`` holds one entry per model call the
  proposer made: the ``model``, the ``messages`` and ``params`` of a
  ``chat`` or the ``body`` of a ``complete``, then the ``reply`` and
  provider ``response`` for a built-in ``chat`` binding, the ``response``
  for ``complete``, or the ``error``, and the ``seconds`` it took.
- ``<scenario>/<step>/patch.json`` holds the proposal's patch, written before
  validation decides, so a refused proposal is on file.
- ``<scenario>/<step>/episodes/<side>-<task index>/`` holds each evaluation
  episode's trajectory files as the adapter writes them (``session.jsonl``
  and ``agents/*.jsonl`` on native, the vendor's own session tree on pi and
  the others), copied out of the throwaway root before it is removed,
  beside an ``episode.json`` with the task, the exit code, stdout and
  stderr, the residue, the score, the failure and the stage path, so a
  scorer can be replayed from the record alone.

Two rules govern what the text may contain. Long text is clipped with a
marker naming what was dropped, and a credential shaped literal anywhere in
the record is replaced by ``[redacted credential]``: the record holds what
the tree boundary has not seen yet.

The provider ``response`` stored in ``<scenario>/<step>/proposer.json``
keeps the reasoning the provider returned, separate from the final reply.
Chat Completions
responses keep ``reasoning``, ``reasoning_content`` and
``reasoning_details`` as returned; Messages keeps thinking content blocks,
and Responses keeps reasoning output items. Streaming responses retain these
fields too. Opaque encrypted blocks and signatures are retained as provider
data, not converted into readable thinking. A provider that returns no
reasoning, an older record, or a custom text-only binding has none to
display; Reef does not reconstruct it.

Not every step writes a full record. A recheck step asks the proposer
nothing, so it writes ``episodes/`` only and counts zero proposer calls; a
step skipped on the step cap or the failure streak writes nothing and names
no ``step_record``. A proposer failure keeps its ``step_record`` directory
on the request's failed commit, including after the trainer reloads, and a
later retry points to its own directory; older failed commits that did not
record this link are not matched to files by directory order or timestamps.
A step directory is never reused: a step retried after a crash lands in
``<step>-2``, so the earlier attempt stays on file, and nothing prunes the
directory.

Together with the commit record, which names the step's directory as
``step_record``, those files let a reader rebuild why the tree changed or
did not. They are the proposer's raw traffic and the episodes' full logs, so
keep the directory where the commit history lives; a copy that fails (a full
disk) aborts the step rather than scoring it.

Edge cases
~~~~~~~~~~

Edge cases in the loop are resolved conservatively, so a step never
publishes accidentally.

- A ``None`` proposal skips the step.
- An episode that could not run ranks below every real score, so a
  candidate cannot win on a crash.
- When both sides fail, the step is a tie.
- A native episode whose turn ended on an error (a tree that cannot load, a
  graph that cannot run) counts as one that could not run, whatever its
  text.
- When the result is a rejection, the head stays as it was: the candidate
  never became a release.

When it fits
------------

Harness evolution fits when the bottleneck is in the text, for example a
prompt that mishandles a task family, a missing skill, or a config default
that is wrong for the deployment. It also fits when there is no weight access
because the model is a closed endpoint, and when iteration speed matters,
since a step needs only one service and one harness binary. It does not fit
when the model itself cannot do the task.

Before you start
----------------

- ``pip install reef-client``: the loop driver imports it.
- An OpenAI-compatible endpoint serving the model under test, hosted or local.
  ``REEF_UPSTREAM_URL`` takes no ``/v1`` suffix.
- Node and ``npm``: deployment startup installs the adapter
  descriptor's pinned ``pi`` under ``~/.local/share/reef-harness/pi`` through
  npm. ``REEF_HARNESS_PREFIX`` moves that root, and ``evolution.binary`` in
  ``serve.yaml`` overrides the whole step with a path of your own. A missing
  ``npm`` or failed vendor install refuses startup with the tool's error.
  Sandboxed episodes mount the adapter's install prefix read-only.

Run the example
---------------

From a Reef checkout:

.. code:: bash

   export REEF_UPSTREAM_API_KEY=sk-...    # only if your endpoint needs one
   cd tutorials/evolve-your-harness
   ./run.sh

To run a harness-evolving deployment without the example's driver, start
the built-in Reefine profile and name the model:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b

The profile is Reefine's own default (loopback, port 8901, no token unless
``REEF_TOKEN`` is set, state under ``.reef/reefine/``); its proposer and evaluator
ship in the wheel, so it needs no checkout. This
example's own stack stays in ``configs/serve.yaml``, which ``run.sh`` passes
with ``-c``. `The CLI reference <../reference/cli.rst>`__ describes provider
settings and legacy shorthand.

``serve.yaml`` holds the endpoint (``http://127.0.0.1:8000``, no ``/v1``
suffix), the model (``qwen3-8b``), and the service token as literals; edit
them there to point at your own. The model name is set once, as
``inference.upstream-model``: the proposer, the evolve episodes and served
traffic all use it, and ``run.py`` repeats it as ``MODEL``; a name the
endpoint does not serve fails the proposer's call, and
the step records ``skipped: no proposal``. The provider key is the one value
``serve.yaml`` does not hold.

`evolve-your-harness.ipynb
<../../tutorials/evolve-your-harness/evolve-your-harness.ipynb>`__ is the same
pass as a notebook, cell by cell, with the service managed as a subprocess.

``run.sh`` copies the recipe config out of ``serve.yaml``, starts the service, and runs
``run.py``: three exact-answer coding tasks go through Reef, each reply is
graded, and every result is reported against its receipt. Every valid scored
report trains, a passing one included, and ``evolution.batch_size: 1`` makes each
report its own step, so every report this run submits triggers one evolve
step. In this example the served model is its own proposer, through Reefine's
text proposer, and it answers with one skill file.

The example's scenario is ``harness-evolve-demo``. ``run.sh`` keeps the
service up only while ``run.py`` runs. When the loop finishes, it prints the
published release, the evaluation metrics, and the evolved ``SKILL.md``,
then stops the service.

Watch it learn
--------------

To follow the same step live, from a second terminal while ``run.sh`` is
still running:


.. code:: bash

   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness            # the seed tree until a step publishes
   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness/releases

The demo service listens on loopback only and runs without a token. On a
shared machine, export ``REEF_TOKEN`` before ``run.sh`` starts it: the
service then requires that token, and every ``curl`` on this page needs
``-H "Authorization: Bearer $REEF_TOKEN"``.

One step is six episodes, three tasks on each of the two trees, and the
reference run finished in 63 s on Qwen3-8B: one reported trace filled the
step, the served model proposed a new skill beside the starter, and the evaluation
scored the candidate 3.0 against 2.0 (1 win, 0 losses, 2 ties). The run has
succeeded when one step's candidate is selected and ``GET /reef/harness``
serves a release other than the seed.
``/reef/harness/releases`` then shows that step's training row with
``selected: true`` in its metrics.

If ``/reef/harness`` still serves the seed after a few minutes, the run has
failed. A server without tool calling can start but fails every episode:
both sides tie, no candidate ever wins, and the head never moves. The step's
row names the cause. Vendor install failures instead refuse deployment
startup. Confirm that
``~/.local/share/reef-harness/pi/node_modules/.bin/pi --version`` runs and
that the server accepts tool calls before suspecting the recipe; vLLM needs
``--enable-auto-tool-choice --tool-call-parser hermes``, and without those
flags it rejects pi's ``tool_choice: "auto"`` requests with a 400 while
still answering plain requests. A missing model server does not produce
this symptom: the record phase raises on its first call and ``run.py``
exits with the upstream error before any evolve step runs.

A model that answers all three tasks correctly still trains one step per
report. Evaluation runs the tasks again on the current and candidate trees,
so passing the initial tasks does not determine whether a candidate will
win, tie, or lose. A step publishes only when its candidate wins; rejected
or skipped steps leave the served tree unchanged. ``run.py`` counts all
submitted reports, waits for the steps they triggered up to its deadline,
and prints how many reports it submitted for steps and how many of them failed.

Use a published tree
--------------------

``GET /reef/harness`` serves the head release as a JSON manifest: ``files``
maps each path the deployment's adapter renders to its text, beside the
``release_id``, its parent, the ``evaluation`` metrics of the step that
published it, and ``requires``, what the release needs where the harness
runs. A fresh scenario serves the recipe's seed as its first release, so a
client can pull before any step has run:

.. code:: bash

   curl -sS -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness | python3 -m json.tool | head

``?release_id=<id>`` serves an older release, which pins or rolls back one
client without moving the head. ``?form=source`` serves the release's source
tree instead: the paths a patch names, for a client that renders the tree
itself or proposes changes to it.

The served files carry no endpoint and no credential. A client writes them
where its agent reads them and binds the agent's model to this Reef. Reef
serves and records a model call that carries the ``x-reef-scenario`` header
and the token, and answers with the receipt a report references. Two ways
reach that today: a harness whose provider config sends custom headers (pi
sends ``providers.reef.headers`` from ``pi-agent/models.json``), or a local
proxy in front of Reef that adds them, as
``reef.harness.capture.CaptureProxy`` does for the native serve process. A
packaged client for each adapter, one command that pulls, binds, reports and
asks, is part of the client design in RFC #918.

Report how a run went with ``POST /reef/report``: a score and feedback over
the receipts of its model calls. A release's ``requires`` items say what the
machine must provide (an ``env`` variable, a ``permission``, a ``service``, a
``binary``), each with an optional ``check`` and a ``prompt`` sentence; the
client checks them before it runs the release. ``tutorials/reefine/run.py``
is a worked client: it pulls the source tree, checks ``requires``, renders the
tree for pi with its own binding at Reef, and runs a session on it.

Ask for a harness change
~~~~~~~~~~~~~~~~~~~~~~~~

You can also ask for a harness change in plain words. A scenario in ``manual``
or ``hybrid`` takes training requests; the tutorial's ``deployment.yaml``
runs in ``hybrid``, and a scenario in ``auto`` takes them after a switch:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"training_mode": "hybrid"}' \
     "$REEF_URL/reef/scenarios/code-repair/update"
   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"text": "run the tests before you report a fix as done", "session": "me", "release_id": ""}' \
     "$REEF_URL/reef/train"

The answer carries the training record id and ``page_path``, the request's
page (``GET /reef/harness/requests/<id>/page`` with the scenario and its page
key as query parameters, so a browser opens it as is). Acceptance does not
mean the change passed the evaluation: the page follows the step while it
runs, then shows the result, what the patch changed, the review, and the next
action. ``GET /reef/harness/requests/<id>/progress`` gives the same state as
JSON for a client without a browser. See the `manual training API
<../reference/http-api.rst#manual-training>`__ for the fields a request
carries and the screens its text passes.

The proposer's ``reads_requests`` must be true. A request may carry
``requires``; the proposer adds items of its own when the change it wrote
needs them, and the backend records ``training_request: {id, session,
release_id, text, requires}`` in the step's commit metrics. A releases row
carries what its own change named; the manifest carries what the release
needs over its whole chain, so a later change that names nothing still needs
what an earlier one added.

An evolved extension runs in the harness's process with your privileges, so
the tutorial's ``configs/deployment.yaml`` sets
``evolution.review_kinds: [code_extension]``: a release that changes one waits
as pending until a person promotes it. ``configs/serve.yaml`` and
``configs/serve-native.yaml`` stay in ``auto``, where an ask is refused. The
built-in ``reef.recipe.reefine:ReefineRecipe`` is available through
``reef serve --recipe reefine``; ``tutorials/reefine/`` runs this path end to
end on one machine, from the ask to the install and a session on the new
tree, with a bug fix demo, a research demo and a measurement of which
requests passed the checks (``./run.sh bugfix``, ``./run.sh research``,
``./run.sh measure``).

Each step has a page, ``GET /reef/harness/releases/<step>/page``, one self
contained HTML page with five sections: Why (the request, else the
proposal's reason), What changed (the patch, file by file; an updated file as
a line diff against the release the step ran on), Result (the evaluation's
result and numbers, and the step record directory when
``evolution.step_record_dir`` is set), Setup (what the release needs: the
step's own items, then those carried from earlier steps) and Chain (the
parent, this release, and its children). ``GET /reef/harness/releases``
lists every row, oldest last, with a ``page_path`` for each.

Promote a pending release
~~~~~~~~~~~~~~~~~~~~~~~~~

A win that touches a kind in ``evolution.review_kinds`` (``code_extension``
in the tutorial's ``deployment.yaml``) or a ``native_loop`` sits in the
catalog with ``pending: true`` and is served to no client. Find its id in
``GET /reef/harness/releases`` (the newest row marked ``pending``), read the
change on its page or with ``GET /reef/harness?release_id=<id>``, and name it
to the promote route. The answer is the new head with a fresh release id,
because a promote republishes the tree as a commit of its own. The tutorial's
deployment listens on port 8901.

.. code:: bash

   curl -sS -H "x-reef-scenario: <scenario>" \
     http://127.0.0.1:8901/reef/harness/releases    # the row with "pending": true
   curl -sS -X POST -H "Content-Type: application/json" \
     -d '{"release_id": "<the pending release id>"}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/promote

Serve the harness as a resident process
---------------------------------------

The native adapter has a second form. ``reef-native -p`` is the episode
form: one process, one turn, what the evaluation runs. ``reef-native serve`` is
the serve form: one resident process that runs a scenario's release and
follows the head while it runs. A publish reaches the process between two
steps of the open turn, or at once when no turn is open. No reinstall, no
restart.

.. code:: bash

   mkdir -p reef-harness/native
   printf '{"api": "openai", "base_url": "http://127.0.0.1:8900", "api_key": "%s", "model": "qwen3-8b"}\n' \
     "${REEF_TOKEN:-reef-no-token}" > reef-harness/native/models.json
   reef-native serve --tree ./reef-harness --scenario harness-evolve-demo &
   reef-native turn --tree ./reef-harness -p "fix the failing test in auth.py"
   reef-native turn --tree ./reef-harness -p "now add a test for it" --session 3f9a1c2b7d4e
   reef-native status --tree ./reef-harness

``--tree`` names the process's directory. ``native/models.json`` there binds
the model: the process reads the Reef URL and the token from it
(``--reef-url`` and ``REEF_TOKEN`` override them). At boot the process reads
the source tree of the release it ran last from ``source/`` beside the
``.reef-harness-release`` metadata file, else it pulls the head's source tree
with ``GET /reef/harness?form=source``. Each mount renders a tree for the
native adapter into its own directory under ``native/mounts/`` and builds the
loop's host from those files. The process starts a capture proxy in process,
so its model calls carry the scenario and the token, and listens on
``native/serve.sock``; ``status`` prints the socket, which moves under
``/tmp`` when the directory's path is too long for a socket address.

``turn`` prints every event of the turn as one JSON line each, then
``turn/result`` with the exit status, the session id, the turn number, the
last assistant text and ``receipts``, the receipts of the turn's model calls
on the served tree (a trial's calls are left out), which a
``POST /reef/report`` for that turn references; ``--quiet`` prints the
text alone. A turn without ``--session`` starts a new session. A session keeps
its messages across turns, its log lands under
``native/sessions/<session>/session.jsonl``, and every turn runs on the
graph's step budget and on ``--turn-timeout`` seconds of wall clock (600),
checked before each model call.

With ``--follow head``, the default, the process polls
``GET /reef/harness/releases`` every ``--poll-interval`` seconds (60) and
reads the ``x-reef-release-id`` header of every inference answer, so a
process with traffic learns of a publish on its next model call. A new head
is mounted between two steps of the open turn, or at once when the process
is idle. The mount is one line in the open turn's session, else in
``native/sessions/serve.jsonl``:

.. code:: text

   {"type": "harness/mount", "seq": 41, "time": 1788600000000, "data": {"release_id": "6f1c...", "parent_release_id": "2a9b...", "source": "release", "files": 8}}

The next step runs on the new tools, hooks, rules, skills and window, and
writes a new ``request/header`` when what the model sees changed; the next
turn runs the new graph. A tree that does not render or load (a tool whose
module binds no ``run``, a hook whose code does not import, a name a self
tool owns) leaves the current host in place: ``harness/mount-failed`` names
the release and the error. On success the process stores the release's source
tree under ``source/`` and names it in the release metadata file, so a restart
boots from it with ``source: boot``.

With ``--follow pinned`` the process logs ``release/available`` with the
new head and waits for a person. ``reef-native mount <release_id> --tree
./reef-harness`` applies one release by hand, an older one included, which
rolls the process back locally; under ``head`` too, a release mounted by
hand stands until the head moves again. A Reef that does not answer logs
``release/poll-failed`` with the error and the next retry, doubling up to
ten minutes, and the process keeps serving. A Reef that is busy is polled
again at the interval: a catalog read waits behind a running evolve step,
so a poll that times out during one retries at the interval and the head
lands as soon as the step ends.

``--self-tools`` gives the model three built-in tools. The tree cannot take
their names, and they are absent in the episode form, so a candidate cannot
pass the checks by calling them:

- ``harness_inspect(what)``: ``tree`` is every file of the live tree and the
  mounted release; ``graph`` is ``main`` and every named graph; ``results``
  is the newest releases with the evaluation metrics that admitted each, and
  the rejected proposals when Reef exposes them; ``status`` is the status
  above.
- ``harness_try(patch)``: mounts the served tree with the patch applied on
  this process for the rest of the turn. The patch passes validation first;
  the change applies from the next step, the model calls carry
  ``x-reef-tag-trial``, and at ``turn/end`` the served tree is mounted back
  (``harness/unmount``). Nothing is published.
- ``harness_propose(patch, reason)``: sends ``POST /reef/harness/proposals``
  with the mounted release and the session id. Reef validates the patch at
  once, and an admitted proposal goes through the evaluation like the
  method's own before it is served. A Reef without the route answers a tool
  error and the turn continues.

The order is inspect, then try, then propose. Every call is a ``tool/call``
and ``tool/result`` pair in the session log, so what the model learned about
itself and what it changed is in the record.

The serve process runs on your machine with your privileges and runs tree
code in process, as the episode form does: every hook imports and listens
there, and a tool call runs there unless ``REEF_NATIVE_ENFORCE=bwrap`` is
set; the evaluation's sandbox does not apply to it. Under ``--follow head``,
whoever can publish to the scenario runs code on the machine the process
serves on. ``--follow pinned`` keeps a person in that loop.

Write a method
--------------

Reef ships no general proposer and no general episode scorer. You supply a
``Proposer``, an ``EpisodeScorer``, and optionally a selection policy; `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ documents the contract, with worked examples.

Connect a different agent
-------------------------

An adapter is one descriptor: where each kind of tree file is written, which
kinds the agent accepts, how the binary is launched, and how a model binds to
it. The bundled descriptors cover these agents:

+---------------+----------------------------------+----------------------------------------+
| Adapter       | Agent                            | Kinds it renders                       |
+===============+==================================+========================================+
| ``pi``        | pi coding agent                  | config, rules, agent_command, skill,   |
|               |                                  | code_extension                         |
+---------------+----------------------------------+----------------------------------------+
| ``opencode``  | OpenCode                         | config, rules, agent_command, skill,   |
|               |                                  | code_extension                         |
+---------------+----------------------------------+----------------------------------------+
| ``claude``    | Claude Code                      | config, rules, agent_command, skill,   |
|               |                                  | code_extension                         |
+---------------+----------------------------------+----------------------------------------+
| ``codex``     | Codex CLI                        | config, rules, agent_command, skill    |
+---------------+----------------------------------+----------------------------------------+
| ``dsh``       | DeepSeek Harness                 | config, rules, agent_command, skill,   |
|               |                                  | code_extension                         |
+---------------+----------------------------------+----------------------------------------+
| ``hermes``    | Hermes Agent                     | config, rules, agent_command, skill,   |
|               |                                  | code_extension                         |
+---------------+----------------------------------+----------------------------------------+
| ``terminus``  | Terminus 2 (Terminal-Bench)      | config, rules, agent_command, skill,   |
|               |                                  | code_extension (sandbox + E2B)         |
+---------------+----------------------------------+----------------------------------------+
| ``native``    | Reef's own loop                  | config, rules, skill, native_tool,     |
|               |                                  | native_hook, native_graph,             |
|               |                                  | native_agent, native_loop              |
+---------------+----------------------------------+----------------------------------------+

`Harness adapters <../developer-guide/harness-adapters.rst>`__ is the descriptor reference and
how to connect an agent that has no adapter yet.

.. seealso::

   `Scenario model configuration <scenario-models.rst>`__ explains scenario-specific custom providers for the entire
   harness evolve model pipeline through Reef API Platform.
