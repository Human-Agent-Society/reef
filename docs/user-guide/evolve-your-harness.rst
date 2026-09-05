Evolve your harness
===================

A harness is everything around the model: the control loop, rules, prompt
templates, skills, tools, config, and extension code. Together, the model and
harness form an agent. Harness evolution improves the harness tree while the
model weights stay fixed. Reef needs no GPU for this. The model stays a fixed
endpoint, hosted or local, and the agent stays online throughout.

Reef supplies the mechanism: it snapshots the tree, applies a mutation, runs
the paired episodes, and publishes or reverts. You supply two Python
callables, ``propose`` (which edit to try) and ``evaluate`` (how an episode
scored). `Write a harness method <../developer-guide/write-a-harness-method.rst>`__ documents the
contract.

At a glance
-----------

+-------------------+--------------------------------------------------------------+
| What evolves      | Config, rules, prompt templates, skills, and extension code; |
|                   | on the native harness also its tools, its hook listeners,    |
|                   | the graph that is its control loop, and the agents the loop  |
|                   | delegates to.                                                |
+-------------------+--------------------------------------------------------------+
| Which agents      | pi, opencode, Claude Code, Codex, DeepSeek Harness, Hermes   |
|                   | Agent, Terminus 2, and ``native``, Reef's own loop. One      |
|                   | adapter per agent maps the tree onto that agent's files.     |
+-------------------+--------------------------------------------------------------+
| What decides      | The candidate and the current tree run the same tasks in     |
|                   | fresh roots. The candidate is published only when it wins    |
|                   | more tasks than it loses, by more than a margin you set.     |
+-------------------+--------------------------------------------------------------+
| What holds it     | A rejected candidate reverts to the snapshot. A failing      |
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

Reef stores the mutable, versioned files of one harness in a single object
called the tree. A tree is a flat list of entries, and each entry has three
fields: ``id`` is unique within the tree, ``name`` selects one of the node
kinds below, and ``config`` holds that kind's own fields. For the named kinds
(``agent_command``, ``skill``, ``code_extension``, ``native_tool``,
``native_hook``), ``config.name`` is the file name the entry renders to.
Nine kinds are registered in
`reef/harness/nodes.py <../../reef/harness/nodes.py>`__:

+--------------------+----------------------------------------------------------+
| ``name``           | Renders as                                               |
+====================+==========================================================+
| ``config``         | a JSON object deep-merged into one of the agent's config |
|                    | files                                                    |
+--------------------+----------------------------------------------------------+
| ``rules``          | text appended to the agent's rules file                  |
+--------------------+----------------------------------------------------------+
| ``agent_command``  | a named prompt template                                  |
+--------------------+----------------------------------------------------------+
| ``skill``          | a named ``SKILL.md``                                     |
+--------------------+----------------------------------------------------------+
| ``code_extension`` | a named code file the harness loads in process           |
+--------------------+----------------------------------------------------------+
| ``native_tool``    | a named tool the native harness loads (schema and code)  |
+--------------------+----------------------------------------------------------+
| ``native_hook``    | a named listener at one event of the native loop (code)  |
+--------------------+----------------------------------------------------------+
| ``native_graph``   | the native loop's control flow: stages and edges (data)  |
+--------------------+----------------------------------------------------------+
| ``native_agent``   | one agent of the native loop: its prompt, graph, tools,  |
|                    | skills, budget, and the agents it hands its text to      |
+--------------------+----------------------------------------------------------+

The table describes what each kind contains. Where each kind is written is
decided by an adapter, which maps every kind to a concrete file for one agent.
Reef bundles adapters for third-party coding agent CLIs (``pi``, ``opencode``,
``claude``, ``codex``, ``dsh`` (DeepSeek Harness), and ``hermes`` (Hermes
Agent)); ``native``, its own agent: a loop inside the reef tree whose tools
are ``native_tool`` nodes, whose loop events listen to ``native_hook``
nodes, whose control flow is a ``native_graph`` node, and whose helpers are
``native_agent`` nodes a graph can call, so the agent can evolve the tools
it runs, how its loop reacts, the loop itself, and who it delegates to, not
only the text around a vendor binary; and ``terminus``, Terminal-Bench's
Terminus 2, run through a Reef-owned Harbor runner. Only ``native`` renders
those four kinds.

Codex and Terminus support ``config``, ``rules``, ``agent_command``, and
``skill``. Both reject ``code_extension``: Codex lifecycle hooks run outside
its command sandbox, and a Terminus module would run in Reef's own runner
rather than the task container. Reef will not activate arbitrary evolved code
without a separate isolation boundary.

With the ``pi`` adapter, ``GET /reef/harness`` serves:

.. code:: text

   pi-agent/
     settings.json             <- config, target "primary"
     models.json               <- config, target "models"
     AGENTS.md                 <- rules
     prompts/<name>.md         <- agent_command
     skills/<name>/SKILL.md    <- skill
     extensions/<name>.ts      <- code_extension

The loop
--------

.. flow::
   :loop: publish the winner, or restore the snapshot

   Batch :: scored reports retained by the score window
   ``propose`` :: one proposal, a mutation or a sequence applied as one, or ``None``
   Episodes* :: run the candidate and current tree on the same tasks
   Verdict :: publish the candidate or restore the snapshot

No evolution runs while traffic flows. A report enters the *window* when it
references at least one receipt and its score is at or below ``max_score``;
the default for harness evolution keeps only failures. A report over one
receipt batches as that exchange; a report over several batches as one
trajectory sample carrying every referenced exchange in order, which is what
``reef-pi report`` sends for a whole run (``--per-receipt`` fans the score
across the receipts as separate reports instead). When ``batch_size``
window entries have accumulated, one step runs the loop once. With
``evolution.promote_failures: true`` a failing trace's prompt is added to the
gate as a permanent task, so the seed tasks are the floor of a suite that
grows from real failures and no later candidate can win while bringing one
back (the method's ``evaluate`` must score an arbitrary prompt). A prompt is
real traffic, so it meets the tree's own credential tripwire first: a prompt
carrying a key-shaped literal is never promoted, never persisted, and never
re-run as a task, and the step goes on without it. A prompt shaped like an
instruction override (``ignore the previous instructions``, a forged system
message, a chat-template control token) is screened the same way, and one
tagged client holds at most ``evolution.max_promoted_per_client`` promoted
tasks, so a single sender cannot fill the suite. Which prompts are
promoted is the method's call: an optional ``evolution.promote`` callable
receives the step's trace samples (and the failure manifest when its
signature names ``manifest``) and returns the prompts to promote; without it
every failing trace's user prompt is promoted. Reef still dedupes, screens,
and caps whatever it returns. ``batch_size``
and ``max_score`` live under ``data:`` in the recipe config, and
``data.batch_policy: records`` drops the report requirement entirely:
recorded traffic alone batches, unscored, for methods that judge for
themselves.

A publish passes the gate as the suite stood at the time, so a suite that
keeps growing can later expose a published tree as a regression on a task the
gate had not seen yet. ``evolution.recheck_every: N`` (0, off, by default)
closes that gap: every ``N`` steps, and at once when the served model or the
adapter version has changed since the publish, the loop re-gates the last
published tree against the tree it replaced on the current suite instead of
proposing. If
the older tree now wins, the loop publishes it, which rolls the deployment
back; if the published tree still wins, nothing changes. Only the tree from
the most recent publish is kept as a rollback target, and a rollback consumes
it, so the recheck reverts one bad publish rather than walking the whole
history back.

Two more settings shape the search itself. ``evolution.min_win_margin: M``
(0 by default) is a noise floor on the verdict: the candidate must win more
than ``M`` task pairings beyond its losses, so on a stochastic episode a
single lucky flip does not publish. ``evolution.max_rejected_history: N``
(25 by default, 0 off) keeps the last ``N`` rejected proposals in the
scenario state, each with its step, its mutations with the options they
carried, and the verdict's reason; a ``propose`` whose signature names
``rejected`` receives them and can stop re-proposing what the gate already
refused.

By default a gate win is served at once. ``evolution.publish: review`` holds
every win as a pending release instead, and ``evolution.review_kinds`` (a
list of node kinds, empty by default) holds only the wins that touch those
kinds, so ``[code_extension]`` lets rules and config auto publish while code
waits for a person. A pending release sits in the catalog with its gate
metrics and is never served until ``POST /reef/scenarios/{scenario}/promote``
names it; the loop keeps evolving from it in the meantime, so promoting the
latest pending release serves everything accumulated since the head.

Most of a step's cost is the evaluation. Every task runs on both trees,
``episode_repeats`` times each (once by default), which makes
``2 x len(tasks) x episode_repeats`` headless episodes, interleaved so both
sides of a pairing see the same upstream conditions. Each episode renders one
side into a throwaway root, runs the agent binary with the task as its prompt
under the ``episode_timeout_s`` limit (600 s by default), reads the
trajectory back, and deletes the root.

Each episode runs through an executor. The default ``local`` executor runs the
binary as a plain subprocess, which is right for development and the tests. A
hosted service that evaluates model-proposed trees sets ``evolution.executor:
sandbox`` so each episode runs in a bubblewrap jail (a fresh non-root
namespace, a read-only base filesystem, no host credentials, resource limits,
and no network unless a model endpoint is allowlisted); a deployment that
requires it refuses to start without the sandbox runtime. On the native
adapter the sandbox also runs each tool call in a nested jail that withholds
the network, workspace writes, or the shell and system binaries a tool did
not declare in its capabilities; it cannot withhold the tool's own
interpreter, or reads. The adapter guide states the full boundary. The
local executor enforces none of this.

The throwaway root contains nothing except the rendered tree: a fresh working
directory and a fresh ``HOME``, with no repository and no files from your
machine. A task must therefore state the whole problem in its prompt. A task
that refers to files the episode cannot see fails on both sides, which ties
the comparison and publishes nothing.

The edge cases resolve conservatively. A ``None`` proposal skips the step. An
episode that could not run ranks below every real score, so a candidate
cannot win on a crash, and when both sides fail the step is a tie; a native
episode whose turn ended on an error (a tree that cannot load, a graph that
cannot run) counts as one that could not run, whatever its text. When the
verdict is a rejection, Reef restores the snapshot it took before the
mutation. Every verdict is recorded in the scenario's commit log together
with its mutation (op, id and the full options, so a rejected rewrite is
readable too), both score vectors, how many model calls the proposer made
and the seconds they took (``proposer_calls``, ``proposer_seconds``), and per
side and task the path each episode took: on the native harness the stage
names the loop exited in order and the reason its turn ended
(``candidate_paths`` and ``current_paths``, one ``{stages, reason}`` per
episode, with ``error`` and ``errored_agent`` when a turn ended on an error,
beside ``candidate_agents``).

The commit log holds the verdict; the step record holds what decided it.
``evolution.step_record_dir`` (off by default) names a directory, made
absolute at build, under which each scenario's steps write
``<scenario>/<step>/proposer.json``, one entry per model call the proposer
made: the ``model``, the ``messages`` and ``params`` of a ``chat`` or the
``body`` of a ``complete``, then the ``reply`` or ``response`` or the
``error``, and the ``seconds`` it took; ``<scenario>/<step>/mutations.json``,
the parsed proposal with its options, written before admission so a refused
proposal is on file; and ``<scenario>/<step>/episodes/<side>-<task index>/``,
each gate episode's trajectory files as the adapter writes them
(``session.jsonl`` and ``agents/*.jsonl`` on native, the vendor's own session
tree on pi and the others) copied out of the throwaway root before it is
removed, beside an ``episode.json`` with the task, the exit code, stdout and
stderr, the residue, the score, the failure and the stage path, so a scorer
can be replayed from the record alone. Long text is clipped with a marker
naming what was dropped, and a credential shaped literal anywhere in the
record is replaced by ``[redacted credential]``: the record holds what the
tree boundary has not seen yet. A recheck step asks the proposer nothing, so
it writes ``episodes/`` only and counts zero proposer calls; a step skipped
on the step cap or the failure streak writes nothing and names no
``step_record``. A step directory is never reused: a step retried after a
crash lands in ``<step>-2``, so the earlier attempt stays on file, and
nothing prunes the directory. A reader can rebuild why the tree changed, or
did not, from those files and the commit record, which names the step's
directory as ``step_record``. The record is the proposer's raw traffic and
the episodes' full logs, so keep the directory where the commit log lives; a
copy that fails (a full disk) aborts the step rather than scoring it.

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
  npm, the same install the served harness script runs on a client.
  ``REEF_HARNESS_PREFIX`` moves that root, and ``evolution.binary`` in
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

``serve.yaml`` holds the endpoint (``http://127.0.0.1:8000``, no ``/v1``
suffix), the model (``qwen3-8b``), and the service token as literals; edit
them there to point at your own. The model name appears twice, as
``model.path`` for the proposer and the evolve episodes and as
``upstream_model`` for served traffic, and ``run.py`` repeats it as
``MODEL``; a name the endpoint does not serve fails the proposer's call, and
the step records ``skipped: no proposal``. The provider key is the one value
``serve.yaml`` does not hold.

`evolve-your-harness.ipynb
<../../tutorials/evolve-your-harness/evolve-your-harness.ipynb>`__ is the same
pass as a notebook, cell by cell, with the service managed as a subprocess;
its committed outputs are a full local run on ollama with no GPU.

``run.sh`` copies the recipe config out of ``serve.yaml``, starts the service, and runs
``run.py``: three exact-answer coding tasks go through Reef, each reply is
graded, and every result is reported against its receipt. Only failures enter
the window, so the first failing report triggers one evolve step. In this
example the served model is its own proposer, and it answers with one skill
mutation.

The example's scenario is ``harness-evolve-demo``. ``run.sh`` keeps the
service up only while ``run.py`` runs. When the loop finishes, it prints the
published release, the gate metrics, and the evolved ``SKILL.md``,
then stops the service.

Watch it learn
--------------

To follow the same step live, from a second terminal while ``run.sh`` is
still running:


.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness            # 404 until a step publishes
   curl -sS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness/releases

One step is six episodes, three tasks on each of the two trees, and the
reference run finished in 63 s on Qwen3-8B: one failing task entered the
window, the served model proposed a new skill beside the starter, and the gate
scored the candidate 3.0 against 2.0 (1 win, 0 losses, 2 ties). The committed
notebook run repeats the arc with no GPU at all, on ollama ``qwen2.5:7b``. The run has succeeded when one
task fails, the failing report opens the window, one evolve step runs, and
``GET /reef/harness`` stops returning 404. ``/reef/harness/releases`` then
shows a published version.

If ``/reef/harness`` still returns 404 after a few minutes, the run has
failed. A server without tool calling can start but fails every episode:
both sides tie, no candidate ever wins, and the route stays 404. The failure
manifest names the cause. Vendor install failures instead refuse deployment
startup. Confirm that
``~/.local/share/reef-harness/pi/node_modules/.bin/pi --version`` runs and
that the server accepts tool calls before suspecting the recipe; vLLM needs
``--enable-auto-tool-choice --tool-call-parser hermes``, and without those
flags it rejects pi's ``tool_choice: "auto"`` requests with a 400 while
still answering plain requests. A missing model server does not produce
this symptom: the record phase raises on its first call and ``run.py``
exits with the upstream error before any evolve step runs.

A model that answers all three tasks correctly also leaves the route at 404,
because nothing fails, so nothing batches and no step runs. ``run.py`` prints
``every task passed: nothing batched, no evolve step runs`` when that
happens.

Install the published tree
--------------------------

Clients pull an evolved harness the way they install any coding agent. A
fresh scenario already serves the recipe's seed as its first release, so
the install works before any step has run:

.. code:: bash

   curl -fsS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     'http://127.0.0.1:8900/reef/harness/install?adapter=pi' | bash

   reef-pi -p "fix the failing test in auth.py"
   reef-pi report --score 0 --feedback "missed the empty-token case"

The script installs the pinned agent, writes the tree, writes the agent's
model binding pointed at the Reef the script came from (the served tree
itself carries no endpoint or credential; the binding takes its token from
``REEF_TOKEN`` in your shell when the script runs), and puts a
``reef-<adapter>`` wrapper (here ``reef-pi``) on your PATH. The wrapper keeps
the receipts from a run, so ``report`` only needs the result. Pinning,
rollback, and the raw manifest routes are in `HTTP API
<../reference/http-api.rst#harness-artifacts>`__.

The native adapter's binary is ``reef-native``, which ships with reef, so
the install route serves no script for it. Pull the tree with the client,
name your Reef URL in its ``native/models.json``, and run the wrapper module
with the same five settings the script bakes into ``reef-pi``:

.. code:: bash

   python3 -c 'from reef_client import ReefClient; ReefClient("http://127.0.0.1:8900", token="reef-local").harness_pull("harness-evolve-demo", "./reef-harness")'
   printf '{"api": "openai", "base_url": "http://127.0.0.1:8900", "api_key": "reef-local", "model": "qwen3-8b"}\n' > reef-harness/native/models.json
   export REEF_HARNESS_BINARY="$(command -v reef-native)" REEF_HARNESS_COMPOSE="$PWD/reef-harness/native"
   export REEF_HARNESS_SCENARIO=harness-evolve-demo REEF_HARNESS_ADAPTER=native REEF_HARNESS_ENV_VAR=REEF_NATIVE_DIR
   python3 -m reef.harness.harness_wrapper -p "fix the failing test in auth.py"
   python3 -m reef.harness.harness_wrapper report --score 0 --feedback "missed the empty-token case"

The wrapper points the loop at its capture proxy through a temp copy of
the tree, keeps the loop's session log under ``native/sessions`` beside
the installed tree, and ``report`` works as for any adapter.

Serve the harness as a resident process
---------------------------------------

The native adapter has a second form. ``reef-native -p`` is the episode
form: one process, one turn, what the gate runs. ``reef-native serve`` is
the serve form: one resident process per installed tree that holds the tree
as a live composition and follows the release Reef serves while it runs. A
publish reaches the process as a mount between two steps of the open turn,
or at once when no turn is open. No reinstall, no restart.

.. code:: bash

   reef-native serve --tree ./reef-harness --scenario harness-evolve-demo &
   reef-native turn --tree ./reef-harness -p "fix the failing test in auth.py"
   reef-native turn --tree ./reef-harness -p "now add a test for it" --session 3f9a1c2b7d4e
   reef-native status --tree ./reef-harness
   python3 -m reef.harness.harness_wrapper report --score 1 --feedback "fixed"

``--tree`` names the pulled tree, the directory that holds ``native/`` and
the release sidecar. The process boots from ``native/tree.json``, the
entries list Reef renders into every native release (a tree pulled before
that file existed runs in the episode form only). It reads the Reef URL and
the token from ``native/models.json``; ``--reef-url`` and ``REEF_TOKEN``
override them. It starts the wrapper's capture proxy in process and listens
on ``native/serve.sock``; ``status`` prints the socket, which moves under
``/tmp`` when the tree's path is too long for a socket address. The receipts
of each turn are spooled as a run of their own, so ``report`` works per
turn, with the wrapper's five settings in the environment as above.

``turn`` prints every event of the turn as one JSON line each, then
``turn/result`` with the exit status, the session id, the turn number and
the last assistant text; ``--quiet`` prints the text alone. A turn without
``--session`` starts a new session. A session keeps its messages across
turns, its log lands under ``native/sessions/<session>/session.jsonl``,
and every turn runs on the graph's step budget and on ``--turn-timeout``
seconds of wall clock (600), checked before each model call.

With ``--follow head``, the default, the process polls
``GET /reef/harness/releases`` every ``--poll-interval`` seconds (60) and
reads the ``x-reef-release-id`` header of every inference answer, so a
process with traffic learns of a publish on its next model call. A new head
is mounted between two steps of the open turn, or at once when the process
is idle. The mount is one line in the open turn's session, else in
``native/sessions/serve.jsonl``:

.. code:: text

   {"type": "harness/mount", "seq": 41, "time": 1788600000000, "data": {"release_id": "6f1c...", "parent_release_id": "2a9b...", "source": "release", "entries": 8}}

The next step runs on the new tools, hooks, rules, skills and window, and
writes a new ``request/header`` when what the model sees changed; the next
turn runs the new graph. A mount that leaves an entry FAILED (a tool whose
code does not import, a kind this reef has no plugin for, a name a self tool
owns) is rolled back whole before the next step: ``harness/mount-failed``
names the release, the entry and the error, and the previous composition
keeps serving. On success the sidecar and ``native/tree.json`` name the new
release, so a restart boots from it with ``source: boot``.

With ``--follow pinned`` the process logs ``release/available`` with the
new head and waits for a person. ``reef-native mount <release_id> --tree
./reef-harness`` applies one release by hand, an older one included, which
rolls the process back locally; under ``head`` too, a release mounted by
hand stands until the head moves again. A Reef that does not answer logs
``release/poll-failed`` with the error and the next retry, doubling up to
ten minutes, and the process keeps serving.

``--self-tools`` gives the model three host plane tools. The tree cannot
remove them or take their names, and they are absent in the episode form,
so a candidate cannot win the gate by calling them:

- ``harness_inspect(what)``: ``tree`` is the live entries and the mounted
  release; ``graph`` is ``main`` and every named graph; ``verdicts`` is the
  newest releases with the gate metrics that admitted each, and the
  rejected proposals when Reef exposes them; ``status`` is the status above.
- ``harness_try(mutations)``: mounts the served entries plus the mutations
  on this process for the rest of the turn. The change applies from the
  next step, the model calls carry ``x-reef-tag-trial`` so ``report`` skips
  their receipts, and at ``turn/end`` the served entries are mounted back
  (``harness/unmount``). Nothing is published.
- ``harness_propose(mutations, reason)``: sends ``POST
  /reef/harness/proposals`` with the mounted release and the session id.
  Reef admits or refuses at once, and an admitted proposal goes through the
  gate like the method's own before it is served. A Reef without the route
  answers a tool error and the turn continues.

The order is inspect, then try, then propose. Every call is a ``tool/call``
and ``tool/result`` pair in the session log, so what the model learned about
itself and what it changed is in the record.

The serve process runs on your machine with your privileges and imports
tree code in process, as the episode form does; the gate's sandbox does not
apply to it. Under ``--follow head``, whoever can publish to the scenario
runs code on the machine the process serves on. ``--follow pinned`` keeps a
person in that loop.

Write a method
--------------

Reef ships no proposer and no episode scorer. You supply ``propose``,
``evaluate``, and optionally a selection policy; `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ documents the contract, with worked examples.

Connect a different agent
-------------------------

An adapter is one descriptor: where each node kind is written, which kinds
the agent accepts, how the binary is launched, and where the proxy captures
the model calls. The bundled descriptors cover these agents:

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
| ``terminus``  | Terminus 2 (Terminal-Bench)      | config, rules, agent_command, skill    |
+---------------+----------------------------------+----------------------------------------+
| ``native``    | Reef's own loop                  | the five above plus native_tool,       |
|               |                                  | native_hook, native_graph,             |
|               |                                  | native_agent                           |
+---------------+----------------------------------+----------------------------------------+

`Harness adapters <../developer-guide/harness-adapters.rst>`__ is the descriptor reference and
how to connect an agent that has no adapter yet.
