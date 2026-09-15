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
``native_hook``, ``native_loop``), ``config.name`` is the file name the entry
renders to. Ten kinds are registered in
`reef/harness/tree/nodes.py <../../reef/harness/tree/nodes.py>`__:

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
| ``native_loop``    | the native loop itself as code: ``run_turn(ctx)`` over   |
|                    | the context API; always reviewed                         |
+--------------------+----------------------------------------------------------+

The table describes what each kind contains. Where each kind is written is
decided by an adapter, which maps every kind to a concrete file for one agent.
Reef bundles adapters for third-party coding agent CLIs (``pi``, ``opencode``,
``claude``, ``codex``, ``dsh`` (DeepSeek Harness), and ``hermes`` (Hermes
Agent)); ``native``, its own agent: a loop inside the reef tree whose tools
are ``native_tool`` nodes, whose loop events listen to ``native_hook``
nodes, whose control flow is a ``native_graph`` node, whose helpers are
``native_agent`` nodes a graph can call, and whose loop can be a
``native_loop`` node written as code, so the agent can evolve the tools it
runs, how its loop reacts, the loop itself, and who it delegates to, not
only the text around a vendor binary; and ``terminus``, Terminal-Bench's
Terminus 2, run through a Reef-owned Harbor runner. Only ``native`` renders
those five kinds.

A ``native_loop`` node goes one step past a graph: its ``code`` defines
``run_turn(ctx)``, and that function runs the root turn in place of the graph
interpreter. ``ctx`` is the context API Reef owns: ``ctx.model()`` takes one
model step, ``ctx.run_tools()`` runs the last message's tool calls,
``ctx.agent(name)`` hands text to a ``native_agent`` and returns its answer,
``ctx.say(text)`` adds a user message, ``ctx.end(reason)`` ends the turn, and
``ctx.log(event, data)`` writes a ``loop/<event>`` line to the session. Hooks,
tools, budgets, the session log and the sandbox stay Reef's; a loop that
raises, or calls the context past its budget of transitions, ends the turn
with ``LOOP_ERROR`` and fails the checks, and the first end of a turn is final
whatever the loop code catches. Loop code runs inside the loop's own
process, so a win that creates, changes or removes a ``native_loop`` is always
a pending release a person promotes, whether or not ``evolution.review_kinds``
names the kind, and ``harness_try`` refuses to mount one on a serving process:
the model proposes a loop, a person serves it.

Codex and Terminus support ``config``, ``rules``, ``agent_command``, and
``skill``. Codex rejects ``code_extension`` because lifecycle hooks run outside
its command sandbox. Terminus accepts one Python module defining
``Agent(Terminus2)`` when Reef's sandbox isolates the runner and Harbor uses
remote E2B tasks. See the adapter guide for the required deployment settings.

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

The loop below runs in the default ``data.training_mode: auto``, from
failures alone; an ask is refused there. A deployment that also takes asks
sets ``data.training_mode: hybrid``, as the tutorial's ``deployment.yaml``
does: ``POST /reef/train`` with ``text``, ``session`` and ``release_id``
(see the `manual training API
<../reference/http-api.rst#manual-training>`__) queues an instruction, and
the next step that reads it runs it first, oldest first, one per step,
with no call to the update route; the failure path continues between
instructions. ``data.training_mode: manual`` runs instructions only:
ordinary traffic never starts evolution in that mode. No inference receipts
or failure report are needed for an ask. A request
whose text is credential shaped or directive shaped is refused with the
rule named, and the scenario must already exist. A request whose step fails
is not retried: it is consumed with a ``skipped`` row in the catalog that
carries the error, and you send it again if you want it run; the failures
beside it in ``hybrid`` stay held for the next step. The step cap and the
failure streak count every step, a request's step included, and stop
automatic steps only; a request still runs past them.
``POST /reef/scenarios/{scenario}/update`` switches a running deployment
between the three.

The proposer must explicitly accept ``requests`` before the recipe builds
in ``manual`` or ``hybrid``. It receives one request mapping containing
``id``, ``text``, ``session``, ``release_id`` and ``untrusted=True``,
together with the current harness and model bindings. In ``hybrid``,
``samples`` carries what an automatic batch would take next, up to
``batch_size`` and possibly none (scored traces, or
records under ``data.batch_policy: records``), so the method answers the
request with the failures beside it; in ``manual`` it is empty. Its mutations pass
through the same evaluation and ``evolution.publish`` policy. Pending agent
proposals and periodic rollback rechecks cannot take the step an
instruction owns. The tutorial's ``propose`` takes ``requests``; a
failure-only proposer must be extended with a ``requests`` branch first.

.. flow::
   :loop: publish the winner, or restore the snapshot

   Batch :: scored reports with existing inference references
   ``propose`` :: one proposal, a mutation or a sequence applied as one, or ``None``
   Episodes* :: run the candidate and current tree on the same tasks
   Result :: publish the candidate or restore the snapshot

With the reports policy, inference traffic alone does not trigger evolution.
Every valid scored report with at least one existing inference reference
contributes a trace, including successful outcomes. A report over one
receipt batches as that exchange; a report over several batches as one
trajectory sample carrying every referenced exchange in order, which is what
``reef-pi report`` sends for a whole run (``--per-receipt`` fans the score
across the receipts as separate reports instead). When ``batch_size``
trace samples have accumulated, one step runs the loop once. With
``evolution.promote_failures: true`` a failing trace's prompt is added to the
evaluation as a permanent task, so the seed tasks are the floor of a suite that
grows from real failures and no later candidate can win while bringing one
back (the method's ``evaluate`` must score an arbitrary prompt); an
instruction step in ``hybrid`` promotes the failures it carries the same way.
A prompt is
real traffic, so it meets the tree's own credential tripwire first: a prompt
carrying a key-shaped literal is never promoted, never persisted, and never
re-run as a task, and the step goes on without it. A prompt shaped like an
instruction override (``ignore the previous instructions``, a forged system
message, a chat-template control token) is screened the same way, and one
tagged client holds at most ``evolution.max_promoted_per_client`` promoted
tasks, so a single sender cannot fill the suite. Which prompts are
promoted is the method's call: an optional ``evolution.promote`` names a
``Promoter`` subclass or instance. Its ``__call__(samples, *, manifest=None)``
receives the step's trace samples and failure manifest and returns the prompts to promote; without it
every failing trace's user prompt is promoted. Reef still dedupes, screens,
and caps whatever it returns. ``batch_size`` lives under ``data:`` in the
recipe config, and
``data.batch_policy: records`` drops the report requirement entirely:
recorded traffic alone batches, unscored, for methods that judge for
themselves.

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

Two more settings shape the search itself. ``evolution.min_win_margin: M``
(0 by default) is a noise floor on the result: the candidate must win more
than ``M`` task pairings beyond its losses, so on a stochastic episode a
single lucky flip does not publish. ``evolution.max_rejected_history: N``
(25 by default, 0 off) keeps the last ``N`` rejected proposals in the
scenario state, each with its step, its mutations with the options they
carried, and the result's reason; a ``propose`` whose signature names
``rejected`` receives them and can stop re-proposing what the evaluation already
refused.

By default a successful evaluation is served at once. ``evolution.publish: review`` holds
every win as a pending release instead, and ``evolution.review_kinds`` (a
list of node kinds, empty by default) holds only the wins that touch those
kinds, so ``[code_extension]`` lets rules and config auto publish while code
waits for a person; a win that touches a ``native_loop`` waits whether or not
the list names it. A pending release sits in the catalog with its evaluation
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
credentials out of candidate compositions. ``egress_hosts`` currently enables
network access; it does not enforce a hostname firewall.

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
result is a rejection, Reef restores the snapshot it took before the
mutation. Every result is recorded in the scenario's commit log together
with its mutation (op, id and the full options, so a rejected rewrite is
readable too), both score vectors, how many model calls the proposer made,
the seconds they took and the tokens the endpoint counted for them
(``proposer_calls``, ``proposer_seconds``, ``proposer_input_tokens``,
``proposer_output_tokens``; the tokens are recorded, never charged), and per
side and task the path each episode took: on the native harness the stage
names the loop exited in order and the reason its turn ended
(``candidate_paths`` and ``current_paths``, one ``{stages, reason}`` per
episode, with ``error`` and ``errored_agent`` when a turn ended on an error,
beside ``candidate_agents``).

The commit log holds the result; the step record holds what decided it.
``evolution.step_record_dir`` (off by default) names a directory, made
absolute at build, under which each scenario's steps write
``<scenario>/<step>/proposer.json``, one entry per model call the proposer
made: the ``model``, the ``messages`` and ``params`` of a ``chat`` or the
``body`` of a ``complete``, then the ``reply`` and provider ``response``
for a built-in ``chat`` binding, the ``response`` for ``complete``, or the
``error``, and the ``seconds`` it took; ``<scenario>/<step>/mutations.json``,
the parsed proposal with its options, written before admission so a refused
proposal is on file; and ``<scenario>/<step>/episodes/<side>-<task index>/``,
each evaluation episode's trajectory files as the adapter writes them
(``session.jsonl`` and ``agents/*.jsonl`` on native, the vendor's own session
tree on pi and the others) copied out of the throwaway root before it is
removed, beside an ``episode.json`` with the task, the exit code, stdout and
stderr, the residue, the score, the failure and the stage path, so a scorer
can be replayed from the record alone. Long text is clipped with a marker
naming what was dropped, and a credential shaped literal anywhere in the
record is replaced by ``[redacted credential]``: the record holds what the
tree boundary has not seen yet. Provider reasoning remains separate from
the final reply: Chat Completions responses keep ``reasoning``,
``reasoning_content`` and ``reasoning_details`` as returned; Messages keeps
thinking content blocks, and Responses keeps reasoning output items.
Streaming responses retain these fields too. Opaque encrypted blocks and
signatures are retained as provider data, not converted into readable
thinking. A provider that returns no reasoning, an older record, or a custom
text-only binding has none to display; Reef does not reconstruct it.
A proposer failure keeps its ``step_record`` directory on the instruction's
failed commit, including after the trainer reloads. A later retry points to
its own directory. Older failed commits that did not record this link are
not matched to files by directory order or timestamps.
A recheck step asks the proposer nothing, so
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

To run a harness-evolving deployment without the example's driver, start
the built-in Reefine profile and name the model:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b

The profile is Reefine's own default (loopback, port 8901, token
``reef-local``, state under ``.reef/reefine/``); its proposer and evaluator
ship in the wheel, so it needs no checkout, and ``--recipe harness-evolve``,
the former name of the profile folded into it, starts the same profile. This
example's own stack stays in ``configs/serve.yaml``, which ``run.sh`` passes
with ``-c``. `The CLI reference <../reference/cli.rst>`__ describes provider
settings and legacy shorthand.

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
published release, the evaluation metrics, and the evolved ``SKILL.md``,
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
window, the served model proposed a new skill beside the starter, and the evaluation
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
model binding pointed at the address the script came from, which behind a
gateway is the gateway's (Reef reads ``x-forwarded-host`` and
``x-forwarded-proto`` when a proxy sets them); the served tree itself carries
no endpoint or credential, and the binding takes its token from
``REEF_TOKEN`` in your shell when the script runs. It also puts a
``reef-<adapter>`` wrapper (here ``reef-pi``) on your PATH; the wrapper runs
through the interpreter that imported reef when the script ran and reads the
token back from the binding, so the shell that runs it later needs neither
on its own. The wrapper keeps
the receipts from a run, so ``report`` only needs the result. ``reef-pi doctor`` prints one line per thing the install needs
(the interpreter and its imports, the service and its token, the binary,
the tools on PATH, the installed release against the served head) and exits
0 when they all hold; it also lists every release that waits for your
review, in the words ``reef-pi harness --wait`` prints. ``reef-pi --help``
(``-h``, ``help``) prints the wrapper's own subcommands (``report``,
``harness``, ``page``, ``doctor``, ``setup``, ``update``; anything else
runs pi) before pi's help. Pinning,
rollback, and the raw manifest routes are in `HTTP API
<../reference/http-api.rst#harness-artifacts>`__.

You can also ask for a harness change in plain words. The tutorial's
``deployment.yaml`` runs in ``data.training_mode: hybrid``, so an ask needs
no mode switch there; a scenario in ``auto`` takes asks after a switch to
``hybrid`` or ``manual``:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"training_mode": "hybrid"}' \
     "$REEF_URL/reef/scenarios/code-repair/update"
   reef-pi harness "run the tests before you report a fix as done"

The wrapper submits to ``POST /reef/train`` with the installed release id
from the release metadata file and the oldest pending session's id, or a fresh session id
when nothing is spooled. A request can execute without inference receipts;
captured receipts remain available for a later feedback report. Acceptance
returns a training record id and does not mean the change has passed the
evaluation: the wrapper prints ``watch it here: <link>``, the request's page
(``GET /reef/harness/requests/<id>/page`` with the scenario and the token
as query parameters, so a browser opens it as is), and says ``reef is
running the step; add --wait to stay here, or check /reef-versions later``.
With ``--wait`` (``--timeout SECONDS``, 1800 by default) it polls the
release catalog every 5 s for the step that consumed the request, says
``the step started; usually one to three minutes`` once the request's
record shows a step took it, and prints one line with the result and the
next action, quoting the request: a selected release to restart ``reef-pi``
for; a pending one with ``This release changes an extension, so it is not
installed until you promote it: /reef-versions <step> promote. Page:
<link>``; a rejected step with the evaluation's reason; a skipped step with why
(the proposer's own reason when the step recorded one, such as a failed
model call); ``not covered: ...`` follows when the step's review lists
points the change left out. The exit status is 0 for a selected or pending
release, 1 for a rejected or skipped step, 2 when the timeout passes first.
On a terminal the wrapper then hands you the next step: a selected release
asks ``Install now? [Y/n]`` and, on yes, runs ``reef-pi setup`` for it and
then ``reef-pi update``, closing with ``Installed release <id>. Restart
reef-pi to use it.``; a pending release names ``reef-pi page <step>`` to
read it, asks ``Promote now? [y/N]`` and, on yes, promotes it and installs
the new head the same way. Declined, or in a script without a terminal,
it prints the commands to run instead.
To return to failure driven
evolution alone, use the same update endpoint with
``{"training_mode": "auto"}``. The commands surface an error when the
scenario is in ``auto``.

With ``evolution.requests: true``, a tree that boots from the seed also
carries the pi ``/reef-harness <request>`` command, which uses the same manual
training API with pi's current session id. In the session the model first
thinks the request through and asks what is unclear, a few options plus a
typed answer per question, then files the request with the answers;
``/reef-harness --direct <request>`` files it as is, and either way the
filing answers with the link to the request's page. A footer status shows
the request queued, then the step running and for how long, and the
result is reported when it settles, with the same next actions as
``--wait`` and the step whose page has the details, as a message the chat
keeps beside a notice. This background result opens no dialog, so you can
keep talking to the agent. When ready, use ``/reef-versions <step> install``
for a published release, or ``/reef-versions <step> promote`` for a pending
one. These explicit actions ask for confirmation before installing or
promoting. Installation runs ``reef-pi update`` for that release, collects
its setup items and ends with ``Installed release <id8>. Type /reload to
load it now.`` You can also use ``reef-pi update`` and ``reef-pi setup``
from a separate terminal. A request filed before a restart, or settled
while you were away, is
reported at the next session start, where the update notice offers the
install. A session start also says the commands exist and counts the
releases awaiting your review, with the ``/reef-versions <step> promote``
that promotes one and then offers its install. Recovered trees keep their
existing entries, as with ``version_check``. The proposer must explicitly
accept ``requests``. The tutorial's proposer asks the served model for a
skill, rules entry, command, or extension, using the bundled
``reef-pi-extension-api`` skill as its extension reference. The native trainer
owns request persistence, scheduling, retry and acknowledgement, and records
``training_request: {id, session, release_id, text, requires}`` in commit
metrics.

Admission screens the proposed mutations and the evaluation evaluates them.
Reef's entries (``reef-version-check``, ``reef-requests``,
``reef-pi-extension-api``) are reserved ids no proposal may change. An
evolved extension runs in pi's process with your privileges, so the
tutorial's ``configs/deployment.yaml`` sets
``evolution.review_kinds: [code_extension]`` beside ``requests: true`` and
``version_check: true``: a release that touches one waits for
``POST /reef/scenarios/{scenario}/promote`` before any session installs it;
promote it as shown below. ``configs/serve.yaml`` and
``configs/serve-native.yaml`` stay in ``auto``, where an ask is refused, and
set none of the three. The built-in ``reef.recipe.reefine:ReefineRecipe`` is available through
``reef serve --recipe reefine``. ``tutorials/reefine/`` runs this path end to
end on one machine, from the ask to the install and a session on the new
tree, with a bug fix flow demo, a research loop demo and a measurement of
which requests passed the checks (``./run.sh bugfix``, ``./run.sh research``,
``./run.sh measure``).

A release can need something from you before it runs. A request may carry
``requires``, a list of ``{name, kind, check, prompt}`` items:
``permission`` (an OS permission you grant), ``env`` (a variable you set;
the extension reads it from the environment, and its value never goes to
reef) or ``service`` (an account or endpoint you connect), each with an
optional ``check``, the variable name for ``env`` and a shell command that
exits 0 once satisfied for the other two, and an optional ``prompt``, one
sentence saying what to enter or grant. The proposer adds items of its own
when the extension it wrote needs them. A releases row carries what its
own change named; the manifest carries what the release needs over its
whole chain, so a later change that names nothing still needs what an
earlier one added. The install script refuses a release with an item you
have not checked off: it prints the setup list and the newest release in
the chain that requires nothing, the one that installs on a machine with
nothing set up (``?release_id=<id>``), and exits 1 before it installs or
writes anything. ``reef-pi setup`` is the one place a check runs: it lists
the newest release's items with each check as written and its prompt,
asks ``run it? [y/N]`` before running a command (``--yes`` answers for
scripts), records what passed in the ``.reef-harness-release`` release
metadata file under ``setup`` with the check it stood for, and exits 0
once every item is met; ``reef-pi setup --mark <name>`` checks an item off
by hand, and ``reef-pi setup --release <id>`` reads a pending release's
items, so you check them off before you promote it. An item whose check
changed since its check off is asked again.

An ``env`` item is met when its variable (the check, else the name) is set
in your environment or in the env file, ``<install root>/.reef-harness-env``
beside the release metadata file. When it is set in neither, ``reef-pi
setup`` shows the prompt and asks you for the value (without echo when the
name contains TOKEN, KEY, SECRET or PASSWORD; ``--yes`` asks nothing) and
stores it there: ``NAME=VALUE`` lines, readable by you alone (mode 0600),
written by ``setup`` only, never in the tree and never sent anywhere. Every
``reef-pi`` session gets each stored variable in its environment unless
your shell already sets it (the shell wins), so an evolved extension reads
``process.env.NAME`` and keeps no file of its own. Three more forms take
one item at a time, for scripts and for the session's extensions:
``reef-pi setup --json`` prints the release to set up and its items as one
JSON object, ``{"release_id": ..., "items": [{name, kind, check, prompt,
met}, ...]}``, and runs nothing; ``reef-pi setup --set NAME=VALUE`` stores
the value of the ``env`` item NAME and checks it off (exit 2 for an
unknown or non-env name; the value is an argument, never shell source);
``reef-pi setup --run NAME`` runs that item's check without asking and
checks it off when it passes (exit 0 when met, 1 otherwise, 2 for an
unknown name). ``reef-pi update`` fetches the install script for the
served head (``--release <id>`` for another release) with your token and
runs it for your install root, printing the installed release; while the
release requires an item you have not met, it prints the items and exits
3 without installing. On a fresh machine install the release the refusal
names first (it requires nothing, so ``reef-pi`` exists), run ``reef-pi
setup`` for the head's list, then ``reef-pi update``. Until every item is
met the update notice, in a session with the ``reef-pi`` wrapper on disk,
asks ``Set up release <id8> now?`` with the list and collects what is
missing the same way (each item once, a check only after your yes), then
offers the install, which runs ``reef-pi update`` and ends with
``Installed release <id8>. Type /reload to load it now.``; without the
wrapper, or headless, it prints the setup list instead of offering the
install. A session that starts on a tree with an unmet item prints the
list once and runs anyway. No check runs at install, and none at session
start without your yes.

See what a version is with ``/reef-versions`` in a ``reef-pi`` session: one
line per catalog row, oldest first, with the step, the first eight characters
of the release id, the result (``selected``, ``rejected``, ``skipped``,
``pending``, ``promoted at step N`` once a later promote serves a pending
release, else the row's operation: ``creation``, ``promote``, ``rollback`` or
``recovery``), ``current`` on the served head and the request text the step
answered. ``/reef-versions <step>`` prints the link to that step's page,
``GET /reef/harness/releases/<step>/page`` with the scenario and the token
as query parameters so a browser opens it as is, one self contained HTML
page with
five sections: Why (the request, else the proposal's reason, else a failure
in the batch), What changed (the mutations; an extension update as a line
diff against the release it ran on), Result (the evaluation's result and numbers,
and the step record directory when ``evolution.step_record_dir`` is set),
Setup (what the release needs from you: the step's own items, then those
carried from earlier steps) and Chain (the parent, this release, and its
children: the steps evaluated on it and any promote or rollback made on it; for
a rejected or skipped step, the head it ran on). For a pending release the
command also prints the promote curl, a trial install with ``?release_id=``
that replaces the tree at your install root, and the head's reinstall to
return to it; ``/reef-versions <step> promote`` runs the promote from the
TUI after you confirm it, then asks ``Install release <id8> now?`` and runs
the install and setup flow. ``/reef-versions <step> install`` starts that
flow for a published step when you are ready. When the step recorded the
proposer's plan and its review, ``/reef-versions <step>`` also prints
``design:`` and ``not covered:``. The command also prints a curl that
fetches the page with the
scenario header and the token into a file, for a hosted deployment where
the link is not enough, and ``reef-pi page <step>`` fetches it the same way
into ``$XDG_CACHE_HOME/reef-harness/<scenario>-step-<step>.html``
(``~/.cache`` by default), prints the path and opens it with ``open`` or
``xdg-open``; ``--print`` prints the path and opens nothing.

The native adapter's binary is ``reef-native``, which ships with reef, so
the install route serves no script for it. Pull the tree with the client,
name your Reef URL in its ``native/models.json``, and run the wrapper module
with the same five settings the script bakes into ``reef-pi``:

.. code:: bash

   python3 -c 'from reef_client import ReefClient; ReefClient("http://127.0.0.1:8900", token="reef-local").harness_pull("harness-evolve-demo", "./reef-harness")'
   printf '{"api": "openai", "base_url": "http://127.0.0.1:8900", "api_key": "reef-local", "model": "qwen3-8b"}\n' > reef-harness/native/models.json
   export REEF_HARNESS_BINARY="$(command -v reef-native)" REEF_HARNESS_COMPOSE="$PWD/reef-harness/native"
   export REEF_HARNESS_SCENARIO=harness-evolve-demo REEF_HARNESS_ADAPTER=native REEF_HARNESS_ENV_VAR=REEF_NATIVE_DIR
   python3 -m reef.harness.client.wrapper -p "fix the failing test in auth.py"
   python3 -m reef.harness.client.wrapper report --score 0 --feedback "missed the empty-token case"

The wrapper points the loop at its capture proxy through a temp copy of
the tree, keeps the loop's session log under ``native/sessions`` beside
the installed tree, and ``report`` works as for any adapter.

Promote a pending release
~~~~~~~~~~~~~~~~~~~~~~~~~

A win that touches a kind in ``evolution.review_kinds`` (``code_extension``
in the tutorial's ``deployment.yaml``) or a ``native_loop`` sits in the
catalog with ``pending: true`` and is served to no session until you promote
it. The notice never offers it either: it offers the newest release that is
not pending, so a pending release shows only under a promote or a trial
install by id. Find its id in ``GET /reef/harness/releases`` (the newest row
marked ``pending``), read the change (``?release_id=<id>`` on the install
route installs that tree for a trial session), and name it to ``POST
/reef/scenarios/{scenario}/promote``. The answer is the new head with a fresh
release id, because a promote republishes the tree as a commit of its own;
the next ``reef-pi`` session offers the update through the notice, and
``/reef-versions <step> promote`` in a session offers it at once. Both
calls name the scenario your install used: the ``x-reef-scenario`` header
you gave the install command or, without one, the generated name the script
baked into ``reef-pi`` as ``REEF_HARNESS_SCENARIO``;
``grep REEF_HARNESS_SCENARIO ./reef-harness/reef-pi`` prints it. The
deployment listens on port 8901.

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: <scenario>" \
     http://127.0.0.1:8901/reef/harness/releases    # the row with "pending": true
   curl -sS -X POST -H "Authorization: Bearer reef-local" \
     -H "Content-Type: application/json" \
     -d '{"release_id": "<the pending release id>"}' \
     http://127.0.0.1:8901/reef/scenarios/<scenario>/promote

Serve the harness as a resident process
---------------------------------------

The native adapter has a second form. ``reef-native -p`` is the episode
form: one process, one turn, what the evaluation runs. ``reef-native serve`` is
the serve form: one resident process per installed tree that holds the tree
as a live composition and follows the release Reef serves while it runs. A
publish reaches the process as a mount between two steps of the open turn,
or at once when no turn is open. No reinstall, no restart.

.. code:: bash

   reef-native serve --tree ./reef-harness --scenario harness-evolve-demo &
   reef-native turn --tree ./reef-harness -p "fix the failing test in auth.py"
   reef-native turn --tree ./reef-harness -p "now add a test for it" --session 3f9a1c2b7d4e
   reef-native status --tree ./reef-harness
   python3 -m reef.harness.client.wrapper report --score 1 --feedback "fixed"

``--tree`` names the pulled tree, the directory that holds ``native/`` and
the ``.reef-harness-release`` metadata file. The process boots from ``native/tree.json``, the
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
module binds no ``run`` at its top level, a hook whose code does not
import, a kind this reef has no plugin for, a name a self tool owns) is
rolled back whole before the next step: ``harness/mount-failed`` names the
release, the entry and the error, and the previous composition keeps
serving. On success the release metadata file and ``native/tree.json`` name the new
release, so a restart boots from it with ``source: boot``.

With ``--follow pinned`` the process logs ``release/available`` with the
new head and waits for a person. ``reef-native mount <release_id> --tree
./reef-harness`` applies one release by hand, an older one included, which
rolls the process back locally; under ``head`` too, a release mounted by
hand stands until the head moves again. A Reef that does not answer logs
``release/poll-failed`` with the error and the next retry, doubling up to
ten minutes, and the process keeps serving. A Reef that is busy is polled
again at the interval: a catalog read waits behind a running evolve step,
so a poll that times out during one retries at the interval and the head
lands as soon as the step ends. A manifest read that times out the same way
logs ``harness/mount-failed`` and is retried by the next poll that names
the head, so a release published between two steps mounts once the steps
are over.

``--self-tools`` gives the model three built-in tools. The tree cannot
remove them or take their names, and they are absent in the episode form,
so a candidate cannot pass the checks by calling them:

- ``harness_inspect(what)``: ``tree`` is the live entries and the mounted
  release; ``graph`` is ``main`` and every named graph; ``results`` (also
  accepted as ``verdicts`` for existing sessions) is the
  newest releases with the evaluation metrics that admitted each, and the
  rejected proposals when Reef exposes them; ``status`` is the status above.
- ``harness_try(mutations)``: mounts the served entries plus the mutations
  on this process for the rest of the turn. The change applies from the
  next step, the model calls carry ``x-reef-tag-trial`` so ``report`` skips
  their receipts, and at ``turn/end`` the served entries are mounted back
  (``harness/unmount``). Nothing is published.
- ``harness_propose(mutations, reason)``: sends ``POST
  /reef/harness/proposals`` with the mounted release and the session id.
  Reef admits or refuses at once, and an admitted proposal goes through the
  evaluation like the method's own before it is served. A Reef without the route
  answers a tool error and the turn continues.

The order is inspect, then try, then propose. Every call is a ``tool/call``
and ``tool/result`` pair in the session log, so what the model learned about
itself and what it changed is in the record.

The serve process runs on your machine with your privileges and runs tree
code in process, as the episode form does: every hook imports and listens
there, and a tool call runs there unless ``REEF_NATIVE_ENFORCE=bwrap`` is
set; the evaluation's sandbox does not apply to it. Under ``--follow head``,
whoever can publish to the scenario
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
| ``terminus``  | Terminus 2 (Terminal-Bench)      | config, rules, agent_command, skill,   |
|               |                                  | code_extension (sandbox + E2B)         |
+---------------+----------------------------------+----------------------------------------+
| ``native``    | Reef's own loop                  | the five above plus native_tool,       |
|               |                                  | native_hook, native_graph,             |
|               |                                  | native_agent, native_loop              |
+---------------+----------------------------------+----------------------------------------+

`Harness adapters <../developer-guide/harness-adapters.rst>`__ is the descriptor reference and
how to connect an agent that has no adapter yet.

.. seealso::

   `Scenario model configuration <scenario-models.rst>`__ explains scenario-specific custom providers for the entire
   harness evolve model pipeline through Reef API Platform.
