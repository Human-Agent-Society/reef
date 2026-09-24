Harness adapters
================

An adapter maps a harness tree into the files expected by a third-party
coding-agent CLI and binds that harness to the served model. The harness and
model together form the running agent. The tree never names a file path; the
adapter does. Reef bundles six, one per third-party coding-agent CLI;
``native``, its own agent, whose loop lives in this tree, whose tools are
``native_tool`` nodes, and whose loop events listen to ``native_hook`` nodes,
so a mutation can add, rewrite, or remove a tool, or change what the loop
does at an event; and ``terminus``, Terminal-Bench's Terminus 2, a Harbor
agent class rather than a CLI, driven by a runner Reef owns.

+--------------+-----------------------------------------------------------+-------------------------------------------+
| Adapter      | Config targets                                            | Install pin                               |
+==============+===========================================================+===========================================+
| ``pi``       | ``primary`` → ``pi-agent/settings.json``,                 | npm ``@earendil-works/pi-coding-agent``   |
|              | ``models`` → ``pi-agent/models.json``                     | 0.84.2                                    |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``opencode`` | ``primary`` → ``opencode/opencode.json``                  | npm ``opencode-ai`` 1.18.18               |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``claude``   | ``primary`` → ``claude/settings.json``                    | npm ``@anthropic-ai/claude-code`` 2.1.257 |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``codex``    | ``primary`` → ``codex/config.toml``                       | npm ``@openai/codex`` 0.152.1             |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``dsh``      | ``primary`` → ``dsh/profiles/headless/cordis.patch.yml``, | npm ``@deepseek-ai/dsh`` 0.1.2-alpha.5    |
|              | ``env`` → ``dsh/.env``                                    |                                           |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``hermes``   | ``primary`` → ``hermes/config.yaml``                      | git ``NousResearch/hermes-agent``         |
|              |                                                           | at ``v2026.8.31`` (0.21.0)                |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``native``   | ``primary`` → ``native/config.json``,                     | none: ``reef-native`` ships with reef     |
|              | ``models`` → ``native/models.json``                       |                                           |
+--------------+-----------------------------------------------------------+-------------------------------------------+
| ``terminus`` | ``primary`` → ``terminus/config.json``                    | none: ``reef-terminus`` ships with reef,  |
|              |                                                           | reef-eval ships with reef-infra           |
+--------------+-----------------------------------------------------------+-------------------------------------------+

The ``terminus`` adapter is the one that does not drive a CLI. Terminus 2 is
a Harbor agent class, so the adapter ships its own runner,
``reef-terminus``: it reads the tree from ``REEF_TERMINUS_DIR``, hands it to
Harbor's own ``terminus-2`` agent as native configuration, runs the task the
prompt names, and writes the verifier's reward and the ATIF trajectory under
``REEF_TERMINUS_SESSION_DIR`` for the ``terminus-atif-json`` reader. It
reaches Harbor through reef-eval, the same primitive the examples under
``recipes/`` use. The prompt is a
Harbor task directory or a registry id, so an episode needs no dataset
location in its environment, which ``run_episode`` would not carry anyway.

``config`` becomes Terminus 2 constructor arguments, refused at render if a
key is not one; ``rules`` becomes an ``extra_instruction_paths`` entry; and
``skill`` and ``agent_command`` become two ``AgentConfig.skills`` roots, so
Harbor keeps its progressive skill loading rather than pasting every body
into the prompt. One ``code_extension`` may define ``Agent(Terminus2)``;
rendering checks syntax without executing it, and the runner uses Harbor's
native ``AgentConfig.import_path`` contract. No extension means stock Terminus 2.

Extensions require ``evolution.executor: sandbox`` to isolate the Python
runner. Harbor then runs the terminal
task remotely. Network access must be enabled with ``sandbox.egress_hosts``;
that setting currently enables networking without enforcing a hostname firewall.
The runtime needs Linux, bubblewrap, Python 3.12+ and ``harbor[e2b]``. The interpreter and local task directories must be visible in
the sandbox (for example under ``/opt``). Ordinary declarative trees can still
use the local executor and Docker. Docker inside bubblewrap and extensions in
an unisolated runner are rejected before process launch.

An adapter quirk can expose an ``ExecutionValidator`` instance as
``validate_execution``. Its ``__call__(files, executor)`` checks the rendered
tree against the configured executor before any episode files are written.
It raises ``EpisodeLaunchError`` for unsupported combinations and
replaces the default ``self_isolating`` nesting restriction. The Terminus quirk
uses this seam; execution, timeout, cleanup and trajectory handling remain shared.

The ``dsh`` adapter runs DeepSeek Harness headless (``dsh --profile headless
"<task>"``) with its whole home relocated by ``DSH_HOME``. dsh composes its
plugin tree from bundle layers plus one user patch layer, a YAML list of
entries addressed by plugin id, so its ``primary`` config target is an
object keyed by plugin id (``{"agent-loop": {"config": {...}}}``, or
``{"disabled": true}``) that the adapter's quirks emit as that list. A
string starting with ``!!js `` becomes a js expression, the form dsh's own
bundles use. The adapter's defaults keep the session log uncompressed and
the telemetry and the LLM title call disabled, and a composition that flips
any of them is refused at render. Rules render to dsh's user global
``AGENTS.md``; skills to ``skills/<name>/SKILL.md`` (dsh needs YAML
frontmatter with ``name`` and ``description``, synthesized when the node
text has none); an ``agent_command`` renders as a user invocable skill
(``disable-model-invocation: true``, run as ``/name``) under the second
skill root ``DSH_AGENTS_HOME``, the only command surface dsh has; a
``code_extension`` renders as a plugin module the patch layer inserts by
relative path. The model binding declares an ``llm-pi-ai`` route whose key
is named by ``apiKeyEnv`` and supplied through the ``env`` target, dsh's
``.env`` launch environment layer.

The ``hermes`` adapter runs Hermes Agent headless (``hermes chat -Q --oneshot
-q "<task>"``) with its whole home relocated by ``HERMES_HOME``. Its
``primary`` config target is ``config.yaml``, which the quirks emit as YAML
from the merged JSON object; the defaults keep an episode hermetic and single
request: the terminal scanner download off (``approval.tirith_enabled``), the
session title call off (``auxiliary.title_generation.enabled``), the memory
nudge that spawns a background review off (``memory.nudge_interval: 0``), and
the per session JSON snapshot on (``sessions.write_json_snapshots``), which is
the trajectory the ``hermes-session-json`` reader parses. A composition that
flips any of them is refused at render. The quirks also write the
``.no-bundled-skills`` marker, so an episode carries the tree's skills and not
hermes's bundled catalog. Rules render to ``SOUL.md``, the one home level
rules file hermes reads (``AGENTS.md`` is project scoped, read from the
working directory chain); skills to ``skills/<name>/SKILL.md`` with the
``name`` and ``description`` frontmatter hermes requires synthesized when the
node text has none; an ``agent_command`` to a second skill root,
``hermes-commands``, that ``skills.external_dirs`` lists, since every hermes
skill is also a ``/name`` slash command in the interactive CLI and hermes has
no other command surface; a ``code_extension`` to a plugin package
(``plugins/<name>/__init__.py`` defining ``register(ctx)``) whose manifest,
``plugins.enabled`` entry, and ``tools.override`` grant the quirks write,
because hermes loads no plugin without that consent; a plugin tool then sits
behind hermes's ``tool_search`` and ``tool_call`` discovery surface. The model
binding is a custom provider with a literal key in ``config.yaml``; only the
``openai`` dialect is bound. hermes's own default approval policy runs tools
inside the working directory with no prompt and refuses a command it flags as
dangerous with a tool error, so no bypass flag is used.

The native adapter also renders the optional ``native_tool`` kind to
``native/tools/{name}.py``: a module holding the node's ``code``, which
defines ``run(args, workdir) -> str``, and after it ``NAME``,
``DESCRIPTION``, ``PARAMETERS`` and ``CAPABILITIES`` from the node config, so
the tree's values are what the module ends with whatever the code assigned.
``capabilities`` is optional: distinct names from ``read``, ``write``,
``exec`` and ``network`` that say what the tool does. The loop reports them
in the session header and hands them to ``pre_execute`` hooks. Under the
``local`` executor nothing enforces them: the tool runs in the loop's own
process. Under the ``sandbox`` executor, which sets
``REEF_NATIVE_ENFORCE=bwrap`` for the episodes it launches, the loop runs
each call in a child process under a bubblewrap profile derived from the
declaration (it binds the episode's ``/proc`` read only, since a jail inside
the episode's cannot mount a fresh one): without ``network`` the call gets an
empty network namespace;
without ``write`` the workspace is bound read only; without ``exec`` only
library directories, the interpreter file running the tool and its prefixes
are bound, so no shell exists in the jail (``/bin`` and ``/usr/bin`` are
absent; the interpreter's own prefix may put an empty ``/usr/local/bin``
there) and ``PATH`` is unset besides. ``subprocess.run(["bash", ...])`` then
fails with a missing file: Python falls back to searching ``/bin:/usr/bin``
when ``PATH`` is absent, and those directories are not there. The absent
directories are the denial; binding one of them for any reason reopens
``exec``. bwrap cannot deny the rest: the tool can still start
``sys.executable``, run an executable installed under a library directory
or under a path it can write (``/tmp`` inside the jail is a private tmpfs),
and read the workspace, so ``read`` is never withheld. The enforcer is
chosen before any module of the tree runs in the loop's process, so a tree
cannot choose it; the loop refuses to start when the variable names
``bwrap`` and no ``bwrap`` is on ``PATH``, and a call the jail could not run
at all ends in ``SANDBOX_FAILED`` rather than passing as a tool failure and
counts as no tool error; the sandbox executor's preflight runs one jail
inside another, so a host that cannot nest them fails at build, not at the
first call. Every ``tool/result`` event carries ``enforcement`` with the
``mode`` (``none`` or ``bwrap``) and ``denied``, the declaration's
complement over ``write``, ``exec`` and ``network`` (empty under ``none``);
``denied`` is what the profile withholds, not an observation of what the
call tried. The seed tools declare theirs; ``run_bash`` declares all three a
shell can do.

The per call jail confines a tool's ``run`` and the module that defines it.
The loop reads a tool module's declaration (``NAME``, ``DESCRIPTION``,
``PARAMETERS`` and ``CAPABILITIES``) from the source with
``ast.literal_eval`` and imports the module only where the call runs: in
the child under the profile, or, under the ``local`` executor, in the
loop's process at the first call. So a tool's import time code no longer
runs in the loop's process at load under any enforcer, and under the
sandbox executor it never runs there at all; the trajectory's
``enforcement`` field describes the profile the call got, which is also
what the module's top level ran under. Hooks are the one thing a tree
carries that still runs in the loop's own process, by design: every hook
module is imported once at start and its ``listen`` runs at every event,
since ``next`` is a call into the layer below and a decision steers the
loop that is running. Under the sandbox executor that process is the
episode jail, which holds the writable workspace and session directory, the
network namespace the model endpoint needs, and the executor's base
directories with their shells; under the local executor it is the host. So
a hook is loop code with the loop's reach: ``review_kinds`` with
``native_hook`` is how a deployment puts a person between a hook proposal
and the tree; under the local executor, where nothing confines a tool
either, ``native_tool`` belongs in that list too.
``reef.harness.runners.native.seed.SEED_TOOLS`` holds the starting ``read_file``,
``write_file``, ``run_bash``, and ``execute`` tools as entries a recipe can
seed and the loop can then evolve; ``execute`` runs a Python block in the
workspace with the other tools importable by name (``import read_file;
read_file.run({"path": "x"}, WORKDIR)``), so a tree can move from one call
per tool to code that calls tools without a loop change. An adapter that declares no ``files.native_tool`` path
refuses to render that kind, so the mutation fails under it instead of
silently dropping the tool. The admission check refuses ``code`` that does not
compile; a tool module the loop cannot read (the file cannot be opened or
does not parse, no top level statement binds ``run``, or the last top level
assignment to a declaration constant is not a literal) ends the episode
with reason ``error`` and code ``LOAD_ERROR`` before any model call, so the
tree that carries it fails the checks instead of running without it; a file
that parses but does not compile fails its first call like a top level that
raises. The read takes the last binding at module scope in source order: it
follows the bodies of ``if``, ``try``, ``with``, ``for``, ``while`` and
``match`` statements, never a function or class body, and a ``def``,
``class`` or ``import``, an assignment, ``for``, ``with``, ``except`` or
walrus target, or a ``match`` capture binds a name; the render writes the
constants last, so they win. A top level that raises, ``SystemExit``
included, or one that binds ``run`` to something not callable, is not found
at load, since nothing runs it there: the first call to that tool fails
with ``TOOL_FAILED`` and the episode goes on. In the loop's process the
module imports once, so every later call fails the same way without running
the top level again; the child imports it afresh at every call.

The loop has four events, and a ``native_hook`` node listens at one of them.
It renders to ``native/hooks/{name}.py`` the same way: ``code`` defining
``listen(payload, next) -> decision``, then ``NAME`` and ``EVENT`` from the
node config. The hooks at one event form a waterfall in file name order: each
``listen`` may call ``next()`` to get the decision of the layer below (the
last layer is the loop's default) and return it, changed or not, or return
its own decision without calling ``next`` and so own the event. ``next`` runs
the layer below at most once however often it is called, and hands the hook
a copy, so an in-place edit is a change like any other. A hook that raises,
or returns anything but a plain object the log can carry, is skipped and the
layer below stands; ``messages`` and ``contexts`` are read as lists of text
and anything else in them is dropped. A hook module that fails to import,
defines no ``listen``, or names an unknown event ends the episode with
``LOAD_ERROR`` like a tool the loop cannot read. Every event takes a plain object and returns one:

.. config::

   pre_step | before each step: ``{step, task, messages}``; returns ``{kind: "enter", messages: [text...]}`` (each text becomes a user message before the request) or ``{kind: "reject", reason}`` (the turn ends with no step)
   pre_execute | before each tool call runs, after its arguments are validated: ``{step, call_id, name, arguments, capabilities}``; returns ``{kind: "allow", arguments?}`` (with ``arguments`` the call runs with the rewrite, validated like the model's own), ``{kind: "deny", reason}`` (the tool does not run and the model reads a ``HOOK_DENIED`` error carrying ``reason``) or ``{kind: "ask", reason}`` (a headless run has no one to ask, so the tool does not run and the model reads an ``APPROVAL_REQUIRED`` error carrying ``reason``); ``post_execute`` still sees the call, with that error as its result
   request_error | after a failed model call: ``{step, attempt, error}`` where ``error`` is ``{code: "MODEL_ERROR", message, status?}`` with ``status`` the HTTP status when the endpoint answered one; returns ``{kind: "retry", delay_ms}`` or ``{kind: "fail"}``; the loop spends at most ``MAX_REQUEST_ATTEMPTS`` (4) attempts a step and waits at most ``MAX_RETRY_DELAY_MS`` (10 s), whatever the hook asks
   post_execute | after each tool call has run: ``{step, call_id, name, arguments, result}``; returns ``{kind: "accept", content?, contexts: [text...]}`` (``content`` replaces what the model reads) or ``{kind: "block", feedback, contexts}`` (the model reads a ``HOOK_BLOCKED`` error carrying ``feedback``; the tool's side effects stand); contexts land as user messages after the step's results, in call order

``reef.harness.runners.native.seed.SEED_HOOKS`` holds the one starting hook,
``loop_guard`` at ``post_execute``, which reminds the model when the same call
repeats three, five, or eight times in a row; it is a node, so a tree can
retune or drop it. ``SEED_NODES`` is the tools and the hooks together, and
``tutorials/evolve-your-harness/configs/serve-native.yaml`` seeds them by reference to
run the tutorial on this adapter.

The native descriptor declares no path for ``agent_command`` or
``code_extension``: the loop never reads either, so a mutation of those kinds
is refused at admission ("does not render") instead of rendering a file
nothing loads. ``config`` keeps both targets, since the render needs
``primary``; the loop reads ``models`` only, and a live tree boot refuses a
``config`` entry with target ``primary`` or one that sets a pinned binding
field (``api``, ``base_url``, ``api_key``, ``model``).

The tree travels as a file
~~~~~~~~~~~~~~~~~~~~~~~~~~

The native descriptor also declares ``files.tree: native/tree.json``. Every
tree the backend renders for this adapter carries that file beside the
rendered ones: the release's entries list, verbatim, as one JSON array of
``{id, name, config}`` objects, the same list the commit log persists under
``algorithm_state["entries"]``. It reaches the evaluation episodes, the
published artifact, the manifest, the install script and a pulled tree
through the existing channel; the base release a seeded recipe serves before
any step carries the seed's list. The binding nodes never enter it: the
pinned model fields stay in ``native/models.json``.

At boot the loop reads the file when it exists: a fresh compose context, a
``Loader`` over ``NATIVE_PLUGINS``, ``root.update(entries)``, every entry
admitted again by its kind's plugin and installed into the host through the
same effects a resident process uses, with the tool, hook and loop modules
written under ``sessions/mounts/boot-<pid>/`` (the one writable path under the
sandbox).
An entry that does not end ACTIVE ends the episode with ``LOAD_ERROR``
naming the entry id, its kind and the fiber's error, or ``no plugin for kind
X`` for a kind the loop never reads, so a hand edited list cannot run
unchecked. Without the file the loop reads the rendered files as before, so
an older pulled tree runs unchanged. The two boots produce the same events;
the session header's ``tree`` field says which ran, ``tree.json`` or
``files``.

The loop's own control flow is a ``native_graph`` node, rendered to
``native/graphs/main.json``: named stages from a closed vocabulary and edges
keyed by each stage's outcome. ``reef.harness.runners.native.seed.SEED_GRAPH`` is
today's loop as that data (``think`` asks the model, ``act`` runs its tool
calls, ``done`` ends the turn), the loop runs it when a tree carries no graph,
and a tree that carries one runs that instead, so a proposal that rewrites
the graph changes what the loop does between its events while hooks keep
deciding at them. Admission refuses a graph that could not run: an unknown
kind or key (no code enters this kind), an outcome without exactly one edge,
a stage not reachable from ``start``, a stage from which no end stage is
reachable, and a cycle with no model stage, so the step budget
(``max_steps``, 1 to 32) ends every run; a ``tools`` allow list naming a
tool the tree lacks fails at render. The stages:

.. config::

   model | one request over the messages with the declared tools; fires ``pre_step`` and ``request_error``; outcomes ``tool_calls``, ``text``
   tools | runs the pending calls of the last assistant message, each behind ``pre_execute`` then ``post_execute``; optional ``allow`` restricts them to named tools; outcome ``done``
   verify | reads the last assistant text: ``check`` is ``last_line_integer``, ``last_line_matches`` with a ``pattern``, or ``nonempty``; an optional ``message`` is appended as a user message on failure; outcomes ``pass``, ``fail``
   message | appends ``text`` as a user message; outcome ``done``
   branch | routes on the run so far: ``cases`` is a list of ``{when, value, outcome}`` (at most 8) where ``when`` is ``steps_used_at_least`` or ``tool_errors_at_least`` with an integer ``value``, or ``last_text_matches`` with a regular expression; the first case that holds names the outcome, none names ``else``; every case outcome and ``else`` need an edge. A pattern, here or in ``verify``, is at most 200 characters and runs in a child process with one second of wall clock, since no static rule tells a pattern that finishes from one that never does; a search that outlives the clock is a case that does not hold or a check that failed, named ``timeout`` in the stage's detail, and a branch matches the last 4096 characters of the text
   subagent | hands the last assistant text (or the task) to the ``native_agent`` named by ``agent``, then down that agent's ``then`` pipeline; the last agent's text comes back as a user message with ``source.kind`` ``agent``; outcomes ``completed``, ``gave_up``, ``budget`` (the agent spent its steps or tool calls), ``ask`` (a ``pre_execute`` hook asked inside the agent's turn, and the reason is what comes back)
   compact | when the messages pass ``fire_ratio`` of the model's context window, one model call summarizes the older span into a user message and the last ``keep_ratio`` of the window stays verbatim (a tool result never opens the kept tail without its call); ``0 < keep_ratio < fire_ratio <= 1``; the window is ``context_window`` in ``models.json`` (a ``config`` node with target ``models`` sets it), 32,768 tokens when unset, at four characters a token; the summary call is not a step, and a cycle must pass a model stage, so a run spends at most one per step; outcome ``done``
   end | ends the turn with ``reason`` ``completed`` or ``gave_up``

Each model stage is one step, so ``max_steps`` bounds model calls as before,
and each call asks for at most 4,096 tokens, so one runaway reply cannot hold
a single slot local server for every other caller;
entering a model stage with the budget spent ends the turn with
``max-steps``. The log names the path: ``stage/enter`` (``step``, ``stage``,
``kind``) and ``stage/exit`` (``outcome``, ``to``, and for a verify stage
``check`` and ``last_line``, for a branch the ``case`` that held, for a
compact whether it ``fired`` and the token counts), a compact that fired
writes ``context/compacted`` (the ``policy``, ``tokens_before``,
``tokens_after``, the ``dropped`` message count, and the ``summary``; a
summary call that failed is logged with its ``error`` and drops nothing),
text a stage injects is a ``user/message`` with
``source.kind`` ``stage``, the session header's ``graph`` says whether
``main`` or the ``seed`` ran, and a graph that cannot load is a
``LOAD_ERROR`` like a tool. A run that somehow exceeds
``(max_steps + 1) * 16`` transitions ends with ``GRAPH_ERROR``; admission
proves that cannot happen, the guard is the backstop.

A ``native_agent`` node is one more agent inside the same tree, rendered to
``native/agents/<name>.json``: its own ``prompt`` (appended to the rules and
skills as its system prompt), the ``graph`` it runs (``seed``, the built in
loop, by default; ``main`` or any graph node by name), the ``tools`` and
``skills`` it alone sees (all of the tree's when unset), ``max_steps`` and
``max_tool_calls``, and ``then``, the agents its final text is handed to in
order, each receiving the previous one's text. A graph calls an agent from a
``subagent`` stage; the tree stays flat, agents are root entries, and render
refuses a name the tree lacks and any cycle through ``then`` lists and
subagent stages, so every delegation is a finite tree. An agent's turn runs
on the parent's remaining step budget (its steps come out of the episode
total) in its own session file under ``sessions/agents/``, numbered in run
order and sorting before the root's ``session.jsonl``, so the trajectory's
last assistant text stays the root's answer and which agent did what is read
off its file; its header names the ``agent``, its ``turn`` and its ``parent``. A
``pre_execute`` hook that answers ``ask`` inside an agent's turn ends the
turn with outcome ``ask`` instead of an ``APPROVAL_REQUIRED`` error, because
the parent graph is the one that can answer. The evaluation's result carries
``candidate_agents`` and ``current_agents``, the turns, steps, tool calls,
tool errors and, when the endpoint reported usage, the input and output
tokens per agent summed over each side's episodes. It also carries
``candidate_paths`` and ``current_paths``, one entry per episode in pairing
order (task by task, then repeat by repeat): the root session's
``stage/exit`` stage names in order and the ``turn/end`` reason kind, plus
``error`` when the turn ended on one and ``errored_agent`` when a delegated
agent's error ended the run before the root wrote its end; a delegated
agent's stages under ``agents/`` stay out of it; an episode that could not
run is ``None`` and a format without stage events gives an empty list and a
``None`` reason.

A ``native_loop`` node is the loop itself as code, rendered to
``native/loops/<name>.py``: the node's ``code``, which defines
``run_turn(ctx)``, then ``NAME`` and ``MAX_STEPS`` from the node config. When
a tree carries one, the root turn calls ``run_turn`` instead of walking
``main``; agents still run their graphs, and the loop reaches them through
``ctx.agent``. One loop per tree: render refuses a second ``native_loop``
node, the host refuses a second ``add_loop``, and the file form refuses two
files under ``loops/``, and admits the file's text before it imports it: a
file that does not parse, or whose ``NAME`` or ``MAX_STEPS`` is bound by
anything but the literal assignment the render wrote, is refused before the
import. Admission reads the code and never runs it: the module compiles,
carries no credential, and leaves ``run_turn`` bound to a plain top level
``def`` with a parameter (the last statement that binds the name at module
scope decides, including one inside an ``if``, ``for``, ``with``, ``try`` or
``match``, which is refused); ``max_steps`` (1 to 32, 12 by default) is the
loop's model step budget. The tree is flat: an entry with ``group`` is
refused at admission, at boot and at every mount, so no loop enters as
another entry's child. The context is the API Reef owns, each call a thin
call into the run:

.. config::

   ctx.prompt, ctx.step, ctx.max_steps, ctx.tools, ctx.messages, ctx.last | the task, the steps spent, the budget, the tool names the run may call, a copy of the messages and a copy of the last assistant message
   ctx.model() | one model step, the ``model`` stage: fires ``pre_step`` and ``request_error``, writes ``step/start``, ``request/header`` when what the model sees changed (always at step 1), ``assistant/message`` and ``step/end``; returns ``tool_calls`` or ``text``; a spent budget ends the turn with ``max-steps``. Returning after ``tool_calls`` without ``run_tools`` leaves those calls unanswered in the conversation for the next turn
   ctx.run_tools(allow=None) | the ``tools`` stage over the last message's calls, each behind ``pre_execute`` then ``post_execute``; ``allow`` narrows them to these names, and an empty or absent ``allow`` is no restriction, as in the stage
   ctx.text() | the last assistant text
   ctx.say(text) | a ``user/message`` with ``source.kind`` ``loop`` and the loop's name; counts as a transition
   ctx.agent(name, text=None) | one agent's turn: runs the named ``native_agent`` alone on ``text`` (the last assistant text, else the task); its ``then`` chain is not followed; appends its answer as a ``user/message`` with ``source.kind`` ``agent``; returns ``(outcome, text)``
   ctx.end(reason="completed") | ends the turn with ``completed`` or ``gave_up``; any other reason is a ``ValueError``
   ctx.log(event, data) | a ``loop/<event>`` line: the name must be a node name other than ``enter`` or ``exit`` and is always prefixed, so this call cannot write a core event; ``data`` is made JSON (keys as their text), and past 4096 serialized characters it is replaced by ``{"text": the first 4096, "truncated": true}``; counts as a transition

Returning from ``run_turn`` ends the turn ``completed``. A loop turn writes
``loop/enter`` (``name``) first and ``loop/exit`` (``reason``) before
``turn/end``, and no ``stage/*`` events, so its stage path is an empty list
with the turn's reason. The transition guard is the graph's: past ``(max_steps
+ 1) * 16`` calls to ``model``, ``run_tools``, ``agent``, ``say`` and ``log``,
or any exception out of ``run_turn`` (``SystemExit`` included;
``KeyboardInterrupt`` propagates), the turn ends with ``LOOP_ERROR`` and exit
status 1, and the evaluation ranks the episode as one that could not run; that leaves
about 16 context calls per model step, ``log`` and ``say`` included. The first
end is final: after ``max-steps``, ``ctx.end`` or an abort, every call into
the context that acts raises the end again and writes nothing, so a turn has
one ``turn/end`` and the exit status it recorded, whatever the loop code
catches. A loop that never calls the context, or catches the end and goes on
without it, is bounded by the episode wall clock in the episode form; in the
serve form it holds the turn until it returns. The session header's ``loop``
names the loop that ran and is null when the graph did; the serve form writes
the header at the first turn of a session, so ``loop`` names the loop of that
first turn; ``graph`` keeps naming the graph the agents fall back to. The loop
code runs in the loop process with that process's privileges, and no enforcer
stands between it and the host: that is why the kind is always reviewed. A
win that touches a ``native_loop`` is a pending release whatever
``review_kinds`` says, and ``harness_try`` refuses to mount one.

The native loop writes its trajectory as ``native-jsonl``: one
``{type, seq, time, data}`` object per line, ``seq`` contiguous from 0. A
``session`` header line names the task, model, tools, hooks (name to
event), the ``enforcement`` mode, ``tree``, where the composition came
from (``tree.json`` or ``files``), ``graph``, ``loop`` (the loop that ran,
null under a graph), and ``agents``; then ``turn/start``, per step
``step/start``, ``request/header`` (the
rendered system prompt and the tool declarations, logged on the first step so
the log holds everything the model saw), ``assistant/message`` (``content``,
``tool_calls``, ``finish``, optional ``usage``, and the provider
``reasoning``, ``reasoning_content``, ``reasoning_details`` and ``thinking``
fields when present), ``tool/call`` (the raw argument string),
``tool/result`` (``content``, ``is_error``, ``enforcement``, and on error a
closed ``code``: ``UNKNOWN_TOOL``, ``INVALID_ARGS``, ``TOOL_FAILED``,
``SANDBOX_FAILED``, ``HOOK_DENIED``, ``APPROVAL_REQUIRED``, ``HOOK_BLOCKED``),
``step/end``, and finally ``turn/end`` with a ``reason`` of ``completed``,
``gave_up``, ``max-steps``, ``max-tool-calls``, ``rejected``, ``ask`` (an
agent's turn a hook escalated), ``turn-timeout`` (the serve form's wall
clock), or ``error`` (its ``error`` code ``MODEL_ERROR``, ``LOAD_ERROR``,
``GRAPH_ERROR``, ``LOOP_ERROR`` under a ``native_loop``, or ``TURN_ERROR`` in
the serve form). Arguments are
validated against the tool's declared
schema before ``run`` sees them. A result over 20,000 characters is saved to a file:
the whole text is written to ``.reef/tool-output/<step>-<call_id>.txt`` under the
workspace, the model reads the head, one marker line naming that file and the
omitted count, and the last 2,000 characters, and ``tool/result`` carries the
file in ``meta.output_file``. A failed model call logs ``request/error``
(``attempt`` and the ``MODEL_ERROR`` failure) before the ``request_error``
event runs. A hook whose decision differs from the layer
below it logs ``hook/decision`` (``event``, ``step``, ``hook``, ``owned``, and
the decision), a hook that raised logs ``hook/error``, and a text a hook
injected lands as ``user/message`` with ``source.kind`` ``hook`` and the
``event``.

The native loop has two forms over the same entries, the same plugins and
the same interpreter. The episode form (``reef-native -p``) is one process
and one turn: ``run_episode`` launches it and the sandbox executor confines
it. The serve form (``reef-native serve``, ``reef/harness/runners/native/serve.py``)
is one resident process per installed tree: it boots a compose ``Loader``
over ``NATIVE_PLUGINS`` from ``native/tree.json``, keeps one ``Run`` per
session across turns, starts the wrapper's capture proxy in process
(``client.wrapper.CaptureProxy``), and follows the head through
``release_client.HeadWatch``, which polls the catalog and reads the
``x-reef-release-id`` header of every inference answer. The interpreter
calls ``loop.before_step(run)`` at the top of every model stage; the
episode form's loop does nothing there, the serve form's lands the queued
mount and checks the turn's wall clock. Tool and hook modules are written
under ``native/mounts/live/`` and unchanged entries keep their modules and
their in memory state across mounts; a changed entry is reinstalled through
its inverse, and a mount whose entries do not all end ACTIVE is rolled back
with ``root.update`` to the served entries.

The serve form adds these events, with the same ``{type, seq, time, data}``
shape, to the open turn's session when there is one and else to
``native/sessions/serve.jsonl``: ``harness/mount`` (``release_id``,
``parent_release_id``, ``source`` of ``boot``, ``release`` or ``try``,
``entries``; a trial adds ``try_id`` and ``mutations``),
``harness/mount-failed`` (``release_id``, ``source``, ``entry``, ``kind``,
``error``), ``harness/unmount`` (``try_id``, ``release_id``, ``source``
``rollback``, ``entries``), ``release/available`` (``release_id``, under
``--follow pinned``) and ``release/poll-failed`` (``error``,
``retry_in_s``). The ``session`` header gains ``mode`` (``serve``),
``session``, ``release_id`` and ``tree``; ``turn/start`` carries the turn
number, the ``prompt`` and the ``cwd``; ``request/header`` repeats whenever
the prompt or the declarations changed since the last one; a turn the wall
clock ended has ``turn/end`` with reason ``turn-timeout``. Steps restart at
1 each turn, so a turn's full tool outputs land under
``.reef/tool-output/t<turn>/``.

The socket protocol is one request per connection, JSON lines, UTF-8, on a
Unix domain socket at ``native/serve.sock`` (or under ``/tmp`` when that
path exceeds 100 bytes). A turn request is ``{"turn": {"prompt": str,
"session": str | null, "workdir": str}}``; the answer is every event of the
turn as written, then ``{"type": "turn/result", "data": {"exit", "session",
"turn", "text"}}``. ``{"control": "status"}`` answers ``{"type":
"control/result", "data": {"release_id", "parent_release_id", "follow",
"entries", "pending_mount", "sessions", "socket", "self_tools"}}`` and
``{"control": "mount", "release_id": str}`` answers ``{"type":
"control/result", "data": {"mounted", "release_id", "error"}}``. A
malformed request answers ``{"type": "error", "data": {"message"}}``.
Turns are served one at a time; a second connection waits. The three self
tools (``reef/harness/runners/native/selftools.py``) are ``ToolModule`` instances
built in code with ``builtin_tool`` set, run in process whatever
``REEF_NATIVE_ENFORCE`` says, and registered only under ``--self-tools``;
a tree entry named like one fails to mount with ``reserved name``.

The descriptor
--------------

One ``descriptor.yaml`` declares how a tree configures and starts a running
agent.

.. config::

   name | the adapter's id
   binary | the executable an episode runs
   argv | the argument list for one headless prompt; ``{prompt}`` is substituted
   files | where each node kind renders, like ``skills/{name}/SKILL.md``; ``rules`` and ``skill`` are required, every other kind is optional and a mutation of a kind left out is refused; ``tree`` names the file the entries list travels in, for a binary that reconciles the tree live
   trajectory | the format and path of the session log Reef reads back
   env | variables pointing the agent's state under the episode root; ``{root}`` is substituted. The install script and the ``reef-<adapter>`` wrapper need one entry that relocates a directory above the primary config target with a ``{root}/<dir>`` value, the composition they write and point the binary at; ``terminus`` relocates the root itself and gets neither
   install | the one-command install pin: ``kind`` (``npm``, or ``git`` for a checkout installed editable into a venv, which adds ``repository`` and ``ref``), ``package``, ``version`` (what ``--version`` must report), and ``binary_path`` under the install prefix
   model_binding | per API dialect (``openai``, ``responses``, ``anthropic``), the config nodes Reef appends at evaluation time; ``{base_url}``, ``{api_key}``, and ``{model}`` substitute into string values
   writable_paths | state directories made writable by the hosted sandbox; rendered inputs within them remain read-only
   client_state | the sessions and settings the ``reef-<adapter>`` wrapper keeps in the installed tree, as ``{path, kind}`` below the relocated composition. The wrapper runs the binary on a temp copy of links that it removes afterwards, so state the binary creates there itself is lost. ``directory`` and ``sqlite`` (an empty database) are created before the run and linked; ``file`` is copied back with its mode after the run when the binary created it, or renamed a new file over its link
   cleanup_whitelist | files the agent itself writes at boot or during the run, tolerated instead of read as drift
   quirks | an optional module for adapter-specific render checks and boot mutations

Connect a new agent
-------------------

To connect an agent that has no adapter yet:

.. steps::

   #. The file it reads configuration from becomes a ``files.config`` target.
   #. The command line that runs one prompt headless becomes ``binary`` and ``argv``.
   #. The path and format of its session log become ``trajectory``. A new format
      subclasses ``TrajectoryReader``
      (`reef/harness/episodes/trajectory.py <../../reef/harness/episodes/trajectory.py>`__) and
      registers with ``@register_trajectory_reader``.
   #. The files its first boot creates go in ``cleanup_whitelist``, so a fresh
      episode root is treated as clean. ``dir/**`` tolerates a whole subtree
      (session storage, ``node_modules``); any other entry is a glob against
      the root-relative path, so anchor a single file with its full path, like
      ``pi-agent/auth.json``. A bare directory name matches nothing under it.

`reef/harness/adapters/descriptor.py <../../reef/harness/adapters/descriptor.py>`__ validates every
descriptor at load, and the bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ are complete references. A third-party adapter
registers on the ``reef.harness_adapters`` entry-point group.
``evolution.client_models`` lists further model names the installed client
may switch to: the install script repeats every ``model_binding`` template
entry that names ``{model}`` (a mapping key, a list item) once per model,
the served model first and still the default, so pi and opencode show them
in their model pickers. Each call names the model it wants and the service
proxies it as is.
``evolution.version_check: true`` in the recipe config writes an update
prompt into the tree and ships for ``pi`` only. The
prompt offers to run the update or skip in interactive mode and prints the
instructions in headless mode. Before the offer, every ``env`` item the
installed release requires (over its chain, as ``reef-pi setup`` reads it)
whose variable, the ``check`` else the ``name``, is unset in the session's
shell gets one warning line, ``reef: <VAR> is not set; the installed harness
needs it (reef-pi setup lists it)``; a check off records that the variable
was set once, not that this shell has it. When the head requires an item
not checked off, an interactive session with the ``reef-pi`` wrapper on
disk (``REEF_HARNESS_WRAPPER``, which ``run_agent`` exports, else
``reef-pi`` beside the release file) asks ``Set up release <id8> now?``
with the list and runs the setup loop described under ``reef-requests``
below before the offer; without a wrapper, or headless, it prints the list
and ``Run reef-pi setup, then start reef-pi again.`` instead of the offer,
and an item the loop leaves unmet is named by the loop, the offer waiting
for the next session start. The update itself runs ``reef-pi update``
through the wrapper when one is on disk (the option says so) and the
install pipeline otherwise, and ends with ``Installed release <id8>. Type
/reload to load it now.``: pi's ``/reload`` re-runs ``session_start`` on
the installed tree, and only the person can type it. An ``opencode`` recipe
that sets it refuses to boot. An evolved tree is adapter-specific:
``config`` node contents follow each adapter's schema.

``evolution.requests: true`` seeds two more reef owned entries for ``pi``
after the notice: the ``code_extension`` ``reef-requests``
(`reef/harness/adapters/pi/requests.ts <../../reef/harness/adapters/pi/requests.ts>`__)
and the ``skill`` ``reef-pi-extension-api``
(`reef/harness/adapters/pi/pi_extension_api.md
<../../reef/harness/adapters/pi/pi_extension_api.md>`__, the pi extension
API reference the service proposer reads before it writes an extension).
The extension registers nothing under ``PI_OFFLINE``; otherwise it registers
two commands, two tools and two event handlers:

- ``/reefine <request>``: with a UI, clarifies the request in the
  background instead of in the session. The command returns at once and a
  loop calls the session's model through ``ctx.modelRegistry.complete`` with
  the last six user and assistant messages of the session as background, the
  request, and the same two tools. The model thinks the request through
  (when it triggers, what state the harness must know and how it learns it,
  what the person must set up, what is ambiguous), asks about an open point
  with ``reef_ask_user`` and files with ``reef_file_request``. The filing,
  a cancel, a reply without a tool call, a failed model call or eight model
  calls end it. While it runs, a widget above the input shows the phase, and
  ``ctrl+q``, or ``/reefine`` with no argument, opens its latest
  steps. When it ends, the chat keeps one
  custom entry (``pi.appendEntry``, type ``reef-harness-clarify``) whose
  line says what happened and whose expanded view (``ctrl+o``) holds the
  whole clarification. The entry stays out of the session model's context.
  Only one clarification runs at a time, and a session without a model is
  told to pick one or use ``--direct``. With ``--direct`` as the first word, or
  without a UI, it files the request as is with ``POST /reef/train`` (the
  scenario runs in ``manual`` or ``hybrid``), leaving captured receipts
  available for feedback. Either way the ``.reef-harness-release`` file
  beside the tree names the release the request runs on; without it nothing
  is sent.
- ``reef_ask_user``, a tool: the questions a reasonable default cannot
  settle, often none and at most four, each with two to four options
  offered through ``ctx.ui.select`` plus ``Other (type an answer)``, which
  opens ``ctx.ui.input``, and ``Cancel this request``. A question may name
  one option as ``recommended``: it is listed first with ``(recommended)``
  after it, and the answer filed is the option alone. Escape is the
  way out of the whole request, not a skipped question: no choice on a
  question, the cancel option, or no text in the free text answer all stop
  the dialogs there, notify ``reef: request cancelled; nothing was filed``
  (the background clarification records it in its entry instead) and return ``the user cancelled this harness request: do not file it, do
  not ask again, and say it was cancelled``, so the model stops instead of
  filing a request the person backed out of. The dialogs carry the turn's
  abort signal, so an aborted turn dismisses them. Otherwise it returns the
  question and answer pairs as JSON; without a UI it returns ``no UI in this
  session: proceed with your best assumptions and list them in the request``.
- ``reef_file_request``, a tool: the request verbatim, then, when there are
  clarifications, a ``Clarifications:`` block of ``- Q:`` / ``A:`` pairs,
  capped at 4000 characters, filed the way the command files it. It returns
  ``filed request <id>; reef is running the step, which usually takes one to
  three minutes, and will report here when it settles. Watch it here:
  <link>`` and throws the command's error messages; the command's own filing
  notifies the same expected time and the same link. The link is the
  request's page, ``GET /reef/harness/requests/<id>/page`` with ``scenario``
  and, when ``REEF_TOKEN`` is set, ``token`` as query parameters, so a
  browser opens it without the headers.
- The spinner, while a step runs: ``ctx.ui.setWidget`` draws one line above
  the input box, an animated frame, the phase in the person's words
  (``queued, waiting for a step``, ``writing the change``, ``checking the
  harness``, ``running the step``, ``saving the result``), the time in the
  step, the request's page as a terminal hyperlink (OSC 8, which pi's TUI
  measures around, so a click opens the page where the terminal offers one)
  and ``ctrl+q or /reefine to look in``. The frames turn every
  250 ms, so the step reads as alive between polls. ``ctrl+q``
  (``pi.registerShortcut``) expands the same widget in place with the
  request asked, its id, the evaluation's episode count and step record when
  the service reports them, the request page link for the full detail, and a
  line saying the step runs in the background; ``ctrl+q`` again closes
  it. pi offers extensions no click event for a widget, so the line names
  three ways in: the hyperlink, the key and the command. The key is a plain
  ``ctrl+<letter>`` pi leaves free: a terminal without the Kitty keyboard
  protocol or xterm's modifyOtherKeys (Apple Terminal among them) sends
  ``ctrl+shift+<letter>`` as the bare control byte, so a shifted key would
  reach pi as its own binding, and ``ctrl+r`` alone renames a session.
  ``/reefine`` with no argument prints the same detail and needs
  neither the key nor a click. Expanding costs no request: it redraws what
  the last poll read. The widget is cleared when the step settles, and a
  headless session draws none.
- The watch, after any filing: ``ctx.ui.setStatus`` shows ``reef: request
  <id> queued`` and, once progress reports the step running, ``reef: step
  for request <id> running for <Nm SSs>``. Record reads only detect requests
  removed from storage. Each poll reads ``GET
  /reef/harness/requests/<id>/progress`` for the step's phase, its episode
  count and its step record, and counts from the step's own
  ``started_at`` when the service reports one; a service without that route
  leaves the spinner at ``queued`` and changes nothing else. Meanwhile the
  extension polls ``GET
  /reef/harness/releases`` every ``REEF_HARNESS_WATCH_MS`` milliseconds
  (5000 by default) for the row whose ``metrics.training_request.id`` is the
  filed record, for at most 30 minutes, checked on every tick; one watch
  runs at a time, a second filing replaces the first, and
  ``session_shutdown`` clears it. Every fetch the extension makes carries an
  abort signal with a 10 s deadline (``REEF_HARNESS_FETCH_MS`` shortens it),
  so a hung read costs one poll, not every later tick. When the row appears,
  the report quotes the request's first 60 characters and names the next
  action by result: a selected release names ``/versions <version> install``;
  a pending one says ``This release changes an extension, so read it before it
  runs: /versions <version> opens the page, /versions <version> install
  serves it.``; a rejected step quotes
  ``selection.reason`` and says to rephrase or split the request; a skipped
  step quotes ``metrics.skipped`` and, when the step recorded one,
  ``proposal_notes.failure``, why the proposer produced nothing. The
  selected, rejected and skipped lines end with ``Details: /versions
  <step>.``; ``Not covered: ...`` follows when the step's
  ``proposal_notes.review.uncovered`` lists items. The report is delivered
  twice on purpose: as a custom message (``pi.sendMessage`` with
  ``customType: "reef-harness"`` and ``triggerTurn: false``), which the chat
  renders and the session file keeps, and as a notice, which the next
  status line may overwrite. Past the cap the watch says ``/versions``
  shows the result when it settles.
- A settled step offers its install, so a win reaches the person who asked
  without them going looking. The dialog waits for a turn to end
  (``ctx.isIdle()``): a busy session keeps the report's commands instead, and
  the next session start offers the same release. ``/versions <version>
  install`` starts the same install on demand after a confirmation linking the
  step's page. A release still held back from the served head is promoted as
  part of installing it, so installing is the one decision; only a rejected or
  skipped step, which published no tree of its own, is refused. The automatic
  update notice at session start remains a separate entry point.
- The install, through the ``reef-pi`` wrapper (``REEF_HARNESS_WRAPPER``,
  which ``run_agent`` exports, else ``reef-pi`` beside the release file;
  with neither on disk the notice is ``reef: no reef-pi wrapper found;
  install it with reef-pi update, then reef-pi setup``): ``reef-pi update
  --release <id>``, then the setup loop, then ``Installed release <id8>.
  Type /reload to load it now.`` (pi's ``/reload`` re-runs
  ``session_start`` on the installed tree; only the person can type it). An
  update the wrapper refuses for unmet items (exit 3) runs the setup loop
  first and then the update again; any other failure stops with ``reef:
  reef-pi update failed (exit N): <stderr>``. If the same installation directory
  was rebound to another service or scenario while the session was running,
  setup and update automatically use the session's original service, scenario,
  and token. The installation is restored to that configuration after a successful
  update. Commands targeting a different install directory use its own configuration.
  Setup values and checks are pinned to the release being installed. If that
  release is absent, the error identifies the queried service and scenario;
  refresh ``/versions`` before choosing a release again.
- The setup loop: ``reef-pi setup --json --release <id>`` lists the
  release's items with ``met``; each unmet item is asked once, an ``env``
  item through ``ctx.ui.input`` titled with its ``prompt`` (else ``Value
  for <NAME>``) and handed over as one argument, ``reef-pi setup --set
  NAME=<value> --release <id>``, a ``permission`` or ``service`` item through
  ``ctx.ui.confirm`` titled with its ``prompt`` (else ``Run this check?``)
  and the check as the message, then ``reef-pi setup --run NAME --release <id>``. One
  line per item: ``reef: NAME set``, ``reef: NAME met``, ``reef: NAME not
  met (exit N)``, or ``reef: NAME skipped`` for a declined check or an
  empty value; at the end, when items stay unmet, ``reef: still to set up:
  A, B (reef-pi setup)``. A listing that fails notifies its stderr and
  stops the loop. The value goes to the wrapper's env file, never into the
  tree or to reef, and an evolved extension reads it from ``process.env``
  at run time.
- The filed requests not yet reported are kept in
  ``.reef-harness-requests.json`` beside the release file, as ``{id, text,
  filed_at}`` entries (the newest ten, none older than a day), and dropped
  once reported. At ``session_start`` each stored id whose row the catalog
  holds gets its report as the custom message and the notice; one the
  catalog does not hold yet gets the watch again. So a restarted pi, or a
  report the person missed, still gets the result in the chat.
- ``session_start``: with a UI, one info line says the two commands exist,
  and a second line counts the releases held back from the served head and
  says how to install them: ``N release(s) ready to install: /versions
  <version>[, <version>] (install with /versions <version> install)``.
- ``/versions [version] [install]``: lists the release chain oldest first in
  aligned version (``v0``, ``v1``, ...), release id, result and status columns.
  Status distinguishes the locally installed version from the served head;
  request summaries appear below their rows, with whitespace collapsed.
  The footer explains the statuses and shows details and install commands. With a version,
  ``v3`` or ``3``, it offers the step's page (``GET
  /reef/harness/releases/{step}/page`` with the scenario and the token as
  query parameters), which holds the design, the review and the numbers;
  taking the offer opens it through the platform's launcher (``open``,
  ``xdg-open``, ``rundll32``), and declining prints the URL. Headless prints
  the summary and the URL instead. ``/versions <version> install`` installs
  the step after a confirmation, promoting a release still held back from the
  served head first.

The writing happens on the service, where the evolve step hands the request
to the recipe's ``propose`` and the commit records it under
``training_request``, the merged ``requires`` list included. The harness
requests RFC (#310) kept the agent side free of tools so that it only asks;
the two tools above change that rule on purpose (issue #435): a request is
clarified from the session where the person asked it, while they are still
there to answer, and the tools still write no mutation and start no step of
their own beyond the filing. Asking needs no extension:
Admission also refuses a ``code_extension`` that writes to the harness
process's own stdout or stderr without a ``ctx.hasUI`` guard
(``console.log``, ``console.error``, ``process.stdout.write``). An extension
runs inside the harness process, which owns the terminal in a session with a
UI, so a raw write lands in a drawn frame and leaves the session without its
input box until the next full redraw; console output stays the fallback for a
session without a UI, and the guard, on the write's own line or the one above
it, is what separates the two. Text for a session with a UI belongs in
``ctx.ui.notify``, ``ctx.ui.setStatus`` or ``ctx.ui.setWidget``.

``reef-<adapter> harness "<request>"`` is a wrapper subcommand on every
adapter. The ids ``reef-version-check``,
``reef-requests`` and ``reef-pi-extension-api`` are ``RESERVED_ENTRY_IDS`` in
`reef/harness/tree/nodes.py <../../reef/harness/tree/nodes.py>`__: the seed
and a recovered state carry them, and admission refuses a mutation that
creates, updates or removes one, the way native tool names are reserved.
Because no step changes them, a scenario would keep the copy it was created
with; instead, whenever the service opens a scenario (at startup or on
creation), ``CordisBackend.shipped_content_update`` renders each reserved
entry of the running Reef's seed and compares its files with the served
release. When one differs or is missing, the service replaces it in the
recorded entries (appending a missing one), renders the whole tree again and
commits it as a training release whose metrics are
``{"shipped_content_update": {"entries": [<ids>]}}``. That release consumes
no records, runs no evaluation and is not held for review, since its
content is Reef's own; the update notice then offers it to installed trees
like any other head. An
evolved extension runs in pi's process with the person's privileges, and
admission screens its text for credential shaped literals only, so the
tutorial's pi deployment
(``tutorials/evolve-your-harness/configs/deployment.yaml``) sets
``evolution.review_kinds: [code_extension]`` beside ``requests: true`` and
``version_check: true``: review is the boundary, and a release that touches
an extension waits for a promote.

A request handed to ``propose`` under ``requests`` carries ``requires``
beside its text, what the person said the change needs from their machine
as ``{name, kind, check}`` items, and the method may add items of the same
shape to the mapping when the change it wrote needs something of its own
(the tutorial's proposer asks the served model for a ``{"requires": [...]}``
object beside the entries); the backend merges them by name into the
commit's ``training_request.requires`` after the shape and text screens
admission runs (a bad item of the method's is dropped alone), and the list
reaches the releases row, the manifest, the install script's refusal and
``reef-<adapter> setup``, never a check on the service.
