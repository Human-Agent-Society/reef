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
|              |                                                           | reef-eval via ``reef-infra[terminus]``    |
+--------------+-----------------------------------------------------------+-------------------------------------------+

The ``terminus`` adapter is the one that does not drive a CLI. Terminus 2 is
a Harbor agent class, so the adapter ships its own runner,
``reef-terminus``: it reads the tree from ``REEF_TERMINUS_DIR``, hands it to
Harbor's own ``terminus-2`` agent as native configuration, runs the task the
prompt names, and writes the verifier's reward and the ATIF trajectory under
``REEF_TERMINUS_SESSION_DIR`` for the ``terminus-atif-json`` reader. It
reaches Harbor through reef-eval, the same primitive the examples under
``recipes/`` use. Nothing of Reef's runs inside the agent. The prompt is a
Harbor task directory or a registry id, so an episode needs no dataset
location in its environment, which ``run_episode`` would not carry anyway.

``config`` becomes Terminus 2 constructor arguments, refused at render if a
key is not one; ``rules`` becomes an ``extra_instruction_paths`` entry; and
``skill`` and ``agent_command`` become two ``AgentConfig.skills`` roots, so
Harbor keeps its progressive skill loading rather than pasting every body
into the prompt. ``code_extension`` is rejected: an evolved module would run
in the runner's process, outside the container that isolates the agent's own
commands, and it is outside Meta-Harness's search space in any case. On an
empty tree every mapping is a no-op and the agent is stock Terminus 2, the
equivalence the measured baseline rests on. Isolation is Harbor's task
container: Docker does not nest in bubblewrap, so ``run_episode`` refuses
this adapter under ``evolution.executor: sandbox``. The extra needs Python
3.12, above Reef's own floor.

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

The per call jail confines a tool's ``run`` and nothing else. Two things a
tree carries still run in the loop's own process: the top level of every
tool and hook module, once, when the loop imports it at start, and every
hook's ``listen`` at every event. Under the sandbox executor that process
is the episode jail, which holds the writable workspace and session
directory, the network namespace the model endpoint needs, and the
executor's base directories with their shells; under the local executor it
is the host. So a hook, and a tool module's import time code, are loop code
with the loop's reach: ``review_kinds`` with ``native_hook`` and
``native_tool`` is how a deployment puts a person between a proposal of
either and the tree, and the trajectory's ``enforcement`` field describes
the profile the run got, not what the module did at import. Moving a tool's
import out of the loop's process is tracked as a follow up.
``reef.harness.native.seed.SEED_TOOLS`` holds the starting ``read_file``,
``write_file``, ``run_bash``, and ``execute`` tools as entries a recipe can
seed and the loop can then evolve; ``execute`` runs a Python block in the
workspace with the other tools importable by name (``import read_file;
read_file.run({"path": "x"}, WORKDIR)``), so a tree can move from one call
per tool to code that calls tools without a loop change. An adapter that declares no ``files.native_tool`` path
refuses to render that kind, so the mutation fails under it instead of
silently dropping the tool. The admission gate refuses ``code`` that does not
compile; a module that fails to import, or defines no ``run``, ends the
episode with reason ``error`` and code ``LOAD_ERROR`` before any model call,
so the tree that carries it loses the gate instead of running without it.

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
``LOAD_ERROR`` like a tool. Every event takes a plain object and returns one:

.. config::

   pre_step | before each step: ``{step, task, messages}``; returns ``{kind: "enter", messages: [text...]}`` (each text becomes a user message before the request) or ``{kind: "reject", reason}`` (the turn ends with no step)
   pre_execute | before each tool call runs, after its arguments are validated: ``{step, call_id, name, arguments, capabilities}``; returns ``{kind: "allow", arguments?}`` (with ``arguments`` the call runs with the rewrite, validated like the model's own), ``{kind: "deny", reason}`` (the tool does not run and the model reads a ``HOOK_DENIED`` error carrying ``reason``) or ``{kind: "ask", reason}`` (a headless run has no one to ask, so the tool does not run and the model reads an ``APPROVAL_REQUIRED`` error carrying ``reason``); ``post_execute`` still sees the call, with that error as its result
   request_error | after a failed model call: ``{step, attempt, error}`` where ``error`` is ``{code: "MODEL_ERROR", message, status?}`` with ``status`` the HTTP status when the endpoint answered one; returns ``{kind: "retry", delay_ms}`` or ``{kind: "fail"}``; the loop spends at most ``MAX_REQUEST_ATTEMPTS`` (4) attempts a step and waits at most ``MAX_RETRY_DELAY_MS`` (10 s), whatever the hook asks
   post_execute | after each tool call has run: ``{step, call_id, name, arguments, result}``; returns ``{kind: "accept", content?, contexts: [text...]}`` (``content`` replaces what the model reads) or ``{kind: "block", feedback, contexts}`` (the model reads a ``HOOK_BLOCKED`` error carrying ``feedback``; the tool's side effects stand); contexts land as user messages after the step's results, in call order

``reef.harness.native.seed.SEED_HOOKS`` holds the one starting hook,
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
same effects a resident process uses, with the tool and hook modules written
under ``sessions/mounts/boot/`` (the one writable path under the sandbox).
An entry that does not end ACTIVE ends the episode with ``LOAD_ERROR``
naming the entry id, its kind and the fiber's error, or ``no plugin for kind
X`` for a kind the loop never reads, so a hand edited list cannot run
unchecked. Without the file the loop reads the rendered files as before, so
an older pulled tree runs unchanged. The two boots produce the same events;
the session header's ``tree`` field says which ran, ``tree.json`` or
``files``.

The loop's own control flow is a ``native_graph`` node, rendered to
``native/graphs/main.json``: named stages from a closed vocabulary and edges
keyed by each stage's outcome. ``reef.harness.native.seed.SEED_GRAPH`` is
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
the parent graph is the one that can answer. The gate's verdict carries
``candidate_agents`` and ``current_agents``, the turns, steps, tool calls and
tool errors per agent summed over each side's episodes. It also carries
``candidate_paths`` and ``current_paths``, one entry per episode in pairing
order (task by task, then repeat by repeat): the root session's
``stage/exit`` stage names in order and the ``turn/end`` reason kind, plus
``error`` when the turn ended on one and ``errored_agent`` when a delegated
agent's error ended the run before the root wrote its end; a delegated
agent's stages under ``agents/`` stay out of it; an episode that could not
run is ``None`` and a format without stage events gives an empty list and a
``None`` reason.

The native loop writes its trajectory as ``native-jsonl``: one
``{type, seq, time, data}`` object per line, ``seq`` contiguous from 0. A
``session`` header line names the task, model, tools, hooks (name to
event), the ``enforcement`` mode, and ``tree``, where the composition came
from (``tree.json`` or ``files``); then ``turn/start``, per step
``step/start``, ``request/header`` (the
rendered system prompt and the tool declarations, logged on the first step so
the log holds everything the model saw), ``assistant/message`` (``content``,
``tool_calls``, ``finish``), ``tool/call`` (the raw argument string),
``tool/result`` (``content``, ``is_error``, ``enforcement``, and on error a
closed ``code``: ``UNKNOWN_TOOL``, ``INVALID_ARGS``, ``TOOL_FAILED``,
``SANDBOX_FAILED``, ``HOOK_DENIED``, ``APPROVAL_REQUIRED``, ``HOOK_BLOCKED``),
``step/end``, and finally ``turn/end`` with a ``reason`` of ``completed``,
``max-steps``, ``rejected``, or ``error`` (its ``error`` code ``MODEL_ERROR``
or ``LOAD_ERROR``). Arguments are validated against the tool's declared
schema before ``run`` sees them. A result over 20,000 characters is spilled:
the whole text is written to ``.reef/spill/<step>-<call_id>.txt`` under the
workspace, the model reads the head, one marker line naming that file and the
omitted count, and the last 2,000 characters, and ``tool/result`` carries the
file in ``meta.spill``. A failed model call logs ``request/error``
(``attempt`` and the ``MODEL_ERROR`` failure) before the ``request_error``
event runs. A hook whose decision differs from the layer
below it logs ``hook/decision`` (``event``, ``step``, ``hook``, ``owned``, and
the decision), a hook that raised logs ``hook/error``, and a text a hook
injected lands as ``user/message`` with ``source.kind`` ``hook`` and the
``event``.

The native loop has two forms over the same entries, the same plugins and
the same interpreter. The episode form (``reef-native -p``) is one process
and one turn: ``run_episode`` launches it and the sandbox executor confines
it. The serve form (``reef-native serve``, ``reef/harness/native/serve.py``)
is one resident process per installed tree: it boots a compose ``Loader``
over ``NATIVE_PLUGINS`` from ``native/tree.json``, keeps one ``Run`` per
session across turns, starts the wrapper's capture proxy in process
(``harness_wrapper.CaptureProxy``), and follows the head through
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
1 each turn, so a turn's spill files land under ``.reef/spill/t<turn>/``.

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
tools (``reef/harness/native/selftools.py``) are ``ToolModule`` instances
built in code with ``host_plane`` set, run in process whatever
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
      (`reef/harness/trajectory.py <../../reef/harness/trajectory.py>`__) and
      registers with ``@register_trajectory_reader``.
   #. The files its first boot creates go in ``cleanup_whitelist``, so a fresh
      episode root is treated as clean. ``dir/**`` tolerates a whole subtree
      (session storage, ``node_modules``); any other entry is a glob against
      the root-relative path, so anchor a single file with its full path, like
      ``pi-agent/auth.json``. A bare directory name matches nothing under it.

`reef/harness/descriptor.py <../../reef/harness/descriptor.py>`__ validates every
descriptor at load, and the bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ are complete references. A third-party adapter
registers on the ``reef.harness_adapters`` entry-point group.
``evolution.version_check: true`` in the recipe config writes an update
prompt into the tree and ships for ``pi`` only. The
prompt offers to run the update or skip in interactive mode and prints the
instructions in headless mode. An ``opencode`` recipe that sets it refuses to
boot. An evolved tree is adapter-specific: ``config`` node contents follow each
adapter's schema.
