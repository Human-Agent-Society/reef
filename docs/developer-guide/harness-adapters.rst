Harness adapters
================

An adapter tells Reef where to render harness tree entries, how to run an
agent with those files and the served model, and how to read its trajectory.
The tree contains entries, not file paths; the adapter chooses the paths.

Reef includes six adapters for third-party coding-agent CLIs. It also includes
``native``, Reef's own agent, and ``terminus``, which runs the Harbor
Terminus 2 agent through a Reef runner. With ``native``, the tree can change
the agent's tools (``native_tool``) and its responses to loop events
(``native_hook``).

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

Terminus 2
~~~~~~~~~~

Terminus 2 is a Harbor agent class, not a CLI. The adapter uses
``reef-terminus`` to read the tree from ``REEF_TERMINUS_DIR``, configure
Harbor's ``terminus-2`` agent, and run the task named by the prompt. The
runner writes the verifier reward and ATIF trajectory under
``REEF_TERMINUS_SESSION_DIR`` for the ``terminus-atif-json`` reader.

The runner reaches Harbor through ``reef-eval``, as the examples under
``recipes/`` do. Its prompt names a Harbor task directory or registry id;
the episode does not need a separate dataset location in its environment.

Tree entries map to Terminus 2 configuration as follows:

- ``config`` supplies constructor arguments. Rendering rejects unknown keys.
- ``rules`` supplies an ``extra_instruction_paths`` entry.
- ``skill`` and ``agent_command`` supply two ``AgentConfig.skills`` roots.
  Harbor loads these skills progressively instead of placing every body in
  the prompt.
- One ``code_extension`` may define ``Agent(Terminus2)``. Rendering checks
  its syntax without executing it. The runner loads it through
  ``AgentConfig.import_path``. Without an extension, it runs stock Terminus 2.

Extensions require ``evolution.executor: sandbox`` to isolate the Python
runner. Harbor runs the terminal task remotely. Enable network access with
``sandbox.egress_hosts``; this setting currently does not enforce a hostname
firewall. The runtime needs Linux, bubblewrap, Python 3.12+, and
``harbor[e2b]``. The interpreter and local task directories must be visible
inside the sandbox, for example under ``/opt``.

Declarative trees can use the local executor and Docker. Reef rejects Docker
inside bubblewrap and extensions in an unisolated runner before launch.

The Terminus quirk supplies ``validate_execution`` as an
``ExecutionValidator``. Its ``__call__(files, executor)`` checks the rendered
tree and configured executor before Reef writes episode files. It raises
``EpisodeLaunchError`` for unsupported combinations and replaces the default
``self_isolating`` nesting restriction. Execution, timeout, cleanup, and
trajectory handling still use the shared episode code.

DeepSeek Harness
~~~~~~~~~~~~~~~~

The ``dsh`` adapter runs ``dsh --profile headless "<task>"`` and relocates
the agent's home through ``DSH_HOME``. dsh combines its bundle layers with a
user patch layer: a YAML list addressed by plugin id. The adapter accepts
the ``primary`` config as an object keyed by plugin id, such as
``{"agent-loop": {"config": {...}}}`` or ``{"disabled": true}``, and its
quirks render that object as the patch list. A string beginning with
``!!js `` becomes a JavaScript expression, as in dsh's own bundles.

The descriptor defaults keep the session log uncompressed and disable
telemetry and the LLM title call. Rendering rejects a tree that changes
those settings. Node paths and transformations are:

- ``rules`` becomes dsh's user-global ``AGENTS.md``.
- ``skill`` becomes ``skills/<name>/SKILL.md``. If the node text lacks the
  required YAML frontmatter, the adapter adds ``name`` and ``description``.
- ``agent_command`` becomes a user-invocable skill under ``DSH_AGENTS_HOME``.
  It uses ``disable-model-invocation: true`` and runs as ``/name``; dsh has no
  separate command surface.
- ``code_extension`` becomes a plugin module referenced by relative path
  from the patch layer.

The model binding uses an ``llm-pi-ai`` route. Its ``apiKeyEnv`` names the
key supplied through the ``env`` config target, dsh's ``.env`` launch layer.

Hermes Agent
~~~~~~~~~~~~

The ``hermes`` adapter runs ``hermes chat -Q --oneshot -q "<task>"`` with
``HERMES_HOME`` relocated. Its ``primary`` target is ``config.yaml``; the
quirks write the merged configuration as YAML. They enforce these defaults
so an episode stays self-contained and makes one request:

- ``approval.tirith_enabled`` disables the terminal scanner download.
- ``auxiliary.title_generation.enabled`` disables the title model call.
- ``memory.nudge_interval: 0`` disables background memory reviews.
- ``sessions.write_json_snapshots`` enables the per-session snapshot read by
  ``hermes-session-json``.

Rendering rejects a tree that changes any of those settings. The quirks
also write ``.no-bundled-skills``, so episodes use the tree's skills instead
of the bundled catalog.

Node paths and transformations are:

- ``rules`` becomes the home-level ``SOUL.md`` that Hermes reads.
  ``AGENTS.md`` is project-scoped and read from the working-directory chain.
  The rules follow Hermes's own default identity: Hermes treats ``SOUL.md``
  as the agent's identity and writes its default there only while the file
  is absent, so rules written alone would replace it.
- ``skill`` becomes ``skills/<name>/SKILL.md``. The adapter adds the required
  ``name`` and ``description`` frontmatter if the node text lacks it.
- ``agent_command`` becomes a skill under ``hermes-commands``, listed in
  ``skills.external_dirs``. Hermes exposes skills as ``/name`` commands and
  has no separate command surface.
- ``code_extension`` becomes a plugin package at
  ``plugins/<name>/__init__.py`` defining ``register(ctx)``. The quirks
  write its manifest, ``plugins.enabled`` entry, and ``tools.override``
  permission. Hermes requires this consent before loading a plugin; plugin
  tools are then available through ``tool_search`` and ``tool_call``.

The model binding uses a custom provider with a literal key in
``config.yaml`` and supports only the ``openai`` dialect. Hermes's default
approval policy runs tools in the working directory without prompting and
returns a tool error for commands it considers dangerous. The adapter does
not use a bypass flag.

Native tools and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The native adapter renders a ``native_tool`` node to
``native/tools/{name}.py``. Its ``code`` defines
``run(args, workdir) -> str``. The renderer writes ``NAME``, ``DESCRIPTION``,
``PARAMETERS``, and ``CAPABILITIES`` from the node config after the code, so
those tree values take precedence over values assigned in the module.

``capabilities`` is an optional list of distinct ``read``, ``write``,
``exec``, and ``network`` names. The loop includes it in the session header
and passes it to ``pre_execute`` hooks. The ``local`` executor does not
enforce the declaration: tools run in the loop process.

The ``sandbox`` executor sets ``REEF_NATIVE_ENFORCE=bwrap``. Each tool call
runs in a child process under a bubblewrap profile based on its capabilities:

- Without ``network``, the call gets an empty network namespace.
- Without ``write``, the workspace is read-only.
- Without ``exec``, the jail omits ``/bin`` and ``/usr/bin`` and unsets
  ``PATH``. It binds library directories, the interpreter running the tool,
  and its prefixes. The interpreter's prefix may still create an empty
  ``/usr/local/bin``. For example, ``subprocess.run(["bash", ...])`` fails
  because Python's fallback search of ``/bin:/usr/bin`` finds neither path.

The profile binds the episode's ``/proc`` read-only because a jail inside
the episode jail cannot mount a new one. The absent binary directories are
what deny ordinary shell execution; binding either directory would reopen
it. This is not a complete ban on execution: a tool can still start
``sys.executable`` or run a program in a bound library directory or a path
it can write, such as the jail's private ``/tmp``. It can also read the
workspace; ``read`` is never withheld.

Reef chooses the enforcer before loading tree modules into the loop. A tree
cannot change it. If ``REEF_NATIVE_ENFORCE`` names ``bwrap`` but the binary
is absent from ``PATH``, the loop refuses to start. If a jail cannot run a
tool call at all, the call ends with ``SANDBOX_FAILED`` and does not count
as a tool error. The sandbox executor checks nested jails before building
the run, so a host that cannot nest them fails before the first tool call.

Each ``tool/result`` event records an ``enforcement`` object. Its ``mode``
is ``none`` or ``bwrap``; ``denied`` lists the undeclared ``write``, ``exec``,
and ``network`` capabilities, or is empty under ``none``. It describes the
profile, not what the tool attempted. Seed tools declare capabilities;
``run_bash`` declares all three.

The per-call jail covers both ``run`` and the code executed when the tool
module is imported. The loop reads ``NAME``, ``DESCRIPTION``, ``PARAMETERS``,
and ``CAPABILITIES`` from the source with ``ast.literal_eval``. It imports
the module only when a call runs: in the child under ``sandbox``, or in the
loop process on the first call under ``local``. Import-time code therefore
never runs when a tool is loaded. Under ``sandbox``, it never runs in the
loop process. The trajectory's ``enforcement`` field describes the profile
used for both import and call.

Hooks do run in the loop process. Each hook module is imported once at
startup, and its ``listen`` function runs at its event. It can call
``next`` to reach the next layer and change the decision that steers the
loop. Under ``sandbox``, the loop process is the episode jail, with a
writable workspace and session directory, network access to the model
endpoint, and the executor's base directories and shells. Under ``local``,
it is the host. Use ``review_kinds: [native_hook]`` to require review before
publishing hook changes. Under ``local``, tools also run without confinement;
include ``native_tool`` in ``review_kinds`` when reviewing those changes.

``reef.harness.runners.native.seed.SEED_TOOLS`` provides the starting
``read_file``, ``write_file``, ``run_bash``, and ``execute`` entries. A recipe
can seed and evolve them. ``execute`` runs Python in the workspace and can
import other tools by name, for example ``import read_file;
read_file.run({"path": "x"}, WORKDIR)``. An adapter without a
``files.native_tool`` path refuses a mutation of that kind.

Admission rejects tool ``code`` that does not compile. Before the first
model call, the loop returns ``LOAD_ERROR`` if it cannot open or parse a
tool file, find a top-level binding of ``run``, or read the last top-level
assignment to a declaration constant as a literal. A file that parses but
does not compile fails on its first call, as does top-level code that raises.

The source reader uses the last module-scope binding in source order. It
follows ``if``, ``try``, ``with``, ``for``, ``while``, and ``match`` bodies,
but not function or class bodies. Definitions, imports, assignment targets,
``for`` and ``with`` targets, exception targets, walrus targets, and match
captures can bind names. The renderer writes declaration constants last,
so those values win.

The loop does not execute top-level code while reading declarations. If
that code raises (including ``SystemExit``) or binds ``run`` to a
non-callable value, the first call fails with ``TOOL_FAILED`` and the
episode continues. Under ``local``, the module is imported once, so later
calls fail the same way without re-running its top level. Under ``sandbox``,
the child imports it on every call.

Native hooks
~~~~~~~~~~~~

A ``native_hook`` listens at one of the loop's four events. It renders to
``native/hooks/{name}.py``: ``code`` defines
``listen(payload, next) -> decision``, followed by ``NAME`` and ``EVENT``
from the node config.

Hooks at the same event run in file-name order. A hook may call ``next()``
to obtain the decision from the next layer, then return that decision with
or without changes. It may instead return its own decision without calling
``next``. The final layer is the loop's default. Reef runs the next layer
at most once even if a hook calls ``next`` again, and gives the hook a copy
of its result.

If a hook raises or returns a value the log cannot store as a plain object,
Reef skips it and keeps the next layer's decision. It reads ``messages`` and
``contexts`` as lists of text and drops other items. If the hook module
cannot import, has no ``listen``, or names an unknown event, the episode
ends with ``LOAD_ERROR``.

Each event takes and returns a plain object:

- ``pre_step`` runs before a step with ``{step, task, messages}``. An
  ``enter`` decision can add user messages before the request. A ``reject``
  decision ends the turn without taking the step.
- ``pre_execute`` runs after tool arguments are validated, before the call.
  Its input is ``{step, call_id, name, arguments, capabilities}``. ``allow``
  may replace ``arguments``, which Reef validates again. ``deny`` returns
  ``HOOK_DENIED`` to the model; ``ask`` returns ``APPROVAL_REQUIRED`` in a
  headless run. Both carry the hook's ``reason``. ``post_execute`` still
  receives the resulting error.
- ``request_error`` runs after a failed model call. Its input contains
  ``step``, ``attempt``, and an ``error`` with ``code: "MODEL_ERROR"``,
  ``message``, and optional HTTP ``status``. It returns ``retry`` with
  ``delay_ms`` or ``fail``. The loop allows at most
  ``MAX_REQUEST_ATTEMPTS`` (4) per step and at most ``MAX_RETRY_DELAY_MS``
  (10 seconds) between attempts, regardless of the hook's request.
- ``post_execute`` runs after a tool call with
  ``{step, call_id, name, arguments, result}``. ``accept`` can replace the
  content the model reads. ``block`` sends ``HOOK_BLOCKED`` with the hook's
  ``feedback``; it does not undo the tool's side effects. Both may return
  ``contexts`` as user messages after the step's tool results, in call order.

``reef.harness.runners.native.seed.SEED_HOOKS`` holds the one starting hook,
``loop_guard`` at ``post_execute``, which reminds the model when the same call
repeats three, five, or eight times in a row. It is a node, so a tree can
retune or remove it. ``SEED_NODES`` combines the starting tools and hooks.
The tutorial's ``serve-native.yaml`` seeds them by reference.

The native descriptor has no path for ``agent_command`` or
``code_extension`` because the loop does not read them. Admission rejects
mutations of those kinds with "does not render". The renderer keeps both
``config`` targets, including ``primary``, but the loop reads only
``models``. A live tree cannot boot with a ``config`` entry targeting
``primary`` or setting a pinned binding field: ``api``, ``base_url``,
``api_key``, or ``model``.

Native tree file
~~~~~~~~~~~~~~~~

The native descriptor declares ``files.tree: native/tree.json``. Reef
renders the release's entries into that file as a JSON array of
``{id, name, config}`` objects. It is the same list stored in the commit
log under ``algorithm_state["entries"]``.

The file accompanies evaluation episodes, published artifacts, manifests,
install scripts, and pulled trees. Before the first evolution step, the
base release carries the seed entries. Model-binding nodes stay out of the
file; pinned model fields go in ``native/models.json``.

At boot, the loop reads ``tree.json`` if present. It creates a compose
context and a ``Loader`` over ``NATIVE_PLUGINS``, then calls
``root.update(entries)``. Each plugin admits its entry again and installs
it using the same effects as a resident process. Tool, hook, and loop
modules are written under ``sessions/mounts/boot-<pid>/``, the sandbox's
writable mount path.

If an entry does not reach ACTIVE, the episode ends with ``LOAD_ERROR``.
The error names its id, kind, and fiber error; an unsupported kind reports
``no plugin for kind X``. This also checks hand-edited lists. Without
``tree.json``, the loop reads rendered files as before, so older pulled
trees still run. Both boot paths produce the same events. The session
header's ``tree`` field identifies the path used: ``tree.json`` or
``files``.

Native graph stages
~~~~~~~~~~~~~~~~~~~

A ``native_graph`` node controls the native loop and renders to
``native/graphs/main.json``. It contains named stages of the kinds below,
with edges keyed by each stage's outcome. When the tree has no graph, the
loop uses ``reef.harness.runners.native.seed.SEED_GRAPH``: ``think`` asks
the model, ``act`` runs tool calls, and ``done`` ends the turn. A tree with
its own graph replaces that flow; hooks still handle their events.

Admission rejects unknown kinds or keys, outcomes without exactly one edge,
stages unreachable from ``start``, stages without a path to an end, and
cycles without a model stage. The step budget (``max_steps``, 1 to 32) thus
bounds each run. Rendering rejects a ``tools`` allow list that names a tool
missing from the tree.

The stage kinds are:

- ``model`` makes one request with the current messages and declared tools.
  It fires ``pre_step`` and ``request_error`` hooks and returns
  ``tool_calls`` or ``text``.
- ``tools`` runs pending calls from the last assistant message, each through
  ``pre_execute`` and ``post_execute``. Optional ``allow`` restricts calls
  to named tools. Its outcome is ``done``.
- ``verify`` checks the last assistant text using ``last_line_integer``,
  ``last_line_matches`` with a ``pattern``, or ``nonempty``. On failure it
  may add a user ``message``. Its outcomes are ``pass`` and ``fail``.
- ``message`` appends ``text`` as a user message and returns ``done``.
- ``branch`` selects an outcome from at most eight
  ``{when, value, outcome}`` cases. ``when`` can be
  ``steps_used_at_least`` or ``tool_errors_at_least`` with an integer value,
  or ``last_text_matches`` with a regular expression. The first matching
  case wins; otherwise the outcome is ``else``. Every case outcome and
  ``else`` need an edge.
- ``subagent`` sends the last assistant text, or the task, to the
  ``native_agent`` named by ``agent`` and then through its ``then`` chain.
  The last agent's text returns as a user message with
  ``source.kind: agent``. Outcomes are ``completed``, ``gave_up``,
  ``budget`` (steps or tool calls exhausted), and ``ask`` (a
  ``pre_execute`` hook asked during the agent's turn).
- ``compact`` summarizes old messages with one model call after they pass
  ``fire_ratio`` of the context window. It keeps the last ``keep_ratio``
  verbatim, without separating a tool result from its call.
  ``0 < keep_ratio < fire_ratio <= 1``. The window comes from
  ``context_window`` in ``models.json``, settable by a ``config`` node
  targeting ``models``; its default is 32,768 tokens at four characters
  per token. The summary call is not a step. A cycle must pass a model
  stage, so it can compact at most once per step. Its outcome is ``done``.
- ``end`` ends the turn with reason ``completed`` or ``gave_up``.

Patterns in ``branch`` and ``verify`` are limited to 200 characters and
run in a child process with a one-second timeout. A timed-out search fails
the case or check and appears as ``timeout`` in the stage detail. A branch
searches only the last 4,096 characters of the text.

Each ``model`` stage uses one step, and each request asks for at most 4,096
tokens. Entering a model stage after spending ``max_steps`` ends the turn
with ``max-steps``.

The trajectory records the graph path through ``stage/enter`` (``step``,
``stage``, ``kind``) and ``stage/exit`` (``outcome``, ``to``). A verify exit
also records ``check`` and ``last_line``; a branch exit records the matching
``case``; a compact exit records whether it ``fired`` and token counts.
When compaction runs, ``context/compacted`` records ``policy``,
``tokens_before``, ``tokens_after``, dropped-message count, and ``summary``.
A failed summary records its ``error`` and drops no messages. Text injected
by a stage appears as ``user/message`` with ``source.kind: stage``. The
session header's ``graph`` identifies ``main`` or ``seed``.

A graph that cannot load ends with ``LOAD_ERROR``. A run exceeding
``(max_steps + 1) * 16`` transitions ends with ``GRAPH_ERROR`` as a fallback
guard, although admission rejects graphs that could reach that limit.

Native subagents
~~~~~~~~~~~~~~~~

A ``native_agent`` is a root entry in the same tree, rendered to
``native/agents/<name>.json``. It can specify:

- ``prompt``, appended to rules and skills as its system prompt;
- ``graph``, which defaults to the built-in ``seed`` loop but can name
  ``main`` or another graph;
- ``tools`` and ``skills`` visible to this agent, defaulting to all in the
  tree;
- ``max_steps`` and ``max_tool_calls``; and
- ``then``, a list of agents that receive its final text in order.

A graph starts an agent through a ``subagent`` stage. Rendering rejects
missing agent names and cycles through ``then`` lists or subagent stages,
so delegation terminates. The agent spends the parent's remaining step
budget. Its session file lives under ``sessions/agents/``, numbered in run
order before the root's ``session.jsonl``. The file header names the agent,
turn, and parent. The root's last assistant text remains the trajectory's
final answer.

If ``pre_execute`` returns ``ask`` during an agent turn, that turn returns
``ask`` to the parent graph rather than reporting ``APPROVAL_REQUIRED`` to
the model. The parent graph can then handle the request.

Evaluation results include ``candidate_agents`` and ``current_agents``:
turns, steps, tool calls and errors, and any reported input and output
tokens per agent, summed across episodes on each side. They also include
``candidate_paths`` and ``current_paths``, one per episode in pairing order.
Each path lists the root session's ``stage/exit`` names and its ``turn/end``
reason. It includes ``error`` if the turn ended with one, or
``errored_agent`` if an agent error stopped the run first. Subagent stages
do not appear in the root path. An episode that could not run has ``None``;
a trajectory format without stage events has an empty path and a ``None``
reason.

Native loop code
~~~~~~~~~~~~~~~~

A ``native_loop`` node renders to ``native/loops/<name>.py``. Its ``code``
defines ``run_turn(ctx)``; the renderer adds ``NAME`` and ``MAX_STEPS``
from the node config. With this node, the root turn runs ``run_turn``
instead of the ``main`` graph. Subagents still run their graphs and are
available through ``ctx.agent``.

Only one loop is allowed per tree. Rendering rejects a second
``native_loop``; the host rejects a second ``add_loop``; the file form
rejects two files under ``loops/``. Before import, Reef checks that the
file parses and that ``NAME`` and ``MAX_STEPS`` have the literal assignments
written by the renderer.

Admission reads code without executing it. It checks that the module
compiles, contains no credential, and leaves ``run_turn`` bound to a plain
top-level ``def`` with a parameter. The last module-scope binding wins,
including bindings inside ``if``, ``for``, ``with``, ``try``, or ``match``;
those other bindings are refused. ``max_steps`` is the model-step budget,
from 1 to 32 (default 12). The tree is flat: entries with ``group`` are
refused at admission, boot, and mount.

``ctx`` exposes the following calls into the run:

- ``ctx.prompt``, ``ctx.step``, and ``ctx.max_steps`` give the task, steps
  spent, and budget. ``ctx.tools`` lists available tool names.
  ``ctx.messages`` and ``ctx.last`` are copies of the messages and last
  assistant message.
- ``ctx.model()`` takes one model step. It fires ``pre_step`` and
  ``request_error`` and writes ``step/start``, ``assistant/message``, and
  ``step/end``. It writes ``request/header`` when the model-visible prompt
  or tools change, always on step 1. It returns ``tool_calls`` or ``text``.
  A spent budget ends the turn with ``max-steps``. If the loop returns
  after ``tool_calls`` without calling ``run_tools``, those calls remain
  unanswered in the conversation for the next turn.
- ``ctx.run_tools(allow=None)`` runs the last message's tool calls through
  ``pre_execute`` and ``post_execute``. ``allow`` restricts them to named
  tools; an empty or absent list imposes no restriction.
- ``ctx.text()`` returns the last assistant text.
- ``ctx.say(text)`` writes a ``user/message`` with the loop's name and
  ``source.kind: loop``. It counts as a transition.
- ``ctx.agent(name, text=None)`` runs one named agent on ``text``, or on
  the last assistant text or task when ``text`` is absent. It does not
  follow the agent's ``then`` chain. It appends the answer as a user message
  with ``source.kind: agent`` and returns ``(outcome, text)``.
- ``ctx.end(reason="completed")`` ends the turn with ``completed`` or
  ``gave_up``; other reasons raise ``ValueError``.
- ``ctx.log(event, data)`` writes a ``loop/<event>`` line and counts as a
  transition. The event must be a node name other than ``enter`` or
  ``exit``; the prefix prevents it from writing a core event. Reef
  converts ``data`` to JSON with text keys. Beyond 4,096 serialized
  characters it writes ``{"text": the first 4096, "truncated": true}``.

Returning from ``run_turn`` completes the turn. A loop turn writes
``loop/enter`` with its name, then ``loop/exit`` with the reason before
``turn/end``. It writes no ``stage/*`` events, so its stage path is empty.

The transition guard allows at most ``(max_steps + 1) * 16`` calls to
``model``, ``run_tools``, ``agent``, ``say``, and ``log``. Exceeding it, or
raising an exception from ``run_turn`` (including ``SystemExit``), ends
the turn with ``LOOP_ERROR`` and exit status 1. Evaluation treats that
episode as unable to run. ``KeyboardInterrupt`` propagates.

The first end is final. After ``max-steps``, ``ctx.end``, or an abort,
subsequent context actions raise the same end without writing anything.
This keeps one ``turn/end`` and its exit status even if loop code catches
the exception. If code never calls the context, or catches the end and
continues without it, the episode wall clock bounds it in episode mode.
In serve mode, it holds the turn until it returns.

The session header's ``loop`` names the loop that ran, or is null under a
graph. In serve mode, it names the loop used on the session's first turn.
``graph`` still names the graph available to subagents. Loop code runs
with the loop process's privileges. For that reason, ``native_loop``
changes always require review: a selected change remains pending regardless
of ``review_kinds``, and ``harness_try`` refuses to mount it.

Native trajectory
~~~~~~~~~~~~~~~~~

The native loop writes ``native-jsonl``: one ``{type, seq, time, data}``
object per line, with ``seq`` contiguous from zero. The ``session`` header
names the task, model, tools, hooks by event, enforcement mode, agents, and
the selected ``tree`` source (``tree.json`` or ``files``). It also names
``graph`` and ``loop``; ``loop`` is null under a graph.

The main event sequence is ``turn/start``, then for each step
``step/start``, ``request/header``, ``assistant/message``, ``tool/call`` and
``tool/result`` as needed, and ``step/end``, followed by ``turn/end``.
``request/header`` records the rendered system prompt and tool declarations
on the first step, so the log includes what the model saw.
``assistant/message`` records ``content``, ``tool_calls``, ``finish``,
optional ``usage``, and provider fields ``reasoning``,
``reasoning_content``, ``reasoning_details``, and ``thinking`` when present.
``tool/call`` records the raw argument string. Reef validates arguments
against the declared schema before calling ``run``.

``tool/result`` contains ``content``, ``is_error``, and ``enforcement``.
On error its ``code`` is one of ``UNKNOWN_TOOL``, ``INVALID_ARGS``,
``TOOL_FAILED``, ``SANDBOX_FAILED``, ``HOOK_DENIED``,
``APPROVAL_REQUIRED``, or ``HOOK_BLOCKED``. ``turn/end`` has a reason of
``completed``, ``gave_up``, ``max-steps``, ``max-tool-calls``, ``rejected``,
``ask`` (from an agent turn), ``turn-timeout`` (serve mode), or ``error``.
Error codes include ``MODEL_ERROR``, ``LOAD_ERROR``, ``GRAPH_ERROR``,
``LOOP_ERROR`` under a native loop, and ``TURN_ERROR`` in serve mode.

For a tool result longer than 20,000 characters, Reef writes the full text
to ``.reef/tool-output/<step>-<call_id>.txt`` in the workspace. The model
receives the head, a line naming the file and omitted count, and the last
2,000 characters. ``tool/result.meta.output_file`` names the saved file.

Other events record failures and hook actions. ``request/error`` contains
the attempt and ``MODEL_ERROR`` failure before ``request_error`` hooks run.
``hook/decision`` records a decision that differs from the next layer,
including ``event``, ``step``, ``hook``, ``owned``, and the decision.
``hook/error`` records a raised exception. Hook-injected text appears as
``user/message`` with ``source.kind: hook`` and the event.

Episode and serve modes
~~~~~~~~~~~~~~~~~~~~~~

Both modes use the same entries, plugins, and interpreter.
``reef-native -p`` runs one episode in one process and turn;
``run_episode`` launches it, and the sandbox executor confines it.

``reef-native serve`` keeps one resident process per installed tree. It
loads ``native/tree.json`` into a compose ``Loader`` over
``NATIVE_PLUGINS`` and keeps a ``Run`` per session across turns. It starts
``client.wrapper.CaptureProxy`` in process. ``release_client.HeadWatch``
follows the served head by polling the catalog and reading the
``x-reef-release-id`` header on inference responses.

The interpreter calls ``loop.before_step(run)`` before each model stage.
In episode mode it does nothing. In serve mode it applies queued mounts
and checks the turn's wall clock. Tool and hook modules live under
``native/mounts/live/``. Unchanged entries keep their modules and memory
state across mounts; changed entries are reinstalled through their inverse.
If any mounted entry fails to reach ACTIVE, ``root.update`` rolls back to
the served entries.

Serve mode writes extra events to the open turn's session, or to
``native/sessions/serve.jsonl`` when no turn is open. They use the same
``{type, seq, time, data}`` shape:

- ``harness/mount``: ``release_id``, ``parent_release_id``, ``entries``,
  and ``source`` (``boot``, ``release``, or ``try``). A trial also records
  ``try_id`` and ``mutations``.
- ``harness/mount-failed``: ``release_id``, ``source``, ``entry``, ``kind``,
  and ``error``.
- ``harness/unmount``: ``try_id``, ``release_id``, ``entries``, and
  ``source: rollback``.
- ``release/available``: ``release_id`` when using ``--follow pinned``.
- ``release/poll-failed``: ``error`` and ``retry_in_s``.

The ``session`` header also records ``mode: serve``, ``session``,
``release_id``, and ``tree``. ``turn/start`` records the turn number,
``prompt``, and ``cwd``. ``request/header`` repeats when the prompt or tool
declarations change. A wall-clock timeout ends the turn with reason
``turn-timeout``. Steps restart at one each turn, so full tool outputs go
under ``.reef/tool-output/t<turn>/``.

The Unix socket uses UTF-8 JSON lines and serves one request per connection.
It is at ``native/serve.sock``, or under ``/tmp`` if that path exceeds 100
bytes. Requests and responses are:

- A turn request is ``{"turn": {"prompt": str, "session": str | null,
  "workdir": str}}``. Reef streams its events, then a ``turn/result``
  object with ``exit``, ``session``, ``turn``, and ``text`` in ``data``.
- ``{"control": "status"}`` returns ``control/result`` with
  ``release_id``, ``parent_release_id``, ``follow``, ``entries``,
  ``pending_mount``, ``sessions``, ``socket``, and ``self_tools``.
- ``{"control": "mount", "release_id": str}`` returns ``control/result``
  with ``mounted``, ``release_id``, and ``error``.
- Malformed input returns an ``error`` object with ``message`` in ``data``.

Turns run one at a time; another connection waits. The three self tools in
``reef/harness/runners/native/selftools.py`` are built-in ``ToolModule``
instances. They run in the process regardless of ``REEF_NATIVE_ENFORCE``
and are registered only with ``--self-tools``. A tree entry using a
reserved self-tool name fails to mount with ``reserved name``.

Descriptor fields
~~~~~~~~~~~~~~~~~

``descriptor.yaml`` describes how Reef renders, starts, and reads an agent:

- ``name`` identifies the adapter. ``binary`` is the executable, and
  ``argv`` supplies its arguments for one headless prompt. Reef substitutes
  ``{prompt}`` in ``argv``.
- ``files`` maps node kinds to output paths, such as
  ``skills/{name}/SKILL.md``. ``rules`` and ``skill`` paths are required.
  Other kinds are optional; Reef rejects a mutation of a kind with no
  render path. ``files.tree`` is an optional entries-list path for agents
  that reconcile the live tree.
- ``trajectory`` specifies the session log's path and reader format.
- ``env`` points the agent's state into the episode root, substituting
  ``{root}``. For install scripts and ``reef-<adapter>`` wrappers, one
  variable must relocate a directory above the primary config file using
  ``{root}/<dir>``. Terminus relocates the root itself and has no wrapper.
- ``install`` pins the vendor install: ``kind`` (``npm`` or editable-venv
  ``git``), ``package``, ``version`` (as reported by ``--version``), and
  ``binary_path`` below the install prefix. A git install also names
  ``repository`` and ``ref``.
- ``model_binding`` contains config nodes for each supported API dialect
  (``openai``, ``responses``, or ``anthropic``). Reef adds the matching nodes
  for evaluation episodes and substitutes ``{base_url}``, ``{api_key}``,
  and ``{model}`` in their string values.
- ``writable_paths`` lists state directories that a hosted sandbox makes
  writable. Rendered inputs within them stay read-only.
- ``client_state`` lists ``{path, kind}`` entries for sessions and settings
  that a ``reef-<adapter>`` wrapper keeps under the relocated composition.
  The wrapper uses a temporary copy of links, then removes it. ``directory``
  and ``sqlite`` entries are created and linked before the run; ``file``
  entries are copied back with their mode if the binary created the file
  or replaced its link. Other state created only in the temporary copy is
  lost.
- ``cleanup_whitelist`` lists agent-written paths allowed after boot or a
  run, rather than reported as drift.
- ``quirks`` names an optional module for adapter-specific render checks
  and boot mutations.

Connect a new agent
~~~~~~~~~~~~~~~~~~~

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

`reef/harness/adapters/descriptor.py <../../reef/harness/adapters/descriptor.py>`__
validates descriptors at load. The bundled adapters under
`reef/harness/adapters/ <../../reef/harness/adapters>`__ provide complete
references. External adapters register through the
``reef.harness_adapters`` entry-point group.

``evolution.client_models`` adds model names that an installed client may
select. The install script repeats each ``model_binding`` template entry
containing ``{model}`` once per model, whether the placeholder appears in
a mapping key or list item. The served model comes first and remains the
default. Pi and opencode show the added models in their pickers; the service
proxies each call to the model the client chose.

Pi version checks
~~~~~~~~~~~~~~~~~

``evolution.version_check: true`` writes an update prompt into a pi tree.
Interactive sessions can run or skip the update; headless sessions print
instructions. An opencode recipe with this setting fails at startup.

Before showing the update, the extension checks the installed release's
required ``env`` items. For each variable absent from the current shell,
it prints ``reef: <VAR> is not set; the installed harness needs it
(reef-pi setup lists it)``. The variable is the item's ``check`` if set,
otherwise its ``name``. A prior successful setup check does not establish
that the variable is set in the current shell.

If a required item has not passed setup, an interactive session with a
``reef-pi`` wrapper asks ``Set up release <id8> now?`` and runs the setup
loop below before offering the update. It finds the wrapper through
``REEF_HARNESS_WRAPPER`` (exported by ``run_agent``), or beside the release
file. Without a wrapper or UI, it prints the unmet items and
``Run reef-pi setup, then start reef-pi again.``. Items still unmet after
setup are reported, and the update offer waits until the next session.

The update runs ``reef-pi update`` through an available wrapper, or uses
the install pipeline. It ends with ``Installed release <id8>. Type /reload
to load it now.`` Only the user can enter pi's ``/reload``; it reruns
``session_start`` on the installed tree. The contents of evolved ``config``
nodes follow the selected adapter's schema.

Pi harness requests
~~~~~~~~~~~~~~~~~~~

``evolution.requests: true`` adds two Reef-owned entries to a pi tree:
the ``code_extension`` ``reef-requests``
(`reef/harness/adapters/pi/requests.ts <../../reef/harness/adapters/pi/requests.ts>`__)
and the ``skill`` ``reef-pi-extension-api``
(`reef/harness/adapters/pi/pi_extension_api.md
<../../reef/harness/adapters/pi/pi_extension_api.md>`__). The skill is the
pi extension API reference read by the service proposer before it writes
an extension. Under ``PI_OFFLINE`` the extension registers nothing.
Otherwise it registers two commands, two tools, and two event handlers.

``/reefine <request>`` files a harness-change request. With a UI, it first
clarifies the request in the background. The command returns immediately;
the clarification calls the session model through
``ctx.modelRegistry.complete`` with the request, the last six user and
assistant messages, and the two request tools. It considers when the change
should run, what harness state it needs, how to obtain that state, setup
requirements, and ambiguities. It can ask a question through
``reef_ask_user`` and file through ``reef_file_request``. Filing,
cancellation, a response without a tool call, a failed model call, or eight
model calls ends the clarification. Only one runs at a time.

The UI widget shows the current phase. ``ctrl+q`` or ``/reefine`` without
an argument opens the latest steps. Once clarification ends, pi stores one
``reef-harness-clarify`` custom entry through ``pi.appendEntry``. Its line
reports what happened; ``ctrl+o`` expands the full clarification. The entry
does not enter the session model's context. A session without a model is
prompted to select one or use ``--direct``.

With ``--direct`` as the first word, or without a UI, ``/reefine`` sends the
request unchanged to ``POST /reef/train``. This requires ``manual`` or
``hybrid`` training mode and leaves captured receipts available for
feedback. Both paths require ``.reef-harness-release`` beside the tree to
identify the release; without it, no request is sent.
The two request tools are:

- ``reef_ask_user`` asks only about points without a reasonable default.
  It asks at most four questions. Each has two to four choices through
  ``ctx.ui.select``, plus ``Other (type an answer)`` through
  ``ctx.ui.input`` and ``Cancel this request``. A choice marked
  ``recommended`` appears first with ``(recommended)``; the filed answer
  contains the choice text without that label.

  Escape, no choice, ``Cancel this request``, or empty free text cancels
  the entire request. The UI reports
  ``reef: request cancelled; nothing was filed``; background clarification
  records cancellation in its entry instead. The tool tells the model not
  to file or retry the cancelled request. The turn's abort signal also
  dismisses the dialogs. Otherwise the tool returns question-answer pairs
  as JSON. Without a UI it tells the model to proceed with stated
  assumptions.
- ``reef_file_request`` files the request verbatim, followed by a
  ``Clarifications:`` block of ``- Q:`` and ``A:`` pairs when present. It
  caps the text at 4,000 characters and uses the same filing path as the
  command. It returns the request id, an expected time of one to three
  minutes, and a link to watch the step; filing errors are returned as
  command errors. The link opens
  ``GET /reef/harness/requests/<id>/page`` with ``scenario`` and, when
  ``REEF_TOKEN`` is set, ``token`` in the query string.
Progress in pi
~~~~~~~~~~~~~~

While a step runs, ``ctx.ui.setWidget`` shows an animated line above the
input box. It includes the phase (``queued, waiting for a step``,
``writing the change``, ``checking the harness``, ``running the step``,
or ``saving the result``), elapsed time, a terminal link to the request
page, and ``ctrl+q or /reefine to look in``. The animation advances every
250 ms between polls. The link uses OSC 8 where the terminal supports it.

``ctrl+q`` expands the widget to show the request, id, reported episode
count and step record, page link, and a reminder that the step runs in the
background. Press it again to collapse. ``/reefine`` without an argument
prints the same detail. Expansion redraws the last poll; it makes no new
request. Pi extensions cannot attach a click handler to the widget, so the
line offers the link, key, and command.

The shortcut uses an available plain ``ctrl+<letter>``. Some terminals,
including Apple Terminal without the Kitty keyboard protocol or xterm's
modifyOtherKeys, send ``ctrl+shift+<letter>`` as the same control byte;
``ctrl+r`` already renames a session. The widget clears when the step
settles. Headless sessions do not show it.
After filing, ``ctx.ui.setStatus`` shows ``reef: request <id> queued``.
Once the step starts, it shows ``reef: step for request <id> running for
<Nm SSs>``. Each poll reads ``GET /reef/harness/requests/<id>/progress``
for the phase, episode count, and step record. Elapsed time starts at the
step's ``started_at`` when supplied. Record reads alone detect only
requests removed from storage. If the service lacks the progress route,
the spinner stays at ``queued``; other behavior is unchanged.

The extension also polls ``GET /reef/harness/releases`` every
``REEF_HARNESS_WATCH_MS`` milliseconds (5,000 by default) for a row whose
``metrics.training_request.id`` matches the request. It stops after 30
minutes. Only one watch runs; a second filing replaces it, and
``session_shutdown`` clears it. Every fetch has an abort signal and a
10-second deadline, shortened by ``REEF_HARNESS_FETCH_MS``. One stalled
fetch therefore cannot stop later polls.

When a result appears, the report quotes the first 60 characters of the
request and gives the next action:

- A selected release names ``/versions <version> install``. A pending
  release first asks the user to read it through ``/versions <version>``
  because it changes an extension.
- A rejected step quotes ``selection.reason`` and suggests rephrasing or
  splitting the request.
- A skipped step quotes ``metrics.skipped`` and, when present,
  ``proposal_notes.failure``.

Each report points to ``/versions <step>`` for details. It adds
``Not covered: ...`` if ``proposal_notes.review.uncovered`` lists items.
The extension sends both a durable custom message (``pi.sendMessage`` with
``customType: "reef-harness"`` and ``triggerTurn: false``) and a transient
notice. After the watch times out, it tells the user to check ``/versions``
for the eventual result.

Install and setup
~~~~~~~~~~~~~~~~~

When a step settles, pi offers to install its release. It waits until the
session is idle (``ctx.isIdle()``); a busy session keeps the report's
commands, and the next session start offers the release again.
``/versions <version> install`` also starts installation after a
confirmation linking to the step page. Installing a release held back from
the served head promotes it first. Rejected or skipped steps have no tree
to install. The separate automatic update notice remains available at
session start.

The install uses the ``reef-pi`` wrapper from ``REEF_HARNESS_WRAPPER``
(exported by ``run_agent``) or beside the release file. If neither exists,
pi reports ``reef: no reef-pi wrapper found; install it with reef-pi
update, then reef-pi setup``. The wrapper runs
``reef-pi update --release <id>``, then setup, and reports
``Installed release <id8>. Type /reload to load it now.`` The user must
type ``/reload``; pi then reruns ``session_start`` on the installed tree.

If the update exits with code 3 because setup items remain unmet, pi runs
setup first and retries. Other failures stop with
``reef: reef-pi update failed (exit N): <stderr>``. If the installation
directory was rebound to another service or scenario during the session,
setup and update still use the session's original service, scenario, and
token. A successful update restores the installation's configuration.
Commands targeting a different directory use that directory's own
configuration. Setup checks and values are tied to the selected release;
if it is absent, the error names the queried service and scenario. Refresh
``/versions`` before selecting again.

The setup loop starts with ``reef-pi setup --json --release <id>`` and
checks each item's ``met`` status. It asks once for each unmet item:

- For ``env``, ``ctx.ui.input`` shows the item's ``prompt`` or
  ``Value for <NAME>``. Pi passes the answer to
  ``reef-pi setup --set NAME=<value> --release <id>``.
- For ``permission`` or ``service``, ``ctx.ui.confirm`` shows the item's
  ``prompt`` or ``Run this check?`` and its check text. Pi then runs
  ``reef-pi setup --run NAME --release <id>``.

Each item reports ``reef: NAME set``, ``reef: NAME met``,
``reef: NAME not met (exit N)``, or ``reef: NAME skipped`` for a declined
check or empty value. Remaining items are listed as
``reef: still to set up: A, B (reef-pi setup)``. If listing fails, pi
reports stderr and stops setup. Environment values go into the wrapper's
env file, never the tree or Reef. An evolved extension reads them from
``process.env`` when it runs.

Requests awaiting reports remain in ``.reef-harness-requests.json`` beside
the release file as ``{id, text, filed_at}`` entries. Pi keeps the newest
ten for at most one day and removes each after reporting it. At
``session_start``, requests already in the catalog produce their custom
message and notice; others restart the watch. A restarted session can
therefore still report a result missed earlier.

With a UI, ``session_start`` announces the two commands and counts
releases held back from the served head, with the install command:
``N release(s) ready to install: /versions <version>[, <version>]
(install with /versions <version> install)``.

``/versions [version] [install]`` lists releases oldest first with version
(``v0``, ``v1``, ...), release id, result, and status. Status distinguishes
the locally installed version from the served head. Request summaries
appear under their rows with collapsed whitespace; the footer explains
statuses and available commands. A version such as ``v3`` or ``3`` offers
the step page at ``GET /reef/harness/releases/{step}/page`` with scenario
and token query parameters. The page includes the design, review, and
numbers. Accepting opens it through ``open``, ``xdg-open``, or
``rundll32``; declining prints the URL. Headless mode prints the summary
and URL. ``/versions <version> install`` confirms before installing and
promotes a held release first.

The service performs the actual edit. Its evolution step passes the filed
request to the recipe's ``propose`` method and records it, including merged
``requires`` items, under ``training_request`` in the commit. The two pi
tools clarify and file the request from the user's session (issue #435);
they do not write mutations or independently start a step. The earlier
harness-requests RFC (#310) described an agent side that only asked.

Admission rejects a ``code_extension`` that writes directly to stdout or
stderr (``console.log``, ``console.error``, or
``process.stdout.write``) without a ``ctx.hasUI`` guard on that line or
the preceding line. In a UI session, the harness owns the terminal; raw
output can disrupt the input box until a full redraw. Without a UI,
console output remains available. For UI text, use ``ctx.ui.notify``,
``ctx.ui.setStatus``, or ``ctx.ui.setWidget``.

Every adapter wrapper offers ``reef-<adapter> harness "<request>"``.
The ids ``reef-version-check``, ``reef-requests``, and
``reef-pi-extension-api`` are ``RESERVED_ENTRY_IDS`` in
`reef/harness/tree/nodes.py <../../reef/harness/tree/nodes.py>`__.
Seed and recovered state may contain them, but admission refuses mutations
that create, update, or remove them.

When a scenario opens, ``CordisBackend.shipped_content_update`` compares
those reserved entries in the running Reef's seed with the served release.
If an entry differs or is missing, Reef replaces or appends it, renders the
tree, and commits a training release with metrics
``{"shipped_content_update": {"entries": [<ids>]}}``. This update
consumes no records, runs no evaluation, and waits for no review because
the content ships with Reef. The update notice offers it to installed
trees like any other served head.

An evolved extension runs in pi's process with the user's privileges.
Admission screens its text only for credential-shaped literals. The pi
tutorial deployment therefore sets
``evolution.review_kinds: [code_extension]`` alongside ``requests: true``
and ``version_check: true``. A selected release that changes an extension
waits for promotion after review.

``propose`` receives the request's ``requires`` list beside its text.
Each item names a requirement from the user's machine as
``{name, kind, check}``. The method may append requirements of the same
shape for its proposed change. The tutorial's proposer, for example,
asks the served model for a ``{"requires": [...]}`` object alongside
the new entries.

After shape and text validation, the backend merges requirements by name
into the commit's ``training_request.requires``. It drops an invalid
method-added item without dropping the mutations. The resulting list
appears in the release row, manifest, install script's refusal, and
``reef-<adapter> setup``. Reef does not run those checks on the service.
