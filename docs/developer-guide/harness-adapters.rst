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
in the session header and hands them to ``pre_execute`` hooks, and the
sandbox will read them when it enforces policy per tool. The seed tools
declare theirs; ``run_bash`` declares all three a shell can do.
``reef.harness.native.seed.SEED_TOOLS`` holds the starting ``read_file``,
``write_file``, and ``run_bash`` tools as entries a recipe can seed and the
loop can then evolve. An adapter that declares no ``files.native_tool`` path
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
``tutorials/harness_evolve/serve-native.yaml`` seeds them by reference to
run the tutorial on this adapter.

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
   end | ends the turn with ``reason`` ``completed`` or ``gave_up``

Each model stage is one step, so ``max_steps`` bounds model calls as before,
and each call asks for at most 4,096 tokens, so one runaway reply cannot hold
a single slot local server for every other caller;
entering a model stage with the budget spent ends the turn with
``max-steps``. The log names the path: ``stage/enter`` (``step``, ``stage``,
``kind``) and ``stage/exit`` (``outcome``, ``to``, and for a verify stage
``check`` and ``last_line``), text a stage injects is a ``user/message`` with
``source.kind`` ``stage``, the session header's ``graph`` says whether
``main`` or the ``seed`` ran, and a graph that cannot load is a
``LOAD_ERROR`` like a tool. A run that somehow exceeds
``(max_steps + 1) * 16`` transitions ends with ``GRAPH_ERROR``; admission
proves that cannot happen, the guard is the backstop.

The native loop writes its trajectory as ``native-jsonl``: one
``{type, seq, time, data}`` object per line, ``seq`` contiguous from 0. A
``session`` header line names the task, model, tools, and hooks (name to
event); then ``turn/start``, per step ``step/start``, ``request/header`` (the
rendered system prompt and the tool declarations, logged on the first step so
the log holds everything the model saw), ``assistant/message`` (``content``,
``tool_calls``, ``finish``), ``tool/call`` (the raw argument string),
``tool/result`` (``content``, ``is_error``, and on error a closed ``code``:
``UNKNOWN_TOOL``, ``INVALID_ARGS``, ``TOOL_FAILED``, ``HOOK_DENIED``,
``APPROVAL_REQUIRED``, ``HOOK_BLOCKED``),
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

The descriptor
--------------

One ``descriptor.yaml`` declares how a tree configures and starts a running
agent.

.. config::

   name | the adapter's id
   binary | the executable an episode runs
   argv | the argument list for one headless prompt; ``{prompt}`` is substituted
   files | where each node kind renders, like ``skills/{name}/SKILL.md``
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
