Harness adapters
================

An adapter maps a harness tree into the files expected by a third-party
coding-agent CLI and binds that harness to the served model. The harness and
model together form the running agent. The tree never names a file path; the
adapter does. Reef bundles five, one per third-party coding-agent CLI;
``native``, its own agent, whose loop lives in this tree and whose tools are
``native_tool`` nodes, so a mutation can add, rewrite, or remove a tool; and
``terminus``, Terminal-Bench's Terminus 2, which is a harbor agent class
rather than a CLI and so is driven by a runner Reef owns.

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
|              |                                                           | reef-eval via ``reef-infra[terminus]``       |
+--------------+-----------------------------------------------------------+-------------------------------------------+

The ``terminus`` adapter is the one that does not drive a CLI. Terminus 2 is
a Harbor agent class, so the adapter ships its own runner,
``reef-terminus``, which reads the tree from ``REEF_TERMINUS_DIR``, binds it
to a Terminus 2 subclass, runs one Terminal-Bench task, and writes the ATIF
trajectory and the verifier's reward under ``REEF_TERMINUS_SESSION_DIR`` for
the ``terminus-atif-json`` reader. It reaches Harbor through reef-eval, the
same primitive the examples under ``recipes/`` use, so Reef has one way in
rather than two. That is an optional extra (``pip install
reef-infra[terminus]``) imported only when the runner runs, so the adapter
registry and the render path need neither Harbor nor Docker.

Each node kind binds to one agent seam. ``config`` becomes Terminus 2
constructor arguments, and a key that is not one is refused at render.
``rules``, ``skill`` and ``agent_command`` join the instruction; Terminus 2
has no slash-command surface, so a command renders under a second skill root
and is named as user-invocable, the resolution the ``dsh`` adapter also
uses. ``code_extension`` is the context seam: one module defining
``assemble(state, request, files)``, loaded in the runner's process and
called before every model call the main loop makes, so an evolved policy can
rewrite history, compact it, or carry notes between turns. It is limited to
one module because the seam is a single call. A policy that raises is logged
and skipped, so a defective candidate degrades to stock assembly and still
earns a score.

On an empty tree every seam is a no-op and the agent is stock Terminus 2.
That equivalence is what makes the stock benchmark score the baseline a
gated change is measured against. Isolation is harbor's task container, not
Reef's jail: Docker cannot nest in bubblewrap, so this adapter does not run
under ``evolution.executor: sandbox``.

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
``native/tools/{name}.py``: a module whose ``NAME``, ``DESCRIPTION``, and
``PARAMETERS`` come from the node config and whose ``code`` defines
``run(args, workdir) -> str``. ``reef.harness.native.seed.SEED_TOOLS`` holds
the starting ``read_file``, ``write_file``, and ``run_bash`` tools as entries
a recipe can seed and the loop can then evolve. An adapter that declares no
``files.native_tool`` path refuses to render that kind, so the mutation fails
under it instead of silently dropping the tool.

The native loop writes its trajectory as ``native-jsonl``: one
``{type, seq, time, data}`` object per line, ``seq`` contiguous from 0. A
``session`` header line names the task, model, and tools; then ``turn/start``,
per step ``step/start``, ``request/header`` (the rendered system prompt and
the tool declarations, logged on the first step so the log holds everything
the model saw), ``assistant/message`` (``content``, ``tool_calls``,
``finish``), ``tool/call`` (the raw argument string), ``tool/result``
(``content``, ``is_error``, and on error a closed ``code``: ``UNKNOWN_TOOL``,
``INVALID_ARGS``, ``TOOL_FAILED``), ``step/end``, and finally ``turn/end``
with a ``reason`` of ``completed``, ``max-steps``, or ``error``. Arguments are
validated against the tool's declared schema before ``run`` sees them, and a
``user/message`` from the ``loop-guard`` plugin lands when the same call
repeats three, five, or eight times in a row.

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
   env | variables pointing the agent's state under the episode root; ``{root}`` is substituted
   install | the one-command install pin: ``kind`` (``npm``, or ``git`` for a checkout installed editable into a venv, which adds ``repository`` and ``ref``), ``package``, ``version`` (what ``--version`` must report), and ``binary_path`` under the install prefix
   model_binding | per API dialect (``openai``, ``anthropic``), the config nodes Reef appends at evaluation time; ``{base_url}``, ``{api_key}``, and ``{model}`` substitute into string values
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
