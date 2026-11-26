Harness adapters
================

An adapter connects a harness source tree to an agent's configuration,
headless command and session format. The same adapter supports evaluation,
Harbor tasks and, optionally, installation on a user's machine.

Start with ``adapter.yaml``. Add Python rendering code only for behavior
that the descriptor cannot express.

The adapter file
----------------

Use an existing descriptor as a starting point, such as
`pi's adapter <../../reef/harness/adapters/pi/adapter.yaml>`__.
Its core fields look like this excerpt:

.. code:: yaml

   name: pi
   binary: pi
   install:
     kind: npm
     package: "@earendil-works/pi-coding-agent"
     version: "0.84.2"
     binary_path: node_modules/.bin/pi
   evolve:
     rules: plugin/rules.md
     skill: plugin/skills/{name}/SKILL.md
     command: plugin/prompts/{name}.md
     extension: plugin/extensions/{name}.ts
   dirs:
     pi-agent: PI_CODING_AGENT_DIR
     sessions: PI_CODING_AGENT_SESSION_DIR
   sessions:
     format: pi-session-jsonl
     path: sessions
   run:
     task:
       argv: ["--mode", "json", "-p", "{task}", "-e", "{root}/plugin"]
       env: {PI_OFFLINE: "1"}
   config:
     primary:
       path: plugin/settings.json

The complete descriptor also supplies model bindings, checks and installation
commands for client machines.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Section
     - Contract
   * - ``install``
     - Pinned npm, uv or git installation. Git uses repository and ref.
   * - ``evolve``
     - Supported tree parts and their rendered paths.
   * - ``dirs``
     - Writable agent directories and environment variables that relocate them.
   * - ``sessions``
     - Session path and trajectory-reader format.
   * - ``model``
     - Binding templates for ``openai``, ``responses`` or ``anthropic``.
   * - ``run.task``
     - Headless arguments, environment, optional task skills and MCP support.
   * - ``run.layer``
     - Client installation/removal commands and session arguments/environment.
   * - ``config``
     - Named target paths, defaults and checks. ``primary`` is required.
   * - ``renderer``
     - Optional ``module:Class`` implementing render steps.

``load_descriptor`` rejects unknown keys.
Self-isolating agents can declare execution validation and host-environment
requirements; see `AdapterDescriptor
<../../reef/harness/adapters/descriptor.py>`__.

How Reef runs an agent
----------------------

Evaluation renders a fresh root, relocates agent state, runs in an empty
workspace, reads the session and removes the root.
Harbor tasks install the pinned agent inside the task environment, render
its tree and run the verifier.
Client installations use ``run.layer`` without reading the user's configuration.

For Harbor tasks:

- Reef stages pinned packages and required runtimes before execution.
  Do not rely on registry access during the agent run.
- ``environment.skills_dir`` requires ``run.task.skills``.
  Task skills are copied first; tree files override matching paths.
- ``environment.mcp_servers`` requires compatible ``run.task.mcp`` entries.
- Missing adapter capabilities fail that rollout permanently; other rollouts continue.
- A reader with ``to_atif`` exposes ``/logs/agent/trajectory.json`` to verifiers.

Task MCP servers
~~~~~~~~~~~~~~~~

Declare only transports the agent supports:

.. code:: yaml

   run:
     task:
       mcp:
         target: agent
         key: mcp
         stdio: {type: local, command: "{argv}"}
         sse: {type: remote, url: "{url}", oauth: false}
         streamable-http: {type: remote, url: "{url}", oauth: false}

``target`` names a config file; ``key`` names its server mapping.
Exact placeholders substitute command, argument list, combined argv or URL:
``{command}``, ``{args}``, ``{argv}``, ``{url}``.
Other values remain literal.

Task servers merge after config checks. A tree can therefore be prohibited
from supplying servers while task configuration supplies them through this interface.
Evaluation rejects unsupported transports, missing MCP declarations and servers
with HTTP headers before running any rollout.

Tree parts
----------

A part kind fixes its source path, format and merge behavior.
The descriptor selects the rendered path:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Kind
     - Source path
     - Merge
   * - ``rules``
     - ``rules.md``
     - By line
   * - ``skill``
     - ``skills/{name}/SKILL.md``
     - By line
   * - ``command``
     - ``commands/{name}.md``
     - By line
   * - ``memory``
     - ``memory/{name}.md``
     - By line
   * - ``extension``
     - ``extensions/{name}`` plus rendered suffix
     - By file
   * - ``native_*``
     - ``native/`` subdirectories
     - By file
   * - ``program``
     - ``program/{name}.py``; ``main`` is the entry point
     - By file
   * - ``config``
     - ``config/<target>.json``
     - Not synced

Names start with a letter or digit, followed by letters, digits, dots,
underscores or hyphens. Override format where needed:

.. code:: yaml

   command: {renders: "plugin/skills/{name}/SKILL.md", format: skill}

``holds`` describes the part to methods through ``material/host.json``.
``review: true`` requires manual activation regardless of ``review_kinds``.
Native loops and Python programs require this review.
Unknown review kinds fail configuration.

Validation rejects empty files, credentials and unsupported paths.
Config files must be JSON objects for declared targets.
Code extensions cannot write raw stdout/stderr without the required UI guard.
Native modules must compile and satisfy their declared contract.

Render paths
~~~~~~~~~~~~

Pi, OpenCode, Claude and Codex render supported parts into ``plugin/``.
Evaluation loads that plugin; client installation registers the same parts.
Codex supports skills and skill-based commands, but no rules or extensions.
Claude plugins do not carry its settings or memory.

Other adapters render into their own directories.
Use each descriptor's ``evolve`` mapping for exact destinations;
`parts.py <../../reef/harness/adapters/parts.py>`__ defines source layouts.
Only declare parts that affect every supported execution mode.

Config targets and a person's machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``config/<target>.json`` merges into the target file.
Checks limit writable settings, while model-binding keys always remain reserved.
For example, pi's primary target permits only an empty ``defaultTools``;
its model and agent targets remain binding-owned.

Only adapters with ``run.layer`` support client installation.
Pi, OpenCode, Claude and Codex provide it. Other bundled adapters run through
Reef without a client layer.

Config checks
-------------

Checks validate merged config before adapter steps. Errors identify keys,
never their values:

.. code:: yaml

   checks:
     - key: includeCoAuthoredBy
       match: IsDefault()
     - key: [model, fallbackModel]
       match: IsAbsent()
       reason: Reef's binding selects the model

``key`` accepts a dot path, path list or ``.`` for the whole file.
``*`` matches entries and ``**`` matches any depth.
Use ``AnyOf(IsAbsent(), X)`` for optional values constrained by ``X``.

Matchers cover defaults/absence, typed equality, membership, numeric bounds,
object/integer types, text matching and transformations, collection entries,
and ``AllOf``, ``AnyOf`` or ``Not`` composition.
See `matchers.py <../../reef/harness/adapters/matchers.py>`__ for their contracts.

Rendering
---------

``render_tree(tree, descriptor, overlays)`` returns root-relative file paths
and contents without writing to disk:

1. Merge target defaults, tree config and overlays, in that order.
2. Place other files at their declared render paths.
3. Apply checks and adapter steps.

Output config uses its suffix's format: JSON, TOML, YAML or ``.env``.
Unknown paths/targets, collisions, failed checks and tree-supplied binding
keys raise render errors.

Model binding checks
~~~~~~~~~~~~~~~~~~~~

Bindings supply endpoint, model and credentials.
Trees cannot set those keys. Add checks for alternate routes such as
fallback models, extra providers and proxies.
These checks inspect configuration; they cannot control a tool that constructs
its own model request.
Use ``tests/harness/test_harness_model_route.py`` for route contract tests.

Token limits
~~~~~~~~~~~~

Binding templates can use:

- ``{context_window}``: ``REEF_CONTEXT_WINDOW``, endpoint model metadata,
  or 128000 tokens, in that order.
- ``{max_output_tokens}``: ``REEF_MAX_OUTPUT_TOKENS`` or 32000 tokens.

Pi and OpenCode write both limits.
Codex writes context only when supplied by environment or model metadata;
it writes no reply limit. Other adapters write neither.

Harness-item ``env`` can set these values for artifact evaluations.
Values must be positive integers. Correct context limits let agents compact
before reaching the endpoint limit.

Adapter steps
~~~~~~~~~~~~~

Subclass ``AdapterRenderer`` with static, stateless methods:

- ``process_config`` adjusts one merged config.
- ``process_skill`` and ``process_command`` process one file's text.
- ``check_model_route`` validates routes that config checks cannot express.
- ``finalize_render`` handles unavoidable cross-file work, such as manifests.
- ``layer_files`` adapts client installation output.

Defaults preserve input. Raise ``RenderError`` to reject it.
Prefer config checks for value constraints and narrow steps for transformations.
Do not pass state between renders or steps.

A person's machine
------------------

``reef-client harness`` downloads the client-form render and runs the adapter's
``register`` command once per installed artifact.
``disconnect`` runs ``unregister``. New selections arrive on the next sync.
Existing user configuration remains separate.

Layer rendering starts config targets empty and skips evaluation checks.
``layer_files`` adds installation files such as marketplace metadata.
Templates support root, model and base URL. Credentials use ``{token}``
in environment variables only, never command-line arguments.

Every declared part must work in both evaluation and installed plugins.
Leave unsupported parts out or reject them during validation.

Bundled adapters
----------------

Bundled adapters include pi, OpenCode, Claude, Codex, dsh, Hermes, native
and Python. Their descriptors own pinned versions, commands and dialect support.
``evolution.binary`` can override the installed binary.
External packages register through ``reef.harness_adapters``.

The Python adapter runs ``program/main.py`` with uv and PEP 723 dependencies.
It receives the task as an argument and reads model binding from
``program/model.json``. Write request/response JSONL under
``REEF_PROGRAM_SESSION_DIR`` for ATIF conversion.

Connect a new agent
-------------------

1. Write the descriptor's paths, installation, model templates and run settings.
2. Add a ``TrajectoryReader`` if the session format is new.
   Implement ``to_atif`` for task verifiers; reuse Harbor converters when available.
3. Register in ``BUILTIN_ADAPTERS`` or the ``reef.harness_adapters`` entry-point group.
   External entries return an ``AdapterDescriptor`` directly or through a zero-argument callable.
4. Add rendering steps only where declarations are insufficient.

Verify loading, golden rendered files, reserved model routes and captured sessions
in ``tests/harness/``. Run a real task against a real model and check
exit status, residue and instruction use.
For every supported part, confirm a marker reaches the model in both evaluation
and client-layer modes. Finally test ``reef-client harness connect``.
External trajectory readers use ``package.module:ClassName`` in ``sessions.format``.

Native runner
-------------

``reef-native`` reads rules, skills and model bindings from ``native/``.
Use ``tutorials/evolve-your-harness/seeds/native/`` as the starter tree.
It also supports:

- Tools: ``native/tools/<name>.py`` defines ``run(args, workdir) -> str``
  and literal ``DESCRIPTION``, ``PARAMETERS`` and ``CAPABILITIES``.
- Hooks: ``listen(payload, next)`` at ``pre_step``, ``pre_execute``,
  ``request_error`` or ``post_execute``.
- Graphs: JSON stages and outcome edges; ``main`` starts a turn.
- Agents: delegated prompt, graph, tools, skills, step budget and continuation.
- Loops: ``run_turn(ctx)`` replaces the main graph and always requires review.

Graphs and loops allow 1–32 steps. Context exposes messages, model calls,
tool execution, delegation and termination.
Local tools run in-process; Linux sandbox tools receive only declared
read/write/exec/network capabilities. Seatbelt retains the outer run profile.
Hooks and loop code run in the loop process.

Trajectory
~~~~~~~~~~

``native-jsonl`` records ``{type, seq, time, data}`` events for turns, steps,
requests, assistant messages and tools. Turn endings include completion,
budget and error reasons. Tool output above 20,000 characters goes to
``.reef/tool-output/``.

Run and serve modes
~~~~~~~~~~~~~~~~~~~

``reef-native -p`` runs one turn.
``reef-native serve`` follows active harness artifacts between steps;
failed loads retain the current tree.
``--follow pinned`` requires explicit ``reef-native mount <artifact_id>``.
Model calls carry scenario and artifact identity.

Tree code runs with user privileges. ``REEF_NATIVE_ENFORCE=bwrap`` confines
tool calls only. ``--self-tools`` adds inspection and trial tools.
The ``native/serve.sock`` socket accepts one JSON request per connection
for a turn, status or artifact mount.
