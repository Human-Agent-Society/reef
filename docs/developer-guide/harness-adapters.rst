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

Also declare model bindings, config checks and any client installation commands.
``primary`` is the required config target. ``load_descriptor`` rejects unknown keys.
The `descriptor contract <../../reef/harness/adapters/descriptor.py>`__ defines
all fields, including self-isolation requirements.

How Reef runs an agent
----------------------

Evaluation renders a fresh root, relocates agent state, runs in an empty
workspace, reads the session and removes the root.
Harbor tasks install the pinned agent inside the task environment, render
its tree and run the verifier.
Client installations use ``run.layer`` without reading the user's configuration.

Harbor tasks require declared skills and MCP support for those capabilities.
Missing capabilities permanently fail the rollout.
Reef stages packages before execution; do not require registry access during runs.
A reader with ``to_atif`` supplies ``/logs/agent/trajectory.json`` to verifiers.

Task MCP servers
~~~~~~~~~~~~~~~~

Declare supported transports in ``run.task.mcp`` with a config ``target``
and server-mapping ``key``. Templates accept ``{command}``, ``{args}``,
``{argv}`` and ``{url}``; other values remain literal.
Task servers merge after config checks.

Unsupported transports, missing declarations and HTTP headers fail evaluation
admission. See `MCP descriptor types <../../reef/harness/adapters/descriptor.py>`__
for the complete mapping.

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

Names start with a letter or digit; later characters can also be ``._-``.
Use ``{renders: PATH, format: FORMAT}`` to override a part's format.
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

Use the descriptor's ``evolve`` mapping for destinations and
`parts.py <../../reef/harness/adapters/parts.py>`__ for source layouts.
Only declare parts that affect every supported execution mode.
For example, Codex supports skills and skill-based commands, but no rules or extensions.

Config targets and a person's machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``config/<target>.json`` merges into the named target.
Checks limit writable settings; model-binding keys remain reserved.
Client installation requires ``run.layer``, which pi, OpenCode, Claude and Codex provide.

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

See `matchers.py <../../reef/harness/adapters/matchers.py>`__ for typed matchers
and ``AllOf``, ``AnyOf`` and ``Not`` composition.

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
Use the `model-route tests <../../tests/harness/test_harness_model_route.py>`__.

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

``reef-client harness`` installs the rendered plugin with ``register``;
``disconnect`` runs ``unregister``. Sync applies new selections.
User configuration remains separate.

Layer rendering starts targets empty and skips evaluation checks.
Use ``layer_files`` for installation metadata.
Pass ``{token}`` only through environment variables, never command arguments.
Every declared part must work in both evaluation and installed plugins.

Bundled adapters
----------------

The `bundled descriptors <../../reef/harness/adapters>`__ define pinned versions,
commands and model dialects. ``evolution.binary`` overrides the installed binary.
The Python adapter runs ``program/main.py`` with uv and PEP 723 dependencies;
its descriptor defines the model binding and session format.

Connect a new agent
-------------------

1. Write the descriptor's paths, installation, model templates and run settings.
2. Add a ``TrajectoryReader`` if the session format is new.
   Implement ``to_atif`` for task verifiers; reuse Harbor converters when available.
3. Register in ``BUILTIN_ADAPTERS`` or the ``reef.harness_adapters`` entry-point group.
   External entries return an ``AdapterDescriptor`` directly or through a zero-argument callable.
4. Add rendering steps only where declarations are insufficient.

Test descriptor loading, rendered files, model routes and captured sessions.
Run a real task and confirm each supported part reaches the model in
evaluation and client modes. Test ``reef-client harness connect`` when supported.
External readers use ``package.module:ClassName`` in ``sessions.format``.

Native runner
-------------

``reef-native`` reads rules, skills and model bindings from ``native/``.
Start from `the native seed <../../tutorials/evolve-your-harness/seeds/native>`__.
Tools, hooks, graphs, delegated agents and custom loops extend execution.
Graphs and loops allow 1–32 steps; custom loops always require review.

Tree code runs with user privileges. Tool capabilities apply under Linux
sandbox execution; Seatbelt retains the outer profile.
See the `runner contracts <../../reef/harness/runners/native>`__ for each extension.

Trajectory
~~~~~~~~~~

``native-jsonl`` records ``{type, seq, time, data}`` events, including turn
termination reasons. Large tool outputs are saved under ``.reef/tool-output/``.

Run and serve modes
~~~~~~~~~~~~~~~~~~~

``reef-native -p`` runs one turn.
``reef-native serve`` follows active harness artifacts between steps;
failed loads retain the current tree.
``--follow pinned`` requires explicit ``reef-native mount <artifact_id>``.
Model calls carry scenario and artifact identity.

``REEF_NATIVE_ENFORCE=bwrap`` confines tool calls only.
``--self-tools`` adds inspection and trial tools.
