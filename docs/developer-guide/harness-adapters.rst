Harness adapters
================

An adapter maps a source tree to an agent's configuration, headless command
and session format. Add Python rendering only where descriptors are insufficient.

The adapter file
----------------

Start from an existing descriptor, such as `pi's <../../reef/harness/adapters/pi/adapter.yaml>`__.
This excerpt declares installation, parts and execution:

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

Also provide model bindings and config checks; ``primary`` is required.
Unknown keys fail loading. Pin installation versions and stage packages before execution;
Harbor runs must not depend on registry access.
See the `descriptor contract <../../reef/harness/adapters/descriptor.py>`__.

Tree parts
----------

Descriptors choose rendered paths; part kinds fix source layouts and merge rules:

.. list-table::
   :header-rows: 1

   * - Kind
     - Source
     - Merge
   * - ``rules``
     - ``rules.md``
     - By line
   * - ``skill``
     - ``skills/{name}/SKILL.md``
     - By line
   * - ``command``, ``memory``
     - ``commands/{name}.md``, ``memory/{name}.md``
     - By line
   * - ``extension``
     - ``extensions/{name}`` plus suffix
     - By file
   * - ``native_*``, ``program``
     - ``native/`` subdirectories, ``program/{name}.py``
     - By file
   * - ``config``
     - ``config/<target>.json``
     - Not synced

Names begin with a letter/digit; later characters can also be ``._-``.
Only declare parts that affect every supported execution mode.
``review: true`` always requires manual activation; native loops and Python programs require it.
Validation rejects unsupported paths, empty files, credentials and invalid declared code contracts.
See `part contracts <../../reef/harness/adapters/parts.py>`__.

Rendering
---------

``render_tree(tree, descriptor, overlays)`` returns relative paths and contents:

1. Merge target defaults, tree config and overlays.
2. Place files at declared render paths.
3. Apply checks and adapter steps.

Output suffixes select JSON, TOML, YAML or env formatting.
Unknown targets, collisions, failed checks and binding-key overrides raise render errors.
Custom ``AdapterRenderer`` methods must be stateless; raise ``RenderError`` for refusal.
Prefer declarative checks over Python transformations.

Client installation requires ``run.layer``. It registers the rendered plugin
without merging user configuration. Layer rendering starts config targets empty
and skips evaluation checks. Credentials use ``{token}`` through environment variables only.
Every declared part must work in evaluation and installed modes.

Config checks
-------------

Checks constrain merged settings before rendering steps:

.. code:: yaml

   checks:
     - key: [model, fallbackModel]
       match: IsAbsent()
       reason: Reef's binding selects the model

Model-binding keys stay reserved. Check alternate providers, fallback models and proxies too.
Errors identify keys, never values.
Use ``AnyOf(IsAbsent(), X)`` for optional constrained values.
See `matchers <../../reef/harness/adapters/matchers.py>`__ for available checks.

Token limits
~~~~~~~~~~~~

``{context_window}`` uses ``REEF_CONTEXT_WINDOW``, endpoint metadata, then 128000.
``{max_output_tokens}`` uses ``REEF_MAX_OUTPUT_TOKENS``, then 32000.
Values must be positive integers.
Pi and OpenCode write both. Codex writes only explicitly supplied context;
other adapters write neither. Artifact-evaluation ``env`` can override these variables.

Connect a new agent
-------------------

1. Define installation, paths, model templates and run settings.
2. Add a ``TrajectoryReader`` for new session formats; ``to_atif`` supplies verifier trajectories,
   and ``agent_work`` the turns, steps, tool calls and tokens of an evaluation step's metrics.
3. Register in ``BUILTIN_ADAPTERS`` or ``reef.harness_adapters`` entry points.
4. Test descriptors, golden renders, model routes and captured sessions.

Harbor skills/MCP require matching ``run.task`` declarations; missing capabilities fail the rollout.
Unsupported MCP transports and HTTP headers fail evaluation admission.
Run a real task and confirm every part reaches the model in each supported mode.
Test client connection when ``run.layer`` is supplied.
See the `bundled adapters <../../reef/harness/adapters>`__ for complete examples.
