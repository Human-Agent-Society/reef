Harness adapters
================

.. page::
   :for: developers who connect a new agent to Reef
   :needs: the harness tree and the loop from `Evolve your harness <../user-guide/evolve-your-harness.rst>`__
   :outcome: an adapter that Reef can render, run in episodes and Harbor tasks, and install on a person's machine

An adapter connects a harness tree to one agent. It is one file,
``reef/harness/adapters/<name>/adapter.yaml``, and an optional ``render.py``
for what the file cannot state. Reef renders the same tree with it in three
places:

.. diagram::
   :caption: The adapter renders one harness tree three ways: as the whole configuration of an evaluation episode, inside the container of a Harbor task, and as a layer over a person's own configuration.

   <img class="fig-light" src="../assets/harness-adapter-modes-light.svg" alt="One adapter.yaml, three ways an agent runs: an evaluation episode in Reef, a Harbor task in its container, and a layer on a person's machine">
   <img class="fig-dark" src="../assets/harness-adapter-modes-dark.svg" alt="One adapter.yaml, three ways an agent runs: an evaluation episode in Reef, a Harbor task in its container, and a layer on a person's machine">

The adapter file
----------------

pi's ``adapter.yaml``, shortened:

.. code-block:: yaml

   name: pi
   binary: pi
   install: {kind: npm, package: "@earendil-works/pi-coding-agent", version: "0.84.2",
             binary_path: node_modules/.bin/pi}
   evolve:                                  # what Reef may evolve, and where pi reads it
     rules: pi-agent/AGENTS.md
     skill: pi-agent/skills/{name}/SKILL.md
     command: pi-agent/prompts/{name}.md
     extension: pi-agent/extensions/{name}.ts
   dirs:                                    # pi's directories, and the variable that moves each one
     pi-agent: PI_CODING_AGENT_DIR
     sessions: PI_CODING_AGENT_SESSION_DIR
   sessions: {format: pi-session-jsonl, path: sessions}
   model:                                   # per API dialect: the config that binds the model
     openai: [{target: models, data: {...}}, {target: primary, data: {...}}]
   run:
     episode: {argv: ["--mode", "json", "-p", "{prompt}"], env: {PI_OFFLINE: "1"}}
     layer: {args: [["-e", "{root}/pi-agent"], ["--model", "reef/{model}"]]}
   config:                                  # the config files, with defaults and checks
     primary: {path: pi-agent/settings.json, checks: [...]}
     models: {path: pi-agent/models.json, checks: [...]}

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Section
     - Says
   * - ``install``
     - the vendor's package and pinned version (``npm``, or ``git`` with
       ``repository`` and ``ref``); ``version`` is what ``--version`` reports
   * - ``evolve``
     - each part Reef may evolve, and where the agent reads it (`Tree parts`_)
   * - ``dirs``
     - the agent's directories under the root, each with the variable that
       moves it there; what the agent writes there is state, not residue
   * - ``sessions``
     - where the agent writes its session log, and the ``TrajectoryReader``
       format that reads it
   * - ``model``
     - config templates per dialect (``openai``, ``responses``,
       ``anthropic``), with ``{base_url}``, ``{api_key}`` and ``{model}``
   * - ``run.episode``
     - the headless command (``{prompt}``) and its variables
   * - ``run.layer``
     - how a person's machine runs the agent: ``args``, ``env``, and the
       agent's own ``register``/``unregister`` commands (`A person's machine`_)
   * - ``config``
     - each config file's ``path``, ``defaults`` and ``checks``
       (`Config checks`_); ``primary`` is required
   * - ``renderer``
     - ``module:Class`` with the adapter's render steps (`Adapter steps`_)

An agent that isolates its own episodes, such as the Terminal-Bench
campaign's ``terminus``, may also set ``self_isolating``,
``execution_validator``, ``host_env``, ``is_root_bind_mounted`` and
``is_prompt_task_directory`` (see `descriptor.py
<../../reef/harness/adapters/descriptor.py>`__). ``load_descriptor`` refuses
an unknown key.

How Reef runs an agent
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Where
     - What Reef does
     - Uses
   * - Evaluation episode
     - Renders the tree with the model binding into a new root, moves the
       agent's directories there, runs ``run.episode`` in an empty
       ``workspace/``, reads the session log, and removes the root.
     - ``dirs``, ``model``, ``run.episode``, ``sessions``
   * - Harbor task
     - Starts the task's container, installs the pinned agent with npm,
       copies in the tree, runs ``run.episode`` with the instruction, and
       scores the tests' reward.
     - an npm ``install``
   * - A person's machine
     - ``reef-client harness`` installs the tree as a layer and runs the
       agent with it.
     - ``run.layer``

Tree parts
----------

A kind fixes a part's tree path, format and merge (``KINDS`` in `parts.py
<../../reef/harness/adapters/parts.py>`__). The adapter gives only the render
path:

.. list-table::
   :header-rows: 1
   :widths: 20 38 16 26

   * - Kind
     - Tree path
     - Format
     - Merge
   * - ``rules``
     - ``rules.md``
     - ``text``
     - by line
   * - ``skill``
     - ``skills/{name}/SKILL.md``
     - ``skill``
     - by line
   * - ``command``
     - ``commands/{name}.md``
     - ``command``
     - by line
   * - ``memory``
     - ``memory/{name}.md``
     - ``text``
     - by line
   * - ``extension``
     - ``extensions/{name}`` + the render path's suffix
     - ``code``
     - by file
   * - ``native_*``
     - ``native/tools/{name}.py`` and so on (`Native runner`_)
     - the kind
     - by file
   * - ``config``
     - ``config/<target>.json``, on every adapter
     - ``config``
     - not synced

``{name}`` is one name: a letter or digit, then letters, digits, ``.``,
``_`` or ``-``. Where a kind's default does not fit, an entry is an object;
Codex's commands are skills:

.. code-block:: yaml

   command: {renders: codex/skills/{name}/SKILL.md, format: skill}

Each part also has ``holds``, one line on what the file holds, which a method
reads in ``material/host.json``, and optionally ``review: true``, which makes
every change to it wait for a person.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Format
     - Validation checks, besides no empty file and no credential
   * - ``text``, ``skill``, ``command``
     - nothing more; ``skill`` and ``command`` files go through
       ``process_skill`` and ``process_command``
   * - ``code``
     - no raw write to stdout or stderr without a ``ctx.hasUI`` guard
   * - ``config``
     - a JSON object of a declared target, without an inline credential
   * - ``native_*``
     - the module compiles and declares its constants, or the JSON is a
       valid graph or agent (`Native runner`_)

Render paths
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Adapter
     - ``rules``
     - ``skill``
     - ``command``
     - ``extension``
   * - ``pi``
     - ``pi-agent/AGENTS.md``
     - ``pi-agent/skills/<name>/SKILL.md``
     - ``pi-agent/prompts/<name>.md``
     - ``pi-agent/extensions/<name>.ts``
   * - ``opencode``
     - ``opencode/AGENTS.md``
     - ``opencode/skill/<name>/SKILL.md``
     - ``opencode/command/<name>.md``
     - ``opencode/plugin/<name>.ts``
   * - ``claude``
     - ``claude/CLAUDE.md``
     - ``claude/skills/<name>/SKILL.md``
     - ``claude/commands/<name>.md``
     - ``claude/extensions/<name>.mjs``
   * - ``codex``
     - ``codex/AGENTS.md``
     - ``codex/skills/<name>/SKILL.md``
     - as a skill
     - none
   * - ``dsh``
     - ``dsh/AGENTS.md``
     - ``dsh/skills/<name>/SKILL.md``
     - ``dsh-agents/skills/<name>/SKILL.md``
     - ``dsh/profiles/headless/extensions/<name>.mjs``
   * - ``hermes``
     - ``hermes/SOUL.md``
     - ``hermes/skills/<name>/SKILL.md``
     - ``hermes-commands/<name>/SKILL.md``
     - ``hermes/plugins/<name>/__init__.py``
   * - ``native``
     - ``native/RULES.md``
     - ``native/skills/<name>/SKILL.md``
     - none
     - none

``claude`` also takes ``memory``: ``claude/memory/<name>.md``, where its
render points Claude Code's automatic memory.

Bundled adapters
----------------

At startup, Reef installs the pinned binary for the episodes unless
``evolution.binary`` names one. Another package adds an adapter through the
``reef.harness_adapters`` entry point, as the Meta-Harness Terminal-Bench
campaign adds ``terminus`` (`its README
<../../recipes/meta_harness/examples/terminal_bench/README.md#the-terminus-adapter>`__).

.. list-table::
   :header-rows: 1
   :widths: 12 26 62

   * - Adapter
     - Pin and command
     - What to know
   * - ``pi``
     - npm 0.84.2; ``pi --mode json -p``
     - Settings admit only ``defaultThinkingLevel`` and an empty
       ``defaultTools``. A skill without frontmatter gets ``name`` and
       ``description``.
   * - ``opencode``
     - npm 1.18.18; ``opencode run --format json --auto``
     - Frontmatter must be a plain YAML mapping. A command's ``agent`` must
       name an enabled agent.
   * - ``claude``
     - npm 2.1.257; ``claude -p --output-format json``
     - An extension is a script that a ``hooks`` entry of the settings runs.
       Only the ``anthropic`` dialect.
   * - ``codex``
     - npm 0.158.0; ``codex exec --json``
     - No extension: Codex hooks run outside its sandbox. Episodes pin
       ``approval_policy="never"`` and ``web_search="disabled"``. Only the
       ``responses`` dialect. On a pin change, update ``bundled_models.json``
       and ``default_instructions.md``.
   * - ``dsh``
     - npm 0.1.2-alpha.5; ``dsh --profile headless``
     - The config is keyed by plugin id; ``!!js `` strings become
       JavaScript. Commands never run on the model's own.
   * - ``hermes``
     - git ``v2026.8.31``; ``hermes chat -Q --oneshot -q``
     - Rules follow Hermes's identity in ``SOUL.md``. An extension is a plugin
       with ``register(ctx)``. Only the ``openai`` dialect.
   * - ``native``
     - ships with Reef; ``reef-native -p``
     - Reef's own loop (`Native runner`_).

The comments in each ``adapter.yaml`` explain its defaults and checks.

Config checks
-------------

A target's ``checks`` say what its merged config, the binding included, must
keep or must not set. The render runs them before any adapter step, and a
refusal names the key, never the value:

.. code-block:: yaml

   checks:
     - key: includeCoAuthoredBy
       match: IsDefault()
     - key: [model, fallbackModel]
       match: IsAbsent()
       reason: Reef's model binding chooses the model
     - key: env.DISABLE_TELEMETRY
       match: Not(Strip(Lowercase(IsIn(["0", "false", "off", "no", ""]))))

``key`` is a dot path, a list of them, or ``.`` for the whole file; ``*`` is
every entry, ``**`` any depth. "If set, it must be X" is ``AnyOf(IsAbsent(),
X)``.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Matcher
     - Matches
   * - ``IsDefault()``, ``IsAbsent()``
     - the target's default for the key; a missing key
   * - ``Eq(v)``, ``IsIn([v, ...])``, ``Ge(n)``
     - an equal value, type included; a number at least ``n``
   * - ``IsObject()``, ``IsInt()``
     - an object; an integer that is not a boolean
   * - ``StartsWith``, ``EndsWith``, ``HasSubstr``, ``MatchesRegex``
     - text, or a scalar's JSON text
   * - ``Lowercase(m)``, ``Uppercase(m)``, ``Strip(m)``, ``ParseJson(m)``
     - text that matches ``m`` after the change
   * - ``Each(m)``, ``Contains(m)``, ``Key(m)``, ``Value(m)``
     - every or some entry; an entry's key or value
   * - ``AllOf``, ``AnyOf``, ``Not``
     - all, any, or not

Rendering
---------

``reef.harness.tree.render_tree(tree, descriptor, overlays)`` returns the
files one agent reads, by root-relative path. It writes no disk.

#. Each config target is its ``defaults``, the tree's
   ``config/<target>.json`` merged in, then the overlays, such as the model
   binding. Each is written in its suffix's format (TOML, YAML, ``.env`` or
   JSON).
#. Every other file goes to its part's render path.
#. The checks run, then the adapter's steps.

The render refuses a path that no part takes, an undeclared target, two
files at one path, a failed check, and a binding key set by the tree.

Model binding checks
~~~~~~~~~~~~~~~~~~~~

Every model call goes to Reef. The binding writes the endpoint, the model
and the credential after the tree, so it wins every key it writes, and the
render refuses a tree that sets one of them. Each adapter's checks also
refuse its agent's other routes, such as a fallback model, a second provider
or a proxy, and ``tests/harness/test_harness_model_route.py`` lists them.
The checks read the config only; a tool that builds its own request can
still name another model.

Adapter steps
~~~~~~~~~~~~~

``renderer`` is a subclass of ``AdapterRenderer``. Each step is a static
method that returns its input unchanged by default and raises
``RenderError`` to refuse:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Step
     - Gets and does
   * - ``process_config``
     - one merged config file; adjusts or refuses it
   * - ``process_skill``, ``process_command``
     - one file's text; adds or checks frontmatter
   * - ``check_model_route``
     - the processed files; refuses a route no check can state
   * - ``finalize_render``
     - every file; work across files (a last resort)
   * - ``layer_args``, ``layer_files``
     - a layer's render (`A person's machine`_)

Prefer a check to a step for a rule on config values.
``reef.harness.adapters.common`` has shared helpers, such as
``with_skill_frontmatter``.

A person's machine
------------------

``reef-client harness`` asks ``GET /reef/harness?form=client&root=<install>``
and gets the layer: the files, the command, and the agent's
``register``/``unregister`` commands. The layer's render starts each config
target from nothing and skips the checks. Then:

- ``layer_args`` returns argument groups for what the agent reads from no
  file of the layer, such as Codex's config keys as ``-c``.
- ``layer_files`` adds what the arguments name, such as a plugin manifest,
  and drops files the agent cannot read from the install.
- The client runs ``register`` once per release, starts the binary with
  ``args`` and ``env``, and runs ``unregister`` on ``disconnect``.

Templates hold ``{root}``, ``{model}``, ``{base_url}``, ``{token}`` (``env``
only) and ``{file:<path>}`` (``args`` only). Every part in ``evolve`` must
take effect both in an episode and in the layer; refuse in the checks what
the layer cannot carry. An adapter without ``layer`` does not run on a
person's machine.

Connect a new agent
-------------------

.. steps::

   #. Write ``adapter.yaml``: ``evolve``, ``install``, ``dirs``,
      ``sessions``, ``model``, ``run``, and ``config`` with the paths alone
      at first.
   #. Add a ``TrajectoryReader`` with ``@register_trajectory_reader`` when the
      session log has a new format (`trajectory.py
      <../../reef/harness/episodes/trajectory.py>`__).
   #. Add the name to ``BUILTIN_ADAPTERS``, or register through the
      ``reef.harness_adapters`` entry point.
   #. Add ``render.py`` only for what the file cannot state.

Then check, in order:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Check
     - How
   * - It loads
     - ``get_adapter('<name>')``
   * - It renders what the agent reads
     - a golden tree in ``tests/harness/test_harness_render.py``; read every
       file
   * - No tree moves the model
     - its route keys in ``tests/harness/test_harness_model_route.py``
   * - The session log reads
     - a captured session in ``tests/harness/test_harness_episode.py``
   * - One real episode runs
     - ``run_episode`` against a real model: exit 0, no residue, a rule in
       ``rules.md`` followed
   * - Every part takes effect
     - a real-binary test in ``tests/smoke/``: a marker in each part reaches
       the model, as an episode and as the layer
   * - The layer works
     - ``BINDINGS`` in ``tests/harness/test_harness_sync.py``, then
       ``reef-client harness connect`` by hand

Native runner
-------------

``reef-native`` is Reef's own agent loop. It reads ``native/``:
``RULES.md``, ``skills/``, ``models.json`` (filled in by the binding), and
five parts that let a method change the loop itself. The starter tree is
``tutorials/evolve-your-harness/seeds/native/``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Part
     - Holds
   * - ``native/tools/<name>.py``
     - ``run(args, workdir) -> str``, with literal ``DESCRIPTION``,
       ``PARAMETERS`` (JSON schema) and ``CAPABILITIES`` (``read``,
       ``write``, ``exec``, ``network``)
   * - ``native/hooks/<name>.py``
     - ``listen(payload, next)`` at one ``EVENT``: ``pre_step``,
       ``pre_execute``, ``request_error`` or ``post_execute``
   * - ``native/graphs/<name>.json``
     - stages and the edges between their outcomes; ``main`` is the root
       turn's graph
   * - ``native/agents/<name>.json``
     - one agent to delegate to: ``prompt``, ``graph``, ``tools``,
       ``skills``, ``max_steps``, ``then``
   * - ``native/loops/<name>.py``
     - ``run_turn(ctx)`` in place of the ``main`` graph; always reviewed

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Graph stage
     - Does (outcomes)
   * - ``model``
     - one request, one step (``tool_calls``, ``text``)
   * - ``tools``
     - the pending calls through the hooks (``done``)
   * - ``verify``
     - checks the last text (``pass``, ``fail``)
   * - ``branch``
     - the first matching case (each case's outcome, ``else``)
   * - ``subagent``
     - delegates to an agent (``completed``, ``gave_up``, ``budget``, ``ask``)
   * - ``message``, ``compact``
     - adds a user message; summarizes old messages (``done``)
   * - ``end``
     - ends the turn as ``completed`` or ``gave_up``

``ctx`` in loop code gives ``prompt``, ``step``, ``tools``, ``messages``,
``model()``, ``run_tools()``, ``text()``, ``say(text)``, ``agent(name)``,
``end(reason)`` and ``log(event, data)``. ``max_steps`` is 1 to 32 for
graphs and loops alike.

Tools run in the loop process under the ``local`` executor. Under
``sandbox``, each call runs in a bubblewrap child that grants only the
declared capabilities. Hooks and loop code always run in the loop process.

Trajectory
~~~~~~~~~~

The loop writes ``native-jsonl``: one ``{type, seq, time, data}`` per line.
A turn is ``turn/start``, then per step ``step/start``,
``request/header``, ``assistant/message``, ``tool/call``, ``tool/result``
and ``step/end``, then ``turn/end`` with its reason (``completed``,
``gave_up``, ``max-steps``, ``error`` and others). A tool result over 20,000
characters goes to ``.reef/tool-output/``.

Episode and serve modes
~~~~~~~~~~~~~~~~~~~~~~~

``reef-native -p`` runs one turn; episodes use it. ``reef-native serve`` is
one resident process that follows the scenario's head (`Serve the harness as
a resident process
<../user-guide/evolve-your-harness.rst#serve-the-harness-as-a-resident-process>`__):

- It mounts each new release between two steps, and keeps the current tree
  when a new one fails to load.
- Its model calls carry the scenario, the token and the ``release`` tag.
- Tree code runs with the user's privileges; only tool calls are confined,
  with ``REEF_NATIVE_ENFORCE=bwrap``.
- ``--self-tools`` adds ``harness_inspect`` and ``harness_try``.
- The socket ``native/serve.sock`` takes one JSON line per connection:
  ``{"turn": {...}}``, ``{"control": "status"}`` or ``{"control": "mount",
  "release_id": ...}``.
