Harness adapters
================

An adapter connects a harness tree to one agent. Its ``tree.yaml`` declares
the parts of the tree: the tree paths a method writes, such as ``rules.md``
and ``skills/<name>/SKILL.md`` (see `The harness tree
<../user-guide/evolve-your-harness.rst#the-harness-tree>`__), and the agent's
own path for each. Its ``descriptor.yaml`` declares the rest: the config
targets, how to run the agent on the rendered files and the served model,
and how to read its trajectory.

This page is for developers. It describes the two files, the rendering rules
every adapter shares, what differs for each bundled agent, the contract of
the native runner, and how to connect a new agent.

Reef bundles six adapters for third-party coding-agent CLIs. It also bundles
``native``, Reef's own agent, and ``terminus``, which runs the Harbor
Terminus 2 agent through a Reef runner. For each adapter, the list gives the
install pin, the headless command, and the config targets. `Render paths`_
gives where the other parts land.

- ``pi`` (npm ``@earendil-works/pi-coding-agent`` 0.84.2):
  ``pi --mode json -p``. ``pi-agent/settings.json`` (``primary``),
  ``pi-agent/models.json`` (``models``).
- ``opencode`` (npm ``opencode-ai`` 1.18.18):
  ``opencode run --format json --auto``. ``opencode/opencode.json``
  (``primary``).
- ``claude`` (npm ``@anthropic-ai/claude-code`` 2.1.257):
  ``claude -p --output-format json --permission-mode auto``.
  ``claude/settings.json`` (``primary``).
- ``codex`` (npm ``@openai/codex`` 0.153.4): ``codex exec --json``.
  ``codex/config.toml`` (``primary``), ``codex/models.json`` (``models``).
- ``dsh`` (npm ``@deepseek-ai/dsh`` 0.1.2-alpha.5):
  ``dsh --profile headless``. Under ``dsh/``:
  ``profiles/headless/cordis.patch.yml`` (``primary``), ``.env``
  (``env``), ``profiles/web/cordis.patch.yml`` (``web``), and
  ``profiles/web/package.json`` (``web_manifest``).
- ``hermes`` (git ``NousResearch/hermes-agent`` at ``v2026.8.31``, 0.21.0):
  ``hermes chat -Q --oneshot -q``. ``hermes/config.yaml`` (``primary``),
  ``hermes/.env`` (``env``).
- ``terminus`` (ships with reef): ``reef-terminus --task``.
  ``terminus/config.json`` (``primary``).
- ``native`` (ships with reef): ``reef-native -p``. ``native/config.json``
  (``primary``), ``native/models.json`` (``models``).

At deployment startup, Reef installs the pinned binary for the evaluation
episodes, unless ``evolution.binary`` names one.

Tree parts
~~~~~~~~~~

The adapter decides which parts a harness tree may hold. Its ``tree.yaml``,
beside ``descriptor.yaml``, declares them under one key, ``parts``.
`reef/harness/adapters/parts.py <../../reef/harness/adapters/parts.py>`__
checks it when ``load_descriptor`` loads the descriptor beside it. This is
pi's ``tree.yaml``:

.. code-block:: yaml

   parts:
     rules:
       path: rules.md
       format: text
       renders: pi-agent/AGENTS.md
       holds: the instructions pi reads at the start of every session
     skill:
       path: skills/{name}/SKILL.md
       format: skill
       renders: pi-agent/skills/{name}/SKILL.md
       holds: an Agent Skill, which pi loads when its description fits the task
     command:
       path: commands/{name}.md
       format: command
       renders: pi-agent/prompts/{name}.md
       holds: a prompt template that a person runs as /name
     extension:
       path: extensions/{name}.ts
       format: code
       renders: pi-agent/extensions/{name}.ts
       holds: a TypeScript extension that pi loads in its process
     config:
       path: config/{name}.json
       format: config
       holds: a JSON object merged into pi-agent/settings.json (primary) or pi-agent/models.json (models)

Each entry under ``parts`` maps a part name to these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - ``path``
     - Required. The tree path that a method writes, relative to the tree
       root. ``{name}`` stands for one name, such as a skill's name: a letter
       or digit, then letters, digits, ``.``, ``_`` or ``-``. A path holds
       ``{name}`` at most once, inside one path segment.
   * - ``format``
     - Required. What Reef checks the file as, and which render step
       processes it (see `Formats`_).
   * - ``renders``
     - Where the agent reads the file, relative to the episode or install
       root. It holds ``{name}`` exactly when ``path`` does. Every part
       except the ``config`` part requires it. The ``config`` part has no
       ``renders``: its ``{name}`` names a config target of the descriptor,
       and its file merges into that target.
   * - ``holds``
     - Required. One line that says what the file holds. A method reads it
       in ``material/host.json``.
   * - ``review``
     - Optional, ``false`` by default. With ``true``, every selected change
       to the part waits for a person, whatever ``evolution.review_parts``
       names, and ``harness_try`` refuses to mount it. Native's
       ``native_loop`` part sets it.
   * - ``merge``
     - Optional, ``file`` by default. How a file that a person and Reef
       both changed since the person's installed release merges:
       ``lines`` takes each side's changed lines and refuses only lines
       both changed; ``file`` refuses. The bundled adapters that run on a
       person's machine merge their text parts (rules, skills, commands,
       memory) by ``lines``.

``load_descriptor`` refuses a ``tree.yaml`` that breaks one of these rules:

- A part name is a lowercase letter, then lowercase letters, digits or
  ``_``.
- A part has no field that the table does not list.
- A tree has at most one ``config`` part, and its path holds ``{name}``.
- No tree path is a path of two parts.

Validation refuses a tree file at a path that no part takes. Its message
names the file and lists the paths the adapter's tree takes. Other settings
and records refer to parts by name: ``evolution.review_parts``, the
``components`` of GEPA and Meta-Harness, the ``parts`` of a method's
``material/host.json``, and the ``patch_parts`` of a step's commit metrics.

Formats
^^^^^^^

A part's format decides which checks its files must pass in validation and
which render step processes them. In every format except ``config``,
validation refuses an empty file and a credential-shaped literal.

.. list-table::
   :header-rows: 1

   * - Format
     - Validation also checks
     - Render
   * - ``text``
     - nothing more
     - written to the ``renders`` path
   * - ``skill``
     - nothing more
     - written to the ``renders`` path, then given to ``process_skill``
       (see `Adapter steps`_)
   * - ``command``
     - nothing more
     - written to the ``renders`` path, then given to ``process_command``
   * - ``code``
     - no write to stdout or stderr (``console.log``,
       ``process.stdout.write``) without a ``ctx.hasUI`` guard on that line
       or the line above, because a raw write breaks a terminal session
     - written to the ``renders`` path
   * - ``config``
     - a JSON object without an inline credential, whose ``{name}`` is a
       config target of the descriptor
     - deep-merged into that config target
   * - ``native_tool``, ``native_hook``
     - the module compiles, binds ``run`` or ``listen``, and declares its
       constants as literals; ``NAME``, when set, is the file name (see
       `Tools`_ and `Hooks`_)
     - written to the ``renders`` path
   * - ``native_graph``, ``native_agent``
     - a JSON object that is a valid graph or agent, whose ``name`` is the
       file name (see `Graphs`_ and `Subagents`_)
     - written to the ``renders`` path
   * - ``native_loop``
     - the module compiles and defines ``run_turn(ctx)`` (see
       `Loop code`_)
     - written to the ``renders`` path

A part's format, not its name, chooses the render step. Codex's ``command``
part has the ``skill`` format, because a Codex command is a skill that a
person runs as ``$name``: its files go through ``process_skill``.

Render paths
^^^^^^^^^^^^

Every bundled adapter takes ``rules.md`` (``rules``),
``skills/<name>/SKILL.md`` (``skill``) and ``config/<target>.json``
(``config``). Every one except ``native`` takes ``commands/<name>.md``
(``command``). Every one except ``codex`` and ``native`` takes
``extensions/<name>.<ext>`` (``extension``), with the suffix the adapter
declares. The parts render to these paths, relative to the episode or
install root. The ``extension`` column gives the tree path's suffix, then the
render path:

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
     - ``.ts``: ``pi-agent/extensions/<name>.ts``
   * - ``opencode``
     - ``opencode/AGENTS.md``
     - ``opencode/skill/<name>/SKILL.md``
     - ``opencode/command/<name>.md``
     - ``.ts``: ``opencode/plugin/<name>.ts``
   * - ``claude``
     - ``claude/CLAUDE.md``
     - ``claude/skills/<name>/SKILL.md``
     - ``claude/commands/<name>.md``
     - ``.js``: ``claude/plugins/<name>/index.js``
   * - ``codex``
     - ``codex/AGENTS.md``
     - ``codex/skills/<name>/SKILL.md``
     - ``codex/skills/<name>/SKILL.md``, as a skill
     - none
   * - ``dsh``
     - ``dsh/AGENTS.md``
     - ``dsh/skills/<name>/SKILL.md``
     - ``dsh-agents/skills/<name>/SKILL.md``
     - ``.mjs``: ``dsh/profiles/headless/extensions/<name>.mjs``
   * - ``hermes``
     - ``hermes/SOUL.md``
     - ``hermes/skills/<name>/SKILL.md``
     - ``hermes-commands/<name>/SKILL.md``
     - ``.py``: ``hermes/plugins/<name>/__init__.py``
   * - ``terminus``
     - ``terminus/AGENTS.md``
     - ``terminus/skills/<name>/SKILL.md``
     - ``terminus-commands/<name>/SKILL.md``
     - ``.py``: ``terminus/context/<name>.py``
   * - ``native``
     - ``native/RULES.md``
     - ``native/skills/<name>/SKILL.md``
     - none
     - none

The five ``native_*`` parts of ``native`` render to their own tree paths,
such as ``native/tools/<name>.py`` (see `Native runner`_). On ``claude``, an
extension is the entry file of a Claude Code plugin; the plugin still needs a
marketplace or settings reference before Claude Code activates it.

``claude`` has one more part, ``memory``: ``memory/<name>.md`` renders to
``claude/memory/<name>.md``. Its renderer sets Claude Code's
``autoMemoryDirectory`` to ``{root}/claude/memory``, so the notes that Claude
Code writes for itself across sessions, and ``MEMORY.md``, their index, are
files of the tree. A client sends them back to Reef, and a method can read
and edit them.

Descriptor fields
~~~~~~~~~~~~~~~~~

``descriptor.yaml`` describes how Reef renders, starts, and reads an agent.
`reef/harness/adapters/descriptor.py
<../../reef/harness/adapters/descriptor.py>`__ validates it at load. The
fields are:

- ``name`` identifies the adapter. ``binary`` is the executable, and
  ``argv`` runs one headless prompt. Reef substitutes ``{prompt}`` in
  ``argv``.
- ``config`` maps each config target's name to the file's root-relative
  ``path``, its ``defaults``, and its ``checks`` (see `Config checks`_). A
  ``primary`` target is required. A file of the tree's ``config`` part
  merges into the target its name names. A config value can name a path
  under the root as ``{root}/...``. The program that writes the rendered
  files replaces ``{root}`` with the directory that holds the render roots:
  the episode root, or a client's install.
- ``trajectory`` gives the session log's ``path`` and reader ``format``.
- ``env`` points the agent's state into the episode root, substituting
  ``{root}``. It also turns off what would make an episode
  non-deterministic, such as the binary's own updater.
- ``layer`` (optional) says how a person's machine runs the agent with a
  release over the person's own configuration, which stays as it is. It
  has two fields, as ``argv`` and ``env`` do for an episode: ``args``, groups
  of arguments added to the binary's own, such as ``["--plugin-dir",
  "{root}/claude"]``, and ``env``, variables such as ``OPENCODE_CONFIG_DIR``.
  Templates hold ``{root}`` (the install), ``{model}`` and ``{base_url}``
  (the model binding's), ``{token}`` (in ``env`` only, so no command line
  shows it), and ``{file:<render path>}``, that file's text as a JSON
  string; a group whose file is empty is left out. Use the agent's own way
  to add configuration, such as a plugin directory or an extra config
  directory. Without ``layer``, the adapter does not run on a person's
  machine (see `Use a published tree
  <../user-guide/evolve-your-harness.rst#use-a-published-tree>`__).
- ``install`` pins the vendor install: ``kind`` (``npm``, or ``git`` for an
  editable virtual environment), ``package``, ``version`` (as ``--version``
  reports it), and ``binary_path`` below the install prefix. A ``git``
  install also names ``repository`` and ``ref``.
- ``model_binding`` holds config templates for each supported API dialect
  (``openai``, ``responses``, or ``anthropic``). Each template names a config
  ``target`` and its ``data``. Reef merges the matching templates after the
  tree when it renders an evaluation episode, and substitutes
  ``{base_url}``, ``{api_key}``, and ``{model}`` in their string values.
- ``host_env`` (optional) lists service variables that an episode under the
  local executor keeps, each with a default for when the service has none.
  ``{home}`` is the service's home directory, and an empty default leaves
  the variable unset. Otherwise the local executor passes only ``PATH``,
  ``SYSTEMROOT``, ``TMPDIR``, and ``CUDA_VISIBLE_DEVICES``.
- ``is_root_bind_mounted`` (optional) is ``true`` when the binary
  bind-mounts paths below the episode root into a container. On macOS the
  local executor then makes the root under ``~/.reef/episodes``, which
  Docker's VM shares, instead of the temp directory.
- ``writable_paths`` lists state directories that a hosted sandbox makes
  writable. Rendered inputs within them stay read-only.
- ``self_isolating`` (optional) is ``true`` when the adapter runs episodes in
  its own container. Reef then refuses to nest it in its jail, unless its
  ``execution_validator`` accepts the configuration.
- ``is_prompt_task_directory`` (optional) is ``true`` when ``{prompt}``
  names a Harbor task directory instead of text the model reads.
- ``cleanup_whitelist`` lists the paths the agent may write during boot or a
  run. Reef reports any other new file as residue. ``dir/**`` allows a whole
  subtree. Any other entry is a glob against the root-relative path, so name
  a single file with its full path, such as ``pi-agent/auth.json``. A bare
  directory name matches nothing below it.
- ``renderer`` (optional) names the class with the adapter's own render
  steps, ``package.module:ClassName``, the form a deployment names a method
  in (see `Adapter steps`_). Each bundled adapter keeps it in ``render.py``.
- ``execution_validator`` (optional) names an ``ExecutionValidator`` class in
  the same form. Reef builds it with no arguments, and calls it with the
  rendered tree and the executor before launch.

Config checks
^^^^^^^^^^^^^

A target's ``checks`` state what its merged config must keep or must not
set. Each check has a ``key``, a ``match`` expression that every value at
the key must match, and an optional ``reason`` that a refusal gives. Use a
check, not an adapter step, for a rule on config values: it sits beside the
defaults it protects.

.. code-block:: yaml

   config:
     primary:
       path: "claude/settings.json"
       defaults:
         includeCoAuthoredBy: false
       checks:
         - key: includeCoAuthoredBy
           match: IsDefault()
         - key: [model, fallbackModel]
           match: IsAbsent()
           reason: Reef's model binding chooses the model
         - key: env.DISABLE_TELEMETRY
           match: |-
             Not(Strip(Lowercase(IsIn(["0", "false", "off", "no", ""]))))
           reason: benchmark episodes keep telemetry off

A check reads the merged target: the defaults, the tree's
``config/<target>.json``, and the model binding when one is applied. The
render runs the checks before any adapter step, so no adapter can skip
them. A refusal names the key, and for ``Each`` the failing key or item. It
never shows the value, which can be a credential. A check without a
``reason`` gives its match expression instead.

**Keys.** ``key`` is required: a dot-separated key path, or a list of them
(one check per key). ``.`` is the whole checked value, for a check on its
top-level key names. A ``*`` segment is every key of an object and every
item of a list. A ``**`` segment is any number of segments, zero included.
A key that the checked value does not have is absent.

**Matchers.** Write a match expression as nested calls with JSON literals
(``true``, ``false``, and ``null`` included). Use ``|-`` in YAML to wrap a
long one.

.. list-table::
   :header-rows: 1

   * - Matcher
     - Matches
   * - ``IsDefault()``
     - The value the target's ``defaults`` give the key, or an absent key
       when they give none. Only for a key without ``*`` or ``**``.
   * - ``IsAbsent()``
     - A key that the checked value does not have. An empty value is set.
   * - ``Eq(v)``, ``IsIn([v, ...])``
     - A value equal to ``v``, or to one of the list, with its type:
       ``false`` matches neither ``0`` nor ``"false"``.
   * - ``IsObject()``, ``IsInt()``, ``Ge(n)``
     - An object; an integer that is not a boolean; a number at least ``n``.
   * - ``StartsWith(s)``, ``EndsWith(s)``, ``HasSubstr(s)``,
       ``MatchesRegex(s)``
     - Text with that prefix, suffix, or substring, or text that the regular
       expression matches as a whole.
   * - ``Lowercase(m)``, ``Uppercase(m)``, ``Strip(m)``
     - Text that matches ``m`` after the change.
   * - ``ParseJson(m)``
     - A string holding JSON whose value matches ``m``.
   * - ``Each(m)``, ``Contains(m)``
     - Every entry of an object or item of a list matches ``m``, or at least
       one does. ``Each`` also matches an absent key.
   * - ``Key(m)``, ``Value(m)``
     - An object entry whose key, or whose value, matches ``m``. Use them
       inside ``Each`` and ``Contains``.
   * - ``AllOf(m, ...)``, ``AnyOf(m, ...)``, ``Not(m)``
     - Every matcher matches, at least one does, or ``m`` does not.

Text matchers read a string as it is, and a number, a boolean, or null as
its JSON text, so ``Lowercase(Eq("0"))`` matches ``0``. They match no
object, list, or absent key. Only ``IsAbsent``, ``Not``, and ``Each`` match
an absent key, so "if it is set, it must be X" is ``AnyOf(IsAbsent(), X)``.
If no matcher fits, contribute one to Reef instead of writing an adapter
step.

Rendering
~~~~~~~~~

Rendering turns a harness tree into the files one agent reads
(``reef.harness.tree.render_tree``). The input is a tree, a descriptor, and
optional config overlays such as a model binding. The output is
root-relative paths with their text; the render writes no disk and starts no
agent. Reef renders a tree to run an episode, to serve a release, and to
validate a candidate or a proposed patch. A release stores the source tree,
and ``GET /reef/harness`` renders it for each request.

Shared rules
^^^^^^^^^^^^

These rules apply to every adapter, including one registered through the
``reef.harness_adapters`` entry-point group. No adapter replaces them.

- Each config target is a JSON object: the descriptor's ``defaults``, then
  the tree's file of the ``config`` part for that target deep-merged over
  them, then the overlays. The merge recurses into objects and replaces
  every other value, lists included. Every target is written even when the
  tree has no file for it, so the defaults always reach the agent.
- Every other file renders to its part's ``renders`` path, with the name
  from its tree path as ``{name}``: ``skills/review/SKILL.md`` is the skill
  ``review``. A part without ``{name}``, such as ``rules``, renders one file
  when the tree holds it, and none otherwise.
- Each config file is written in the format its suffix names: ``.toml`` as
  TOML, ``.yaml`` or ``.yml`` as YAML, ``.env`` as ``KEY=value`` lines, and
  any other suffix as JSON.

The render refuses a tree, before any agent runs, when:

- a path is one that no part of the adapter's tree takes, such as
  ``extensions/notes.ts`` under ``native``;
- a ``config/<target>.json`` names an undeclared target or does not hold a
  JSON object;
- two files render to the same path;
- a merged config target fails one of its `Config checks`_;
- the tree sets a key that the binding writes (see `Model binding checks`_).

Validation (``reef.harness.tree.validate_tree``) runs before an episode. It
adds the checks of each file's format (see `Formats`_), and a tree that
passes them also renders. A ``native_graph`` or ``native_agent`` must also
not name a tool, skill, agent, or graph the tree does not hold, agents must
not delegate in a cycle, and a tree holds at most one ``native_loop``.

Model binding checks
^^^^^^^^^^^^^^^^^^^^

Every model call must go to Reef. Reef's model binding writes the endpoint,
the model, and the credential after the tree, so it wins every key it
writes. The served tree is rendered without a binding, so it names no
endpoint and holds no credential.

- The render refuses a tree that sets a key the binding writes, unless the
  binding's credential (the value holding ``{api_key}``) is set beside it.
  Validation refuses an inline credential, so only the binding can hold
  one. No adapter can turn this check off.
- Each adapter also refuses the other routes its agent reads, such as a
  fallback model, a second provider, a proxy, or a model in a request body.
  The agent sections below list them.

These checks read the rendered config only. They do not authorize requests:
the proxy forwards ``model`` and ``models`` as sent, so a tool or a plugin
that builds its own request can still name another model.

Adapter steps
^^^^^^^^^^^^^

The ``renderer`` is a subclass of ``AdapterRenderer``
(``reef.harness.adapters.descriptor``). After the shared checks, the render
calls its steps in this order:

#. ``process_config`` gets one merged config file as a JSON object. It
   adjusts the config, or refuses one that a check cannot describe.
#. ``process_skill`` and ``process_command`` get the text of one file of a
   part whose format is ``skill`` or ``command``, to add or check
   frontmatter.
#. ``check_model_route`` reads the processed config, skill, and command
   files, and refuses a route that no check can describe, such as a model
   in frontmatter.
#. ``finalize_render`` gets every file as text, each config file already
   written in its format. It is for work that spans several files, or for a
   layout that the suffix's format cannot produce.
#. ``layer_files`` runs only for a layer (``render_tree`` with ``layer``),
   which starts each config target from nothing instead of the episode
   defaults and skips the targets' checks. It adds what the layer's
   ``args`` name and the tree may lack: Claude's plugin manifest, an empty
   rules file, or pi's extension that registers the release's models.

Each step defaults to returning its input unchanged, and refuses a tree by
raising ``RenderError``. Every step must be a static method; an ordinary
method fails when the class is created. No step can pass state to a later
one, so keep module-level state out of the steps too. Prefer the narrow
steps: ``finalize_render`` can rewrite every file, which makes its effect
hard to review. Shared helpers are in ``reef.harness.adapters.common``; for
example, ``with_skill_frontmatter(path, text)`` adds the ``name`` and
``description`` that most agents need to list a skill.

A step sees paths, text, and config objects only, not the tree file that
produced them. Restrictions that depend on how an episode runs are checked
at launch by the ``execution_validator`` instead.

Run a rendered tree
^^^^^^^^^^^^^^^^^^^

``reef-client harness`` runs an agent on a published tree for an adapter
with a ``layer``. To do it yourself, write the files of
``GET /reef/harness?form=client&service_url=<reef>&root=<directory>`` under
``<directory>``, and replace ``REEF_TOKEN_PLACEHOLDER`` with your token in
the files and in ``run.env``. Then run ``run.binary`` with ``run.args`` and
``run.env``. Those files bind the model to
``<reef>/reef/scenarios/<scenario>``, which names the scenario in the path,
so the agent needs no custom header.

A client that renders the source tree (``GET /reef/harness?form=source``)
binds the model in the same render:

.. code:: python

   from reef.harness.adapters import get_adapter
   from reef.harness.episodes.model_binding import ModelBinding
   from reef.harness.tree import HarnessTree, render_tree

   descriptor = get_adapter("pi")
   binding = ModelBinding(base_url="http://127.0.0.1:8901", model="qwen3-8b", api_key=token)
   files = render_tree(HarnessTree(source_files), descriptor, binding.config_overlays(descriptor))

Point ``base_url`` at ``<reef>/reef/scenarios/<scenario>``, or send the
``x-reef-scenario`` header: ``/v1/*`` answers only a call that names its
scenario. A tree must not set provider headers, so a client that sends the
header adds it to its own binding after the render. pi sends ``providers.reef.headers``
from ``pi-agent/models.json``, as ``recipes/beta/reefine/examples/run.py``
writes them. For an agent that sends no custom header, point ``base_url``
at a local proxy that adds it, such as
``reef.harness.capture.CaptureProxy``, which also keeps the receipts that a
report references. See `Use a published tree
<../user-guide/evolve-your-harness.rst#use-a-published-tree>`__ for the
workflow.

pi
~~

``PI_CODING_AGENT_DIR`` relocates pi's whole configuration.
``PI_OFFLINE`` and ``PI_SKIP_VERSION_CHECK`` keep startup off the network.

- A skill without ``name`` and ``description`` frontmatter gets both: the
  directory name and the first line of the text. pi does not list a skill
  without them.
- pi rewrites some old ``settings.json`` keys when it loads them. The render
  writes the new form, so the installed file is the one pi keeps.
- Routes refused: a provider in ``models.json`` other than ``reef``, fields
  on ``reef`` that the binding does not write, ``enabledModels``, and
  ``httpProxy``.

opencode
~~~~~~~~

The defaults keep autoupdate and sharing off, allow every permission, and
set ``enabled_providers`` to ``["reef"]``. Rendering refuses a change to
those update, sharing, and provider-list settings.

- Skill and command frontmatter must be a YAML mapping between an opening
  and a closing ``---`` line, without a byte order mark, YAML tags, or
  another format such as JSON. Quote a string that holds a colon and a
  space, or numeric-looking text such as ``"1e5"``; ``yes`` is a string. A
  skill without frontmatter gets ``name`` and ``description`` from its
  directory name and first line. Your own frontmatter must have both.
- Command fields: ``name`` (optional) must match the file name;
  ``description`` and ``variant`` are strings; ``subtask`` is a boolean;
  ``agent`` names an enabled tree agent or a built-in one (``build``,
  ``plan``, ``general``, ``explore``, ``title``, ``summary``,
  ``compaction``).
- Tree agents are under ``agent`` (or the older ``mode``). ``disable`` and
  ``hidden`` are booleans, ``mode`` is ``subagent``, ``primary``, or
  ``all``, and an agent must not set a different ``name``.
  ``default_agent`` must name an enabled agent that is neither hidden nor a
  subagent; without it, keep at least one such agent.
- Routes refused: ``provider``, ``model``, ``disabled_providers``,
  ``small_model``, and a ``model`` in an agent, a command, or frontmatter.

Claude Code
~~~~~~~~~~~

``CLAUDE_CONFIG_DIR`` relocates the configuration, and the episode ``env``
turns off the updater, telemetry, and non-essential traffic. Claude Code
applies the ``env`` block of ``settings.json`` over the environment, so the
checks refuse a tree that sets the updater variables there or turns
telemetry or non-essential traffic back on.

Routes refused:

- in the ``env`` block: ``ANTHROPIC_`` variables other than the binding's,
  cloud-provider switches and credentials (Bedrock, Vertex, Foundry, and
  others), proxies, endpoints, sockets, and model names, matched without
  case;
- ``model``, ``fallbackModel``, ``availableModels``, ``modelOverrides``,
  ``advisorModel``, and related model keys;
- credential helpers (``apiKeyHelper``, ``awsAuthRefresh``,
  ``awsCredentialExport``, ``gcpAuthRefresh``, ``proxyAuthHelper``) and the
  login method;
- ``model`` in command or skill frontmatter, or an unreadable frontmatter
  block that holds the word ``model`` or an escape;
- a ``CLAUDE_CODE_EXTRA_BODY`` that is not a JSON object, or that holds
  ``model`` or ``models``.

The binding supports only the ``anthropic`` dialect.

Codex CLI
~~~~~~~~~

``CODEX_HOME`` relocates the user home. The defaults turn off the update
check, analytics, feedback, telemetry, hooks, plugins, and apps, and give
the ``workspace-write`` sandbox no network. Rendering refuses a change to
them, and ``config.toml`` admits only a fixed set of top-level keys.

- A command is a skill that a person types as ``$name``: the ``command``
  part has the ``skill`` format, so it gets the same frontmatter as a skill.
  A skill and a command with one name render to one path, so Reef refuses
  them.
- Codex takes no ``extension`` part, because Codex hooks run outside its
  command sandbox. Validation refuses a file under ``extensions/``.
- The tree may set ``web_search`` (``disabled``, ``cached``, ``indexed``,
  or ``live``) but not ``approval_policy``. The episode ``argv`` pins
  ``approval_policy="never"`` and ``web_search="disabled"``. A person's
  session keeps Codex's ``on-request`` approvals.
- Routes refused: a ``model_provider`` other than ``reef``, and fields on
  it that the binding does not write. The binding supports only the
  ``responses`` dialect.

**Model metadata.** Before it routes Codex calls, Reef reads the model's
``context_length`` and ``supported_parameters`` from the upstream
``GET /v1/models`` (the OpenRouter shape), with a five-second timeout, and
caches the result in the service process. For another endpoint, set
``evolution.model_metadata`` (see `Configuration
<../reference/configuration.rst>`__); it takes precedence over discovery.
Without metadata, Codex keeps its own fallback; Reef does not invent a
context window. The binding renders ``codex/models.json`` from the pinned
CLI's bundled catalog with the metadata applied. A tree may set the same
capability fields in ``config/models.json``, but not another catalog path.
When you change the install pin, update ``bundled_models.json`` and
``default_instructions.md`` in the adapter package to match.

DeepSeek Harness
~~~~~~~~~~~~~~~~

``DSH_HOME`` relocates the agent's home. dsh reads a user patch layer: a
YAML list of entries addressed by plugin id. The ``primary`` target is an
object keyed by plugin id, such as ``{"agent-loop": {"config": {...}}}`` or
``{"disabled": true}``, and the renderer writes it as that list. A string that
starts with ``!!js `` becomes a JavaScript expression.

- The defaults keep the session log uncompressed and turn off telemetry and
  the title call. Rendering refuses a change to them.
- A command is a skill under ``DSH_AGENTS_HOME`` that a person runs as
  ``/name``. The adapter always sets ``disable-model-invocation: true``, so
  the model never runs a command. dsh lists the tree's commands only when
  ``DSH_AGENTS_HOME`` names ``dsh-agents/``, as the episode ``env`` does.
- Frontmatter is read as YAML 1.2, as dsh reads it, so ``Yes`` and ``1:30``
  are strings. A missing or invalid ``name`` or ``description`` is filled
  in. Frontmatter that does not parse, holds a tag other than ``!`` or
  ``!!str``, or is not a mapping is refused.
- The ``web`` target is the patch layer of the profile that ``dsh web``
  boots for a person's session, with the same defaults, checks, and binding
  as the headless one. ``web_manifest`` keeps ``patchReload: startup``,
  because with ``live`` the web profile exits at start. Both profiles load
  an extension from its one rendered path.
- Routes refused: ``llm-pi-ai`` routes other than ``reef`` and fields on it
  that the binding does not write; ``agent-default-model``, ``llm-deepseek``,
  and web-search endpoint or model overrides; provider or model choices for
  the title call, subagents, declared agents, or compaction; a patch entry
  that names another package; and ``!!js`` values in the routing plugins,
  which rendering cannot inspect.

Hermes Agent
~~~~~~~~~~~~

``HERMES_HOME`` relocates the configuration. The defaults keep an episode to
its own model calls and files, and rendering refuses a change to them:
``security.tirith_enabled`` (scanner download),
``auxiliary.title_generation.enabled`` (title call), ``memory.nudge_interval`` and
``skills.creation_nudge_interval`` (background reviews that call the model
and write skills), ``curator.enabled`` (archives the release's skills), and
``sessions.write_json_snapshots`` (the snapshot the trajectory reader
reads).

- ``rules`` becomes ``SOUL.md`` after Hermes's default identity, because
  rules written alone would replace that identity.
- The renderer writes ``.no-bundled-skills``, so episodes use only the tree's
  skills.
- A command is a skill under ``hermes-commands``, which the renderer adds to
  ``skills.external_dirs``. Hermes runs it as ``/name``.
- An extension (``extensions/<name>.py``) is a plugin that defines
  ``register(ctx)``. The renderer writes its manifest, its ``plugins.enabled``
  entry, and its ``tools.override`` permission, because Hermes loads no
  plugin without that consent.
- The tree's own entries in ``skills.external_dirs``, ``plugins.enabled``,
  and a plugin's ``granted_capabilities`` stay ahead of the added ones. Any
  value there other than a list of strings is refused, except one string
  in ``skills.external_dirs``.
- Hermes writes some files into ``skills/`` that no key turns off. In a
  person's session they land in the rendered tree.
- Routes refused: ``providers``, ``custom_providers``, ``fallback_model``,
  ``fallback_providers``, ``model_aliases``, the ``moa`` presets,
  ``auxiliary.openrouter_model``, top-level ``provider``, ``base_url``, and
  ``api_base``; endpoint, model, and credential fields under ``model``;
  ``model.api_mode`` and ``model.openai_runtime``, which can bypass the
  custom provider; provider, endpoint, credential, model,
  ``fallback_chain``, or ``prefer_fast_model`` settings for auxiliary
  tasks, delegation, cron, or the curator (an auxiliary provider may stay
  ``auto`` or ``main``); and ``model`` or ``models`` in an ``extra_body``
  or ``delegation.request_overrides``.

The binding supports only the ``openai`` dialect. Hermes runs tools without
prompting and returns a tool error for commands it considers dangerous; the
adapter uses no bypass flag.

Terminus 2
~~~~~~~~~~

Terminus 2 is a Harbor agent class, not a CLI. ``reef-terminus`` reads the
tree from ``REEF_TERMINUS_DIR``, configures Harbor's ``terminus-2`` agent
through ``reef-eval``, and runs the one Harbor task (a directory or a
registry id) that ``--task`` names. It writes the verifier reward and the
ATIF trajectory under ``REEF_TERMINUS_SESSION_DIR``.

- ``config`` supplies constructor arguments; unknown keys are refused.
  ``rules`` supplies an ``extra_instruction_paths`` entry. ``skill`` and
  ``command`` are two ``AgentConfig.skills`` roots, and Harbor does not mark
  commands as commands.
- One extension file, ``extensions/<name>.py``, may define
  ``Agent(Terminus2)``. Rendering checks its syntax without running it.
- The binding keeps the served name in ``model_name`` and sets
  ``llm_kwargs.custom_llm_provider`` to ``litellm_proxy``, so litellm sends
  the name unchanged, whatever its vendor prefix, and passes
  ``reasoning_effort``, the thinking budget, and ``llm_call_kwargs`` in the
  body. Harbor looks up the context limit under ``model_name``; a name that
  litellm does not list gets 1,000,000 tokens.
- Routes refused: ``llm_kwargs`` fields that the binding does not write;
  ``llm_call_kwargs`` arguments that select an endpoint, provider,
  credential, model, fallback, or logging callback (such as ``base_url``,
  ``model``, and ``fallbacks``); and ``model`` or ``models`` in
  ``llm_call_kwargs.extra_body``.

To run a published tree yourself, save the ``files`` of ``GET /reef/harness``
under a tree root. Evaluation episodes call the upstream directly, but your
own run through Reef needs the scenario header, or Reef answers HTTP 400.
Write ``terminus/config.json`` with the service's ``REEF_TOKEN`` (any text
when the service has no token), then run one task:

.. code:: json

   {
     "model_name": "<served model>",
     "api_base": "http://127.0.0.1:8901/v1",
     "llm_kwargs": {
       "api_key": "<REEF_TOKEN>",
       "custom_llm_provider": "litellm_proxy",
       "extra_headers": {"x-reef-scenario": "<scenario>"}
     }
   }

.. code:: bash

   TREE_ROOT="$HOME/reef-harness/terminus"
   REEF_TERMINUS_DIR="$TREE_ROOT" \
     REEF_TERMINUS_SESSION_DIR="$TREE_ROOT/terminus/sessions" \
     REEF_TERMINUS_TRIALS_DIR="$HOME/reef-trials" \
     reef-terminus --task /path/to/harbor/task

The trial record under ``$TREE_ROOT/terminus/sessions`` holds the rewards
and the trajectory. Docker must share the trials directory with the host;
with colima on macOS, use a path under your home directory. A trial with
neither reward nor verifier output is reported as a possible mount problem.

Docker or a remote environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Episodes run the Harbor task in local Docker by default. The episode keeps
the service's ``DOCKER_HOST``, ``DOCKER_CONTEXT``, and ``DOCKER_CONFIG``
(default ``~/.docker``), because ``HOME`` points into the episode. On macOS
the episode root is under ``~/.reef/episodes``. Elsewhere it is in the temp
directory; if Docker's VM does not share it, set ``TMPDIR`` for the service.

To run tasks in a remote Harbor environment, use the sandbox executor and
pass the switch and the credentials into it. For E2B:

.. code:: yaml

   evolution:
     executor: sandbox
     sandbox:
       egress_hosts: [api.e2b.dev]
       env_from: [REEF_TERMINUS_ENVIRONMENT, E2B_API_KEY]

Start the service with ``REEF_TERMINUS_ENVIRONMENT=e2b`` and
``E2B_API_KEY``. For Modal, allow ``api.modal.com``, pass
``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, and set
``REEF_TERMINUS_ENVIRONMENT=modal``. The runtime needs Linux, bubblewrap,
Python 3.12+, and Harbor with the environment's extra (``harbor[e2b]``, or
the ``modal`` package). The interpreter and local task directories must be
visible inside the sandbox, for example under ``/opt``. ``egress_hosts``
does not yet enforce a hostname firewall.

Reef refuses before launch: local Docker under the sandbox executor, a
remote environment without ``egress_hosts`` or its credentials in
``env_from``, and an ``extension`` file under any executor other than
``sandbox``. Such a tree still renders; the refusal depends on how the
episode runs, not on what the tree holds.

Native runner
~~~~~~~~~~~~~

``reef-native`` is Reef's own agent loop. It reads ``native/`` from the
rendered tree: ``RULES.md``, ``skills/``, ``models.json``, and the five
native parts below. Native's ``tree.yaml`` has no ``command`` or
``extension`` part, because the loop does not read them; validation refuses
a file under ``commands/`` or ``extensions/`` as not a path of the native
harness tree. Its binding fields in ``models.json`` (``api``, ``base_url``,
``api_key``, and ``model``) come from the model binding. The starter tree,
``tutorials/evolve-your-harness/seeds/native/``, holds four tools, the hook
``loop_guard``, and a ``main`` graph.

Tools
^^^^^

A ``native_tool`` file defines ``run(args, workdir) -> str`` and assigns
literal ``DESCRIPTION``, ``PARAMETERS`` (a JSON schema of the arguments),
and optional ``CAPABILITIES`` (distinct names from ``read``, ``write``,
``exec``, and ``network``) and ``NAME`` (the file name). The loop reads
the last module-scope assignment of each constant from the source. It
imports the module only when a call runs, so top-level code never runs at
load.

- Validation refuses a tool that does not compile, whose ``NAME`` is not
  the file name, or whose name a self tool reserves.
- The loop ends with ``LOAD_ERROR`` before the first model call if it
  cannot parse a tool, find a top-level ``run``, or read a constant as a
  literal.
- A call fails with ``TOOL_FAILED``, and the episode continues, if the
  module's top level raises (``SystemExit`` included) or ``run`` cannot be
  called.

The executor chooses how tools are confined; a tree cannot change it:

- Under the ``local`` executor, tools run in the loop process without
  confinement. ``CAPABILITIES`` is only recorded.
- The ``sandbox`` executor sets ``REEF_NATIVE_ENFORCE=bwrap``. Each call,
  import included, runs in a child process under a bubblewrap profile.
  Without ``network`` it has no network; without ``write`` the workspace is
  read-only; without ``exec``, ``/bin`` and ``/usr/bin`` are absent and
  ``PATH`` is unset. ``read`` is never withheld. The ``exec`` rule stops
  ordinary shell commands but is not a complete ban: a tool can still start
  ``sys.executable`` or a program in a path it can write.
- If ``bwrap`` is required but absent, the loop refuses to start. A call
  that the jail cannot run ends with ``SANDBOX_FAILED``, which is not a
  tool error.

Each ``tool/result`` records ``enforcement``: ``mode`` (``none`` or
``bwrap``) and ``denied``, the undeclared capabilities.

Hooks
^^^^^

A ``native_hook`` file defines ``listen(payload, next) -> decision`` and
assigns ``EVENT`` and optional ``NAME``. Hooks at one event run in
file-name order. A hook may call ``next()`` for the next layer's decision
and return it, changed or not, or return its own decision. The last layer
is the loop's default, and ``next`` runs at most once.

- ``pre_step`` (``{step, task, messages}``): ``enter`` may add user
  messages; ``reject`` ends the turn.
- ``pre_execute`` (``{step, call_id, name, arguments, capabilities}``):
  ``allow`` may replace ``arguments``, which are validated again; ``deny``
  returns ``HOOK_DENIED`` to the model; ``ask`` returns
  ``APPROVAL_REQUIRED`` in a headless run.
- ``request_error`` (``{step, attempt, error}``): ``retry`` with
  ``delay_ms``, or ``fail``. The loop allows at most 4 attempts per step
  and 10 seconds between attempts, whatever the hook asks.
- ``post_execute`` (``{step, call_id, name, arguments, result}``):
  ``accept`` may replace the content the model reads; ``block`` returns
  ``HOOK_BLOCKED`` with ``feedback`` but does not undo side effects. Both
  may return ``contexts``, added as user messages after the tool results.

A hook that raises or returns a value the log cannot store is skipped. A
module that cannot import, has no ``listen``, or names an unknown event
ends the episode with ``LOAD_ERROR``.

Hooks run in the loop process under both executors: the episode jail under
``sandbox``, which can reach the model endpoint and write the workspace,
and the host under ``local``. To require review before publishing hook
changes, add ``native_hook`` to ``evolution.review_parts``; under
``local``, add ``native_tool`` too.

Graphs
^^^^^^

A ``native_graph`` file holds named stages and edges keyed by each stage's
outcome. ``main`` is the root turn's graph, and an agent may name another.
Without ``main``, the loop uses a built-in ``seed`` graph: ask the model,
run its tool calls, and end.

Validation refuses unknown stage kinds or keys, an outcome without exactly
one edge, a stage unreachable from ``start`` or without a path to an end, a
cycle without a model stage, and a ``tools`` allow list that names a
missing tool. ``max_steps`` is 1 to 32, so the step budget bounds every
run.

The stage kinds and their outcomes are:

- ``model`` (``tool_calls``, ``text``) makes one request, of at most 4,096
  output tokens, fires ``pre_step`` and ``request_error``, and uses one
  step.
- ``tools`` (``done``) runs the pending calls through the hooks. Optional
  ``allow`` restricts them by name.
- ``verify`` (``pass``, ``fail``) checks the last text with
  ``last_line_integer``, ``last_line_matches`` (with ``pattern``), or
  ``nonempty``, and may add a ``message`` on failure.
- ``message`` (``done``) appends ``text`` as a user message.
- ``branch`` (each case's outcome, or ``else``) tests up to eight
  ``{when, value, outcome}`` cases: ``steps_used_at_least``,
  ``tool_errors_at_least``, or ``last_text_matches``. The first match wins.
- ``subagent`` (``completed``, ``gave_up``, ``budget``, ``ask``) sends the
  last text, or the task, to the agent named by ``agent`` and its ``then``
  chain. The answer returns as a user message.
- ``compact`` (``done``) summarizes old messages with one model call, not
  a step, after they pass ``fire_ratio`` of the context window, and keeps
  the last ``keep_ratio``; ``0 < keep_ratio < fire_ratio <= 1``. The window
  is ``context_window`` in ``config/models.json``, 32,768 tokens by
  default.
- ``end`` ends the turn as ``completed`` or ``gave_up``.

Patterns are at most 200 characters and time out after one second, which
fails the case or check. A branch searches only the last 4,096 characters.
A model stage entered after ``max_steps`` ends the turn with
``max-steps``. A graph that cannot load ends with ``LOAD_ERROR``, and a run
over ``(max_steps + 1) * 16`` transitions ends with ``GRAPH_ERROR``.

Subagents
^^^^^^^^^

A ``native_agent`` file defines one agent to delegate to: ``prompt`` (added
after the rules and skills), ``graph`` (``seed`` by default), the
``tools`` and ``skills`` it may use (all by default), ``max_steps``,
``max_tool_calls``, and ``then`` (the agents that receive its final text,
in order). Validation refuses unknown names and delegation cycles. An agent
spends the parent's remaining step budget and logs under
``sessions/agents/``. An ``ask`` from ``pre_execute`` during an agent turn
returns ``ask`` to the parent graph.

Loop code
^^^^^^^^^

A ``native_loop`` file defines ``run_turn(ctx)`` and may assign the
literals ``NAME`` and ``MAX_STEPS`` (1 to 32, default 12). The root turn
runs it instead of the ``main`` graph; agents still run their graphs. A
tree holds at most one loop. Validation reads the code without running it:
it must compile, hold no credential, and bind ``run_turn`` to a plain
top-level ``def`` with a parameter.

``ctx`` is the loop's only access to the run:

- ``ctx.prompt``, ``ctx.step``, ``ctx.max_steps``, ``ctx.tools``, and
  copies of ``ctx.messages`` and ``ctx.last``;
- ``ctx.model()``, one model step that returns ``tool_calls`` or ``text``;
- ``ctx.run_tools(allow=None)``, the last message's tool calls through the
  hooks;
- ``ctx.text()``, the last assistant text;
- ``ctx.say(text)``, a user message;
- ``ctx.agent(name, text=None)``, one agent without its ``then`` chain,
  which returns ``(outcome, text)``;
- ``ctx.end(reason="completed")``, with ``completed`` or ``gave_up``;
- ``ctx.log(event, data)``, a ``loop/<event>`` line, with ``data``
  truncated beyond 4,096 serialized characters.

Returning from ``run_turn`` completes the turn. The first end is final:
after it, every ``ctx`` call raises the same end and writes nothing, even if
the code catches it. More than ``(max_steps + 1) * 16`` calls to
``model``, ``run_tools``, ``agent``, ``say``, and ``log``, or an exception
from ``run_turn`` (``SystemExit`` included), ends the turn with
``LOOP_ERROR`` and exit status 1, and evaluation counts the episode as
unable to run. Code that never calls ``ctx`` is bounded only by the
episode wall clock.

Loop code runs with the loop process's privileges. Native's ``tree.yaml``
therefore marks the ``native_loop`` part ``review: true``: a change that
creates, changes, or removes a loop always waits for review, whatever
``evolution.review_parts`` says, and ``harness_try`` refuses to mount it.

Trajectory
^^^^^^^^^^

The loop writes ``native-jsonl``: one ``{type, seq, time, data}`` object per
line, with ``seq`` contiguous from zero. The ``session`` header names the
task, model, tools with capabilities, hooks, enforcement mode, agents,
``graph``, and ``loop``.

A turn is ``turn/start``; per step ``step/start``, ``request/header``,
``assistant/message``, ``tool/call`` and ``tool/result`` as needed, and
``step/end``; then ``turn/end``. ``request/header`` records the system
prompt and tool declarations on the first step and whenever they change.

- ``tool/result`` error codes: ``UNKNOWN_TOOL``, ``INVALID_ARGS``,
  ``TOOL_FAILED``, ``SANDBOX_FAILED``, ``HOOK_DENIED``,
  ``APPROVAL_REQUIRED``, and ``HOOK_BLOCKED``.
- ``turn/end`` reasons: ``completed``, ``gave_up``, ``max-steps``,
  ``max-tool-calls``, ``rejected``, ``ask``, ``turn-timeout``, and
  ``error``, with codes such as ``MODEL_ERROR``, ``LOAD_ERROR``,
  ``GRAPH_ERROR``, ``LOOP_ERROR``, and ``TURN_ERROR``.
- A graph writes ``stage/enter`` and ``stage/exit``; a loop writes
  ``loop/enter`` and ``loop/exit``. ``context/compacted``,
  ``request/error``, ``hook/decision``, and ``hook/error`` record the rest.
  Injected text is a ``user/message`` with ``source.kind`` ``hook``,
  ``stage``, ``loop``, or ``agent``.

A tool result over 20,000 characters is saved under ``.reef/tool-output/``
in the workspace and named in ``tool/result.meta.output_file``. The model
reads the head, a line naming the file, and the last 2,000 characters.

Episode and serve modes
^^^^^^^^^^^^^^^^^^^^^^^

``reef-native -p`` runs one turn in one process; evaluation episodes use it.
``reef-native serve`` keeps one resident process that follows the
scenario's served release across sessions and turns; `Serve the harness as
a resident process
<../user-guide/evolve-your-harness.rst#serve-the-harness-as-a-resident-process>`__
describes its commands. Both read the same files with the same loader. In
serve mode:

- Each mount renders a source tree under ``native/mounts/``, between two
  steps or at once when idle. A tree that fails to render or load leaves
  the current one in place. A mount imports every hook again, so
  module-level hook state starts again.
- Model calls go through an in-process capture proxy, so they carry the
  scenario, the token, and the ``release`` tag (and ``trial`` during a
  ``harness_try``).
- Tree code runs on the host with the user's privileges; only tool calls
  are confined, and only when ``REEF_NATIVE_ENFORCE=bwrap`` is set. Under
  ``--follow head``, whoever can publish to the scenario runs code there.
- ``--self-tools`` adds ``harness_inspect`` and ``harness_try``, which run
  in the process whatever the enforcer. A tree file that takes one of their
  names fails validation and fails to mount. Episodes never have them.
- A turn over its wall-clock timeout ends with ``turn-timeout``. Code that
  never calls ``ctx`` holds the turn.

Serve mode logs ``harness/mount`` (``release_id``, ``parent_release_id``,
``source`` of ``boot``, ``release``, ``try``, or ``rollback``, and
``files``), ``harness/mount-failed``, ``harness/unmount``,
``release/available`` (under ``--follow pinned``), and
``release/poll-failed``. They go to the open turn's session, or to
``native/sessions/serve.jsonl`` when no turn is open.

The Unix socket is ``native/serve.sock``, or a path under ``/tmp`` when that
one exceeds 100 bytes. Each connection sends one UTF-8 JSON line, and turns
run one at a time:

- ``{"turn": {"prompt": str, "session": str | null, "workdir": str}}``
  streams the turn's events, then ``turn/result`` with ``exit``,
  ``session``, ``turn``, ``text``, and ``receipts`` (a trial's calls are
  left out).
- ``{"control": "status"}`` returns ``control/result`` with
  ``release_id``, ``parent_release_id``, ``follow``, ``files``,
  ``pending_mount``, ``sessions``, ``socket``, and ``self_tools``.
- ``{"control": "mount", "release_id": str}`` returns ``control/result``
  with ``mounted``, ``release_id``, and ``error``.
- Malformed input returns an ``error`` object with ``message``.

Connect a new agent
~~~~~~~~~~~~~~~~~~~

To connect an agent that has no adapter yet, write a ``tree.yaml`` (see
`Tree parts`_) and a ``descriptor.yaml`` (see `Descriptor fields`_):

.. steps::

   #. The files it reads rules, skills, commands and extensions from become
      parts in ``tree.yaml``: each part's tree path, format, render path,
      and what it holds.
   #. The file it reads configuration from becomes a ``config`` target in
      ``descriptor.yaml``, with defaults and checks that keep an episode
      self-contained and the model on Reef.
   #. The command line that runs one prompt headless becomes ``binary`` and
      ``argv``, and ``env`` relocates its state into the episode root.
   #. The path and format of its session log become ``trajectory``. A new
      format subclasses ``TrajectoryReader``
      (`reef/harness/episodes/trajectory.py
      <../../reef/harness/episodes/trajectory.py>`__) and registers with
      ``@register_trajectory_reader``.
   #. The files its first boot creates go in ``cleanup_whitelist``, so a
      fresh episode root counts as clean.
   #. ``model_binding`` writes the endpoint, model, and credential for each
      dialect the agent supports.

Add a ``renderer`` only for what the declaration cannot state. The
bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ are complete references. External adapters
register through the ``reef.harness_adapters`` entry-point group, as an
``AdapterDescriptor`` or a function that returns one; ``load_descriptor``
builds one from a ``descriptor.yaml`` and the ``tree.yaml`` beside it.
