Harness adapters
================

An adapter tells Reef where each file of a harness tree lands for one agent,
how to run that agent on the rendered files and the served model, and how to
read its trajectory. The tree holds layout paths such as ``rules.md`` and
``skills/<name>/SKILL.md`` (see `The harness tree
<../user-guide/evolve-your-harness.rst#the-harness-tree>`__); the adapter
chooses the agent's own paths.

Reef includes six adapters for third-party coding-agent CLIs. It also includes
``native``, Reef's own agent, and ``terminus``, which runs the Harbor
Terminus 2 agent through a Reef runner. With ``native``, the tree can change
the agent's tools (``native_tool``) and its responses to loop events
(``native_hook``).

.. list-table::
   :header-rows: 1

   * - Adapter
     - Config targets
     - Install pin
   * - ``pi``
     - ``primary`` -> ``pi-agent/settings.json``, ``models`` ->
       ``pi-agent/models.json``
     - npm ``@earendil-works/pi-coding-agent`` 0.84.2
   * - ``opencode``
     - ``primary`` -> ``opencode/opencode.json``
     - npm ``opencode-ai`` 1.18.18
   * - ``claude``
     - ``primary`` -> ``claude/settings.json``
     - npm ``@anthropic-ai/claude-code`` 2.1.257
   * - ``codex``
     - ``primary`` -> ``codex/config.toml``, ``models`` ->
       ``codex/models.json``
     - npm ``@openai/codex`` 0.153.4
   * - ``dsh``
     - ``primary`` -> ``dsh/profiles/headless/cordis.patch.yml``, ``env`` ->
       ``dsh/.env``, ``web`` -> ``dsh/profiles/web/cordis.patch.yml``,
       ``web_manifest`` -> ``dsh/profiles/web/package.json``
     - npm ``@deepseek-ai/dsh`` 0.1.2-alpha.5
   * - ``hermes``
     - ``primary`` -> ``hermes/config.yaml``, ``env`` -> ``hermes/.env``
     - git ``NousResearch/hermes-agent`` at ``v2026.8.31`` (0.21.0)
   * - ``native``
     - ``primary`` -> ``native/config.json``, ``models`` ->
       ``native/models.json``
     - none: ``reef-native`` ships with reef
   * - ``terminus``
     - ``primary`` -> ``terminus/config.json``
     - none: ``reef-terminus`` ships with reef, reef-eval ships with reef-infra

A tree's ``config/<target>.json`` merges into the target of that name. At
deployment startup, Reef installs the pinned binary for the evaluation
episodes, unless ``evolution.binary`` names one.

Rendering
~~~~~~~~~

Rendering turns a harness tree into the files one agent reads: the tree and
an adapter descriptor in, root-relative paths with their text out
(``reef.harness.tree.render_tree``). It touches no disk and starts no agent.
Reef renders a tree to run an episode, to serve a release, and to validate a
proposed patch. A release stores the source tree, and ``GET /reef/harness``
renders it for each request.

Reef and the adapter own the rendering together. The shared rules come
first: they are the same for every agent, and they refuse a tree that breaks
them. The adapter gets the last word: its declared paths decide where each
file goes, and the ``quirks`` module it may declare receives the laid-out
files and returns the files that are actually written.

What every adapter shares
^^^^^^^^^^^^^^^^^^^^^^^^^

`reef/harness/tree/render.py <../../reef/harness/tree/render.py>`__ runs for
every adapter. No adapter replaces it, and an adapter registered through the
``reef.harness_adapters`` entry-point group reaches it on the same path as a
bundled one. It turns tree files into agent files by these rules:

- Each config target is a JSON object: the descriptor's ``defaults``, then
  the tree's ``config/<target>.json`` deep-merged over them, then the model
  binding when one is applied. Every target is written even when the tree
  has no file for it, so the defaults always reach the agent.
- ``rules.md`` renders to the descriptor's rules path. A tree without
  ``rules.md`` renders no rules file.
- Every other file renders through its kind's path template, with the name
  from its tree path filling ``{name}``: ``skills/review/SKILL.md`` is the
  skill ``review``.

The render refuses a tree before any agent runs. These checks apply to
whatever the adapter declares; for example:

- A path off the layout, or a file of a kind the descriptor gives no path,
  such as ``extensions/notes.ts`` under ``native``.
- A ``config/<target>.json`` that names a target the descriptor does not
  declare, or that does not hold a JSON object.
- Two files that render to the same path.
- A merged config target that fails one of the ``checks`` its descriptor
  declares (see `Config checks`_). The checks run before the adapter's own
  steps, so no adapter can skip them.

Validation (``reef.harness.tree.validate_tree``) runs before an episode and
adds the checks a single render does not make. Each file must pass its
kind's content checks: JSON objects where the kind needs one, no
credential-shaped literal, and for a ``code_extension`` no write to the
harness's stdout or stderr (``console.log``, ``process.stdout.write``)
without a ``ctx.hasUI`` guard on that line or the line above, because a raw
write breaks the terminal of a session with a UI. A ``native_graph`` or
``native_agent`` must not name a tool, skill, agent, or graph the tree does
not hold, agents must not delegate in a cycle, and a tree holds at most one
``native_loop``. A tree that passes validation also renders.

What the adapter owns
^^^^^^^^^^^^^^^^^^^^^

An adapter is configured mostly by declaration. ``descriptor.yaml`` supplies
the path template for each kind, the named config targets, the defaults
merged under every tree, and the checks the merged config must pass
(``checks``). An adapter whose agent differs only in where files go needs no
code at all, although its descriptor must follow these rules:

- ``files.config`` must declare a ``primary`` target.
- ``files.rules`` and ``files.skill`` are required.
- every named template must contain ``{name}``.

Steps
^^^^^

An adapter may also declare a ``quirks`` module to further customize the
rendering process. Before any of its steps run, the render runs the config
checks the descriptor declares, which `Config checks`_ explains below. The
module defines one subclass of ``AdapterRenderer`` from
``reef.harness.adapters.descriptor``; the loader finds it in the module. The
render calls its steps in this order, and each step is given only its own
files:

.. list-table::
   :header-rows: 1

   * - Step
     - Given
     - Use it to
   * - ``process_config``
     - One merged config file, as a JSON object
     - Adjust config, or refuse a config a check cannot describe
   * - ``process_skill``
     - The text of one ``skill`` file
     - Add or check skill frontmatter
   * - ``process_command``
     - The text of one ``agent_command`` file
     - Add or check command frontmatter
   * - ``check_model_route``
     - The processed config files, skill files and command files
     - Refuse a model route that no check can describe, such as a model in
       the frontmatter of a skill or a command
   * - ``finalize_render``
     - Every rendered file as text, each config file already written in its
       format
     - Work that spans several files, or a config file its suffix's format
       does not fit; it parses a config file again to change it

Each step defaults to leaving its input unchanged, so an adapter overrides only
the ones it needs. A step refuses a tree by raising ``RenderError``.

Steps that several adapters need call helpers in
``reef.harness.adapters.common``. For example, most harnesses list a
``SKILL.md`` only when it has ``name`` and ``description`` frontmatter, so
their ``process_skill`` returns ``with_skill_frontmatter(path, text)``. That
helper adds the skill's directory name and the first line of its text when the
text has no frontmatter. An adapter that writes the header in another form
gets the same values from ``skill_frontmatter(path, text)``.

.. important::

   Write a rule on config values as a config check, not as a step (see
   `Config checks`_). When you need to customize the tree processing, prefer
   the narrow steps over ``finalize_render``. A ``process_*`` step sees
   one file of one kind, and ``check_model_route`` only reads, so a reader can
   tell what each one may change and a reviewer can check it in isolation.
   ``finalize_render`` sees and may rewrite every file, including config files
   it has to parse again, so its effects are hard to follow and easy to break
   when the render changes. Use it only for work no narrow step can do:
   output that depends on several files, or a file layout its suffix's format
   cannot produce. Keep what it does small, and move a check into a narrow
   step whenever that step sees everything the check needs.

Every model call must go to Reef. Before any step, the render refuses a tree
that sets a key the descriptor's ``model_binding`` writes, unless the binding's
credential (the value holding ``{api_key}``) is set beside it. The binding is
merged after the tree and wins every key it writes, so only a tree rendered
without the binding, as validation renders a proposal, can set one;
validation refuses an inline credential, so the credential tells the two
apart. No adapter can turn this check off. Each adapter refuses the other
routes its harness reads, such as a fallback model or a second provider, in
its config checks, and in ``check_model_route`` what the checks cannot
describe.

Every step is a static method, and defining one as an ordinary method fails
when the class is created. The render uses the class without making an
instance, so a step's output depends only on its arguments and no step passes
state to a later one. Keep module-level state out of the steps as well.

Before ``finalize_render``, the render writes each config file in the format
its suffix names: ``.json`` as JSON, ``.toml`` as TOML (Codex's
``config.toml``), ``.yaml`` or ``.yml`` as YAML (Hermes's ``config.yaml``), and
``.env`` as ``KEY=value`` lines; any other suffix is written as JSON. An
adapter whose file needs another layout writes it again in ``finalize_render``.

Examples:

- dsh's ``finalize_render`` turns an object keyed by plugin id into the YAML
  patch list dsh actually loads, a shape no deep merge could produce directly,
  and adds the rendered extensions to that list, since only the rendered files
  name them.
- Codex writes ``models.json`` and points ``config.toml`` at it in
  ``finalize_render``, since that spans two config files. It refuses
  ``code_extension`` there too; refusals like this far outnumber the shared
  ones.

Note that ``native`` declares no ``quirks`` and runs on the shared rules
alone.

Where the shared rules end
^^^^^^^^^^^^^^^^^^^^^^^^^^

- The merge rule is shared. It recurses into objects and replaces
  everything else, lists included, so an agent whose configuration appends to a
  list cannot say so in the descriptor; its quirks rebuild the list instead.
- The renderer steps receive paths, text, and config objects. They cannot see
  which tree file produced what.
- A tree has one rules file. An agent that reads rules from several files
  needs its quirks to split the text.

Config checks
^^^^^^^^^^^^^

A descriptor's ``checks`` state what a config target must keep or must not
set. This is particularly useful during harness evolution. Each check has a
``key``, a ``match`` expression that every value at the key must match, and
an optional ``reason`` that a refusal gives.

Adapter authors should use ``checks`` rather than custom steps to enforce rules
on configuration values: a check is declared beside the defaults it protects,
and it is more discoverable and maintainable than a snippet of code in a corner
of ``quirks.py``.

The checks are declared on the target, ``files.config.<target>.checks``, and
match the merged target: the defaults, the tree's ``config/<target>.json``,
and the model binding when one is applied. This is what lets ``IsDefault()``
compare a key with the target's default:

.. code-block:: yaml

   files:
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

The render runs each target's checks on the merged target, before any adapter
step. Reef owns the matchers. If the existing ones are not enough, consider
contributing a matcher to Reef instead of overriding a step. The render refuses
a tree whose merged config fails a check, for example::

   claude harness tree fails the check on env key 'AWS_REGION' in
   claude/settings.json: Reef's model binding chooses the endpoint, ...

A refusal names the key, and for ``Each`` the failing key or item, but never
the value, which can be a credential the binding wrote. A check without a
``reason`` gives its match expression instead.

**Keys.** ``key`` is required: dot-separated key names, or a list of them
(one check per key). ``.`` names the whole checked value, for a check on its
top-level key names; a check without a key fails to load. A ``*`` segment is
every key of an object and every item of a list. A ``**`` segment is any
number of segments, zero included. A key the checked value does not have is
absent. A ``*`` or ``**`` segment visits only the values the checked value
has.

**Matchers.** Write a match expression as nested calls with JSON literals
(``true``, ``false`` and ``null`` included). Use ``|-`` in YAML to wrap a long
one over several lines.

.. list-table::
   :header-rows: 1

   * - Matcher
     - Matches
   * - ``IsDefault()``
     - For a config target: the value its ``defaults`` give the key, or an
       absent key when they give none. Only for a key without ``*`` or ``**``.
   * - ``IsAbsent()``
     - A key the checked value does not have. An empty value is a set one.
   * - ``Eq(v)``, ``IsIn([v, ...])``
     - A value equal to ``v``, or to one of the list, with its type: ``false``
       matches neither ``0`` nor ``"false"``.
   * - ``IsObject()``, ``IsInt()``, ``Ge(n)``
     - An object; an integer that is not a boolean; a number at least ``n``.
   * - ``StartsWith(s)``, ``EndsWith(s)``, ``HasSubstr(s)``,
       ``MatchesRegex(s)``
     - Text with that prefix, suffix or substring, or text that the regular
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

Text matchers read a string as it is, and a number, a boolean or null as its
JSON text, so ``Lowercase(Eq("0"))`` matches ``0``. They match no object, list
or absent key. Only ``IsAbsent``, ``Not`` and ``Each`` match an absent key, so
"if it is set, it must be X" is ``AnyOf(IsAbsent(), X)``.

Rendering and publication
^^^^^^^^^^^^^^^^^^^^^^^^^

Rendering is not publication. An episode's files are written into a throwaway
root that is removed when the episode ends, and harness evolution renders both
the candidate and the current tree every step in order to compare them; at most
one of the two becomes a release.

Restrictions that depend on how an episode runs, rather than on what a tree
contains, are checked at launch instead. Terminus's requirement for remote
tasks is one: a tree carrying a ``code_extension`` renders, then fails to
launch under the wrong executor.

Run a rendered tree
^^^^^^^^^^^^^^^^^^^

``GET /reef/harness`` renders a release without a model binding, so the
served files name no endpoint and hold no credential. To run the agent on
them yourself, write the files under one directory, point the agent at it the
way the descriptor's ``env`` does (for pi, ``PI_CODING_AGENT_DIR`` set to
``<directory>/pi-agent``), and bind the model. A client that renders the
source tree itself (``GET /reef/harness?form=source``) binds the model in the
same render:

.. code:: python

   from reef.harness.adapters import get_adapter
   from reef.harness.episodes.model_binding import ModelBinding
   from reef.harness.tree import HarnessTree, render_tree

   descriptor = get_adapter("pi")
   binding = ModelBinding(base_url="http://127.0.0.1:8901", model="qwen3-8b", api_key=token)
   files = render_tree(HarnessTree(source_files), descriptor, binding.config_overlays(descriptor))

Reef's inference routes answer only a call that carries the
``x-reef-scenario`` header. The render refuses a tree that sets provider
headers, so the client adds the header to its own binding after the render.
pi sends the headers of its provider entry, ``providers.reef.headers`` in
``pi-agent/models.json``, as ``recipes/beta/reefine/examples/run.py`` writes them. For an
agent that sends no custom header, point ``base_url`` at a local proxy that
adds it, such as ``reef.harness.capture.CaptureProxy``, which also keeps the
receipts a report references. See `Use a published tree
<../user-guide/evolve-your-harness.rst#use-a-published-tree>`__ for the
workflow around a run.

Codex model metadata
~~~~~~~~~~~~~~~~~~~~

For Codex, Reef reads the selected upstream model's ``context_length`` and
``supported_parameters`` from ``GET /v1/models`` before routing calls through
Reef. This is the metadata shape served by OpenRouter. Successful lookups are
cached in the service process; discovery uses the upstream credential and a
five-second timeout. Other adapters do not make this discovery request.

Endpoints without these fields can provide explicit values under
``evolution.model_metadata``, keyed by the exact model name:

.. code:: yaml

   evolution:
     adapter: codex
     model_metadata:
       local-model:
         context_window: 64000
         reasoning: false

A Python ``ModelBinding`` accepts ``metadata=ModelMetadata(64000, False)``; a
named ``evolution.models`` binding accepts the same fields under ``metadata``.
Values in ``evolution.model_metadata`` take precedence over discovery. Missing
or unavailable metadata leaves Codex's existing fallback behavior intact; Reef
does not invent a model's context window. Restart the service to refresh cached
provider metadata. A scenario model override resolves the new model separately.

The binding renders ``codex/models.json`` and a relative ``model_catalog_json``
reference, so an episode reads the same capabilities wherever its root is. Codex retains its own context safety margin and
compaction policy. Custom-model entries keep the pinned Codex unknown-model
instructions, standard shell tools, and low/medium/high effort when reasoning
is supported. The catalog includes all bundled models, so selecting another
model with ``--model`` or the interactive picker retains its native
instructions and tools, including ``gpt-6-astra``. Provider-prefixed names such
as ``openai/gpt-6-astra`` retain the same native configuration. Explicit and
discovered capabilities override the selected model's context window and
reasoning support even when its name matches a bundled model; other native
fields and supported reasoning levels remain intact.
``codex/default_instructions.md`` records the effective unknown-model prompt
exported with this adapter configuration. It is derived from `OpenAI Codex
rust-v0.153.4
<https://github.com/openai/codex/blob/rust-v0.153.4/codex-rs/models-manager/prompt.md>`__,
under the Apache-2.0 license, with SHA-256
``3b08633fa672906666659d764864dfda1d7af5b5111ea5817c8f46e5de4e1a8d``. Codex
removes instructions for tools disabled by the adapter from the source prompt.
A real-binary regression compares the resulting request instructions with and
without metadata. ``codex/bundled_models.json`` is configuration data exported
from the same Apache-2.0-licensed CLI with ``codex debug models --bundled``.
Its SHA-256 is
``661ed96cf0542e8ee117f1ddfd879f416cc05dc96dd1955249a13c04a313ba5e``. Keep both
resources synchronized with the Codex install pin; the real-binary regression
compares the complete bundled catalog, including instructions and tools.
Rendering reads these packaged resources without starting Codex or making a
network request. Tree rules and skills are still added normally. A tree may
supply the same capability fields in ``config/models.json``, but may not set an
arbitrary catalog path or inject native Codex model fields.

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

Tree files map to Terminus 2 configuration as follows:

- ``config`` supplies constructor arguments. Rendering rejects unknown keys.
- ``rules`` supplies an ``extra_instruction_paths`` entry.
- ``skill`` and ``agent_command`` supply two ``AgentConfig.skills`` roots.
  Harbor loads these skills progressively instead of placing every body in
  the prompt.
- One ``code_extension`` may define ``Agent(Terminus2)``. Rendering checks
  its syntax without executing it. The runner loads it through
  ``AgentConfig.import_path``. Without an extension, it runs stock Terminus 2.

Terminus is a batch runner, not a session: ``reef-terminus`` takes only
``--task`` and runs one Harbor task, so the adapter declares no ``install``
section. ``GET /reef/harness`` serves the tree, and each evaluation episode
renders it with the model binding.

Run a published tree
^^^^^^^^^^^^^^^^^^^^

Before running a declarative Terminus tree locally, prepare:

- A running Reef service and the scenario whose published tree you want to use.
- A Harbor task directory, and either a working Docker installation or the
  credentials of a remote task environment: an E2B API key, or a Modal token.
- For Docker, a trial directory shared with Docker. With colima on macOS, use
  a path under your home directory; ``$TMPDIR`` is not shared by default.
- A local tree root containing the ``files`` returned by ``GET /reef/harness``
  for the scenario, saved under their relative paths.

Evaluation episodes call the upstream directly. A tree that you run through
Reef yourself also needs the scenario header in ``llm_kwargs.extra_headers``,
or Reef answers HTTP 400 ``missing or empty x-reef-scenario``. Write this in
``terminus/config.json`` under the tree root, with the service's
``REEF_TOKEN`` as the key (any text when the service has no token):

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

Set the paths to your saved tree and task, then run one task:

.. code:: bash

   TREE_ROOT="$HOME/reef-harness/terminus"
   TASK_DIR="/path/to/harbor/task"
   TRIALS_DIR="$HOME/reef-trials"
   REEF_TERMINUS_DIR="$TREE_ROOT" \
     REEF_TERMINUS_SESSION_DIR="$TREE_ROOT/terminus/sessions" \
     REEF_TERMINUS_TRIALS_DIR="$TRIALS_DIR" \
     reef-terminus --task "$TASK_DIR"

To run the task remotely instead of in local Docker, add
``REEF_TERMINUS_ENVIRONMENT=e2b`` and ``E2B_API_KEY=<key>`` to the command, or
``REEF_TERMINUS_ENVIRONMENT=modal`` with ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET``.

The JSON trial record under ``$TREE_ROOT/terminus/sessions`` contains the
verifier rewards and ATIF trajectory. The health task of the beta Reefine
recipe, ``recipes/beta/reefine/health``, succeeds with reward 1. If a local
Docker trial has neither reward nor verifier output, the runner reports a
possible mount problem: check that Docker shares ``$TRIALS_DIR`` with the
host.

Evaluation directories and Docker context
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reef prepares an evaluation episode for Docker itself. On macOS its root is
made under ``~/.reef/episodes`` (``is_root_bind_mounted``), which colima and
Docker Desktop share with their VM by default. On Linux, WSL, and Windows
the root stays in the temp directory, so the service needs no writable
home. Where Docker there runs in a VM that does not share the temp
directory, set ``TMPDIR`` for the service to a path it shares. The episode
keeps the service's ``DOCKER_HOST``, ``DOCKER_CONTEXT``, and
``DOCKER_CONFIG`` (``host_env``; ``DOCKER_CONFIG`` defaults to
``~/.docker``), because ``HOME`` points into the episode and the docker CLI
reads its current context (colima, Docker Desktop) and the compose plugin
from that directory.

Docker or a remote environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation episodes run the Harbor task in local Docker unless the
deployment runs them in a remote Harbor environment: ``e2b`` with
``E2B_API_KEY``, or ``modal`` with ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET``. For a remote environment, use the sandbox executor and
pass the switch and the credentials into it. For E2B:

.. code:: yaml

   evolution:
     executor: sandbox
     sandbox:
       egress_hosts: [api.e2b.dev]
       env_from: [REEF_TERMINUS_ENVIRONMENT, E2B_API_KEY]

Start the service with ``REEF_TERMINUS_ENVIRONMENT=e2b`` and ``E2B_API_KEY``
set. For Modal, list ``REEF_TERMINUS_ENVIRONMENT``, ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET`` in ``env_from``, allow ``api.modal.com``, and start the
service with ``REEF_TERMINUS_ENVIRONMENT=modal``. The runtime needs Linux,
bubblewrap, Python 3.12+, and Harbor with the environment's extra
(``harbor[e2b]``, or the ``modal`` package).
The interpreter and local task directories must be visible inside the
sandbox, for example under ``/opt``. ``egress_hosts`` currently does not
enforce a hostname firewall. The episode root under ``~/.reef/episodes`` and
the ``DOCKER_*`` variables above apply only to local Docker; an episode in a
remote environment gets neither.

A Python extension (``code_extension``) runs only this way. Reef rejects
Docker inside bubblewrap and an extension in an unisolated runner before
launch.

The Terminus quirk supplies ``validate_execution`` as an
``ExecutionValidator``. Its ``__call__(files, executor)`` checks the rendered
tree and configured executor before Reef writes episode files. It raises
``EpisodeLaunchError`` for unsupported combinations and replaces the default
``self_isolating`` nesting restriction. Execution, timeout, cleanup, and
trajectory handling still use the shared episode code.

Model binding and provider compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Terminus 2 calls the model through litellm. The model binding keeps the
served name in ``model_name`` and sets ``llm_kwargs.custom_llm_provider`` to
``litellm_proxy``, litellm's route to an OpenAI-compatible proxy. litellm
then sends that name unchanged to ``api_base``, whatever vendor prefix it
carries, and puts the tree's call arguments into the request body:
``reasoning_effort``, the ``thinking`` budget Harbor sends for a Claude
model under ``max_thinking_tokens``, and ``llm_call_kwargs`` fields such as
``provider`` or ``top_k``.

Without that provider, a vendor prefix litellm does not know, such as
``qwen/qwen3-coder``, fails with ``LLM Provider NOT provided``, and one it
knows, such as ``deepseek/``, goes to that vendor's own client without its
vendor prefix. ``custom_openai`` also keeps the name, but it drops
``reasoning_effort`` and ``thinking``, and a call with an
``llm_call_kwargs`` field the OpenAI SDK does not take fails. Harbor looks
up the context limit that Terminus 2 summarizes against under
``model_name``, so a served name that litellm lists, such as
``openai/gpt-4o-mini``, keeps its limit. A name litellm does not list gets
Harbor's fallback of 1,000,000 tokens.

opencode
~~~~~~~~

The ``opencode`` adapter runs ``opencode run --format json --auto "<task>"``
headless. It renders configuration to ``opencode/opencode.json``, skills to
``opencode/skill/<name>/SKILL.md``, and commands to
``opencode/command/<name>.md``.

Write a skill or command
^^^^^^^^^^^^^^^^^^^^^^^^

For a skill named ``notes``, use a body such as:

.. code:: text

   ---
   name: notes
   description: Summarize the changes made during a task.
   ---
   List the changed files and the checks that ran.

If a skill has no frontmatter, Reef supplies its directory name and first
line as quoted ``name`` and ``description`` strings. If you provide your own
frontmatter, include both fields.

A command named ``review`` can select an existing agent:

.. code:: text

   ---
   name: review
   description: Review the current changes.
   agent: plan
   subtask: false
   ---
   Review the changes and explain any correctness problems.

When present, frontmatter must be a YAML mapping between opening and closing
``---`` lines, without a byte order mark, YAML tags, or an alternate format
such as JSON. Quote strings containing a colon followed by a space, or
numeric-looking text such as ``"1e5"``. Use ``true`` or ``false`` for
booleans; ``yes`` is a string.
Reef rejects unsupported frontmatter forms and fields with the wrong types
during rendering.

Command fields have these constraints:

.. list-table::
   :header-rows: 1

   * - Field
     - Type and constraint
   * - ``name``
     - Optional; must match the command's file name. The name replaces any
       existing command with that name.
   * - ``description``, ``variant``
     - Strings when present.
   * - ``agent``
     - A string naming an enabled tree agent or a built-in agent: ``build``,
       ``plan``, ``general``, ``explore``, ``title``, ``summary``, or
       ``compaction``.
   * - ``subtask``
     - A boolean when present.
   * - ``model``
     - Not allowed; Reef's model binding selects the model.

Tree agents are configured under ``agent`` (or the older ``mode``). Their
``disable`` and ``hidden`` fields are booleans, and ``mode`` is ``subagent``,
``primary``, or ``all``. An agent may not set a different ``name``.
``default_agent`` must name an enabled, non-hidden agent that is not a
subagent. Without ``default_agent``, keep at least one agent meeting those
conditions. Reef rejects invalid configurations before opencode can fail to
start or load a command.

Configuration and model binding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The defaults keep autoupdate and sharing off, allow every permission, and set
``enabled_providers`` to ``["reef"]``. Rendering rejects changes to those
update, sharing, and provider-list settings.

The deployment's model binding selects the provider and model. The tree must
not contain inline credentials, set ``provider``, ``model``,
``disabled_providers``, or ``small_model``, or choose a model in an agent or
command. Reef rejects these overrides.

DeepSeek Harness
~~~~~~~~~~~~~~~~

Rendered files and profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``dsh`` adapter runs ``dsh --profile headless "<task>"`` and relocates
the agent's home through ``DSH_HOME``. dsh combines its bundle layers with a
user patch layer: a YAML list addressed by plugin id. The adapter accepts
the ``primary`` config as an object keyed by plugin id, such as
``{"agent-loop": {"config": {...}}}`` or ``{"disabled": true}``, and its
quirks render that object as the patch list. A string beginning with
``!!js `` becomes a JavaScript expression, as in dsh's own bundles.

The descriptor defaults keep the session log uncompressed and disable
telemetry and the LLM title call. Rendering rejects a tree that changes
those settings. The tree's files render as follows:

- ``rules`` becomes dsh's user-global ``AGENTS.md``.
- ``skill`` becomes ``skills/<name>/SKILL.md``. If the file lacks the
  required YAML frontmatter, the adapter adds ``name`` and ``description``.
- ``agent_command`` becomes a user-invocable skill under ``DSH_AGENTS_HOME``.
  It uses ``disable-model-invocation: true`` and runs as ``/name``; dsh has no
  separate command surface.
- ``code_extension`` becomes a plugin module referenced by relative path
  from the patch layer.

The model binding uses an ``llm-pi-ai`` route. Its ``apiKeyEnv`` names the
key supplied through the ``env`` config target, dsh's ``.env`` launch layer.

The ``web`` target is the patch layer of the profile ``dsh web`` boots, a
person's session in the browser; evaluation episodes use the headless
profile. It carries the same defaults as the headless patch, is checked the
same way, and gets the model binding too. The upstream web template compresses
session logs. Reef disables compression in both profiles so they can share the
sessions root; a compressed profile would reject the headless profile's plain
logs. The ``web_manifest`` target is the profile's ``package.json`` with
``patchReload: startup``: the manifest dsh writes for a new web profile sets
``live``, and with it ``dsh web`` exits at start.

A ``code_extension`` renders once; the web patch inserts it from the
headless profile's directory (``../headless/extensions/<name>.mjs``). A
config file reaches one profile, the one its target names. Commands render
under ``dsh-agents/``, so a session lists the tree's commands only when
``DSH_AGENTS_HOME`` names that directory, as the episode ``env`` does;
otherwise dsh lists the person's ``~/.agents/skills``.

Command frontmatter
^^^^^^^^^^^^^^^^^^^

For an ``agent_command`` named ``review``, a minimal body is:

.. code:: text

   ---
   name: review
   description: Review the current changes.
   ---
   Review the changes and report correctness problems.

The adapter enforces ``disable-model-invocation: true``. The command remains
available to the user as ``/review`` and is not available for the model to
invoke.

Parsing and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^

When the command text carries its own frontmatter, the adapter reads it
the way dsh does: between two ``---`` lines that may end in a carriage
return, as YAML 1.2, where ``Yes`` and ``1:30`` are strings. It keeps the
keys, sets ``disable-model-invocation: true``, and removes
``user-invocable`` and the camelCase keys dsh rejects (``userInvocable``,
``disableModelInvocation``, ``modelInvocable``), so the model never runs a
command and the person always can. A ``name`` that is not a skill name, or
a ``description`` that is empty or not a string, is written the way a
missing one is, since dsh ignores the file otherwise. Every value is
written so that YAML 1.2 reads it back with its type. A value tagged ``!``
or ``!!str`` is a string, as it is to dsh, and an empty or null block
counts as a mapping with no keys. Rendering rejects frontmatter that does
not parse, nests too deeply to read, holds any other tag, or is not a
mapping.

Hermes Agent
~~~~~~~~~~~~

The ``hermes`` adapter runs ``hermes chat -Q --oneshot -q "<task>"`` with
``HERMES_HOME`` relocated. Its ``primary`` target is ``config.yaml``, which
the render writes as YAML. The descriptor's checks keep these defaults so an
episode stays self-contained and makes one request:

- ``security.tirith_enabled`` disables the terminal scanner download. The
  scanner would also block a command it cannot resolve.
- ``auxiliary.title_generation.enabled`` disables the title model call.
- ``memory.nudge_interval: 0`` and ``skills.creation_nudge_interval: 0``
  disable the background memory and skill reviews, which make model calls
  and write skills into the tree.
- ``curator.enabled: false`` disables the curator. It writes its state into
  the tree's ``skills/`` at the first start, and later archives and backs up
  the skills there, which in a person's session are the release's own.
- ``sessions.write_json_snapshots`` enables the per-session snapshot read by
  ``hermes-session-json``.

Rendering rejects a tree that changes any of those settings, or that puts a
value that is not an object where a section holding one belongs. The quirks
also write ``.no-bundled-skills``, so episodes use the tree's skills instead
of the bundled catalog.

The tree's files render as follows:

- ``rules`` becomes the home-level ``SOUL.md`` that Hermes reads.
  ``AGENTS.md`` is project-scoped and read from the working-directory chain.
  The rules follow Hermes's own default identity: Hermes treats ``SOUL.md``
  as the agent's identity and writes its default there only while the file
  is absent, so rules written alone would replace it.
- ``skill`` becomes ``skills/<name>/SKILL.md``. The adapter adds the required
  ``name`` and ``description`` frontmatter if the file lacks it.
- ``agent_command`` becomes a skill under ``hermes-commands``, which the
  quirks add to ``skills.external_dirs``. Hermes exposes skills as ``/name``
  commands and has no separate command surface.
- ``code_extension`` becomes a plugin package at
  ``plugins/<name>/__init__.py`` defining ``register(ctx)``. The quirks
  write its manifest, ``plugins.enabled`` entry, and ``tools.override``
  permission. Hermes requires this consent before loading a plugin; plugin
  tools are then available through ``tool_search`` and ``tool_call``.

Command and plugin configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quirks add the commands root beside the home
(``${HERMES_HOME}/../hermes-commands``) to ``skills.external_dirs``.

The tree's own entries in ``skills.external_dirs``, ``plugins.enabled``, and
a rendered plugin's ``granted_capabilities`` stay ahead of the ones the
quirks add, because a config file's list replaces the list below it. A
string in ``skills.external_dirs`` is one entry, as Hermes reads it.
Rendering rejects any other value in these three settings that is not a
list of strings.

Session state
^^^^^^^^^^^^^

Hermes writes files of its own into ``skills/`` that no config key turns
off: the bundled skill manifest it rewrites at every start, the one
essential skill it seeds (``autonomous-ai-agents/hermes-agent``), and the
usage counts it updates when a skill is loaded (``.usage.json`` and its
lock). An episode lists them in ``cleanup_whitelist``; in a person's session
they land in the rendered tree's ``skills/``.

Model binding and approvals
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model binding uses a custom provider with a literal key in
``config.yaml`` and supports only the ``openai`` dialect. Hermes's default
approval policy runs tools in the working directory without prompting and
returns a tool error for commands it considers dangerous. The adapter does
not use a bypass flag.

Codex CLI
~~~~~~~~~

The ``codex`` adapter runs ``codex exec --json`` headless with its user
home relocated through ``CODEX_HOME``. Its ``primary`` target is
``config.toml``, which the render writes as TOML. The descriptor's checks
keep these defaults so an episode stays self-contained:
the update check, analytics, feedback, and telemetry are off, and the
``workspace-write`` sandbox has no network. Rendering rejects a tree that
changes them.

The tree's files render as follows:

- ``rules`` becomes ``AGENTS.md``.
- ``skill`` becomes ``skills/<name>/SKILL.md`` below ``CODEX_HOME``, the root
  that both ``codex exec`` and the interactive CLI list. The adapter adds the
  required ``name`` and ``description`` frontmatter if the file lacks it.
- ``agent_command`` becomes a skill in the same root, which the person types
  as ``$name``. Codex 0.153.4 loads no custom prompts, and the interactive
  CLI rejects an unknown ``/name``. A skill and an ``agent_command`` with one
  name render to one path, so Reef rejects them.
- ``code_extension`` is rejected, because Codex hooks run outside its
  command sandbox.

The tree may set ``web_search`` (``disabled``, ``cached``, ``indexed``, or
``live``) for a person's session, but not ``approval_policy``. The episode
argv pins ``--config approval_policy="never"`` and
``--config web_search="disabled"``, which win over ``config.toml``, so an
episode never waits for an approval and never searches the web. A person's
session keeps Codex's own ``on-request`` approvals and the sandbox without
network.

The model binding supports only the ``responses`` dialect.

Native tools and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``native_tool`` file is the tool's Python module, and it renders to
``native/tools/<name>.py`` unchanged. It defines ``run(args, workdir) -> str``
and declares the tool with top-level literal assignments: ``DESCRIPTION``,
``PARAMETERS`` (a JSON schema of the arguments), ``CAPABILITIES``, and
optionally ``NAME``, which must be the file name.

``CAPABILITIES`` is an optional list of distinct ``read``, ``write``,
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

Hooks do run in the loop process. Each hook module is imported when the
loop loads the tree (once per episode, once per mount in serve mode), and its
``listen`` function runs at its event. It can call
``next`` to reach the next layer and change the decision that steers the
loop. Under ``sandbox``, the loop process is the episode jail, with a
writable workspace and session directory, network access to the model
endpoint, and the executor's base directories and shells. Under ``local``,
it is the host. Use ``review_kinds: [native_hook]`` to require review before
publishing hook changes. Under ``local``, tools also run without confinement;
include ``native_tool`` in ``review_kinds`` when reviewing those changes.

The tutorial's starter tree, ``tutorials/evolve-your-harness/seeds/native/``,
holds the tools ``read_file``, ``write_file``, ``run_bash``, and ``execute``,
the hook ``loop_guard``, and a ``main`` graph; ``evolution.seed`` names it, and
the steps evolve it. ``execute`` runs Python in the workspace and can import
other tools by name, for example ``import read_file;
read_file.run({"path": "x"}, WORKDIR)``. An adapter without a
``files.native_tool`` path refuses a file of that kind.

Validation rejects a tool file that does not compile, whose ``NAME`` is not
the file name, or whose name a self tool reserves. Before the first
model call, the loop returns ``LOAD_ERROR`` if it cannot open or parse a
tool file, find a top-level binding of ``run``, or read the last top-level
assignment to a declaration constant as a literal. A file that parses but
does not compile fails on its first call, as does top-level code that raises.

The source reader uses the last module-scope binding in source order. It
follows ``if``, ``try``, ``with``, ``for``, ``while``, and ``match`` bodies,
but not function or class bodies. Definitions, imports, assignment targets,
``for`` and ``with`` targets, exception targets, walrus targets, and match
captures can bind names. The file's last assignment to a declaration
constant therefore wins.

The loop does not execute top-level code while reading declarations. If
that code raises (including ``SystemExit``) or binds ``run`` to a
non-callable value, the first call fails with ``TOOL_FAILED`` and the
episode continues. Under ``local``, the module is imported once, so later
calls fail the same way without re-running its top level. Under ``sandbox``,
the child imports it on every call.

Native hooks
~~~~~~~~~~~~

A ``native_hook`` file listens at one of the loop's four events and renders
to ``native/hooks/<name>.py`` unchanged. It defines
``listen(payload, next) -> decision`` and assigns ``EVENT``, and optionally
``NAME``, which must be the file name.

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

The starter tree's one hook, ``loop_guard`` at ``post_execute``, reminds the
model when the same call repeats three, five, or eight times in a row. It is
a tree file, so a step can retune or remove it.

The native descriptor has no path for ``agent_command`` or
``code_extension`` because the loop does not read them. Validation rejects
files of those kinds with "does not render". The renderer writes both
config targets, but the loop reads only ``models``. Its binding fields
(``api``, ``base_url``, ``api_key``, and ``model``) come from the model
binding, and the model-route check refuses a tree that sets them.

Native graph stages
~~~~~~~~~~~~~~~~~~~

A ``native_graph`` file controls the native loop. ``native/graphs/main.json``
is the root turn's graph, and an agent may name another. A graph contains
named stages of the kinds below, with edges keyed by each stage's outcome.
When the tree has no ``main`` graph, the loop uses
``reef.harness.runners.native.graph.SEED_GRAPH``: ``think`` asks the model,
``act`` runs tool calls, and ``done`` ends the turn. A tree with its own
graph replaces that flow; hooks still handle their events.

Validation rejects unknown kinds or keys, outcomes without exactly one edge,
stages unreachable from ``start``, stages without a path to an end, and
cycles without a model stage. The step budget (``max_steps``, 1 to 32) thus
bounds each run. Validation also rejects a ``tools`` allow list that names a
tool missing from the tree.

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
  ``context_window`` in ``models.json``, which the tree sets in
  ``config/models.json``; its default is 32,768 tokens at four characters
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
guard, although validation rejects graphs that could reach that limit.

Native subagents
~~~~~~~~~~~~~~~~

A ``native_agent`` file, ``native/agents/<name>.json``, defines one agent
the loop delegates to. It can specify:

- ``prompt``, appended to rules and skills as its system prompt;
- ``graph``, which defaults to the built-in ``seed`` loop but can name
  ``main`` or another graph;
- ``tools`` and ``skills`` visible to this agent, defaulting to all in the
  tree;
- ``max_steps`` and ``max_tool_calls``; and
- ``then``, a list of agents that receive its final text in order.

A graph starts an agent through a ``subagent`` stage. Validation rejects
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

A ``native_loop`` file renders to ``native/loops/<name>.py`` unchanged. It
defines ``run_turn(ctx)`` and may assign the literals ``NAME`` (the file
name) and ``MAX_STEPS``, the model-step budget, from 1 to 32 (default 12).
With a loop file, the root turn runs ``run_turn`` instead of the ``main``
graph. Subagents still run their graphs and are available through
``ctx.agent``.

Only one loop is allowed per tree: validation rejects a second
``native_loop`` file, and the loader refuses two files under ``loops/``.
Before import, the loader checks that the file parses and that ``NAME`` and
``MAX_STEPS``, where the file binds them, are plain literal assignments.

Validation reads the code without executing it. It checks that the module
compiles, contains no credential, and leaves ``run_turn`` bound to a plain
top-level ``def`` with a parameter. The last module-scope binding wins,
including bindings inside ``if``, ``for``, ``with``, ``try``, or ``match``;
those other bindings are refused.

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
  transition. The event must be a tree name (a letter or digit, then
  letters, digits, ``.``, ``_`` or ``-``) other than ``enter`` or ``exit``;
  the prefix prevents it from writing a core event. Reef
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
names the task, model, tools with their capabilities, hooks by event,
enforcement mode, and agents. It also names ``graph`` and ``loop``; ``loop``
is null under a graph.

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
~~~~~~~~~~~~~~~~~~~~~~~

Both modes read the same rendered files with the same loader and
interpreter. ``reef-native -p`` runs one episode in one process and turn;
``run_episode`` launches it, and the sandbox executor confines it. It builds
one ``NativeHost`` from the rendered ``native/`` directory at boot and never
changes it.

``reef-native serve`` keeps one resident process per directory and a ``Run``
per session across turns. It starts ``reef.harness.capture.CaptureProxy`` in
process, so every model call carries the scenario, the token, and the
``release`` tag (``trial`` too during a ``harness_try``).
``release_client.HeadWatch`` follows the served head by polling the catalog
and reading the ``x-reef-release-id`` header on inference responses. Each
mount renders a source tree into its own directory under
``native/mounts/`` and builds a new ``NativeHost`` from those files. A tree
that fails to render or load leaves the current host in place.

The interpreter calls ``loop.before_step(run)`` before each model stage.
In episode mode it does nothing. In serve mode it checks the turn's wall
clock and swaps in the host of a queued mount. A new host imports every
hook module again, so module-level state, such as ``loop_guard``'s repeat
count, starts again after a mount.

Serve mode writes extra events to the open turn's session, or to
``native/sessions/serve.jsonl`` when no turn is open. They use the same
``{type, seq, time, data}`` shape:

- ``harness/mount``: ``release_id``, ``parent_release_id``, ``source``
  (``boot``, ``release``, ``try``, or ``rollback``), and ``files``, the
  number of files in the mounted tree. A trial also records ``try_id`` and
  the ``paths`` its patch changes.
- ``harness/mount-failed``: ``release_id``, ``source``, and ``error``.
- ``harness/unmount``: ``try_id`` and ``release_id``, after the served tree
  is mounted back with ``source: rollback``.
- ``release/available``: ``release_id`` when using ``--follow pinned``.
- ``release/poll-failed``: ``error`` and ``retry_in_s``.

The ``session`` header also records ``mode: serve``, ``session``, and
``release_id``. ``turn/start`` records the turn number, ``prompt``, and
``cwd``. ``request/header`` repeats when the prompt or tool declarations
change. A wall-clock timeout ends the turn with reason ``turn-timeout``.
Steps restart at one each turn, so full tool outputs go under
``.reef/tool-output/t<turn>/``.

The Unix socket uses UTF-8 JSON lines and serves one request per connection.
It is at ``native/serve.sock``, or under ``/tmp`` if that path exceeds 100
bytes. Requests and responses are:

- A turn request is ``{"turn": {"prompt": str, "session": str | null,
  "workdir": str}}``. Reef streams its events, then a ``turn/result``
  object with ``exit``, ``session``, ``turn``, ``text``, and ``receipts``,
  the receipts of the turn's model calls on the served tree, in ``data``;
  the calls of a ``harness_try`` trial are left out.
- ``{"control": "status"}`` returns ``control/result`` with
  ``release_id``, ``parent_release_id``, ``follow``, ``files``,
  ``pending_mount``, ``sessions``, ``socket``, and ``self_tools``.
- ``{"control": "mount", "release_id": str}`` returns ``control/result``
  with ``mounted``, ``release_id``, and ``error``.
- Malformed input returns an ``error`` object with ``message`` in ``data``.

Turns run one at a time; another connection waits. The three self tools in
``reef/harness/runners/native/selftools.py`` are built-in tools that join
every host. They run in the process regardless of ``REEF_NATIVE_ENFORCE``
and are registered only with ``--self-tools``. A tree file that takes a
self tool's name fails validation, and a mount of such a tree fails to load.

Descriptor fields
~~~~~~~~~~~~~~~~~

``descriptor.yaml`` describes how Reef renders, starts, and reads an agent:

- ``name`` identifies the adapter. ``binary`` is the executable, and
  ``argv`` supplies its arguments for one headless prompt. Reef substitutes
  ``{prompt}`` in ``argv``.
- ``files`` maps tree kinds to output paths, such as
  ``skills/{name}/SKILL.md``. ``rules`` and ``skill`` paths are required.
  Each ``files.config.<target>`` gives a config file's ``path``, its
  ``defaults`` and its ``checks`` (see `Config checks`_).
  Other kinds are optional; Reef rejects a tree file of a kind with no
  render path.
- ``trajectory`` specifies the session log's path and reader format.
  ``final_assistant_text`` extracts assistant replies from flat messages,
  pi's ``message`` events, and Codex's ``response_item.payload`` messages,
  including ``output_text`` parts. Episode scorers use this text as the answer;
  user messages, tool outputs, and reasoning events are not answers.
- ``env`` points the agent's state into the episode root, substituting
  ``{root}``. It also turns off what would make an episode
  non-deterministic, such as the binary's own updater while Reef pins its
  version: ``PI_SKIP_VERSION_CHECK`` for ``pi``, and ``DISABLE_AUTOUPDATER``
  and ``DISABLE_UPDATES`` for ``claude``. Claude Code applies the ``env``
  block of ``settings.json`` over the environment, so the ``claude``
  descriptor's checks reject a tree that sets either one there.
- ``install`` pins the vendor install: ``kind`` (``npm`` or editable-venv
  ``git``), ``package``, ``version`` (as reported by ``--version``), and
  ``binary_path`` below the install prefix. A git install also names
  ``repository`` and ``ref``. Harness evolution startup installs the pin
  unless ``evolution.binary`` names a binary.
- ``host_env`` optionally lists service variables that an episode under the
  local executor keeps, each with a default for when the service has none.
  ``{home}`` is the service's home directory, and an empty default leaves the
  variable unset. It is for a host tool that the relocated ``HOME`` would
  hide. Otherwise the local executor passes only ``PATH``, ``SYSTEMROOT``,
  ``TMPDIR``, and ``CUDA_VISIBLE_DEVICES`` from the service.
- ``is_root_bind_mounted`` is optional and ``true`` when the binary
  bind-mounts paths below the episode root into a container. On macOS, where
  Docker runs in a VM that shares the home directory, the local executor
  then makes the root under ``~/.reef/episodes`` rather than the temp
  directory. On other platforms the root stays in the temp directory.
- ``model_binding`` contains config templates for each supported API
  dialect (``openai``, ``responses``, or ``anthropic``). Reef merges the
  matching templates after the tree when it renders an evaluation episode,
  and substitutes ``{base_url}``, ``{api_key}``, and ``{model}`` in their
  string values.
- ``writable_paths`` lists state directories that a hosted sandbox makes
  writable. Rendered inputs within them stay read-only.
- ``self_isolating`` is optional and ``true`` when the adapter runs episodes
  in its own container, so nesting in Reef's jail is refused unless the
  quirks' execution check accepts the configuration.
- ``is_prompt_task_directory`` is optional and ``true`` when the ``{prompt}``
  slot names a Harbor task directory rather than text the model reads.
- ``cleanup_whitelist`` lists agent-written paths allowed after boot or a
  run, rather than reported as residue.
- ``quirks`` names an optional module for adapter-specific render steps,
  execution checks, and cleanup patterns.

Model binding checks
~~~~~~~~~~~~~~~~~~~~

These checks restrict the configuration rendered from a harness tree for
``claude``, ``dsh``, ``hermes``, ``pi``, and ``terminus``. They are not
request-time model authorization: the proxy forwards ``model`` and ``models``
as sent, so a tool or plugin that constructs its own request can still name
another model.

For rendered configuration, Reef's model binding supplies the endpoint, model,
and credential after the tree and replaces every value it writes. A tree may
not contain an inline credential. The renderer accepts the binding's complete
configuration shape and rejects tree files that independently change model
routing. The following groups describe the adapter-specific restrictions.

Claude Code routing
^^^^^^^^^^^^^^^^^^^

Rendering rejects these settings:

- In ``settings.json`` ``env``: ``ANTHROPIC_`` variables, cloud provider
  switches and credentials (including Bedrock, Vertex, and Foundry), proxies,
  endpoints, and model names. Names are matched without case, as on Windows.
- Model choices such as ``model``, ``fallbackModel``, ``availableModels``,
  ``modelOverrides``, and ``advisorModel``.
- Credential helpers: ``apiKeyHelper``, ``awsAuthRefresh``,
  ``awsCredentialExport``, ``gcpAuthRefresh``, and ``proxyAuthHelper``, plus
  the login method.
- ``model`` in command or skill frontmatter. An unreadable frontmatter block
  is also rejected when it contains the word ``model`` or an escape.

The binding writes the credential as ``ANTHROPIC_AUTH_TOKEN``.

Hermes routing
^^^^^^^^^^^^^^

Rendering rejects alternative routes and model choices:

- ``providers``, ``custom_providers``, ``fallback_model``,
  ``fallback_providers``, ``moa`` presets (``presets``, or the older
  ``reference_models`` and ``aggregator``), and ``auxiliary.openrouter_model``.
- Aliases (``model_aliases``, ``model.aliases``) and endpoint, model, or
  credential fields such as ``model.model``, ``model.name``,
  ``model.api_base``, and ``model.key_env``. Top-level ``provider``,
  ``base_url``, and ``api_base`` are included because Hermes moves them into
  ``model``.
- ``model.api_mode`` and ``model.openai_runtime``. These can bypass the custom
  provider: ``bedrock_converse`` calls AWS Bedrock, while ``codex_app_server``
  delegates to a ``codex app-server`` subprocess.
- Provider, endpoint, credential, ``api_mode``, model, ``fallback_chain``, or
  ``prefer_fast_model`` settings for auxiliary tasks, delegation, cron, or
  ``curator.auxiliary``. An auxiliary provider may remain ``auto`` or ``main``
  to use the main model.

The binding writes the credential as ``model.api_key``.

DeepSeek Harness routing
^^^^^^^^^^^^^^^^^^^^^^^^

Rendering rejects:

- ``llm-pi-ai`` routes other than ``reef``, or fields on that route that the
  binding does not write.
- ``agent-default-model``, ``llm-deepseek``, and web-search endpoint/model
  overrides.
- Provider or model choices for the title call, subagents, declared agents,
  or compaction summaries.
- A patch entry naming another package, or JavaScript expressions in the
  routing plugins, whose values cannot be inspected during rendering.

The binding's credential is ``REEF_API_KEY``.

Pi routing
^^^^^^^^^^

Rendering rejects providers in ``models.json`` other than ``reef``, fields on
``reef`` that the binding does not write, ``enabledModels``, and ``httpProxy``.
The latter would forward all calls through another host. The binding writes
``providers.reef.apiKey``.

Terminus routing
^^^^^^^^^^^^^^^^

Rendering rejects ``llm_kwargs`` fields that the binding does not write. It
also rejects ``llm_call_kwargs`` arguments that select an endpoint, provider,
credential, model, fallback, or logging callback, including ``base_url``,
``api_base``, ``custom_llm_provider``, ``model``, and ``fallbacks``. The
Terminus quirk lists the recognized routing arguments. The binding writes
``llm_kwargs.api_key``.

Request-body fields
^^^^^^^^^^^^^^^^^^^

A request body that a tree passes to the bound endpoint reaches the
provider with the bound key, so rendering rejects one that names a model:
``model``, or ``models``, which OpenRouter reads as fallback models. These
bodies are the ``claude`` ``CLAUDE_CODE_EXTRA_BODY`` env value (which must
be a JSON object), the hermes ``extra_body`` of an auxiliary task or of the
curator, ``delegation.request_overrides`` with its ``extra_body``, and the
terminus ``llm_call_kwargs`` (litellm sends a key it does not read in the
body) with its ``extra_body``. Other body fields, such as OpenRouter's
``provider`` preferences, stay admitted.

Connect a new agent
~~~~~~~~~~~~~~~~~~~

To connect an agent that has no adapter yet:

.. steps::

   #. The file it reads configuration from becomes a ``files.config`` target.
   #. The command line that runs one prompt headless becomes ``binary`` and
      ``argv``.
   #. The path and format of its session log become ``trajectory``. A new
      format subclasses ``TrajectoryReader``
      (`reef/harness/episodes/trajectory.py
      <../../reef/harness/episodes/trajectory.py>`__) and registers with
      ``@register_trajectory_reader``.
   #. The files its first boot creates go in ``cleanup_whitelist``, so a fresh
      episode root is treated as clean. ``dir/**`` tolerates a whole subtree
      (session storage, ``node_modules``); any other entry is a glob against
      the root-relative path, so anchor a single file with its full path, like
      ``pi-agent/auth.json``. A bare directory name matches nothing under it.

`reef/harness/adapters/descriptor.py
<../../reef/harness/adapters/descriptor.py>`__ validates descriptors at load.
The bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ provide complete references. External adapters
register through the ``reef.harness_adapters`` entry-point group.
