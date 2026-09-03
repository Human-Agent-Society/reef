Harness adapters
================

An adapter maps a harness tree into the files expected by a third-party
coding-agent CLI and binds that harness to the served model. The harness and
model together form the running agent. The tree never names a file path; the
adapter does. Reef bundles five, one per third-party coding-agent CLI.

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

The ``codex`` adapter runs ``codex exec --json`` with its Codex state root
relocated by ``CODEX_HOME`` (``run_episode`` separately relocates ``HOME``).
Reef's JSON ``config`` nodes render as Codex
TOML; rules render to ``AGENTS.md``; skills use the shared user skill root
under ``$HOME/.agents/skills``; and ``agent_command`` uses Codex's legacy
custom-prompt directory. Custom prompts remain supported but are deprecated
upstream in favor of skills. Codex ``code_extension`` nodes are rejected for
now: native hooks run outside Codex's command sandbox, so activating arbitrary
evolved JavaScript would cross Reef's isolation boundary. The model binding
uses Codex's Responses wire API, so set
``reef.upstream_api: responses``; the default Chat Completions dialect fails
at recipe construction. Its bearer token is added only to the transient
Reef-owned proxy; Codex receives only the proxy's temporary loopback address
and a short-lived capability, so neither the evaluation render nor the
committed tree contains the upstream token.
Codex config evolution is limited to model-behavior fields; Reef pins or
rejects settings that can load host paths, launch integrations, add egress, or
override the transient model, provider, or endpoint.

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
   install | the one-command install pin: ``kind`` (``npm`` only), ``package``, ``version``, and ``binary_path`` under the install prefix
   model_binding | per API dialect (``openai``, ``responses``, ``anthropic``), the config nodes Reef appends at evaluation time; ``{base_url}``, ``{api_key}``, and ``{model}`` substitute into string values
   model_binding_proxy | keep the upstream credential in Reef and render a temporary capability-authenticated loopback endpoint; its binding-owned config fields are reserved from evolved nodes
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
