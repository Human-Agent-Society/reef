Harness adapters
================

Adapters define installation, configuration, headless execution and session formats.

Connect a new agent
-------------------

#. Adapt `pi's descriptor <../../reef/harness/adapters/pi/adapter.yaml>`__. Include model bindings and ``config.primary``. Unknown keys prevent loading.
#. Pin package versions and stage packages before Harbor runs. Runs must not require registry access.
#. Add a ``TrajectoryReader`` for new session formats and verifier trajectories (``to_atif``).
#. Register in ``BUILTIN_ADAPTERS`` or ``reef.harness_adapters`` entry points.
#. Test renders, routing and sessions. Run a real task in each supported mode.

Client installation needs ``run.layer`` and keeps user configuration separate.
Pass ``{token}`` only through environment variables.
Declare Harbor skills/MCP in ``run.task``. Unsupported capabilities fail.
See the `descriptor contract <../../reef/harness/adapters/descriptor.py>`__ for rendering hooks.

Tree parts
----------

Declare only parts that work in every supported mode.

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

Native loops and Python programs require ``review: true`` and manual activation.
See `part contracts <../../reef/harness/adapters/parts.py>`__ for naming and validation rules.

Config checks
-------------

Reserve model-binding keys and block alternate model routes.

.. code:: yaml

   checks:
     - key: [model, fallbackModel]
       match: IsAbsent()
       reason: Reef's binding selects the model

Errors identify keys, never values. Use ``AnyOf(IsAbsent(), X)`` for optional constrained values.
See `matchers <../../reef/harness/adapters/matchers.py>`__.
``AdapterRenderer`` methods must be stateless. Raise ``RenderError`` to refuse a tree.

Token limits
~~~~~~~~~~~~

Set ``REEF_CONTEXT_WINDOW`` or ``REEF_MAX_OUTPUT_TOKENS`` to a positive integer in the
`evaluation environment <../reference/http-api-evaluation.rst#the-harness-field>`__.
Each arrow points to the next fallback.

.. list-table::
   :header-rows: 1

   * - Adapter
     - Context window
     - Max output tokens
   * - pi, OpenCode
     - Override → model metadata → 128,000
     - Override → 32,000
   * - Codex
     - Override → model metadata → agent default
     - Agent default
   * - Other built-in adapters
     - Agent default
     - Agent default

Use ``{context_window}`` and ``{max_output_tokens}`` in templates that set these limits.
