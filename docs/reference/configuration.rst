Configure Reef serving and training
===================================

Local inference needs no YAML file:

.. code:: bash

   uv run reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct

This starts SGLang on an automatically selected loopback port, waits for its
health endpoint, then starts Reef on ``127.0.0.1:8900``. Use
``--inference.tensor-parallel-size 2`` for two visible GPUs; the default is one.
``--inference.backend sglang`` makes the default backend explicit. The
service interpreter (``REEF_PYTHON``, otherwise the launcher's interpreter)
must have SGLang and GPU-enabled PyTorch installed. SGLang validates its GPU environment,
model compatibility and available device memory during startup. Its output
is available in ``.reef/run/sglang.log``. The managed path currently supports a single GPU node and SGLang.
Use the `SGLang installation guide <https://docs.sglang.io/docs/get_started/install>`__
to prepare the inference environment; the base Reef installation stays CPU-only.

Local model paths and Hugging Face IDs use the existing model resolver.
Reef resolves a downloaded snapshot once and preserves the original model
identifier as SGLang's served model name. Startup has a one-hour readiness
deadline for SGLang and 30 seconds for Reef. On failure or interruption,
Reef cleans up both processes. Logs live under ``.reef/run/``.
Local ``--inference.model-path`` cannot be combined with upstream URL/model selection;
``--model`` remains provider shorthand. Native engine options use ``inference.options`` as described below. Training
still requires an explicit stack file.

An external-provider deployment also needs no YAML file:

.. code:: bash

   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

Reef starts its core record-only recipe, listens on ``127.0.0.1:8900``, and
stores state under ``.reef/`` in the launch directory. It records inference
and feedback without training weights. Use ``--reef.host`` or ``--reef.port`` to change
the bind address. Logs live under ``.reef/run/``. Reef checks its own HTTP
readiness, runs in the foreground, and cleans up its process on Ctrl-C;
it does not launch or stop the upstream provider. Readiness does not verify
provider credentials or model availability.

``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and
``REEF_TOKEN`` supply optional environment fallbacks for this mode. Explicit
CLI settings win. ``--model ollama/my-model`` fills the Ollama endpoint and
model; ``--model openai/my-model`` uses ``REEF_UPSTREAM_API_KEY``. A model ID
with any other prefix still needs an upstream URL.

Versioned configuration layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical CLI paths match the sections of a ``schema-version: 2`` file.
For example, the local inference command above can also use:

.. code:: yaml

   schema-version: 2
   reef:
     host: 127.0.0.1
     port: 8900
   inference:
     backend: sglang
     model-path: Qwen/Qwen2.5-1.5B-Instruct
     tensor-parallel-size: 1
     options:
       mem-fraction-static: 0.8
   recipe:
     implementation: recipe

Run it with ``reef serve -c reef.yaml --inference.options.mem-fraction-static 0.7``.
An explicit CLI value overrides the corresponding YAML value. Omitted public
fields use their declared defaults. Hyphens and underscores are accepted in
declared YAML fields; supplying both spellings of one field is an error.
Unknown sections and undeclared public fields are rejected.

The public layout groups fields by their owner:

* ``reef``: HTTP host, port, authentication, run directory and readiness deadline.
* ``inference``: model path, backend, TP size, provider connection and native options.
* ``recipe.implementation``: the selected recipe class or preset.
* ``recipe.config``: fields declared by that recipe.
* ``recipe.runtime``: the runtime type and its declared settings.
* ``training``: backend selection, bridge startup/request timeouts, Ray connection and native Slime ``options``.
* ``storage``: artifact repository, work/cache directories and record retention.
* ``execution`` / ``executors``: role placement and named executor profiles.
* ``evaluation`` / ``observability``: their existing component-owned settings.

Version 2 contains no process definitions. ``service`` and ``services`` are
rejected; HTTP settings belong in ``reef``. The selected Recipe and inference
or training backend determine the processes, dependencies and connections.
``training.config`` holds workload variables such as checkpoint directories;
native backend flags belong in ``training.options``.

With the default Slime backend, Reef starts a local driver, waits for its healthy
bridge, then starts HTTP and obtains the inference connection from that bridge.
Slime owns the SGLang model workers; Reef does not start a second inference
server. Both processes share the Ray address, namespace, actor name and resolved
model path. With no Ray address, Reef owns the shared runtime and stops it on
exit; an existing cluster is left running. Model topology, optimizer settings
and checkpoint paths still need the complete options for the selected recipe.

The same path supports CLI-only training with
``--recipe.implementation package.module:WeightRecipe``,
``--inference.model-path`` and the corresponding ``--training.options.*`` flags.
``training.backend`` defaults to ``slime`` for compatibility. It also accepts an
installed ``reef.training_backends`` entry-point name or an importable
``package.module:Deployment`` class. The selected definition owns the process
plan and HTTP runtime connection; other backends do not inherit Slime's Ray,
SGLang or native-argument requirements.

An in-process integration can use ``InProcessTrainingDeployment``: it starts
only Reef HTTP and constructs its registered training runtime inside that
process. ``training.options`` is parsed by that runtime factory, with CLI leaf
overrides taking precedence over YAML. Runtime type, model and shared timeout
settings cannot be overridden inside the options map. Such integrations reject
Ray, training/rollout executor and standalone inference-engine settings. MLX
support remains in `PR #325 <https://github.com/Human-Agent-Society/reef/pull/325>`__;
this extension contract alone does not install or implement MLX.
``training.ready-timeout`` controls bridge startup (default 3600 seconds);
``reef.ready-timeout`` controls HTTP startup (default 30 seconds).
Slime-owned inference uses ``training.options.sglang-*`` for native engine
settings; upstream provider settings and standalone ``inference.options`` cannot
be combined with it. Reef binds ``training.options.hf-checkpoint`` to
``inference.model-path``; an explicit value must agree. ``ready-file`` is managed
by Reef and cannot be supplied through native options.

Reef coordinates native inference and training, alongside its HTTP service.
PRM and user-simulation services are independently deployed by OpenClawRL;
Reef does not discover, launch, schedule, probe or stop them. The recipe consumes
``recipe.config.prm-url`` and ``recipe.config.prm-tokenizer-path``, with the same
CLI-over-YAML precedence as other recipe fields. Its client owns request timeouts
and error handling. The example's Docker Compose owns auxiliary model commands,
health checks and GPU allocation, with separate devices from Reef/Slime.

Managed engine launches use one generic builder. A backend definition supplies
its command template, public parameter bindings, reserved aliases and HTTP
health path. Adding an engine with this launch contract does not require a
backend-specific deploy module or a second process lifecycle implementation.
Currently only the SGLang definition is supplied.

The launcher translates public paths to the existing internal service and
recipe contracts before starting children. Config references such as
``${reef.port}`` and ``${inference.model-path}`` use the same field mapping.
Existing unversioned files retain their ``reef``, ``training`` and ``services``
layout and defaults, including the HTTP service's ``0.0.0.0`` bind address.
Version 2 defaults to loopback. Its ``reef`` section contains only declared
HTTP settings; legacy model/recipe fields move to their owning public sections.

Native backend options
~~~~~~~~~~~~~~~~~~~~~~

Public fields and backend-specific flags share the same CLI-over-YAML merge.
For managed SGLang, use:

.. code:: bash

   uv run reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8 \
     --inference.options.trust-remote-code true

These become native ``--mem-fraction-static=0.8`` and ``--trust-remote-code``
arguments to ``python -m sglang.launch_server``. SGLang owns their types,
defaults and validation. Reef does not duplicate the engine argument schema.
For argv-based engines such as Slime, ``true`` emits a switch and ``false``
or ``null`` omits it. In-process training passes values to its runtime parser;
``false`` stays false and null follows the declared field type. To disable
an engine feature enabled by default, use that engine's native disabling
flag. Lists supply multiple argument values; objects are passed as JSON.
Use native flag names without their leading ``--`` inside ``options``.

A field override preserves its YAML siblings; an explicit whole object,
such as ``--inference.options '{}'``, replaces the entire object. Model,
served-model name, TP size, bind address, authentication and unsupported
multi-node launch controls cannot be overridden through native options in
managed serving. Use public fields; explicit custom process definitions belong only to unversioned legacy deployments.

For Slime, a versioned training stack can contain:

.. code:: yaml

   training:
     options:
       lr: 0.000001
       use-critic: true

Override an individual native flag with
``reef serve -c training.yaml --training.options.lr 0.000002``. The normalized
options reach ``reef.service.slime_driver`` through the same effective config
as the HTTP child. The driver passes them through its existing recipe-specific
argument handling and Slime's native parser. Automatic training launches use
only this effective config and ignore an ambient ``SLIME_ARGS_FILE``. Explicit
unversioned ``services`` stacks retain ``SLIME_ARGS_FILE`` and driver command flags, which
take precedence over the options object; avoid specifying a flag in both places.

``inference.backend-config`` has a different owner: it configures Reef's
selected ``inference.backend-factory`` adapter, for example its tool parser.
It does not configure the managed SGLang process. Executor ``options`` and
recipe-owned option objects likewise stay with their selected components.

Explicit file selection and legacy compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relative config paths resolve from the directory where you run the command.
With no ``-c`` or explicit ``--recipe`` profile, Reef uses CLI inference inputs;
it does not discover ``REEF_CONFIG`` or ``./reef.yaml``. File deployments must
use ``reef serve -c reef.yaml`` or ``reef serve -c "$REEF_CONFIG"`` instead
of relying on the previous implicit discovery. ``REEF_CONFIG`` remains the
internal way the launcher passes effective settings to its HTTP child.
A selected missing or invalid file is an error, even when provider flags are
present. Reef reports the selected config source and does not search the Reef
installation for your config. From outside a checkout,
pass an absolute path to a cookbook config. Relative state paths still use
the launch directory. Unversioned legacy stacks can use ``services[].cwd``
to override a process working directory.

.. code:: yaml

   reef:
     host: 0.0.0.0
     port: 8900
     recipe: recipe
     token: ${REEF_TOKEN}
     upstream_url: ${REEF_UPSTREAM_URL}
     upstream_api_key: ${REEF_UPSTREAM_API_KEY}

   services:
     - name: reef
       command: ["${REEF_PYTHON}", "-m", "reef.service"]
       ready: curl -sf http://127.0.0.1:${reef.port}/healthz

Values interpolate from the environment with ``${VAR}`` and from the config
itself with ``${dotted.path}``. Use full public paths for command-line overrides,
including legacy files: ``--inference.model-path /models/demo`` or
``--training.config.checkpoint_dir /tmp/ckpt``. Legacy files default to
``/tmp/reef-stack/`` logs; version 2 uses ``.reef/run/``. Set ``reef.run-dir``
in version 2 (legacy ``run_dir``) to move them.

Configuration-free startup accepts public settings and the selected weight
recipe's declared fields, plus native inference or training options for the
selected launch mode. Unknown public flags are rejected. Declared recipe runtimes are constructed by the HTTP child. Method-specific
processes are prepared by the selected recipe's Python deployment hook.
The effective settings are handed to the child using a private temporary
config, removed when the launcher exits. No user YAML file is created.

Public service settings use the same argument parser for YAML and CLI values.
Their types, defaults, and help are declared on ``ServiceConfig``. Explicit
CLI values override YAML values; omitted values use the setting's default.
Run ``reef serve --help`` to see these options. Use ``--inference.upstream-model``
as the canonical spelling. Compatibility aliases include ``--upstream-model``,
``--upstream_model`` and ``--reef.upstream_model``. The last
explicit CLI spelling of a setting wins. ``--recipe`` still selects a launcher
profile; ``--recipe.implementation`` overrides the deployment's recipe setting.

String settings retain their text: ``--inference.upstream-model 00123`` remains ``00123``.
Numeric and boolean settings are parsed according to their declared type;
invalid values fail before model downloads or process startup. Booleans accept
an explicit value or a bare flag; a negative flag can disable a YAML setting:

.. code:: bash

   reef serve -c stack.yaml --reef.port 9000 --no-reef.allow-implicit-scenario-creation

List and object options take one quoted JSON/YAML value. Empty lists and
objects are preserved, and an explicit container replaces the YAML value:

.. code:: bash

   reef serve -c stack.yaml --reef.tokens '[]' \
     --inference.backend-config '{"tool_call_parser": "qwen25"}'

The parsed public values are also supplied to service commands and the HTTP
child's config. Existing empty/null service values retain their defaulting
behavior. ``reef.token`` and ``reef.tokens`` remain distinct inputs whose
credentials are combined. The selected Recipe, runtime adapter and executor
settings use the same field parser. Only undeclared custom-stack mappings
retain generic YAML coercion; the ``services`` layout is unchanged.

Component configuration
~~~~~~~~~~~~~~~~~~~~~~~

Shipped examples and profiles use version 2, including standalone recipe files
loaded by embedding scripts. ``recipe_config_from_mapping`` / ``load_recipe_config``
translate their public envelope into the existing recipe construction contract.
Recipes may declare opaque sections in ``config_sections``; for example, Cordis
owns ``recipe.config.evolution``. Those sections use object/leaf CLI overrides,
then their recipe validates the payload. Executor placement remains shared with
the deployment and reaches dotted recipes as well as named presets.

After selecting ``recipe.implementation`` (legacy ``reef.recipe``), Reef loads that class's declarations without
constructing the recipe. A dotted weight-training recipe exposes its fields
as ``--recipe.config.batch-size``, with legacy aliases
``--batch-size`` / ``--batch_size`` / ``--reef.batch_size``. Other dotted
recipes use the same ``--recipe.config.*`` namespace; their internal ``data``
section and ``--reef.data.*`` spellings remain compatibility details. ``reef serve -c stack.yaml --help`` includes the
selected component's flags; basic ``reef serve --help`` does not load a recipe.
The selected package must be importable in the launcher and child environments.
When a profile file also declares its recipe ``implementation``, its fields
use the top-level ``--data.<field>`` and ``--runtime.<field>`` paths. The HTTP
child receives that merged preset instead of reloading the original file.

Declared component fields follow **CLI > YAML > declared environment fallback
> dataclass default**. False, zero, empty strings and empty containers remain
explicit values. Strings are not guessed as YAML scalars. The last CLI alias
wins, and boolean fields support ``--no-...``. A declaration that conflicts
with a public option is rejected. The normalized values retain their types
when handed to the HTTP child.

For compatibility, an empty/null flat weight-recipe key remains omitted.
Structured component fields accept null only if their annotation is optional;
the literal string ``null`` remains text for string fields. Recipe floats
retain their historical non-finite support; a recipe can declare
``allow_nonfinite=False`` to require finite values. Service fields and backend
resource/timeouts require finite values. Errors identify the field and expected
type without echoing its value.

Executor fields are declared by ``ExecutorSettings`` and ``WorkerResources``:

.. code:: bash

   reef serve -c stack.yaml \
     --execution.evolution.workers 4 \
     --execution.evolution.resources.cpus-per-worker 0.5

A nested override of a named executor profile makes a local copy for that
role; it does not modify the shared profile. Backend ``options`` objects
remain owned by the selected backend. Slime's native model/optimizer flags
continue through Slime's own parser; Reef does not duplicate that schema.

A non-weight dotted recipe can supply ``recipe.runtime`` to select a registered
or dotted runtime factory. Its declared fields use names such as
``--recipe.runtime.timeout-s``. Inference proxy, Ray training, and executor
training adapters declare their connection settings. Custom ``RuntimeFactory``
implementations can opt in with ``config_type()``; legacy callable factories
keep receiving their existing mapping. Unknown fields in a declared component
schema fail validation. Recipe-specific sections and opaque adapter option
objects continue to be validated by their owning component.

If a service exits before readiness or exceeds its ``ready_timeout``, Reef
stops the stack and reports the service, the failure reason, and the local
log directory. An exited service's message includes its exit code. CLI
startup failures exit with status 1; invalid configuration exits with status
2. Child output is retained in the logs and forwarded to the terminal.
Interrupting startup with SIGINT or SIGTERM also stops the services already
launched, including when they are still loading a model.

The basic and SAO example ``run.sh`` launchers wait for the HTTP health
endpoint and stop waiting if Reef exits. Startup deadlines are configured
through ``reef.ready-timeout``; the scripts do not add a
second deadline. Each HTTP probe has a five-second timeout. Startup errors
are recorded in ``work/reef.log``, and exiting the script stops the Reef
process it started.

Use ``${VAR:?}`` for a required environment variable, for example
``upstream_model: ${REEF_UPSTREAM_MODEL:?}``. If it is unset, empty, or only
whitespace, Reef reports the missing variable names and their config fields
before downloading models or starting processes. Command-line overrides are
applied before this check, so ``--inference.upstream-model <model-id>`` can supply the
value instead. Plain ``${VAR}`` keeps resolving to an empty string when unset;
use it for optional values such as an API key for a provider without authentication.

``REEF_PYTHON`` defaults to the interpreter that launched ``reef serve`` and
can be overridden in the environment. Use it when a service must share Reef's
Python environment. A literal ``python`` keeps its normal meaning and is
resolved from that service's ``PATH``; Reef never rewrites command names.

Scenario-specific model settings are supplied through the existing scenario
create/update API. See `Scenario model configuration <../user-guide/scenario-models.rst>`__
for model selection, persistence and platform upgrades. Omitting the model
setting preserves deployment-wide configuration.

Start from a cookbook stack
---------------------------

The source checkout's runnable stacks live under the ``recipes/`` cookbook
and ``tutorials/``: the learn-nothing ones use the core ``recipe``
implementation in ``recipes/basic/``, each weight-training method owns its
examples, and harness evolution ships as a tutorial.

+-------------------------------------------------------------+----------------------------------------------------------+
| File                                                        | What it starts                                           |
+=============================================================+==========================================================+
| ``recipes/basic/external-provider.yaml``                    | no GPU, no local model: one Reef process proxying an     |
|                                                             | HTTP provider                                            |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/basic/local-sglang.yaml``                         | local inference: an SGLang server plus Reef, no training |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/<method>/examples/<example>/serve.yaml``          | weight training: Ray head, Slime driver, Reef, and the   |
|                                                             | method's own services                                    |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``tutorials/evolve-your-harness/configs/serve.yaml``        | harness evolution: one Reef process, no GPU; ``run.sh``  |
|                                                             | materializes its recipe preset and starts the stack      |
+-------------------------------------------------------------+----------------------------------------------------------+

Each weight-training example ships its stack as ``serve.yaml``.
``recipes/sao/examples/sao/serve.yaml`` is the smallest, two GPUs for one
actor and one rollout engine; ``recipes/tttd/examples/tttd/serve.yaml`` adds
LoRA training, and ``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds
a PRM engine and a student model.

Legacy ``reef`` section
----------------------

The fields below describe the unversioned compatibility contract. New files
use ``recipe.implementation``, ``reef``, ``inference`` and ``storage`` as
shown above; the repository examples all use version 2.

.. config::

   reef.recipe | the recipe this deployment serves. Required.
   reef.host | 0.0.0.0 | bind address
   reef.port | 8900 | bind port
   reef.console_origins | [] | exact browser console origins allowed to access the HTTP service; disabled by default
   reef.token | the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.
   reef.model_path | a local HF model directory or a repo id, downloaded on start
   reef.upstream_url | the OpenAI-compatible provider, with no ``/v1`` suffix
   reef.upstream_api_key | its credential. Reef is the only party that sees it.
   reef.upstream_model | the model to request upstream
   reef.upstream_api | openai | the provider dialect: ``openai`` for Chat Completions, ``responses`` for OpenAI Responses, or ``anthropic`` for an Anthropic-style endpoint
   reef.inference_url | the address the training backend reports | the local engine; set only to front the engines with something else
   reef.inference_timeout_s | 300.0 | per-request timeout
   reef.allow_implicit_scenario_creation | true | when false, an unknown scenario is HTTP 404
   reef.checkpoint_every_n_versions | 1 | how often a version becomes durable

Storage paths default under ``.reef/``, which the basic and sao stacks keep;
the openclawrl stack overrides them to ``/var/lib/reef``. Point them somewhere
persistent.

.. config::

   reef.artifact_repository | .reef/artifacts.git | the Git-backed release chain
   reef.artifact_work_dir | .reef/artifact-work | materialization scratch
   reef.artifact_cache_dir | .reef/artifact-cache | fetched artifact cache
   reef.agent_record_dir | .reef/agent-record | the record store
   reef.agent_record_retention_days | 7.0 | compacted trace bodies expire this many days after compaction
   reef.agent_record_retention_max_bytes | 21474836480 | 20 GiB shared across compacted trace bodies in the record directory

.. warning::

   On ephemeral storage, a restart loses the record store, the commit logs, and
   every version.

The record store keeps trace bodies after training compaction. Compaction marks
records as retired from training; it does not remove their requests, responses,
or feedback from SQLite immediately. The HTTP service starts retention cleanup
at startup and repeats it every 60 seconds, outside the inference and training
request paths. It first removes bodies older than 7 days, then the oldest
remaining bodies until their total fits within 20 GiB. Both limits are
configurable above and must be positive; the time limit must also be finite.

The byte budget counts UTF-8 JSON payloads, references, and artifact references
across all scenario databases, including databases under ``archived/``. It is
shared across the directory, not allocated separately to each scenario. Deletes
commit in batches of 256. Active records, retry hashes, and commit/receipt
metadata are retained. Cleanup failures are logged and retried on the next sweep.

This is a retained-body budget, not a hard disk quota. Incoming compaction can
exceed the budget between sweeps; active records, indexes, hashes, commit logs,
and WAL files take additional space. SQLite reuses pages freed by cleanup but
does not automatically shrink the database file. Allow additional disk headroom.
Standalone Python stores do not start a maintenance task; see `Python API <python-api.rst>`__
for explicit retention and purge methods.

Existing stores gain ``compacted_at`` and ``body_bytes`` columns when opened.
The migration measures retained JSON byte sizes once. Already
deleted bodies cannot be recovered by this migration. Older Reef versions do
not filter that column: stop the service and restore a pre-upgrade backup for
rollback, or purge all compacted bodies with the new version before downgrading.
Do not share a migrated store between old and new writers.

Recipe settings such as ``batch_size`` sit beside these in the same section,
along with any others the recipe declares with ``config_field``. When
``reef.recipe`` is a dotted weight-training class, keys the service does not
recognize are handed to the recipe, and the recipe rejects any key it does
not declare. With the core ``recipe`` or a named preset, the recipe reads its
configuration from the preset, and unrecognized keys here are silently
ignored.

Recipe configuration
--------------------

Version 2 puts declared recipe fields under ``recipe.config`` and the runtime
under ``recipe.runtime``. Harness settings live in ``recipe.config.evolution``.
The ``data``, ``evolution`` and ``runtime`` paths below also name the existing
Python recipe contract and remain accepted by legacy presets.

``data.training_mode`` is shared by all recipes and defaults to ``auto``.
In ``auto``, the recipe's processor decides when its data can form a batch,
and ``POST /reef/train`` is refused. In ``manual``, inference and reports
cannot authorize training by themselves; ``POST /reef/train`` supplies the
user instruction, and harness evolution runs it alone. In ``hybrid``, the
processor batches as in ``auto`` and runs instructions too, a queued
instruction first; harness evolution hands the proposer, beside the
instruction, the units an automatic batch would take next, up to
``batch_size`` and possibly none: failing traces in the score window, or
records under ``data.batch_policy: records``. The processor defines
what an instruction batch carries, independently of its automatic batching
policy.
For a dotted weight-training deployment this field is also accepted as
``reef.training_mode``. Named presets set it in their own ``data`` section.

The processor receives ``ProcessorContext.training_mode`` as its initial
batching mode. The modes share ingestion and retention; the
``make_training_batch(batch_number, request)`` hook selects batch inputs.
Processors declare ``supported_training_modes``; unsupported modes or missing
instruction assembly raise ``NotImplementedError``.
Harness evolution supports the three modes and requires a proposer that
explicitly accepts ``requests`` for ``manual`` and ``hybrid``. Setting either
on an inference-only recipe does not create a training backend.

.. code:: yaml

   data:
     training_mode: hybrid

The mode controls training initiation, independently of
``evolution.publish: auto | review``. It supplies the initial processor mode.
Use ``POST /reef/scenarios/{scenario}/update`` to select another mode
at runtime. This changes subsequent batches; a reserved batch completes under
its original mode. Mode changes are not persisted: a service restart uses the
recipe's configured mode again, while a scenario reload after a failed step
keeps the selected mode. In ``manual`` the reported-feedback processor holds
at most four batches of units and releases the oldest beyond that with a
warning, at the switch to ``manual`` and as reports arrive.

.. config::

   data.training_mode | auto | ``manual`` waits for ``POST /reef/train`` instructions instead of batching by the recipe's rules; ``hybrid`` batches by the recipe's rules and runs a queued instruction first

A recipe is selected three ways:

- **The core record-only recipe:** ``recipe: recipe``
- **A dotted class:** ``recipe: "my_pkg.my_method:MyMethodRecipe"``
- **A named preset:** ``recipe: my-preset``, resolved to ``my-preset.yaml``
  under ``REEF_RECIPE_CONFIG_DIR``

There is no recipe-implementation registry. A bare name other than ``recipe``
is always a preset name; it never imports a learning method implicitly.

A dotted class does not have to be pip-installed. ``reef serve`` looks for
its top-level package beside the config, walking up from the config file's
directory to the nearest ancestor that holds ``<package>/__init__.py``, and
appends that directory to ``PYTHONPATH`` for every service it starts. This is
how ``recipes.sao.recipe:SAORecipe`` resolves from a source checkout: the
``recipes/`` cookbook sits next to the example's ``serve.yaml``, so the
launcher does not export ``PYTHONPATH`` itself. Entries already in
``PYTHONPATH`` keep their precedence. Python process definitions and legacy
stacks can set a process environment explicitly.

``REEF_RECIPE_CONFIG_DIR`` is the directory preset YAML is read from, and it has
**no default**: a bare recipe name resolves to a preset only when it is set.
The one kind of preset reef bundles is a recipe's profile under
``reef/service/profiles/``: ``reef serve --recipe <name>`` points this
variable at that directory and reads the profile as both the deployment
config and the preset (see `the CLI reference <cli.rst>`__).

A preset is read as-is. ``${VAR}`` interpolates in a deployment config, never
in a preset. A preset carries its own ``implementation``, ``model``, and
``data`` sections, plus an optional ``runtime`` section when the recipe
builds its own runtime instead of using the deployment's upstream proxy.
When using the deployment's runtime, a preset may omit ``model.path`` to
inherit that runtime's model (``reef.upstream_model`` for an upstream proxy).
An explicit ``model.path`` takes precedence. A preset with its own ``runtime``
section must still supply its own non-empty ``model.path``.
Harness-evolution presets also carry an ``evolution`` section:

.. code:: yaml

   implementation: reef.recipe.cordis:CordisRecipe
   model:
     path: qwen3-8b
   data:
     batch_size: 1
     max_score: 0.0
   evolution:
     adapter: pi
     propose: methods.mine:propose
     evaluate: methods.mine:evaluate
     tasks: ["..."]

The preset's ``implementation`` is ``recipe`` or a dotted recipe class. Weight-training
recipes are selected directly by dotted class in the deployment config, so the
service can assemble their Ray training runtime; their fields are flat
``reef.<name>`` keys. Presets suit recipes whose runtime can be built from the
preset or the deployment's upstream proxy. There, ``data`` holds batching
fields and a recipe-specific section holds the rest.

A preset's ``runtime.type: executor_training`` selects an executor-backed
training coordinator. Its ``executor`` mapping accepts ``backend`` (default
``auto``, resolving to ``uni``, ``mp`` or ``ray``, or a custom executor import path), ordered ``workers`` and backend
``options``. ``coordinator_rank`` defaults to zero. Existing ``ray_training``
configuration still connects to a named bridge. The Slime driver's separate
``--reef-executor-backend`` selects the model worker executor and defaults to
``auto`` (currently Slime's Ray launcher). See `Worker executors <../developer-guide/executors.rst>`__ for the full
contracts and examples.

Stack execution backends
------------------------

Version 2 chooses native process placement in backend implementation code;
``execution.services`` is rejected. Unversioned custom stacks retain
``execution.services`` and per-process ``services[].executor`` overrides.
``execution.training`` and ``execution.rollout``
select the Slime training-worker and rollout-control executors (both default
``auto``, resolving to ``ray``). All selectors accept a backend name/import path or a profile under
``executors``. Inline objects/profiles accept ``backend`` (default ``auto``),
``options``, ``workers``, and ``resources`` containing ``cpus_per_worker`` and
``gpus_per_worker``. Counts must be positive integers; resource quantities must
be finite nonnegative numbers. Numeric environment interpolation is accepted.

``execution.evolution`` selects harness evaluation workers (default ``auto``).
Unlike Slime, ordinary harness evolution calls external model endpoints and
does not need local GPUs: one worker selects ``uni``, multiple workers select
``mp``. Local GPU worker requirements follow the same topology rule after
checking visible CUDA capacity; insufficient GPUs fail with a prompt to
explicitly select ``ray``. Multiple workers inside a Ray placement group or
declared cluster/actor options select ``ray``. Local allocations require whole
GPUs; fractional reservations require explicit Ray. The former worker-level
``local`` executor is removed (the separate episode-isolation option is unchanged).
Omitted resources retain component defaults (normally one CPU and no GPUs).
CPU-only local executors do not reserve cores or enforce CPU quotas.

Service executors accept the same resources, mapped to per-service launch
requests, but ``workers`` must be one: service replicas are not implemented.
Slime training/rollout reject these generic workers/resources fields; their
model-parallel topology and placement groups still come from Slime's existing
training configuration. Resource declarations are never silently treated as
model-parallel resizing.

For services, ``auto`` selects ``uni`` unless resource/worker options
or an existing Ray placement group call for Ray. Explicit local CUDA visibility
selects ``uni`` and cannot be combined with cluster resource options.
Explicit service selectors and Slime CLI flags override the corresponding role
defaults. Backend startup failures never silently fall back to another backend.

For Ray services, ``resources`` supplies actor launch options such as
``num_gpus`` and ``num_cpus``; do not also set ``cuda``. A service can publish
``endpoint: http://{host}:23001`` and dependents can use
``${endpoints.SERVICE_NAME}``. Readiness runs on the service's execution node,
and dependencies are topologically sorted. Local services advertise localhost
unless ``advertise_host`` is set. See the `whole-stack examples and backend
contracts <../developer-guide/executors.rst#whole-stack-deployment-configuration>`__
before moving services across nodes.

Harness evolution keys
~~~~~~~~~~~~~~~~~~~~~~

``batch_size`` and ``max_score`` go under ``data:``; the rest goes under
``evolution:``. `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ describes what each one changes.

.. config::

   data.batch_size | 1 | traces per mutation attempt
   data.max_score | 0.0 | upper bound of the score window that batches
   data.batch_policy | reports | ``records`` batches recorded traffic alone, every ``batch_size`` requests, with unscored samples

The window has no lower bound, so the default keeps only traces at or below
zero.

.. config::

   evolution.propose | a ``Proposer``, a plain callable, or a dotted ``module:attribute``
   evolution.evaluate | an ``EpisodeScorer``, likewise
   evolution.selection | score_comparison | ``always``, or a dotted reference to an object with ``decide``
   evolution.tasks | non-empty list of episode prompts, scored once per tree per step
   evolution.adapter | pi | ``opencode``, ``claude``, ``codex``, ``dsh`` (DeepSeek Harness), ``hermes`` (Hermes Agent), ``native`` (Reef's own agent, whose tools are ``native_tool`` nodes, whose loop events listen to ``native_hook`` nodes, and whose loop is a ``native_graph`` node or, as code, a ``native_loop`` node), ``terminus`` (Terminal-Bench's Terminus 2, through a Reef-owned Harbor runner), or an entry-point adapter
   evolution.binary | a path to the harness binary; unset, backend construction installs the adapter's pinned version through the vendor's channel under ``$REEF_HARNESS_PREFIX`` (default ``~/.local/share/reef-harness``)
   evolution.episode_timeout_s | 600 | seconds one evaluation episode may run
   evolution.episode_repeats | 1 | episode pairings per task per step; each repeat tallies on its own
   evolution.forbid_residue | false | when true, an episode leaving files outside the cleanup whitelist scores as one that could not run
   evolution.max_steps | 0 | stop automatic evolve steps once this many steps ran, instruction steps included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs past it
   evolution.max_failure_streak | 0 | stop automatic evolve steps after this many consecutive rejected steps, instruction steps included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs while the breaker is open
   evolution.max_model_calls_per_step | 0 | cap the proposer's model calls in one step; 0 disables the limit
   evolution.executor | local | ``local`` runs episodes as a plain subprocess (development, hermetic tests); ``sandbox`` runs each in a bubblewrap jail for a hosted service and refuses to start without it; it also refuses every episode of a ``self_isolating`` adapter such as ``terminus``, whose Docker task container cannot nest in the jail
   execution.evolution.workers | 1 | fixed worker-group size; CPU auto selects ``uni`` for one and ``mp`` for multiple
   execution.evolution.backend | auto | worker placement, independent of the ``local/sandbox`` episode isolation policy; ``local`` retains shared-memory callbacks
   execution.evolution.resources | | ``cpus_per_worker`` and ``gpus_per_worker``; omitted values retain component defaults; GPUs select Ray under ``auto`` and cannot reduce declared GPU needs
   evolution.episode_workers / worker_executor / worker_resources | | deprecated compatibility aliases; conflicting resource values are rejected; legacy worker_executor cannot accompany role-level workers/resources
   evolution.sandbox | | the sandbox executor's policy: ``egress_hosts`` (allowlisted model endpoints; empty denies network) and ``limits`` (``cpu_seconds``, ``memory_bytes``, ``processes``, ``file_bytes``)
   evolution.promote_failures | false | when true, a failing trace's prompt becomes a permanent gate task, so no later candidate can win while bringing the failure back; the seed tasks stay the floor
   evolution.max_promoted_tasks | 50 | the cap on promoted tasks; admission stops there so the suite is bounded
   evolution.max_promoted_per_client | 5 | the cap on promoted tasks from one tagged client (its ``x-reef-tag-client``, else session, tag); untagged traffic has no identity to count under and meets only ``max_promoted_tasks``; 0 disables the cap
   evolution.promote | | optional callable or dotted ``module:attribute`` choosing which trace prompts to promote; receives the step's samples (and the failure manifest when its signature names ``manifest``); without it every failing trace's user prompt is promoted, and the caps and the credential and directive screens still apply
   evolution.publish | auto | ``review`` holds every gate win as a pending release until ``POST /reef/scenarios/{scenario}/promote`` names it
   evolution.review_kinds | [] | node kinds whose wins wait for a promote while the rest publish at once; a win that touches a ``native_loop`` waits whether or not the list names it
   evolution.seed | entry options loaded into the tree on first boot, or a dotted ``module:attribute`` naming a sequence of them (``reef.harness.runners.native.seed:SEED_NODES`` is the native harness's shipped tools and hook); recovered state takes precedence
   evolution.models | auxiliary models for the method: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name
   evolution.version_check | appends the adapter's update notice; an interactive pulled tree offers to run the update or skip when behind
   evolution.requests | false | appends the adapter's harness requests extension and its extension API skill after the notice (the reserved entries ``reef-requests`` and ``reef-pi-extension-api``), so a ``reef-pi`` session gets ``/reef-harness <request>`` in the TUI (submits to ``POST /reef/train``, which needs ``data.training_mode: hybrid`` or ``manual``) and the method reads the API reference before it writes an extension; ``pi`` only, other adapters refuse boot (a seed entry: a deployment that boots from a recovered state keeps its tree, as with ``version_check``); the tutorial's ``tutorials/evolve-your-harness/configs/deployment.yaml`` sets it, with ``version_check: true`` and ``review_kinds: [code_extension]``
   evolution.proposals_dir | .reef/proposals | where agent proposals from ``POST /reef/harness/proposals`` wait for the next evolve step: one directory per scenario under it (``<dir>/<scenario>``, made absolute at build, created when the first proposal arrives), with ``claimed/``, ``refused/`` and ``settled/`` beside the pending files
   evolution.max_pending_proposals | 8 | how many admitted proposals one scenario holds; the route answers ``admitted: false`` with reason ``inbox full`` beyond it, and with reason ``manual mode takes instructions only`` on a scenario in ``data.training_mode: manual``
   evolution.step_record_dir | | off by default; when set, every step writes its record under ``<dir>/<scenario>/<step>`` (the path is made absolute at build): ``proposer.json`` (each model call the proposer made: ``model``, ``messages`` and ``params`` for a ``chat`` or ``body`` for a ``complete``, then ``reply`` and the provider ``response`` for a built-in ``chat`` binding, ``response`` for ``complete``, or ``error``, and ``seconds``; the response retains provider reasoning/thinking fields when returned; long text is clipped with a marker and a credential shaped literal is replaced by ``[redacted credential]``), ``mutations.json`` (the parsed proposal with its full options, refused or not, redacted the same way) and ``episodes/<side>-<task index>/`` (each gate episode's trajectory files as the adapter writes them, copied out of its root before the root is removed, plus ``episode.json`` with the task, the exit code, stdout and stderr, the residue, the score, the failure and the stage path; a repeat adds ``-<repeat>``); a recheck step writes ``episodes/`` only and has no proposer files; a step skipped on the step cap or the failure streak writes nothing; a step directory is never reused, so a retried step lands in ``<step>-2``, then ``<step>-3``; nothing prunes the directory; an unwritable path refuses boot and a record copy that fails aborts the step instead of scoring it

The served model's binding is appended at render time; it never enters the
published files. The seed defines the baseline the first mutation is measured
against. The step record holds the proposer's raw traffic and every gate
episode's session log, so treat its directory like the commit log.

Legacy process definitions
--------------------------

The following fields apply only to unversioned legacy files. Version 2 rejects
``services`` and automatically assembles the selected components. Move custom
process dependencies into the owning recipe package, or use an external
deployment tool for infrastructure orchestration.

Each entry is one process. ``command`` can be a command-line string or an argv
list. Prefer the list form when exact argument boundaries matter; existing
string commands retain their current ``shlex`` parsing.

.. config::

   services[].name | the service's id, used by ``depends_on``; unique within one stack
   services[].command | the command line string or argv list to run
   services[].ready | a shell command or argument list that succeeds once the service is up; lists run without a shell
   services[].ready_timeout | seconds to wait for ``ready`` before giving up; the top-level ``ready_timeout`` sets the default
   services[].depends_on | services that must be ready first
   services[].cuda | optional ``CUDA_VISIBLE_DEVICES`` for local services; Ray services must declare ``resources.num_gpus`` instead
   services[].env | extra environment variables

The ``training`` section
------------------------

Read by the weight-training stack. See `Evolve your model
<../user-guide/evolve-your-model.rst>`__ for how to size it.

.. config::

   training.backend | slime | built-in, installed entry-point name, or dotted TrainingDeployment class
   training.ready-timeout | 3600 | backend-owned component startup deadline; in-process model loading is covered by reef.ready-timeout
   training.config.num_gpus | example-specific GPU count passed to Slime's model topology flags; does not reserve GPUs for the driver or set the Ray cluster's capacity
   training.config.global_batch_size | samples in one optimizer step. Must equal the recipe's ``batch_size``.
   training.config.checkpoint_dir | where Megatron and HF checkpoints are written
   training.config.megatron_checkpoint_path | optional pre-converted torch_dist checkpoint, to skip HF conversion on every start
   training.config.checkpoint_retention | storage-fraction bounds and the retention policy
   training.options | native backend flags as a mapping: GPU layout, optimizer, sequence length, and loss settings

Slime fills architecture flags such as layer counts and hidden sizes from
``inference.model-path``. Do not put them in the config.

The ``evaluation`` section
--------------------------

Only weight-training recipes read this section; a deployment that pairs it
with any other recipe fails at startup, because a harness recipe builds its
evaluator in code. Absent by default, in which case a successful
weight-training step publishes without a gate. When present, Reef calls the
named factory once per scenario and hands the plugin the exported but
unpublished checkpoint.

.. config::

   evaluation.module | a ``package.module:factory`` reference to the plugin factory. Required.
   evaluation.config | opaque mapping handed to the factory; Reef never reads it

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:build_evaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

The plugin interface is in `Write a recipe
<../developer-guide/write-a-recipe.rst#gate-a-candidate>`__.

Experiment tracking
-------------------

Tracking is optional, off by default, and belongs to a Reef *scenario* rather
than to one training backend. The same provider-neutral logger is shared by the
recipe, the processor, backend results, and the commit lifecycle. Install
``reef[wandb]`` when the training extra does not already provide it.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team             # optional
       group_prefix: prod-us-east    # optional scenario-group namespace
       name_prefix: baseline         # optional run-name prefix
       tags: [openclawrl, qwen]
       mode: online                  # online, offline, or disabled
       directory: /var/lib/reef/wandb
       upload_checkpoints: false

Export ``WANDB_API_KEY`` before starting, or log in once with the credential
store on the cluster.

.. warning::

   There is no API-key field here. Reef rejects Slime's ``--wandb-key`` flag and
   never writes a credential into metrics or run config. Do not put one in the
   YAML, in ``training.options``, in a tag, or in a run name.

``online`` sends data to the project. ``offline`` makes no network calls and
writes syncable data below ``directory`` for a later ``wandb sync``.
``disabled`` makes no calls even when ``enabled`` is true.

Each scenario maps to a group named after the scenario or
``<group_prefix>/<scenario>``. Within it, Reef opens one run when the scenario
binds and another after each rollback. The deterministic run id includes those
identities, so restarting resumes the same run with ``resume=allow``. A rollback
finishes the current run, marks its summary with the source and target, and
resets ``train/step`` to zero; the globally monotonic ``reef/step`` stays
attached for joining a run back to the commit log.

Recipe and processor code logs through the same object without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own
``<namespace>/event`` axis. Only finite numeric values are sent.

Durable commit metrics carry ``experiment/provider``, ``experiment/project``,
``experiment/group``, and ``experiment/run_id``. Use them to open the run from
a Reef version, and use the run's ``reef/training_job_id`` to go the other way.
Checkpoint paths are metadata only unless ``upload_checkpoints: true``.

Import, initialization, logging, summary, and upload failures are reported in
the service log and never fail a training step or its commit.
