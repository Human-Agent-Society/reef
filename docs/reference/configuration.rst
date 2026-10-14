Configure Reef serving and training
===================================

Local inference needs no YAML file:

.. code:: bash

   uv run reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct

This starts SGLang on an automatically selected loopback port, waits for its
health endpoint, then starts Reef on ``127.0.0.1:8900``. Use
``--inference.tensor-parallel-size 2`` for two visible GPUs; the default is one.
``--inference.backend sglang`` makes the default backend explicit;
``--inference.backend vllm`` starts ``vllm.entrypoints.openai.api_server``
instead. The service interpreter (``REEF_PYTHON``, otherwise the launcher's
interpreter) must have the selected engine and GPU-enabled PyTorch installed.
The engine validates its GPU environment, model compatibility and available
device memory during startup. Its output is available in
``.reef/run/sglang.log`` or ``.reef/run/vllm.log``. The managed path currently
supports a single GPU node. Use the `SGLang installation guide
<https://docs.sglang.io/docs/get-started/install>`__ or the `vLLM installation
guide <https://docs.vllm.ai/en/latest/getting_started/installation/>`__ to
prepare the inference environment; the base Reef installation stays CPU-only.

Local model paths and Hugging Face IDs use the existing model resolver.
Reef resolves a downloaded snapshot once and preserves the original model
identifier as SGLang's served model name. Startup has a one-hour readiness
deadline for SGLang and 30 seconds for Reef. On failure or interruption,
Reef cleans up both processes. Logs live under ``.reef/run/``.
Local ``--inference.model-path`` cannot be combined with upstream URL/model selection;
``--model`` remains provider shorthand. Native engine options use ``inference.options`` as described below.

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

Configuration file layout
~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical CLI paths match the sections of a configuration file.
For example, the local inference command above can also use:

.. code:: yaml

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
* ``pipeline``: the stage classes that replace a weight-training recipe's own, with their config.
* ``generator``: the generator service that writes, checks and plays Harbor tasks.

A configuration file contains no process definitions. ``service`` and
``services`` are rejected; HTTP settings belong in ``reef``. The selected Recipe and inference
or training backend determine the processes, dependencies and connections.
``training.config`` holds workload variables such as checkpoint directories;
native training flags belong in ``training.options`` and engine flags in ``inference.options``.

With the default Slime backend, Reef starts a local driver, waits for its healthy
coordinator, then starts HTTP and obtains the inference connection from that coordinator.
For managed full-weight and LoRA training, including colocated configurations,
the backend-neutral Reef model driver owns separate
resource, inference and training components. Reef selects the two backend
definitions independently, starts each component, then attaches a weight-transfer
session. Training workers send directly to inference workers; batch processing
uses Slime's adapter inside Reef's coordinator. Publication, version verification,
LoRA residency and colocated memory handoffs live in ``reef.runtime``.
Native engine launch and control live in ``reef.inference.sglang``. Reef
reserves the model GPUs itself (``reef.runtime.executor.placement``): one
placement group per deployment, ordered by node and device, sliced for the
training and inference components; ``training.colocate`` gives both the same
bundles.
External-engine paths use the same component lifecycle while borrowing their
external engines; automatic cold-rebuild supervision remains disabled for them.
HTTP and the driver share the Ray address, namespace, actor name and resolved
model path. With no Ray address, Reef owns the shared runtime and stops it on
exit; an existing cluster is left running. Model topology, optimizer settings
and checkpoint paths still need the complete options for the selected recipe.

The same path supports CLI-only training with
``--recipe.implementation package.module:WeightRecipe``,
``--inference.model-path`` and the corresponding ``--training.options.*`` flags.
``training.backend`` defaults to ``slime`` for compatibility. The optional
``tinker`` backend provides remote LoRA training and immutable sampling without
local GPUs, or trains behind Reef's coordinator for a local SGLang engine when
``inference.backend: sglang`` is selected; see `Train with Tinker <../user-guide/tinker.rst>`__. It also accepts an
installed ``reef.training_backends`` entry-point name or an importable
``package.module:Deployment`` class. The selected definition describes the process
plan and HTTP runtime connection; Reef owns the managed component lifecycle; other backends do not inherit Slime's Ray,
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

``training.timeout-s`` sets the training coordinator RPC wait in seconds. With
Slime, it also sets the internal worker RPC waits for training, checkpoint saves,
weight export and publication. For example, set ``training: {timeout-s: 43200}`` in
YAML or pass ``--training.timeout-s 43200`` to allow waits of up to 12 hours.
The value must be a positive finite number. If omitted, the coordinator wait
follows ``inference.timeout-s`` (default 300 seconds), while Slime's internal
worker waits retain their four-hour default. Worker initialization keeps its
existing timeout independently of this setting.

Each RPC has its own wait budget. The outer coordinator call still bounds the
combined training and checkpoint-save wait; giving each internal RPC 12 hours
does not allow those phases to take 12 hours each inside the same outer call.
A timeout stops waiting, not worker execution, and does not automatically retry
the job or save an intermediate checkpoint. A job left in ``RUNNING`` needs
operator recovery before it can be replayed; see
`Training-step coordination <../developer-guide/executors.rst#training-step-coordination>`__.

Slime-integrated inference uses the same ``inference`` fields as standalone
serving. ``inference.num-gpus`` is the total inference GPU budget;
``inference.tensor-parallel-size`` is the GPU count per engine (default 1).
The total defaults to the per-engine count and must be a positive multiple of
it. For example, 4 GPUs with tensor parallel size 2 creates two engines.
Standalone serving currently supports one engine, so its total must equal its
tensor parallel size. An external provider does not accept local GPU requests.

.. code:: yaml

   inference:
     model-path: Qwen/Qwen2.5-1.5B-Instruct
     num-gpus: 1
     tensor-parallel-size: 1
     options:
       mem-fraction-static: 0.6
       router-port: 30000  # Slime-integrated inference only
   training:
     backend: slime
     colocate: false  # true trains on the inference GPUs
     options:
       actor-num-nodes: 1
       actor-num-gpus-per-node: 1
       # Add the recipe's optimizer, model and checkpoint options here.

CLI overrides use the same parser, for example
``--inference.num-gpus 4 --inference.tensor-parallel-size 2`` or
``--inference.options.mem-fraction-static 0.7``. Native engine options use
SGLang's names without a ``sglang-`` prefix. The Slime integration translates
these only when constructing driver arguments; generated inference flags are
not stored in ``training.options``. Router bind settings use ``router-ip`` and
``router-port``; other supported router flags retain their native ``router-*``
names. The standalone engine launcher does not include a router.

In Slime-integrated inference, and for the local SGLang engines that the
Tinker backend starts, each engine probes free ports on its host at startup.
A probe reserves nothing, so another process, such as another stack on the
same host network, can bind a probed port first. If the engine then fails to
start and another process holds one of its ports, Reef stops the engine and
starts it again on new ports. Reef does the same when another process listens
on the engine's HTTP port and answers the readiness check. Reef makes at most
three startup attempts per engine. Other startup failures, and any failure of
an encoder engine, stop the launch.

When Reef starts a router, it probes a free metrics port on the host, from
port 4000 up. It also probes a free HTTP port, unless the router uses
``router-port``. Only the first model's router uses ``router-port``. The
router serves both ports on the host's address. If the router then fails to
start and another process holds one of these ports, Reef stops the router and
starts it again on new ports. Reef does the same when another process listens
on the router's HTTP port and answers the readiness check. Reef makes at most
three attempts. A configured ``router-port`` does not change, so a conflict on
it stops the launch. Another stack's metrics port can take a ``router-port``
near 4000, so use a port outside that range.

Encoder engines keep the weights they started with. Weight updates go only to
the other engines of the model. When a model with an encoder group trains,
keep its vision encoder frozen.

Managed launches reject inference flags in ``training.options``, such as
``rollout-num-gpus``, ``rollout-num-gpus-per-engine``, ``sglang-*`` and
``colocate``, even if their values agree with the ``inference`` fields and
``training.colocate``. Native options cannot override
managed model, placement or parallelism settings. Pipeline/data parallel and
prefill/decode-disaggregated inference topologies are not supported by this
managed path yet. Reef binds ``training.options.hf-checkpoint`` to
``inference.model-path``; an explicit value must agree. ``ready-file`` is managed
by Reef and cannot be supplied through native options.

The training-capable SGLang implementation lives in ``reef.inference.sglang``.
Its native engine launch and control do not depend on Slime. Slime converts its
training requirements to plain configuration data and supplies the weight transport;
the inference component receives ordinary configuration and borrowed GPU
reservations. Custom inference executors now receive ``config`` and ``pg``
instead of Slime's argument namespace. See `Worker executors
<../developer-guide/executors.rst#independent-sglang-backend>`__ for the boundary.

This continues `RFC #425 <https://github.com/Human-Agent-Society/reef/issues/425>`__.
Training GPU capacity remains in ``training.options.actor-num-*``; the shared
physical node size remains ``training.options.num-gpus-per-node``. Inference
and training share one allocation plan, with no duplicate model-GPU
reservations. Full-weight, LoRA and colocated training borrow Reef-owned
inference. ``training.colocate`` shares the GPU reservation between them; Reef
derives the native training and inference offload flags from it, and managed
launches reject their spellings in ``training.options``.
``training.options.keep-lora-base-resident`` retains the frozen inference base
during later colocated LoRA steps; cold startup still releases all inference
memory before training initializes. The separate inference control actor requires
one Ray CPU and zero GPUs. Batch processing runs locally in the training
coordinator, so no separate batch-manager CPU is reserved. The HTTP endpoint is
still discovered through the training bridge. Managed deployments, including
LoRA and colocated modes, automatically rebuild both components after failure,
rerun checkpoint recovery and rediscover the endpoint
without restarting the HTTP service. Explicit gateway URLs stay fixed. This
recovery does not replay ambiguous optimizer steps and stops if old resources
cannot be confirmed retired. A rejected Slime candidate also requires restoring
the committed checkpoint before restart; its training checkpoint must not be
used to reconstruct serving. See `Worker executors <../developer-guide/executors.rst>`__ for the
recovery policy and compatibility limits.

``reef.inference_backend: vllm`` selects the vLLM engine integration for
training-time serving. Reef launches one vLLM server per engine on the
reserved GPUs with ``VLLM_SERVER_DEV_MODE=1``, Reef's connector and
``--logprobs-mode processed_logprobs``, and drives publication through vLLM's
control routes: ``/pause?mode=keep`` (a retracting pause adds
``/reset_prefix_cache?reset_running_requests=true``), ``/sleep`` and
``/wake_up`` for colocated memory release, ``/update_weight_version``,
``/collective_rpc reload_weights`` and the ``/v1/load_lora_adapter`` routes.
Engines are single-node; more than one engine needs ``router_url``. Each
engine takes one port, probed upward from ``engine_port_base`` (default
15000) on its host. If the engine then fails to start and another process
holds that port, or another process listens on that port when the engine
becomes ready, Reef stops the engine and starts it again on a new port, with
at most three startup attempts per engine. Engine
options use vLLM's engine-argument names; Reef sets ``model``, ``host``,
``port``, ``tensor_parallel_size`` and ``enable_sleep_mode`` itself, defaults
``generation_config`` to ``vllm`` so the model's own generation defaults cannot
add truncation or a temperature the trainer never sees, rejects
``kv_offloading_size`` (list ``OffloadingConnector`` in ``kv_transfer_config``
instead) and enables prefix caching only under a retracting pause. vLLM
releases the KV cache only together with the weights, so
``keep-lora-base-resident`` is unavailable on it, and it resumes scheduling by
itself once every region is resident, so the engine restores the KV cache
when Reef resumes generation rather than when the coordinator calls
``onload_kv``: generation stays paused until the coordinator's commit.

The Slime backend pairs with ``inference.backend: vllm`` through the disk
weight path: ``training.options.update-weight-transport: disk`` in full mode,
with no ``update-weight-local-checkpoint-dir``, no LoRA rank and no
``check-weight-update-equal``, since Slime's other transports call SGLang-only
engine routes. The trainer writes each checkpoint under
``update-weight-disk-dir`` and the engines reload it with their weight version.
Engine options come from ``inference.options`` in vLLM's own names and never
pass through Slime's parser. One engine per stack: ``inference.num-gpus`` must
equal ``inference.tensor-parallel-size`` until Slime can name a ``router_url``.
The Tinker backend still produces SGLang engine options and keeps
``inference.backend: sglang``.

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
Reef supplies the SGLang and vLLM definitions; ``--inference.backend`` selects
one.

The launcher translates public paths to the existing internal service and
recipe contracts before starting children. Config references such as
``${reef.port}`` and ``${inference.model-path}`` use the same field mapping.
The HTTP service binds to loopback by default. The ``reef`` section contains
only declared HTTP settings; model and recipe fields belong to their owning
sections.

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
managed serving. Use the public fields for those settings.

For Slime, a training stack can contain:

.. code:: yaml

   training:
     options:
       lr: 0.000001
       use-critic: true

Override an individual native flag with
``reef serve -c training.yaml --training.options.lr 0.000002``. The normalized
options reach ``reef.service.training_driver`` through the same effective config
as the HTTP child. The driver passes them through its existing recipe-specific
argument handling and Slime's native parser. Automatic training launches use
only this effective config and ignore an ambient ``SLIME_ARGS_FILE``.

``inference.handler-config`` has a different owner: it configures Reef's
selected ``inference.handler-factory`` adapter, for example its tool parser.
The factory path must name an ``InferenceHandler`` subclass with a
``from_config`` class method; handler functions are not supported.
It does not configure the managed SGLang process. Executor ``options`` and
recipe-owned option objects likewise stay with their selected components.

Two token-native handlers record exact samples: the sampled token ids, their
rollout log probabilities and the weight version that produced each token.
``reef.inference.sglang.chat.SGLangInferenceHandler`` calls SGLang
``/generate`` and relays its incremental stream.
``reef.inference.vllm.chat.VLLMInferenceHandler`` calls vLLM
``/inference/v1/generate``; it serves buffered responses and answers a
streaming request with the completed turn as one burst of protocol frames.
Both accept ``tool_call_parser`` (the engine's parser name), ``capture_topk``,
``sampling_defaults`` and ``force_reasoning``; per-request sampling extras pass
through ``sglang_sampling_params`` or ``vllm_sampling_params``. A positive
``capture_topk`` also records the sampler's top-K ids and log-probs for every
response token. Score centering (``training.options.score-centering``) needs
at least ``score-centering-top-k`` of them (see the loss families guide). A vLLM engine
serving a training stack also needs Reef's connector, which stamps every
sampled token with the weight version that produced it::

   --kv-transfer-config '{"kv_connector": "ReefConnector", "kv_connector_module_path": "reef.inference.vllm.connector", "kv_role": "kv_both"}'

The connector moves no KV and composes with another connector under vLLM's
``MultiConnector``. To combine it with vLLM's native CPU offloading, list
``OffloadingConnector`` and ``ReefConnector`` as ``MultiConnector`` children
in ``--kv-transfer-config``; the ``--kv-offloading-size`` flag replaces the
configured connector, so it cannot be combined with this one. A local release
without the connector serves under its release id; a live release without it
is rejected.

``rollout_log_probs`` must mean the same thing on every engine as in the
trainer. Sampling runs through::

   raw logits -> penalties, logit_bias -> / temperature -> [A] -> top-k, top-p, min-p -> [B] -> sample

SGLang and Slime's trainer both read at [A] (full vocabulary, trainer with
``rollout_temperature``, no penalties), so they agree as long as a recipe uses
no penalties or ``logit_bias``. vLLM ``--logprobs-mode processed_logprobs``
reads at [B], so it matches only with top-k, top-p and min-p off, and Reef's
vLLM client rejects a request whose effective ``top_p`` is below 1, ``top_k``
above 0 or ``min_p`` above 0; replaying vLLM's sampling mask in the trainer
would lift that restriction. Verify with Slime's
``train_rollout_logprob_abs_diff`` on identical weights before training.

For both handlers, set ``inference.handler-config.force_reasoning`` to
``true`` if the chat template pre-opens ``<think>``, or ``false`` if it does
not. When omitted, the handler detects this by rendering the template and
caches only a successful result. Tokenizer or template errors propagate to the
request; they do not silently disable reasoning separation. An explicit value
bypasses this detection.

Explicit file selection
~~~~~~~~~~~~~~~~~~~~~~~

Relative config paths resolve from the directory where you run the command.
With no ``-c`` or explicit ``--recipe`` profile, Reef uses CLI inference inputs;
it does not discover ``REEF_CONFIG`` or ``./reef.yaml``. File deployments must
use ``reef serve -c reef.yaml`` or ``reef serve -c "$REEF_CONFIG"`` instead
of relying on the previous implicit discovery. ``REEF_CONFIG`` remains the
internal way the launcher passes effective settings to its HTTP child.
A selected missing or invalid file is an error, even when provider flags are
present. Reef reports the selected config source and does not search the Reef
installation for your config. From outside a checkout,
pass an absolute path to a cookbook config.

Before any model download or process start, the launcher logs the selected
source (file, profile, or command line) and then every resolved setting with
where its value came from:

.. code:: text

   [reef] config: /srv/reef/stack.yaml
   [reef] resolved settings:
   [reef]   recipe.implementation = recipes.sao.recipe:SAORecipe  (file)
   [reef]   reef.host = 127.0.0.1  (default)
   [reef]   reef.port = 9000  (command line)
   [reef]   reef.tokens = ["****"]  (file)
   [reef]   inference.model-path = /models/demo  (file)
   [reef]   inference.backend = sglang  (automatic)
   [reef]   recipe.config.max-staleness = 2  (environment REEF_MAX_STALENESS)
   [reef]   training.backend = slime  (automatic)

Sources are ``file``, ``command line``, ``environment`` (the ``REEF_*``
variables a configuration-free start reads, or a field's declared fallback
variable), ``automatic`` for a choice Reef made because the field was omitted,
and ``default``. Settings left at their defaults are not listed except the
recipe and the HTTP bind. Tokens, API keys, passwords and database URLs are
masked by key name at any depth, including inside native ``options`` objects.

``reef serve ... --print-config`` prints the same report on standard output,
defaults included, and exits with status 0 (or 2 for an invalid config)
without downloading a model, allocating GPUs or starting a process. It takes
the same ``-c``, ``--recipe``, ``--model`` and override flags as a real start,
so it shows exactly what that start would use. A Hugging Face model path is
shown as written; the snapshot is resolved only at startup.

.. code:: bash

   reef serve -c stack.yaml --reef.port 9000 --print-config

Relative state paths use the launch directory. Values interpolate from the
environment with ``${VAR}`` and from the config itself with ``${dotted.path}``.
Use full public paths for command-line overrides:
``--inference.model-path /models/demo`` or
``--training.config.checkpoint_dir /tmp/ckpt``. Logs go to ``.reef/run/``;
set ``reef.run-dir`` to move them.

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
     --inference.handler-config '{"tool_call_parser": "qwen25"}'

The parsed public values are also supplied to service commands and the HTTP
child's config. An empty string, as an unset ``${VAR}`` reference expands to,
is treated as an omitted service value and keeps its default. An explicit
``null`` is a value: optional fields resolve to null and any other field
rejects it (see below). ``reef.token`` and ``reef.tokens`` remain
distinct inputs whose credentials are combined. The selected Recipe, runtime adapter and executor
settings use the same field parser.

Component configuration
~~~~~~~~~~~~~~~~~~~~~~~

Standalone recipe files loaded by embedding scripts use the same layout.
``load_recipe_config`` and ``recipe_config_from_file_layout`` translate it into the
existing recipe construction contract.
Recipes may declare opaque sections in ``config_sections``; for example, Cordis
owns ``recipe.config.evolution``. Those sections use object/leaf CLI overrides,
then their recipe validates the payload. Executor placement remains shared with
the deployment and reaches dotted recipes as well as named presets.

After selecting ``recipe.implementation``, Reef loads that class's declarations without
constructing the recipe. A dotted weight-training recipe exposes its fields
as ``--recipe.config.max-staleness``, with the aliases
``--max-staleness`` / ``--max_staleness`` / ``--reef.max_staleness``. Other dotted
recipes use the same ``--recipe.config.*`` namespace; their ``--reef.data.*``
spellings remain accepted aliases. ``reef serve -c stack.yaml --help`` includes the
selected component's flags; basic ``reef serve --help`` does not load a recipe.
The selected package must be importable in the launcher and child environments.

Declared component fields follow **CLI > YAML > declared environment fallback
> dataclass default**. False, zero, empty strings and empty containers remain
explicit values. Strings are not guessed as YAML scalars. The last CLI alias
wins, and boolean fields support ``--no-...``. A declaration that conflicts
with a public option is rejected. The normalized values retain their types
when handed to the HTTP child.

``null`` is distinct from omission for every declared field: an optional
field (``str | None`` and similar) resolves to null, and writing ``null`` for
any other field is an error naming the field, rather than a silent fall back
to its default. Omit the field to use the default. The literal string
``null`` remains text for string fields. A file that repeats a YAML key,
``reef.port`` twice for example, is rejected with both line numbers instead of
the last occurrence silently winning, and two spellings of one field
(``model-path`` and ``model_path``) are a duplicate field error. Recipe floats
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
``recipes/sao/examples/imo_answerbench/serve.yaml`` is the smallest, two GPUs for one
actor and one rollout engine; ``recipes/tttd/examples/tttd/serve.yaml`` adds
LoRA training, and ``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds
a PRM engine and a student model.

Service and storage settings
----------------------------

The HTTP service reads these fields from the ``recipe``, ``reef``,
``inference`` and ``storage`` sections.

.. config::

   recipe.implementation | recipe | the recipe this deployment serves
   reef.host | 127.0.0.1 | bind address
   reef.port | 8900 | bind port
   reef.served-url | | the URL a composite recipe's own evaluation calls (episodes and the proposer) reach this service at; the default is loopback on the bind port, so set it when episodes run on another host. A recipe of one component calls its runtime's endpoint directly and ignores it
   reef.console-origins | [] | exact browser console origins allowed to access the HTTP service; disabled by default
   reef.token | the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.
   reef.allow-implicit-scenario-creation | true | when false, an unknown scenario is HTTP 404
   inference.model-path | a local HF model directory or a repo id, downloaded on start
   inference.upstream-url | the OpenAI-compatible provider, with no ``/v1`` suffix
   inference.upstream-api-key | its credential. Reef is the only party that sees it.
   inference.upstream-model | the model to request upstream
   inference.upstream-api | openai | the provider dialect: ``openai`` for Chat Completions, ``responses`` for OpenAI Responses, or ``anthropic`` for an Anthropic-style endpoint
   inference.url | the address the training backend reports | the local engine; set only to front the engines with something else
   recipe.config.checkpoint-every-n-versions | 1 | how often a version becomes durable, for a weight-training recipe

Storage paths default under ``.reef/``, which the basic and sao stacks keep;
the openclawrl stack overrides them to ``/var/lib/reef``. Point them somewhere
persistent.

.. config::

   storage.artifact-repository | .reef/artifacts.git | the Git-backed release chain
   storage.artifact-work-dir | .reef/artifact-work | materialization scratch
   storage.artifact-cache-dir | .reef/artifact-cache | fetched artifact cache
   storage.agent-record-dir | .reef/agent-record | the record store
   storage.agent-record-retention-days | 7.0 | deprecated compatibility setting; automatic cleanup no longer expires data by age
   storage.agent-record-retention-max-bytes | 21474836480 | 20 GiB shared across all record bodies, including unconsumed records

.. warning::

   On ephemeral storage, a restart loses the record store, the commit logs, and
   every version.

Training completion updates consumption progress without retiring stored bodies.
Storage manages cleanup independently: the HTTP service runs a sweep at startup
and every 60 seconds. When the total record-body size exceeds the configured
budget, storage evicts the oldest records by stored creation time, breaking ties
by append sequence. Both consumed and unconsumed data are eligible; processors
cannot protect disk records from capacity eviction. The deprecated
``storage.agent-record-retention-days`` option is accepted but has no cleanup effect.

The budget measures UTF-8 JSON payloads, references and artifact references
across active and archived scenarios. SQLite shares it across the record
directory; PostgreSQL shares it across the deployment schema. Each eviction
logs a warning with the scenario, record count, sequence range and body bytes.
Per-scenario loss totals survive restart. The pipeline status exposes
``record_data_incomplete``, ``evicted_record_count`` and ``evicted_body_bytes``
after a loss; metrics also expose ``records/evicted_count`` and
``records/data_incomplete``. These describe missing stored data, not proof that
an already committed model missed those examples. Pending reports whose inputs
are no longer stored are skipped with a warning; remaining records continue
through the usual processor.

Retry hashes and commit metadata survive eviction, so retrying an upload does
not restore evicted records or train them again. A consumer needing multiple
passes must tolerate missing records once capacity eviction occurs.

This is a body budget, not a hard filesystem quota or an emergency disk-full
handler. Concurrent writes can exceed it between sweeps. Indexes, retry hashes,
commit logs and WAL require additional disk headroom. SQLite reuses freed pages
without automatically shrinking its file. Cleanup failures are logged and
retried on the next sweep. Standalone stores need an explicit maintenance call;
see `Python API <python-api.rst>`__.

Existing stores are upgraded transactionally. Older retirement markers and
receipts become consumption records without deleting original bodies. The old
columns, indexes and receipt table are removed; PostgreSQL advances its schema
version to 3. Training commits store only consumption progress. Already deleted
bodies cannot be recovered. Do not share stores between old and new writers, or
downgrade without restoring a compatible backup.

Recipe settings such as ``max_staleness``, and any others the recipe declares
with ``config_field``, go under ``recipe.config``. A field the selected recipe
does not declare is an error. With a named preset, the recipe reads its
configuration from the preset file.

Recipe configuration
--------------------

A configuration file puts declared recipe fields under ``recipe.config`` and
the runtime under ``recipe.runtime``. Harness settings live in
``recipe.config.evolution``.

``recipe.config.training_mode`` is shared by all recipes and defaults to ``auto``.
In ``auto``, the recipe's processor decides when its data can form a batch,
and ``POST /reef/train`` is refused. In ``manual``, inference and reports
cannot authorize training by themselves; ``POST /reef/train`` supplies the
user instruction, and harness evolution runs it alone. In ``hybrid``, the
processor batches as in ``auto`` and runs instructions too, a queued
instruction first; harness evolution hands the proposer, beside the
instruction, the units an automatic batch would take next, up to
``pipeline.task2batch.config.batch_size`` and possibly none: scored traces, or
unscored records when the ``pipeline`` section names the inference stages. The processor defines
what an instruction batch carries, independently of its automatic batching
policy.

The training pipeline receives ``PipelineContext.training_mode`` as its
initial batching mode. The modes share ingestion and retention. In ``manual``
an instruction's batch has no record sets; in ``hybrid`` it takes the sets that
an automatic batch would. A recipe's ``Task2Batch`` declares
``supported_training_modes``; an unsupported mode raises ``NotImplementedError``.
Harness evolution supports the three modes and requires a proposer that
explicitly accepts ``requests`` for ``manual`` and ``hybrid``. Setting either
on an inference-only recipe does not create a training backend.

.. code:: yaml

   recipe:
     config:
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

   recipe.config.training_mode | auto | ``manual`` waits for ``POST /reef/train`` instructions instead of batching by the recipe's rules; ``hybrid`` batches by the recipe's rules and runs a queued instruction first

A recipe is selected three ways:

- **The core record-only recipe:** ``recipe.implementation: recipe``
- **A dotted class:** ``recipe.implementation: "my_pkg.my_method:MyMethodRecipe"``
- **A named preset:** ``recipe.implementation: my-preset``, resolved to
  ``my-preset.yaml`` under ``REEF_RECIPE_CONFIG_DIR``

There is no recipe-implementation registry. A bare name other than ``recipe``
is always a preset name; it never imports a learning method implicitly.

A dotted class does not have to be pip-installed. ``reef serve`` looks for
its top-level package beside the config, walking up from the config file's
directory to the nearest ancestor that holds ``<package>/__init__.py``, and
appends that directory to ``PYTHONPATH`` for every service it starts. This is
how ``recipes.sao.recipe:SAORecipe`` resolves from a source checkout: the
``recipes/`` cookbook sits next to the example's ``serve.yaml``, so the
launcher does not export ``PYTHONPATH`` itself. Entries already in
``PYTHONPATH`` keep their precedence.

``REEF_RECIPE_CONFIG_DIR`` is the directory preset YAML is read from, and it has
**no default**: a bare recipe name resolves to a preset only when it is set.
The one kind of preset reef bundles is a recipe's profile under
``reef/service/profiles/``: ``reef serve --recipe <name>`` points this
variable at that directory and starts the profile as the deployment config
(see `the CLI reference <cli.rst>`__).

A preset uses the configuration file layout and is read as-is. ``${VAR}``
interpolates in a deployment config, never in a preset. A preset carries its
``recipe.implementation`` and ``recipe.config``, plus an optional
``recipe.runtime`` when the recipe builds its own runtime instead of using the
deployment's upstream proxy. ``inference.upstream-model`` or
``inference.model-path`` names the preset's model. When using the deployment's
runtime, a preset may omit the model to inherit that runtime's model
(``inference.upstream-model`` for an upstream proxy). An explicit model takes
precedence. A preset with its own ``recipe.runtime`` must still name its own
model. Harness-evolution presets also carry ``recipe.config.evolution``:

.. code:: yaml

   recipe:
     implementation: reef.recipe.cordis:CordisRecipe
     config:
       evolution:
         adapter: pi
         propose: methods.mine:MyProposer
         evaluate: methods.mine:MyScorer
         tasks: ["..."]
   pipeline:
     task2batch:
       config:
         batch_size: 1
   inference:
     upstream-model: qwen3-8b

The preset's ``recipe.implementation`` is ``recipe`` or a dotted recipe class.
Weight-training recipes are selected directly by dotted class in the deployment
config, so the service can assemble their Ray training runtime. Presets suit
recipes whose runtime can be built from the preset or the deployment's
upstream proxy. There, ``recipe.config`` holds the training mode and the
recipe's own sections, and ``pipeline`` holds the stage config.

A composite recipe serves and evolves several release components in one
scenario, one recipe per component. Its ``components`` object maps each
component name to a section with the shape of the top-level ``recipe``
section: ``implementation`` names the recipe class, and ``config`` holds the
recipe's fields and sections. A component also keeps its own ``pipeline``
stage config under ``config``. A component has no ``runtime``: every
component uses the deployment's model, runtime and training runtime. A
composite whose component trains weights is deployed the way that recipe is:
``training.backend`` selects the training backend, and the runtime pair
reaches every component. A top-level ``pipeline`` or ``evaluation`` section
goes to the component that trains weights. The repository base keeps one
directory per component: the recipes' seeds are written there, a bootstrap
model snapshot goes under the weight-training component's directory, and a
component whose recipe seeds nothing starts empty. Each component's trainer
runs as its own worker and commits into the same release chain, so every
step checkpoints and the components share one ``training_mode``. A report
is admitted when any component's contract accepts it, and each trainer
keeps the reports its own contract parses:

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         harness:
           implementation: reef.recipe.cordis:CordisRecipe
           config:
             evolution:
               adapter: pi
               propose: methods.mine:MyProposer
               evaluate: methods.mine:MyScorer
               tasks: ["..."]
         config:
           implementation: my_pkg.config:ConfigRecipe
   inference:
     upstream-model: qwen3-8b

The same composite with a weight-training component is a deployment file:
the composite is selected by its dotted class, ``training.backend`` picks the
backend, and the weight component's stage config lives under its ``config.pipeline``:

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {task2batch: {config: {batch_size: 8}}}
         harness:
           implementation: reef.recipe.cordis:CordisRecipe
           config:
             evolution: {adapter: pi, propose: methods.mine:MyProposer, evaluate: methods.mine:MyScorer, tasks: ["..."]}
   inference:
     model-path: Qwen/Qwen3-8B
   training:
     backend: slime

A preset's ``recipe.runtime.type: executor_training`` selects an executor-backed
training coordinator. Its ``executor`` mapping accepts ``backend`` (default
``auto``, resolving to ``uni``, ``mp`` or ``ray``, or a custom executor import path), ordered ``workers`` and backend
``options``. ``coordinator_rank`` defaults to zero. Existing ``ray_training``
configuration still connects to a named bridge. The Slime driver's separate
``--reef-executor-backend`` selects the model worker executor and defaults to
``auto`` (currently Slime's Ray launcher). See `Worker executors <../developer-guide/executors.rst>`__ for the full
contracts and examples.

Stack execution backends
------------------------

Reef chooses native process placement in backend implementation code;
``execution.services`` is rejected. ``execution.training`` and ``execution.rollout``
select the Slime training-worker and rollout-control executors (both default
``auto``, resolving to ``ray``). All selectors accept a backend name/import path or a profile under
``executors``. Inline objects/profiles accept ``backend`` (default ``auto``),
``options``, ``workers``, and ``resources`` containing ``cpus_per_worker`` and
``gpus_per_worker``. Counts must be positive integers; resource quantities must
be finite nonnegative numbers. Numeric environment interpolation is accepted.

``execution.generator`` selects the executor of the generator service a
``generator`` section adds (default ``auto``, one local process); it accepts
the service executor settings below, with one worker.

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
``num_gpus`` and ``num_cpus``.

Harness evolution keys
~~~~~~~~~~~~~~~~~~~~~~

The batch size goes in the deployment's top-level ``pipeline`` section; the
rest goes under ``recipe.config.evolution``. `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ describes what each one changes.
The built-in Reefine recipe instead defaults to ``selection: reefine``;
see its `independent evaluation configuration
<../user-guide/recipes/reefine.rst#independent-evaluation>`__ for request
contexts, protected tasks and reviewer bindings.

.. config::

   pipeline.task2batch.config.batch_size | 1 | traces per mutation attempt

The default stages batch every valid scored report as a trace, including
successful outcomes. To batch recorded traffic alone, every ``batch_size``
requests with unscored samples, set
``pipeline.record2record.module: reef.record2record:InferenceRecord2Record`` and
``pipeline.record2task.module: reef.train.cordis_backend.processor:InferenceTasks``.
Reef refuses ``batch_size`` and ``batch_policy`` under ``recipe.config``.

.. config::

   evolution.propose | a ``Proposer`` class: ``package.module:ClassName``, or a mapping with ``module`` and ``config`` that the class method ``from_config`` receives
   evolution.evaluate | an ``EpisodeScorer`` class, in the same form; ``reef.evaluation.harness:VerifierRewardScorer`` scores a task directory episode by its Harbor verifier's reward, for an evaluation fed by ``evolution.task_manifest``
   evolution.selection | score_comparison | ``floor`` (the candidate alone must score at least ``evolution.floor_score`` on every evaluation task; the current release is not run, and ``recheck_every`` is refused since a recheck compares two trees), ``always``, or a ``SelectionPlugin`` class in the same form as ``evolution.propose``
   evolution.floor_score | 1.0 | the score every evaluation task must reach under ``selection: floor``; a positive number, refused with any other selection, as ``min_win_margin`` is outside ``score_comparison``
   evolution.tasks | non-empty list of episode prompts, scored once per tree per step
   evolution.task_manifest | a split manifest written by ``reef.core.tasks``; the eval split names the evaluation's tasks as directories under ``evolution.tasks_root``, each passed to the adapter as its path; set instead of ``evolution.tasks`` and only with an adapter whose prompt is a task directory (``terminus``); ``promote_failures`` cannot be combined with it
   evolution.tasks_root | the directory the manifest's task names live under
   evolution.adapter | pi | ``opencode``, ``claude``, ``codex``, ``dsh`` (DeepSeek Harness), ``hermes`` (Hermes Agent), ``native`` (Reef's own agent, whose tools are ``native_tool`` nodes, whose loop events listen to ``native_hook`` nodes, and whose loop is a ``native_graph`` node or, as code, a ``native_loop`` node), ``terminus`` (Terminal-Bench's Terminus 2, through a Reef-owned Harbor runner), or an entry-point adapter
   evolution.model_metadata | {} | optional map of exact model names to ``{context_window: <positive token count>, reasoning: <boolean>}``; Codex uses these values before provider discovery
   evolution.binary | a path to the harness binary; unset, backend construction installs the adapter's pinned version through the vendor's channel under ``$REEF_HARNESS_PREFIX`` (default ``~/.local/share/reef-harness``)
   evolution.episode_timeout_s | 600 | seconds one evaluation episode may run
   evolution.episode_repeats | 1 | episode pairings per task per step; each repeat tallies on its own
   evolution.on_stale | merge | what a step's result becomes when another component's commit (a weights step, in a composite) replaced the release it was evaluated against: ``merge`` commits it onto the release served now, since each pairing compared candidate and current under the same conditions when it ran (the commit metrics then carry ``merged_onto``); ``reevaluate`` keeps the candidate and runs its episodes again against the new release, under the next attempt directory of the step record; ``refuse`` drops the result and proposes again
   evolution.forbid_residue | false | when true, an episode leaving files outside the cleanup whitelist scores as one that could not run
   evolution.max_steps | 0 | stop automatic evolve steps once this many steps ran, instruction steps included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs past it
   evolution.max_failure_streak | 0 | stop automatic evolve steps after this many consecutive rejected steps, instruction steps included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs while the breaker is open
   evolution.max_model_calls_per_step | 0 | cap the proposer's model calls in one step; 0 disables the limit
   evolution.multimodal | | reefine only; the gateway Reef relays a scenario's ``/v1/images``, ``/v1/embeddings``, ``/v1/audio/speech`` and ``/v1/decisions`` to, unrecorded, and the agent proposer's trials reach: ``preset`` (``openrouter``, the default, or ``openai-compatible`` for OrcaRouter, LiteLLM, ...), ``url`` (the gateway's address, no ``/v1``; required for ``openai-compatible``) and ``api_key`` (its key; the profile reads ``REEF_MULTIMODAL_API_KEY``, and ``api_key_env`` names a variable instead; empty, the upstream's key when the upstream is the same address). Without a key those routes answer 501
   evolution.proposer_agent | | off unless set (the reefine recipe sets it); a proposer whose ``runs_agent`` is true runs a coding agent under it: ``sandbox`` (``bwrap`` jails it with pasta networking and refuses to start where the host cannot; ``e2b`` runs it in an E2B cloud sandbox that reaches the gateway through a tunnel, with ``e2b_api_key`` (else ``E2B_API_KEY``) and ``e2b_template`` (else ``reef-pi-<version>-episodes-v1`` with bubblewrap, built on first use), and needs the ``e2b`` extra; ``none`` runs it unisolated and must be chosen; unset, or ``REEF_PROPOSER_SANDBOX``, jails it where the host can and leaves it off where it cannot), ``timeout_s`` (1800, the whole agent run) and ``trial_timeout_s`` (300, each run of the candidate harness). See the reefine recipe guide
   evolution.executor | local | ``local`` runs episodes as a plain subprocess; ``sandbox`` uses a local bubblewrap jail; ``e2b`` creates a cloud VM per episode and checks provider access, template, binary and namespace support at startup. E2B uses remote binaries; Terminus retains its own Harbor execution
   execution.evolution.workers | 1 | fixed worker-group size; CPU auto selects ``uni`` for one and ``mp`` for multiple
   execution.evolution.backend | auto | worker placement, independent of the ``local/sandbox`` episode isolation policy; ``local`` retains shared-memory callbacks
   execution.evolution.resources | | ``cpus_per_worker`` and ``gpus_per_worker``; omitted values retain component defaults; GPUs select Ray under ``auto`` and cannot reduce declared GPU needs
   evolution.episode_workers / worker_executor / worker_resources | | deprecated compatibility aliases; conflicting resource values are rejected; legacy worker_executor cannot accompany role-level workers/resources
   evolution.sandbox | | for ``sandbox``: ``egress_hosts`` and ``limits`` (``cpu_seconds``, ``memory_bytes``, ``processes``, ``file_bytes``); for ``e2b``: ``e2b_api_key_env`` (default ``E2B_API_KEY``), optional ``e2b_api_key``, ``e2b_template`` and ``forward_ports`` (explicit worker-loopback TCP ports). E2B rejects nonempty ``egress_hosts``/``limits``; template resources and outbound internet apply. Both accept ``env_from``
   evolution.promote_failures | false | when true, a failing trace's prompt becomes a permanent evaluation task, so no later candidate can win while bringing the failure back; the seed tasks stay the floor
   evolution.max_promoted_tasks | 50 | the cap on promoted tasks; admission stops there so the suite is bounded
   evolution.max_promoted_per_client | 5 | the cap on promoted tasks from one tagged client (its ``x-reef-tag-client``, else session, tag); untagged traffic has no identity to count under and meets only ``max_promoted_tasks``; 0 disables the cap
   evolution.promote | | optional ``Promoter`` class, in the same form as ``evolution.propose``; its ``promote(samples, *, manifest=None)`` chooses which trace prompts to promote; without it every failing trace's user prompt is promoted, and the caps and the credential and directive screens still apply
   evolution.publish | auto | ``review`` holds every successful evaluation as a pending release until ``POST /reef/scenarios/{scenario}/promote`` names it
   evolution.review_kinds | [] | node kinds whose wins wait for a promote while the rest publish at once; a win that touches a ``native_loop`` waits whether or not the list names it
   evolution.seed | entry options loaded into the tree on first boot, or a dotted ``module:attribute`` naming a sequence of them (``reef.harness.runners.native.seed:SEED_NODES`` is the native harness's shipped tools and hook); recovered state takes precedence
   evolution.models | auxiliary models for the method: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name
   evolution.version_check | appends the adapter's update notice; an interactive pulled tree offers to run the update or skip when behind
   evolution.requests | false | appends the adapter's harness requests extension and its extension API skill after the notice (the reserved entries ``reef-requests`` and ``reef-pi-extension-api``), so a ``reef-pi`` session gets ``/reefine <request>`` in the TUI (submits to ``POST /reef/train``, which needs ``recipe.config.training_mode: hybrid`` or ``manual``) and the method reads the API reference before it writes an extension; on ``claude``, ``codex``, ``opencode``, ``hermes`` and ``dsh`` it appends one ``agent_command`` named ``reefine`` under the same reserved id, which has the session's model file the request with ``reef-<adapter> evolve``, wait with ``reef-<adapter> wait`` and offer ``reef-<adapter> update``; an adapter with no command surface (``terminus``, ``native``) refuses boot (a seed entry: a deployment that boots from a recovered state keeps its tree, as with ``version_check``); the tutorial's ``tutorials/evolve-your-harness/configs/deployment.yaml`` sets it, with ``version_check: true`` and ``review_kinds: [code_extension]``
   evolution.proposals_dir | .reef/proposals | where agent proposals from ``POST /reef/harness/proposals`` wait for the next evolve step: one directory per scenario under it (``<dir>/<scenario>``, made absolute at build, created when the first proposal arrives), with ``claimed/``, ``refused/`` and ``settled/`` beside the pending files
   evolution.max_pending_proposals | 8 | how many admitted proposals one scenario holds; the route answers ``admitted: false`` with reason ``inbox full`` beyond it, and with reason ``manual mode takes instructions only`` on a scenario in ``recipe.config.training_mode: manual``
   evolution.step_record_dir | | off by default; when set, every step writes its record under ``<dir>/<scenario>/<step>`` (the path is made absolute at build): ``proposer.json`` (each model call the proposer made: ``model``, ``messages`` and ``params`` for a ``chat`` or ``body`` for a ``complete``, then ``reply`` and the provider ``response`` for a built-in ``chat`` binding, ``response`` for ``complete``, or ``error``, and ``seconds``; the response retains provider reasoning/thinking fields when returned; long text is clipped with a marker and a credential shaped literal is replaced by ``[redacted credential]``), ``mutations.json`` (the parsed proposal with its full options, refused or not, redacted the same way) and ``episodes/<side>-<task index>/`` (each evaluation episode's trajectory files as the adapter writes them, copied out of its root before the root is removed, plus ``episode.json`` with the task, the exit code, stdout and stderr, the residue, the score, the failure and the stage path; a repeat adds ``-<repeat>``); a recheck step writes ``episodes/`` only and has no proposer files; a step skipped on the step cap or the failure streak writes nothing; a step directory is never reused, so a retried step lands in ``<step>-2``, then ``<step>-3``, and a kept candidate evaluated again under ``on_stale: reevaluate`` writes its ``episodes/`` under the next attempt directory with ``reevaluation.json`` naming the first; nothing prunes the directory; an unwritable path refuses boot and a record copy that fails aborts the step instead of scoring it

The served model's binding is appended at render time; it never enters the
published files. The seed defines the baseline the first mutation is measured
against. The step record holds the proposer's raw traffic and every evaluation
episode's session log, so treat its directory like the commit log.

The ``inference`` section
-------------------------

Read by every serving mode. ``inference.timeout-s`` limits one inference
request. Buffered inference attempts share ``inference.retry-timeout-s``;
when it is omitted, it follows ``inference.timeout-s``.

.. config::

   inference.timeout-s | 300.0 | maximum time for one inference request
   inference.retry-timeout-s | ``inference.timeout-s`` (300.0 by default) | total deadline shared by inference attempts and retry delays
   inference.retry-initial-s | 0.05 | delay before the first retry
   inference.retry-max-s | 1.0 | maximum delay between retries

The ``training`` section
------------------------

Read by the weight-training stack. See `Evolve your model
<../user-guide/evolve-your-model.rst>`__ for how to size it.

.. config::

   training.backend | slime | built-in (slime or tinker), installed entry-point name, or dotted TrainingDeployment class
   training.colocate | false | train on the inference GPUs: Reef reserves one shared allocation and derives the native offload flags
   training.ready-timeout | 3600 | backend-owned component startup deadline; in-process model loading is covered by reef.ready-timeout
   training.config.num_gpus | example-specific GPU count passed to Slime's model topology flags; does not reserve GPUs for the driver or set the Ray cluster's capacity
   training.config.global_batch_size | samples in one optimizer step. Must equal ``pipeline.task2batch.config.batch_size``.
   training.config.checkpoint_dir | where Megatron and HF checkpoints are written
   training.config.megatron_checkpoint_path | optional pre-converted torch_dist checkpoint, to skip HF conversion on every start
   training.config.checkpoint_retention | storage-fraction bounds and the retention policy
   training.options | native training flags: actor GPU layout, optimizer, sequence length, and loss settings

Slime fills architecture flags such as layer counts and hidden sizes from
``inference.model-path``. Do not put them in the config.

The ``evaluation`` section
--------------------------

Only weight-training recipes read this section; a deployment that pairs it
with any other recipe fails at startup, because a harness recipe builds its
evaluator in code. Absent by default, in which case a successful
weight-training step publishes without an evaluation. When present, Reef calls the
named class's ``from_config`` method once per scenario and hands the plugin the exported but
unpublished checkpoint.

.. config::

   evaluation.module | a ``package.module:ClassName`` reference to a ``ConfiguredEvaluationPlugin`` subclass. Required.
   evaluation.config | opaque mapping handed to ``from_config``; Reef never reads it

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:BenchmarkEvaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

The class method ``from_config(config, *, runtime, training_runtime, scenario,
environ)`` returns the scenario's plugin. Reef checks the class when it loads
the recipe config. It builds the plugin later, so allocate model resources in
``from_config``. Reef rejects functions, abstract classes and structural
lookalikes.

The plugin interface is in `Write a recipe
<../developer-guide/write-a-recipe.rst#gate-a-candidate>`__.

The ``pipeline`` section
------------------------

A recipe's pipeline has three stages (see `Training pipeline
<../developer-guide/processors.rst>`__). The recipe names a default class for
each stage. This section gives each stage its config, and it can name another
class for a stage. Every stage reads only its own ``config``; the recipe's
``recipe.config`` holds no stage settings.

.. config::

   pipeline.record2record | the record-to-record stage: ``config``, and optionally ``module``, a ``Record2Record`` subclass
   pipeline.record2task | the record-to-task stage: ``config``, and optionally ``module``, a ``Record2Task`` subclass
   pipeline.task2batch | the task-to-batch stage: ``config``, and optionally ``module``, a ``Task2Batch`` subclass

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2record:
       config:
         groups_per_step: 8
         rollouts_per_group: 64

A stage without a key is the recipe's class with no config. Each recipe's page
lists the keys of its stages. Keys are written as the stage reads them; a
hyphen spelling is also accepted.

Reef checks each stage when it loads the recipe config, before a scenario
starts: it imports a named class, rejects functions and abstract classes, and
parses the stage's ``config``. An unknown key or a value out of range fails
the startup. A setting that moved from ``recipe.config`` to a stage fails
with the key's new place, for example ``move groups_per_step to
pipeline.record2record.config.groups_per_step``.

For each scenario, Reef builds the stages in pipeline order with the class
method ``from_config``. A later stage receives the stages built before it, and
it reads a value that an earlier stage owns from that stage. For example, the
TTTD step grid is set once, in the record stage. A stage class without
``from_config`` cannot be named here.

Generated tasks
~~~~~~~~~~~~~~~

The built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, binds the ``task_group``
objective on Tinker's ``importance_sampling`` loss, so it requires the
``tinker`` training backend. Its stages are ``GeneratedTaskRecord2Record``,
``GeneratedTaskTasks``, and ``GeneratedTaskBatches``. Without
``pipeline.record2task.config.rounds`` no round runs, and the recipe trains on
the training arm's reports only.

.. code:: yaml

   recipe:
     implementation: reef.recipe.generated_tasks:GeneratedTaskRecipe
   pipeline:
     record2record:
       config:
         plays_per_task: 4
         batches_per_round: 2
     record2task:
       config:
         generator_url: ${endpoints.generator}
         rounds: 20
         description: shell tasks with hidden state under /var and /etc
         state_dir: ${REEF_STATE_DIR}/rounds
         generator:
           module: recipes.beta.spade:SpadeTaskGenerator
           config:
             generator_url: ${endpoints.generator}
             skills: [inspection, repair]
             state_dir: ${REEF_STATE_DIR}/generations
     task2batch:
       config:
         tasks_per_step: 2

``GeneratedTaskRecord2Record`` groups the training plays by task and requests
the rounds:

.. config::

   pipeline.record2record.config.plays_per_task | 4 | training plays of each task; a group is complete at this count, or at the plays that reported when fewer did. At least 2.
   pipeline.record2record.config.batches_per_round | 1 | trained batches after which the next round starts
   pipeline.record2record.config.source_records_per_round | 0 | recorded sessions that each round gives the generator, at most; 0 generates from the description only
   pipeline.record2record.config.source_tag | the ``name=value`` tag that marks a source session: each of its model calls carries the header ``x-reef-tag-<name>: <value>``. Required when ``source_records_per_round`` is above 0.

``GeneratedTaskTasks`` runs the rounds and turns the training plays into
tasks. It plays each task ``plays_per_task`` times, as its record stage
groups them:

.. config::

   pipeline.record2task.config.rounds | 0 | rounds of task generation; 0 generates nothing
   pipeline.record2task.config.generator | the ``TaskGenerator`` class (``module``) and its ``config``. Required with rounds.
   pipeline.record2task.config.description | what the tasks are about; every generation request carries it. Required with rounds.
   pipeline.record2task.config.generator_url | the generator service, ``${endpoints.generator}``. Required with rounds.
   pipeline.record2task.config.state_dir | where each scenario keeps its round summaries. Required with rounds.
   pipeline.record2task.config.served_model | the model the task player asks for; the deployment's by default
   pipeline.record2task.config.scaffold_tolerance | 8 | masked tokens the sample assembly may realign before a response, for a thinking template's scaffold

``GeneratedTaskBatches`` batches complete task groups:

.. config::

   pipeline.task2batch.config.tasks_per_step | 4 | complete task groups in one batch

Reef calls the generator class's ``from_config(config, *, scenario, environ)``
once for each scenario. A generator that calls a remote service reads its
credentials through environment references in ``config``. The generator
interface and the round are in `Training pipeline
<../developer-guide/processors.rst#task-generation-contract>`__.

A source session is a report, without a training arm, whose inference
records all carry ``source_tag``. The generator service accepts at most
512 KiB of records and files in one request, so keep
``source_records_per_round`` small when the sessions are long.

The ``generator`` section
-------------------------

Absent by default. When present, ``reef serve`` starts the generator service
(``python -m reef.service.deploy.generator_process``) before the HTTP service, which depends on
it, and publishes its address as ``${endpoints.generator}`` for the
generated-task stages to consume. The generator writes, checks and plays Harbor tasks for the
generated-task recipe and for a task generator such as SPADE, under the same
interpreter as the Reef service (``REEF_PYTHON``, otherwise the launcher's). Its host needs Docker and the
``harbor`` command line; the Reef service itself does not. ``execution.generator`` selects its executor, so a deployment can
place it on the host that has Docker.

.. config::

   generator.tasks-root | the directory generated tasks, manifests and Harbor job files live under. Required.
   generator.host | 127.0.0.1 | bind address
   generator.port | 8910 | bind port
   generator.work-dir | ``<tasks-root>/.play`` | where the task player keeps trials
   generator.agent | terminus-2 | the Harbor agent the task player runs, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. ``import_path: reef.record2task.generator.terminus:VerifiedTerminus2`` in place of ``name: terminus-2`` also scores an episode whose agent filled its context or lost its terminal, which Terminus 2 leaves without a reward
   generator.agent-host | an address of the host the task container can reach, for an agent that runs inside the container
   generator.harbor | ``harbor`` on PATH | the harbor command line for the oracle check
   generator.concurrency | 2 | episodes in flight per play request
   generator.designer-url | a Reef service the designer calls go to instead of this deployment's
   generator.designer-token | the token for ``designer-url``
   generator.designer-model | the served model the designer asks for; the deployment's by default
   generator.designer-timeout-s | 1800 | seconds one designer call may take; ``inference.timeout-s`` must allow it too
   generator.designer-api | the API the designer asks through: ``openai`` (Chat Completions) or ``responses`` (the Responses API). The default is ``responses`` when designer calls go to this deployment and its ``inference.upstream-api`` is ``responses``, and ``openai`` otherwise
   generator.designer-options | extra fields of the designer's request, e.g. ``{"reasoning_effort": "none"}``; with ``designer-api: responses``, Responses fields such as ``{"reasoning": {"effort": "low"}}``
   generator.ready-timeout | 60 | seconds ``reef serve`` waits for the generator to answer

.. code:: yaml

   generator:
     tasks-root: ${REEF_SPADE_STATE_DIR}/tasks
   execution:
     generator: auto
   pipeline:
     record2task:
       config:
         generator_url: ${endpoints.generator}

The generator reads the resolved deployment through ``REEF_CONFIG`` like the
other children: the Reef address from ``reef.host`` and ``reef.port``, the
token from ``reef.token``, and the served model from ``inference.model-path``
or ``inference.upstream-model``. Without ``reef serve``,
``python -m reef.service.deploy.generator_process -c serve.yaml`` runs it from the same file.

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
attached for joining a run back to the commit log. A scenario with several
trainers shares the run: each step's metrics carry its component name as a
prefix (``harness/train/loss``) on the same ``train/step`` axis with
``reef/component`` on the row, its optimizer step rows land under
``<component>/step/*`` on their own counter, and the run config lists each
component's backend under ``reef.components`` and ``backend.<component>``
with ``reef.backend`` null. The names ``train``, ``step``, ``reef`` and
``operations`` are these prefixes, so a composite recipe refuses them as
component names.

Recipe and processor code logs through the same object without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own
``<namespace>/event`` axis. Only finite numeric values are sent.

Operational metrics
~~~~~~~~~~~~~~~~~~~

With W&B enabled, the dispatcher samples each loaded scenario, including
inference-only scenarios, every
10 seconds and once during graceful shutdown. Samples use the existing scenario
run under ``operations/*``, with Unix time in ``operations/time_seconds`` as
their horizontal axis. They do not advance ``train/step`` and do not wait for a
successful commit or a ``/reef/status`` request. Offline mode records the same
samples locally; disabled tracking starts no sampling thread.

The following names are relative to ``operations/``:

.. list-table:: Operational measurements
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Meaning
   * - ``serve/request/*``
     - Inference requests after scenario resolution, including admission,
       backend calls, retries, and record acceptance. Streaming requests remain
       active until their completion record is accepted; an incomplete stream
       or cancelled request counts as failed.
   * - ``serve/admission/*``
     - Time acquiring runtime admission. ``active`` is the number waiting for
       admission; immediate admissions also contribute to count and duration.
   * - ``evaluate/request/*``, ``evaluate/admission/*``
     - The same measurements for calls on the evaluation route, a step's
       episodes and proposer calls, which keep no record; counted apart so a
       step does not read as served traffic.
   * - ``serve/retries_total``, ``serve/timeouts_total``
     - Additional buffered inference attempts and requests that exhaust the
       inference retry deadline. Retries do not create extra request counts.
       Calls on the evaluation route count under ``evaluate/retries_total``
       and ``evaluate/timeouts_total``.
   * - ``serve/version_mismatch_total``
     - Responses rejected by runtime-load-ID verification, including missing
       engine version information and buffered or deferred streaming responses.
       This is a counter of observed rejections, not a background drift probe.
       Calls on the evaluation route count under ``evaluate/version_mismatch_total``.
   * - ``ingest/accepted_total``, ``ingest/duplicates_total``
     - New records appended and identical retries acknowledged by the dispatcher.
       Accepted records include incomplete-stream diagnostics; acceptance does
       not imply a trainable record or successful training.
   * - ``ingest/rejected_report_total``, ``ingest/rejected_request_total``
     - Report schema/reference violations and invalid training instructions
       detected during dispatcher record validation.
   * - ``ingest/rejected_conflict_total``
     - Record identities reused with different content.
   * - ``ingest/write/*``
     - Record-store append calls, including identical retries that reach the
       store. ``failed_total`` counts append exceptions, including conflicts;
       validation rejections before append do not count as write failures.
   * - ``runtime/stale_batches_total``
     - Training submissions the runtime rejects as stale, such as an exact
       version mismatch or an exceeded bounded-staleness window. This does not
       count compatible older samples as errors or increment on status reads.
   * - ``records/unread_count``
     - Training-visible records after the processor's read cursor. These may
       still need feedback or filtering; this is not a ready-batch count.
   * - ``records/oldest_unread_age_seconds``
     - Age of the first unread record, measured from its recorded creation
       time; zero when none remain.
   * - ``processor/unreserved_reports``, ``processor/reserved_reports``
     - Reported-feedback records waiting outside, or held inside, the reserved
       batch. An incomplete group still counts as waiting.
   * - ``processor/oldest_report_wait_seconds``
     - Age of the oldest unreserved report; zero when none remain.
   * - ``processor/tracked_records``, ``processor/judging_records``
     - Computed-feedback records waiting for more traffic or a judgment.
   * - ``processor/unreserved_candidates``, ``processor/reserved_candidates``
     - Computed-feedback candidates outside or inside the reserved batch.
   * - ``processor/buffered_requests``
     - Explicit training instructions already buffered by the processor.
   * - ``training/reserved_batches``
     - Zero or one. A reservation can be executing or waiting for settlement;
       it is not necessarily a queued batch.
   * - ``training/auto_enabled``
     - One for auto/hybrid training, zero for manual training. Waiting data in
       manual mode does not by itself indicate a stalled worker.
   * - ``training/error``, ``training/failed_attempts_total``
     - Current recorded training error (zero/one) and cumulative failures
       recorded by the dispatcher, including retries and worker failures.
       Recovery clears the current error but retains the counter. The counter
       resets on dispatcher restart or scenario deletion.
   * - ``training/checkpoint_storage_blocked``
     - Whether the dispatched training worker is blocked on checkpoint storage.
   * - ``training/execution/*``
     - Backend preparation, evaluation, and settlement measurements.
   * - ``runtime/weight_sync/*``
     - Runtime-scheduler weight activation/update calls, including resumed
       transfers. This covers the call's full duration, not only network time.

All operation families expose ``started_total``, ``active`` (in-flight count),
``elapsed_seconds`` (age of the oldest active call, zero when idle),
``completed_total``, ``failed_total``, ``duration_seconds_total``, and
``last_duration_seconds`` after a call finishes. Execution and weight-sync calls
are serial, so their active count is zero or one. Request and admission calls
can overlap. Durations are in seconds; cumulative durations sum individual
calls, so concurrent work may accumulate faster than wall time. Divide the
change in total duration by the change in completed plus failed count to obtain
a mean completed-call latency for an interval.
An execution returning a skip, stale drop, or storage retry is a completed
backend call, not necessarily a committed training step. These measurements
reset when their trainer or scheduler is rebuilt, including recovery; the
corresponding ``training/started_at_seconds`` and ``runtime/started_at_seconds``
identify that reset. Stale-batch counts share the scheduler lifetime. Request
and ingestion measurements reset when the scenario is rebuilt, including
recovery, with ``operations/started_at_seconds`` identifying that reset. They
are not persisted training history. Malformed HTTP payloads/headers and
failures before scenario resolution are outside these scenario measurements.
No arbitrary scenario names, rejection messages, or record IDs become metric
keys. Ingestion counters start at the dispatcher's typed-record boundary;
wire-payload normalization failures are outside that boundary.

Additional finite numeric processor status fields appear under
``operations/processor/``. Sampling does not advance processor readiness or
consume its queue. If ingestion or commit holds the trainer lock, that sample
omits queue gauges while still reporting execution measurements. Operational
samples contain no request bodies, record IDs, or error messages. They are
low-frequency service and training diagnostics; underlying inference engines retain their
own high-frequency monitoring. Uploading these values does not configure
alert notifications.

Commit correlation
~~~~~~~~~~~~~~~~~~

Durable commit metrics carry ``experiment/provider``, ``experiment/project``,
``experiment/group``, and ``experiment/run_id``. Use them to open the run from
a Reef version, and use the run's ``reef/training_job_id`` to go the other way.
Checkpoint paths are metadata only unless ``upload_checkpoints: true``.

Import, initialization, logging, summary, and upload failures are reported in
the service log and never fail a training step or its commit.

Record tracing
--------------

Record tracing exports every accepted record and every committed training
step as OpenTelemetry spans, so a tracing backend shows what an agent did in a
scenario and which version it trained. It is optional, off by default, and
independent of experiment tracking. Reef speaks OTLP over HTTP and names no
vendor: point it at Langfuse, Arize Phoenix, Jaeger, Grafana Tempo or an
OpenTelemetry Collector. Install ``reef-infra[opentelemetry]``.

.. code:: yaml

   observability:
     tracing:
       enabled: true
       endpoint: https://cloud.langfuse.com/api/public/otel/v1/traces  # full OTLP/HTTP traces URL
       authorization: ${LANGFUSE_AUTH}  # the backend credential, sent as the Authorization header
       service_name: reef              # optional resource service.name

``endpoint`` is the complete traces URL. ``authorization`` is the credential
the backend expects in its ``Authorization`` header: ``Basic <base64
public:secret>`` for Langfuse, ``Bearer <token>`` for most others. It is
handled like ``inference.upstream_api_key``: write it as an environment
reference in the YAML, or omit it and export ``REEF_TRACING_AUTHORIZATION``;
the startup report masks it and it never appears in a span or a log line. A
backend that expects its credential under another header name, such as
``x-api-key``, takes it in the ``headers`` mapping; the startup report masks
every header value, and ``Authorization`` itself is rejected there so the
credential has one place. When no endpoint, credential or header is configured the
exporter reads the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
``OTEL_EXPORTER_OTLP_HEADERS`` environment variables instead.

Each accepted inference record becomes the root span of its own trace, named
``chat <model>``. Its trace and span ids derive from the scenario and record
id, so a restart or a second Reef host produces the same ids. The span carries:

* ``session.id`` and ``reef.scenario``: the scenario.
* ``reef.agent_record_id`` and ``reef.request_type``: the receipt and record kind.
* ``gen_ai.request.model``, ``gen_ai.response.model``, ``gen_ai.response.id``,
  ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens`` and
  ``gen_ai.response.finish_reasons`` from the provider request and response,
  following the OpenTelemetry GenAI semantic conventions.
* ``reef.release_id``, ``reef.content_id`` and ``reef.runtime_load_id``: the
  version that served the request, when the record names one.
* ``reef.tags``: the ``x-reef-tag-*`` request tags.

A feedback record becomes a ``feedback`` span inside the trace of the first
inference it references, with ``reef.score`` and ``reef.references``; further
references are span links. A training instruction becomes a
``training request`` span. A committed step adds a ``training step N`` span
with the commit's step, release, job id, consumed record counts,
its component as ``reef.component`` when the scenario runs several trainers,
and its scalar metrics as ``reef.metrics.*``, plus one ``trained in step N``
child span below every record the step consumed, linked back to the commit
span. Records carry one timestamp, so their spans have zero duration and start
at the record's creation time.

For streamed provider responses, token usage is read from the captured SSE
events, including Chat Completions usage chunks, Responses terminal events
and Anthropic message usage. Cumulative counts are not summed across chunks.
If the provider sends no usage, token counts remain absent; for Chat
Completions, request ``stream_options: {include_usage: true}`` when supported.
The stored response and the forwarded stream remain unchanged.

Spans carry the exchange itself: the request messages and the reply as JSON
in ``gen_ai.input.messages`` and ``gen_ai.output.messages``, feedback text in
``reef.feedback`` and instruction text in ``reef.instruction``. The backend
therefore sees the scenario's traffic; point tracing only at one trusted with
it. Export failures are reported in the service log and never
fail record acceptance or a commit; the exporter batches spans in a background
thread and flushes them during graceful shutdown.
