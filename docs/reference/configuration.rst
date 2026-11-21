Configure Reef serving and training
===================================

This page lists every setting of ``reef serve``: the sections of a
configuration file, their keys and defaults, and the matching command-line
flags. Each flag is the YAML path of its setting, for example
``--inference.model-path``.

Local inference needs no YAML file:

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct

This command starts SGLang on a free loopback port and waits for its health
endpoint. It then starts Reef on ``127.0.0.1:8900``.

- ``--inference.backend vllm`` starts ``vllm.entrypoints.openai.api_server``
  instead of SGLang (the default, ``sglang``).
- ``--inference.tensor-parallel-size 2`` uses two visible GPUs; the default is
  one. The managed path supports a single GPU node.
- The service interpreter (``REEF_PYTHON``, otherwise the launcher's) must have
  the engine and GPU-enabled PyTorch installed. See the `SGLang installation
  guide <https://docs.sglang.io/docs/get-started/install>`__ or the `vLLM
  installation guide
  <https://docs.vllm.ai/en/latest/getting_started/installation/>`__. The base
  Reef installation stays CPU-only.
- A local path or a Hugging Face ID names the model. Reef resolves a downloaded
  snapshot once and keeps the identifier that you wrote as the served model
  name.
- The engine has a one-hour readiness deadline and Reef has 30 seconds. On
  failure or interruption, Reef stops both processes. Logs, including
  ``sglang.log`` or ``vllm.log``, are under ``.reef/run/``.
- ``--inference.model-path`` cannot be combined with an upstream URL or model.
  Native engine options go in ``inference.options`` (`Native backend
  options`_).

An external-provider deployment also needs no YAML file:

.. code:: bash

   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

Reef starts its core record-only recipe on ``127.0.0.1:8900`` and stores state
under ``.reef/`` in the launch directory. It records inference and feedback
and trains no weights. Reef runs in the foreground and stops on Ctrl-C. It does
not start or stop the provider, and readiness does not check the provider's
credentials or models.

``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and
``REEF_TOKEN`` are optional environment fallbacks for this mode; explicit CLI
settings win. ``--model ollama/my-model`` fills the Ollama endpoint and model.
``--model openai/my-model`` uses ``REEF_UPSTREAM_API_KEY``. A model ID with
another prefix still needs an upstream URL.

Scenario routing between students and teachers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.routing_manifest`` (CLI: ``--reef.routing-manifest``) routes each
Scenario's ``/v1/chat/completions`` requests, ordinary and SSE, among external
targets. It is unset by default. Each configured Scenario serves its default
target (usually the student). A request that matches a keyword or domain rule
goes to another target (usually the teacher). A request can also pin one of
the Scenario's targets. The `Semantic Router tutorial
<../../tutorials/semantic-router/README.md>`__ shows a complete deployment,
and `Routed external-model requests
<http-api.rst#routed-external-model-requests>`__ gives the request and record
contract.

Requirements:

- Select ``recipe.implementation: recipe``. Routing records inference and
  feedback; it does not train the routed models.
- Set ``reef.tokens`` and ``reef.host: 127.0.0.1``.
- Clients call Reef with a service token and ``x-reef-scenario``. They may not
  send Reef's or the router's selection headers.

For ``model: auto``, Reef asks an unmodified Semantic Router (SR) for one
preview and checks the selected rule and target against the manifest. A pin
is checked locally and never calls SR. If a provider aborts a buffered
request, Reef retries the same target:

.. container:: config

   ``inference.retry-initial-s`` : 0.05
       the first retry delay of a routed request, in seconds

   ``inference.retry-max-s`` : 1.0
       the longest retry delay, in seconds

   ``inference.retry-timeout-s`` : ``inference.timeout-s``
       the deadline of the retries, in seconds

The manifest (``schema_version: 2``) has three parts:

* ``targets`` maps target IDs to provider model IDs, endpoints and immutable
  external deployment revisions. An optional ``api_key_env`` names the
  environment variable of the provider credential.
* ``scenarios`` lists, for each Scenario, its ``targets``, its
  ``default_target`` and optional ``rules``. Each rule has a ``name``, a
  ``target`` and exactly one of ``keywords`` or ``domains``. If several rules
  match, the first one listed is used.

  - A keyword rule matches a request whose text contains any of its keywords
    (``match: any``, the default) or all of them (``match: all``). It ignores
    case unless ``case_sensitive: true``. A keyword of letters, digits or
    underscores matches only as a whole word. SR checks only the latest user
    message that has text. Only when no user message has text or an inline
    image does SR check the system, developer and assistant text instead. SR
    never checks tool results.
  - A domain rule matches when the deployment's domain classifier predicts
    one of its labels. It needs ``deployment.domain_classifier``. Within one
    Scenario, a label may appear in only one rule, ignoring case.

* ``deployment`` holds two optional keys.

  - ``semantic_router_image`` names the SR image by repository digest
    (``repository@sha256:<hex>``) or local image ID (``sha256:<hex>``).
    Without it, a managed deployment builds the pinned image (`Semantic
    Router image`_).
  - ``domain_classifier`` names a local pretrained domain classifier:
    ``model_directory`` (an absolute path without symlinks), ``threshold`` (a
    number from 0 to 1) and ``files``, the ``sha256:`` digest of each of
    exactly ``config.json``, ``tokenizer.json``, ``model.safetensors`` and
    ``category_mapping.json``. Reef checks the files and labels and mounts
    only these four files, read-only, into SR. See the `domain classifier
    section
    <../../tutorials/semantic-router/README.md#route-by-a-pretrained-domain-classifier>`__.

Target IDs and ``automatic_model`` must not start with ``reef-entrypoint/``,
in any letter case: that prefix names the private router entrypoints.

``reef.routing_deployment.lifecycle`` selects who runs SR: ``managed`` (the
default, `Managed routing lifecycle`_) or ``external`` (`External router
lifecycle`_).

Managed routing lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^

With ``reef.routing_manifest`` set, ``reef serve`` runs the SR container and a
routed Reef child. ``reef serve --component`` refuses a managed manifest. The
lifecycle needs Linux, a local Docker daemon with Compose, and the local
``uni`` service executor. You run the model services yourself.

.. code-block:: yaml

   reef:
     host: 127.0.0.1
     routing_manifest: /path/to/input-manifest.yaml
     routing_deployment:
       deployment_id: code-opd-local
       restart_policy: fail_stop

These keys go under ``reef.routing_deployment``. An unknown key is an error.
YAML accepts underscores or hyphens. The CLI takes the complete object through
``--reef.routing-deployment``.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - lifecycle
     - managed
     - ``managed``: this section. ``external``: see `External router lifecycle`_.
   * - deployment_id
     - required with routing_manifest
     - 1-48 lowercase letters, digits or hyphens; must start with a letter or digit.
   * - restart_policy
     - fail_stop
     - ``bounded`` permits a limited number of SR container recoveries.
   * - max_restarts
     - 2
     - Attempts per explicit deployment; failed startup attempts count and the budget never resets automatically.
   * - restart_backoff_seconds
     - [1.0, 2.0]
     - Delay before each recovery; later attempts reuse the last delay.
   * - ready_timeout_seconds
     - 120
     - Complete startup or recovery readiness deadline.
   * - drain_timeout_seconds
     - 30
     - Deadline for accepted requests, including uploads, and local response sends to finish.
   * - finalization_timeout_seconds
     - 5
     - Additional deadline for record writes after cancellation.
   * - container_stop_timeout_seconds
     - 10
     - Container stop operation timeout.
   * - image_build_timeout_seconds
     - 3600
     - Longest startup build of the SR image, when the manifest names none and no built image is cached.
   * - sr_management_port
     - 50053
     - SR's authenticated HTTP API (previews and configuration), published on host loopback.
   * - selection_timeout_seconds
     - 10
     - Whole seconds SR may spend on one preview; Reef waits five seconds longer for SR's reply.
   * - preview_max_concurrency
     - 32
     - Concurrent SR previews; more are refused with 429. Changes need a restart.
   * - sr_cpus / sr_memory_bytes
     - 4 / 4294967296
     - SR CPU and memory limits.

**Startup.** The container publishes only SR's HTTP API, on
``127.0.0.1:<sr_management_port>``. Reef generates SR's credentials at each
start, and they never reach clients. Reef opens admission after SR reports the
generated runtime as active and ready, the providers answer, and one preview
per Scenario passes. ``/healthz`` stays a liveness check; it does not express
routing readiness.

**Request limits.** Reef accepts request bodies up to 64 MiB, and a body must
finish uploading within 30 seconds (408 otherwise). The preview body of an
automatic request must fit SR's 10 MiB limit (413 otherwise); the same body as
a pin is served. The tutorial lists every `error
<../../tutorials/semantic-router/README.md#errors>`__.

**Policy updates.** Edit the configured ``reef.routing_manifest`` and send
``SIGHUP`` to the ``reef serve`` process. Reef and SR do not restart. `Change
the routing policy
<../../tutorials/semantic-router/README.md#change-the-routing-policy>`__ gives
the procedure. The rules:

- An update can change targets (add, remove, enable, disable, or another
  provider model, revision or endpoint), rules, a Scenario's default target
  and its target membership. It cannot remove or disable a target that a
  default or a rule still names.
- The automatic model, the Scenario names, the domain classifier, the image,
  the deployment settings and the preview settings need a restart. An update
  never builds an image.
- A target's ``api_key_env`` can name only a variable that the Reef child had
  when it started.
- Reef first checks the new manifest's providers. A rejected update is logged
  as ``policy_rejected``, and the current policy keeps serving.
- During the update, new inference requests receive HTTP 503. Requests that
  already selected a provider finish on their original endpoint, so keep
  replaced model endpoints available until they finish.
- If an admitted request is still selecting when ``drain_timeout_seconds``
  expires, or SR rejects the change, Reef reopens the old policy. Once SR runs
  the new configuration, a failed final check keeps admission closed; Reef
  does not return to the old policy.
- If ``reef serve`` stops while an update is pending, the next start renders
  the last committed manifest, not the edited source file, and records
  ``pending_policy_recovered``. Send ``SIGHUP`` again to apply the source.
- A ``SIGHUP`` during startup or recovery is applied once the deployment is
  ready. One that arrives after a stop began is ignored.

The generation's ``lifecycle-events.jsonl`` records the results
(``policy_committed``, ``policy_rolled_back``, ``policy_rejected``).

**Shutdown and recovery.** Ctrl-C or SIGTERM closes acceptance, drains
accepted requests and pending record writes within the configured deadlines,
and then removes the container and its anonymous volumes. New inference
requests receive 503 during the drain. ``bounded`` recovery recreates the SR
container without replaying requests or restarting Reef or the model
services. A Reef failure, unresolved record writes or exhausted recovery
attempts stop the deployment.

**Ownership.** Reef records ownership under
``${XDG_STATE_HOME:-~/.local/state}/reef/routing/`` and locks it per
deployment ID, so a different run directory does not permit a second owner.
Reef removes only the containers and networks that it owns, and keeps
records, model caches and provider containers. A killed parent may leave the
SR container behind until the next start or an explicit cleanup. An
interrupted Docker operation blocks the next start until the operator
acknowledges it. See `Stop, inspect and clean up
<../../tutorials/semantic-router/README.md#stop-inspect-and-clean-up>`__.

Semantic Router image
^^^^^^^^^^^^^^^^^^^^^

``deployment.semantic_router_image`` is optional for a managed deployment.

* **The manifest names no image.** Startup builds the image from the SR commit
  pinned by this release, with ``docker buildx build``, and tags it
  ``reef-semantic-router:<commit prefix>-<recipe>``. Later starts reuse it.
  ``image_build_timeout_seconds`` bounds the build, and the Reef HTTP port is
  not open until the image is ready.
* **The manifest names an image.** Reef runs that image and builds none.

In both cases, the image's ``org.opencontainers.image.revision`` label must
match the pinned commit, or Reef creates no container. Reef removes no image
and no build cache. The tutorial's `Semantic Router image
<../../tutorials/semantic-router/README.md#1-the-semantic-router-image>`__
section gives the build's requirements and limits.

Startup reads the manifest once and fixes the image for the generation. Policy
updates keep the image, and recovery restarts it without a build. The
generation's ``snapshot_version`` covers the image. Without a named image, the
same manifest therefore gets another version on another host, or after a
rebuild, and ``python -m reef.service.deploy.routing.cli validate`` prints
``"snapshot_version": null``. Name an image when hosts with the same Reef
release and settings must get the same version.

External router lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``reef.routing_deployment.lifecycle: external`` when each Reef service runs
in its own process and another system, such as Kubernetes, runs SR. Reef then
needs no Docker. Each replica of the ``inference`` service uses one SR on its
own loopback. Every service can read the same configuration, because the other
services ignore the manifest. ``reef serve`` without ``--component`` refuses
this lifecycle.

.. code-block:: yaml

   reef:
     routing_manifest: /etc/reef/manifest.yaml
     routing_deployment:
       lifecycle: external
       sr_management_port: 50053
   recipe:
     implementation: recipe
   inference:
     # The launcher needs a reachable default; routed requests use the manifest's targets.
     upstream_url: http://127.0.0.1:8101
     upstream_model: code-student

To start a replica:

1. Set ``REEF_SR_MANAGEMENT_TOKEN`` and ``REEF_SR_SELECTION_TOKEN`` to two
   different secrets, for both SR and the replica.
2. Write SR's configuration, before SR starts:

   .. code-block:: bash

      python -m reef.service.deploy.routing.cli render \
        --config /etc/reef/reef.yaml --output /app/config/config.yaml

3. Run SR with ``/app/config/config.yaml`` as its argument, in the replica's
   network namespace. SR listens on ``127.0.0.1:<sr_management_port>``. The render
   step writes the file with mode 0600, so run SR as the user that wrote it.
4. Run ``reef serve --component inference``. Clients reach it on ``reef.host``.

Use ``GET /readyz`` as the replica's readiness check. It needs no token. It
answers 200 after SR reports its generated runtime as active and ready, every
enabled target lists its model, and one preview per Scenario passes. It answers
503 otherwise, including for requests of Scenarios outside the manifest. The
replica reads SR every five seconds and checks a new runtime hash in the same
way before requests use it.

This lifecycle differs from ``managed``:

* **Policy updates.** The replica reads the manifest once and has no ``SIGHUP``
  update. To change the policy, replace the replicas. During a rollout, two
  policies can serve at the same time. Each record's ``snapshot_version`` shows
  which one served it.
* **Scope.** A manifest Scenario's ``POST /v1/chat/completions`` follows the
  routing rules. Every other Scenario and route keeps the ordinary
  ``inference`` behavior. Clients must call the ``inference`` service, because
  the other services do not route.
* **Stopping.** On SIGTERM the replica stops accepting connections and waits up
  to 60 seconds for requests. It then cancels the rest and waits
  ``finalization_timeout_seconds`` for their records. Give the process more
  time than this before it is killed.
* **Your responsibilities.** Reef does not build, start, restart or stop SR,
  and it does not check the image's revision label. ``render`` and the replica
  both require ``deployment.semantic_router_image``. Name an image built from
  this Reef release's SR commit; one that a managed start built qualifies.
* **Settings.** Only ``sr_management_port``, ``selection_timeout_seconds``,
  ``preview_max_concurrency`` and ``finalization_timeout_seconds`` apply. Set
  SR's CPU and memory limits where you run it.

Configuration file layout
~~~~~~~~~~~~~~~~~~~~~~~~~

The sections of a configuration file match the CLI paths. For example, the
local inference command above can also use:

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
An explicit CLI value overrides the YAML value, and an omitted field uses its
declared default. Section names and declared fields accept hyphens or
underscores; both spellings of one field in one file are an error. Reef
rejects unknown sections and undeclared fields.

.. list-table::
   :header-rows: 1

   * - Section
     - Holds
   * - ``reef``
     - HTTP host, port, authentication, run directory, readiness deadline, routing and scenario limits
   * - ``inference``
     - model path, backend, GPUs, provider connection, serving mode and native engine ``options``
   * - ``recipe``
     - ``implementation`` (the recipe class or preset), ``config`` (the recipe's declared fields) and ``runtime`` (a runtime type and its settings)
   * - ``training``
     - backend, timeouts, Ray connection, workload variables (``config``) and native training ``options``
   * - ``teacher``
     - a frozen SGLang teacher engine
   * - ``evaluation``
     - sandbox limits and plugins of the evaluation service
   * - ``storage``
     - artifact, record, data and task-file stores, and record retention
   * - ``execution`` / ``executors``
     - executor selection for each role, and named executor profiles
   * - ``activation``
     - the rule that decides which new artifacts serve
   * - ``pipeline``
     - the config, and optionally the class, of each pipeline stage
   * - ``generator``
     - the generator service that writes and checks Harbor tasks and runs their rollouts
   * - ``data``, ``task-import``
     - the data service's limits, and the Harbor tasks that each scenario imports
   * - ``harness``
     - how harness clients reach the deployment's services
   * - ``observability``
     - experiment tracking (``wandb``) and record tracing (``tracing``)

A configuration file contains no process definitions: ``service``,
``services`` and ``execution.services`` are rejected. The selected recipe and
the inference and training backends determine the processes.

**Processes of a weight-training stack.** With the default Slime backend and
an engine (``inference.backend: sglang`` or ``vllm``), ``reef serve`` starts:

- ``inference``: a local inference component (``python -m
  reef.inference.remote``) on a free loopback port. It starts the serving
  engines and loads each published snapshot into them. An ``inference.remote``
  section replaces it with a component on another host.
- ``distill-teacher``: a frozen SGLang teacher engine, only when a ``teacher``
  section is set.
- ``reef``: the HTTP service. It hosts the training component, which starts a
  Slime driver on Ray for each job.

Training and serving use separate processes and GPUs. The training component
saves checkpoints. Reef publishes them as snapshots, and the serving engines
load only published snapshots, from files. The inference component and the
HTTP service join the same Ray cluster, so the engines and the training
workers reserve separate GPUs. `Training components
<../developer-guide/training-components.rst>`__ and `Serving runtimes
<../developer-guide/serving-runtimes.rst>`__ describe the job and serving contracts.

.. container:: config

   ``training.backend`` : slime
       ``slime``, ``tinker`` (remote LoRA training without local GPUs, `Train with Tinker <../user-guide/tinker.rst>`__), ``spindle_full`` (all the parameters of one Hugging Face revision on a Spindle deployment, `Train all the parameters on Spindle <../user-guide/tinker.rst#train-all-the-parameters-on-spindle>`__), an installed ``reef.training_backends`` entry-point name, or the ``package.module:ClassName`` of a ``TrainingDeployment`` class. The selected backend validates its own settings. Other backends do not inherit Slime's Ray or native-argument requirements

   ``training.ray-address`` : ``RAY_ADDRESS``
       the Ray cluster that the inference component and the HTTP service join. Without an address, Reef starts a local cluster and stops it on exit; Reef leaves an existing cluster running

   ``training.ray-namespace`` : reef
       the Ray namespace of the training component

   ``training.ready-timeout`` : 3600
       startup deadline, in seconds, of the services that the training backend adds, such as the inference component and the teacher engine

   ``reef.ready-timeout`` : 30
       HTTP startup deadline in seconds. When the HTTP service loads the model into training workers or starts serving engines before it answers, as with Slime and whenever Reef's engines serve, Reef raises it to at least ``training.ready-timeout``

   ``training.max-jobs`` : 1
       training jobs that one process of the training service runs at the same time for each training component. Several jobs of one scenario run together only when the component allows it; the Tinker component does (`Training components <../developer-guide/training-components.rst>`__). Each running job needs its own compute

   ``training.manager`` : unset
       a ``TrainingManager`` plugin (``reef.train.manager``), as ``package.module:ClassName`` or as ``module`` and ``config``. Without it, a scenario in ``auto`` mode starts a job when ``job_samples`` untrained samples wait. With it, the manager decides (`Decide what a scenario trains <../developer-guide/training-components.rst#decide-what-a-scenario-trains>`__)

   ``training.timeout-s`` : four hours with Slime
       bound, in seconds, of each call to the training workers (Slime: each worker call of a job, such as training and checkpoint saves; Tinker: each remote call). A positive finite number. A timeout stops the wait, not the work on the workers. The job then fails, Reef does not retry it, and the next job starts from the last saved checkpoint

   ``teacher.model-path`` : unset
       the teacher engine's Hugging Face model or directory

   ``teacher.num-gpus`` : 1
       the teacher engine's GPUs, also its tensor parallel size

   ``teacher.port`` : 30001
       the teacher engine's internal HTTP port

   ``teacher.timeout`` : 300
       deadline of one teacher scoring request, in seconds

   ``teacher.options`` : {}
       native SGLang options of the teacher engine

Training without a file uses the same path: pass
``--recipe.implementation package.module:WeightRecipe``,
``--inference.model-path`` and the ``--training.options.*`` flags that the
recipe needs.

An in-process integration can use ``InProcessTrainingDeployment``. It starts
only the HTTP service, whose runtime factory builds the training component in
that process and parses ``training.options``. The options map cannot override
the runtime type, the model or the shared timeouts. Such an integration
rejects Ray and training executor settings.

Serving of a weight-training deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``inference`` section alone decides how a weight-training deployment
serves its snapshots. The training backend builds only its training
component. That component declares the format of its snapshots: full weights,
or a PEFT adapter with its rank and target modules.

.. container:: config

   ``inference.serving`` : ``engines`` when an engine setting is set, else the training backend's default
       ``engines``: Reef's engines on an inference component load each snapshot (Slime's default); ``trainer``: the training backend serves its own snapshots (Tinker's sampler, Tinker's default); ``none``: nothing serves; ``module:Factory``: a serving ``RuntimeFactory``

   ``inference.serving-config`` : {}
       the settings of a ``module:Factory`` serving runtime; it also receives the artifact format as ``artifact_format``, ``null`` when the snapshots hold no weight files

   ``inference.num-gpus`` : ``inference.tensor-parallel-size``
       the total inference GPU budget; with ``inference.remote.slot-urls``, the budget of each component. A positive multiple of the tensor parallel size: 4 GPUs with size 2 make two engines on one component. Standalone serving supports one engine, so its budget must equal the size. An external provider accepts no GPU request

   ``inference.tensor-parallel-size`` : 1
       GPUs of each engine

   ``inference.max-loaded-adapters`` : 1
       adapters the engines keep resident when the snapshots are PEFT adapters. Each scenario keeps its served adapter resident, so set at least the number of training scenarios; one more lets a scenario's new adapter load beside its served one

   ``inference.prefix-sharing`` : false
       ``true``: requests on the same weights share the engines' prefix cache. The engines finish their requests and clear the cache before each change of weights

The engine settings are ``inference.backend``, ``inference.num-gpus``,
``inference.tensor-parallel-size``, ``inference.options``,
``inference.remote``, ``inference.handler-factory``,
``inference.handler-config``, ``inference.max-loaded-adapters`` and
``inference.prefix-sharing``. Another serving mode refuses them.

.. code:: yaml

   inference:
     model-path: Qwen/Qwen2.5-1.5B-Instruct
     num-gpus: 1
     tensor-parallel-size: 1
     options:
       mem-fraction-static: 0.6
       router-port: 30000  # SGLang engines of a training stack only
   training:
     backend: slime
     options:
       actor-num-nodes: 1
       actor-num-gpus-per-node: 1
       global-batch-size: 32
       # Add the recipe's optimizer, model and checkpoint options here.

**What the engines serve.** The HTTP service uploads each published snapshot
to the inference component as files:

- Full-weight training publishes the checkpoint's Hugging Face export. The
  component reloads it in place of the served weights. Each load moves the
  whole model, for example about 16 GB for an 8B bf16 model.
- LoRA training (``training.options.megatron-lora-rank``) publishes the
  scenario's PEFT adapter. The component loads it as the scenario's adapter
  beside the base model. Only ``inference.backend: sglang`` serves adapters.
- Snapshots without weight files, such as Tinker's, are served by engines only
  when the training backend names how to fetch their weights
  (``TrainingDeployment.engine_weights``); Tinker's does. Otherwise the HTTP
  service refuses the pairing when it starts.

The training service stores each checkpoint in the artifact backend from the
checkpoint directory, and every other service reads it from the artifact
backend. With multi-node training, put the checkpoint directory on storage
that the training service's host can read.

**Request pinning.** With ``engines``, the inference component keeps each
production request on the weights that the request selected (`Request pinning
<../developer-guide/serving-runtimes.rst#request-pinning>`__). No setting turns this
off. It needs ``inference.backend: sglang`` or ``vllm`` with the engine's own
request handler, so Reef refuses another backend or
``inference.handler-factory`` here. Another engine needs its own serving
runtime (``inference.serving: module:Factory``). With ``trainer``, Tinker's
sampler serves each snapshot as an immutable revision and needs no pinning.

**Prefix sharing.** With ``inference.prefix-sharing: true``, the engines clear
their prefix cache before each change of weights or adapters, after the
admitted requests finish. The clear covers the engine's own cache, which
includes SGLang's host-memory tier (``enable_hierarchical_cache``). Engines
that do not confirm the clear stay unavailable until the control plane
replaces them. For adapter artifacts, SGLang keeps the cache only when it keys
the entries by adapter; otherwise the engines run without a prefix cache.

The clear does not cover KV that an engine keeps outside itself. Engine
serving therefore refuses these ``inference.options``, with or without prefix
sharing: SGLang's ``hicache_storage_backend`` and ``enable_lmcache``, and
vLLM's ``kv_transfer_config``, ``kv_offloading_size`` and
``kv_offloading_backend``. Reef checks these option names only. Do not give
the engines another KV store through a plugin or an environment variable.

**Engine failures.** The control plane checks the inference component every
30 seconds. When the component has restarted, or an engine has failed, the
control plane pauses generation, replaces the component's engine stack, loads
each scenario's served artifact again, and resumes (`Engine supervision
<../developer-guide/serving-runtimes.rst#engine-supervision>`__).

**SGLang engines.** Native engine options use SGLang's names without a
``sglang-`` prefix, for example ``--inference.options.mem-fraction-static
0.7``. ``router-port`` binds the router's HTTP port, and other router flags
keep their native ``router-*`` names. The standalone engine launcher includes
no router. `Independent SGLang backend
<../developer-guide/serving-runtimes.rst#independent-sglang-backend>`__ describes the
integration's boundary.

Each engine and the router probe free ports on the host at startup; the
router's metrics port is probed from 4000 up. If a start fails because another
process took a probed port, Reef starts the engine or the router again on new
ports, at most three attempts each. Other startup failures stop the launch. A
configured ``router-port`` does not change, so a conflict on it stops the
launch. Another stack's metrics port can take a ``router-port`` near 4000, so
use a port outside that range.

**vLLM engines.** ``inference.backend: vllm`` launches one vLLM server per
engine with ``VLLM_SERVER_DEV_MODE=1`` and ``--logprobs-mode
processed_logprobs``. It loads each published snapshot through vLLM's control
routes: ``/pause?mode=keep``, ``/update_weight_version``, ``/collective_rpc
reload_weights`` and, with ``inference.prefix-sharing: true``,
``/reset_prefix_cache?reset_running_requests=true``.

- Engine options use vLLM's engine-argument names. Reef sets ``model``,
  ``host``, ``port`` and ``tensor_parallel_size`` itself and refuses
  ``enable_sleep_mode``.
- Reef defaults ``generation_config`` to ``vllm``, so the model's own
  generation defaults cannot add truncation or a temperature that the trainer
  never sees.
- Reef enables prefix caching only with ``inference.prefix-sharing: true``.
- vLLM engines serve full-weight artifacts only. The HTTP service refuses
  adapter artifacts, such as Slime's with
  ``training.options.megatron-lora-rank`` or Tinker's, when it starts.
- Reef starts no vLLM router, so one single-node engine serves the stack:
  ``inference.num-gpus`` must equal ``inference.tensor-parallel-size``.
- Each engine takes one port, probed upward from ``engine_port_base`` (default
  15000). Reef restarts an engine that loses its port to another process, at
  most three attempts.

**Slime's training options.** Slime's own engine and placement options in
``training.options`` (``rollout-num-gpus``, ``colocate``, ``sglang-*``,
``router-*`` and the rest) reach each job's Slime driver unread. They configure
the rollout engines of Slime's serving session (`Task jobs on Slime`_), never
the serving engines.

- Reef's serving never shares GPUs with training, so ``training.colocate`` is
  refused.
- Native options cannot override the managed model, placement or parallelism
  settings. Each serving engine runs on one node. The managed path supports no
  pipeline or data parallel topology and no prefill/decode disaggregation.
- Reef binds ``training.options.hf-checkpoint`` to ``inference.model-path``;
  an explicit value must agree.
- Slime needs an explicit ``training.options.global-batch-size``, the samples
  in one optimizer step.
- ``training.options.actor-num-*`` set the training GPUs, and
  ``rollout-num-gpus`` gives rollout engines GPUs of their own. The physical
  node size is ``training.options.num-gpus-per-node``.
- Until the first task job, each job's driver reserves the training GPUs only
  while it runs. From the first task job on, Slime's serving session holds
  them until the component closes.
- The inference component reserves ``inference.num-gpus`` GPUs on the same Ray
  cluster. The inference control process takes one Ray CPU and no GPU.
- When a training worker fails, the running job fails, and the next job starts
  again from the latest checkpoint. A configuration error that only Slime's
  workers detect fails the first job.

**Other services.** Reef starts only its own inference, training and HTTP
services. OpenClawRL deploys its PRM and user-simulation services itself; the
recipe reads ``recipe.config.prm-url`` and
``recipe.config.prm-tokenizer-path``.

Train without an inference engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``inference.backend: none`` runs no inference engine. Use it for supervised
fine-tuning on data that does not come from the served model, such as uploaded
trajectory items (``POST /reef/namespaces/{owner}/data``) or imported records
(``POST /reef/records``). Only the Slime backend supports it. The trainer
takes every GPU that ``training.options.actor-*`` names:

.. code:: yaml

   inference:
     model-path: /models/Qwen3-8B   # the weights to train; nothing serves them
     backend: none
   training:
     backend: slime
     options:
       actor-num-nodes: "1"
       actor-num-gpus-per-node: "8"
       global-batch-size: "32"
       # Add the recipe's optimizer, model and checkpoint options here.

Training in this mode works as follows:

- Each job trains its data on Slime's workers, which stay on their GPUs
  between jobs.
- Each job saves one checkpoint: the Megatron checkpoint and its HF export.
  This mode refuses an ``evaluation`` section, so Reef publishes every
  checkpoint as a snapshot. See `Set the checkpoint cadence
  <../user-guide/operate.rst#set-the-checkpoint-cadence>`__.
- A restart continues from the last saved checkpoint, as with an engine.

The deployment serves no inference. ``/v1/chat/completions``,
``/v1/responses`` and ``/v1/messages`` answer 404 for every scenario. To use a
trained version, serve its published checkpoint from a separate deployment.

``reef serve`` refuses the settings that configure an engine or call the
model: ``inference.num-gpus`` (other than 0),
``inference.tensor-parallel-size``, ``inference.options``, ``inference.url``,
``inference.remote``, ``inference.handler-factory``,
``inference.handler-config``, ``training.colocate``, a ``teacher`` engine, an
``evaluation`` section, a ``generator`` section and a composite recipe.

Native backend options
~~~~~~~~~~~~~~~~~~~~~~

``inference.options`` and ``training.options`` hold the selected backend's
native flags, without the leading ``--``. The backend owns their types,
defaults and validation; Reef does not copy that schema.

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8 \
     --inference.options.trust-remote-code true

These become ``--mem-fraction-static=0.8`` and ``--trust-remote-code``
arguments of ``python -m sglang.launch_server``.

- For engines that take command-line arguments, such as SGLang and Slime,
  ``true`` emits a switch and ``false`` or ``null`` omits it. To disable a
  feature that the engine enables by default, use the engine's own disabling
  flag. A list supplies several argument values, and an object is passed as
  JSON.
- In-process training passes the values to its runtime parser: ``false`` stays
  false, and null follows the declared field type.
- A field override keeps its YAML siblings. An explicit whole object, such as
  ``--inference.options '{}'``, replaces the entire object.
- In managed serving, native options cannot override the model, the served
  model name, the tensor parallel size, the bind address, authentication or
  multi-node launch controls. Use the public fields for those.

For Slime, ``training.options`` holds flags such as ``lr: 0.000001`` and
``use-critic: true``; override one with ``--training.options.lr 0.000002``.
The training component parses the options once at startup, so a malformed
option fails before the first job. The driver takes out Reef's own options,
such as ``--reef-checkpoint-policy``, and the loss family's options, and
passes the rest to Slime's parser.

**Request handlers.** ``inference.handler-factory`` names an
``InferenceHandler`` subclass with a ``from_config`` class method, as
``package.module:ClassName``; handler functions are not supported.
``inference.handler-config`` holds that handler's options, for example its
tool parser. It does not configure the engine process.

Two token-native handlers render the prompt with the model's chat template
and call the engine's native generate route:

.. list-table::
   :header-rows: 1

   * - Handler
     - Engine route
     - Streaming
   * - ``reef.inference.sglang.chat:SGLangInferenceHandler``
     - SGLang ``/generate``
     - relays the engine's incremental stream
   * - ``reef.inference.vllm.chat:VLLMInferenceHandler``
     - vLLM ``/inference/v1/generate``
     - answers a streaming request with the completed turn as one burst of protocol frames

Both handlers answer with the provider response, without training tensors. A
client gets token log probabilities only when it sends the OpenAI ``logprobs:
true``, and token IDs only when it sends vLLM's ``return_token_ids: true``.
Per-request sampling extras pass through ``sglang_sampling_params`` or
``vllm_sampling_params``. Both handlers accept these options; any other
option, such as ``capture_topk``, fails the handler's start.

.. container:: config

   ``tool_call_parser``
       the engine's parser name

   ``sampling_defaults``
       sampling fields for a request that does not set them. A scenario can set ``temperature``, ``top_p`` and ``top_k`` over them with its `serving config <http-api.rst#serving-config>`__; a scenario's value is also a default, and it applies alike to OpenAI chat and Anthropic messages requests

   ``chat_template_kwargs``
       chat template arguments for a request that omits them, for example ``{enable_thinking: false}``; a request's own ``chat_template_kwargs`` win key by key

   ``force_reasoning`` : unset
       ``true`` if the chat template pre-opens ``<think>``, ``false`` if it does not. Unset, the handler checks whether each request's rendered prompt ends with ``<think>``, so the check follows the request's ``chat_template_kwargs``

Both handlers, and the Tinker sampler, split each reply into ``content`` and
``reasoning_content`` with the rules of SGLang's Qwen3 reasoning parser:
``<think>`` starts reasoning, and the first ``</think>`` or ``<tool_call>``
ends it. A reply that ends inside reasoning has an empty ``content``. The
Tinker sampler takes no ``force_reasoning`` and always checks the prompt.

**Log probabilities of task rollouts.** ``rollout_log_probs`` are the sampled
tokens' log probabilities that a training component's own task rollouts
record; serving records none. They must mean the same thing as the trainer's.
Sampling runs through::

   raw logits -> penalties, logit_bias -> / temperature -> [A] -> top-k, top-p, min-p -> [B] -> sample

SGLang and Slime's trainer both read at [A] (full vocabulary, trainer with
``rollout_temperature``, no penalties), so they agree as long as a recipe uses
no penalties or ``logit_bias``. vLLM ``--logprobs-mode processed_logprobs``
reads at [B], so it matches only with top-k, top-p and min-p off. Reef's vLLM
client rejects a request whose effective ``top_p`` is below 1, ``top_k``
above 0 or ``min_p`` above 0. Verify with Slime's
``train_rollout_logprob_abs_diff`` on identical weights before training.

Chat template arguments
~~~~~~~~~~~~~~~~~~~~~~~

Serving records the provider exchange only. The training component renders
each recorded call with the model's chat template, and must render it as the
serving engine did. Give the training component the same chat template
arguments that the engine renders with: the handler's
``chat_template_kwargs``, or the arguments that the clients send.

- Slime renders with the tokenizer of ``--hf-checkpoint`` and
  ``training.options.apply-chat-template-kwargs``, a JSON object.
- Tinker renders with the base model's tokenizer and
  ``training.options.chat_template_kwargs``, a mapping.

.. code:: yaml

   inference:
     handler-config:
       chat_template_kwargs: {enable_thinking: false}
   training:
     backend: slime
     options:
       apply-chat-template-kwargs: '{"enable_thinking": false}'

A sample that cannot render, or that is longer than a limit of its stage,
does not train. The job metric ``unrenderable_samples`` counts these samples.
A chat template that drops the reasoning of earlier turns, as Qwen3's does,
renders a multi-turn sample as several segments. Each segment trains as one
sample (see `Processors <../developer-guide/processors.rst>`__).

Explicit file selection
~~~~~~~~~~~~~~~~~~~~~~~

``reef serve -c reef.yaml`` selects a configuration file. Without ``-c``, Reef
uses the CLI settings alone: it does not read ``REEF_CONFIG`` or
``./reef.yaml``, and it does not search the Reef installation. A selected file
that is missing or invalid is an error. Relative config and state paths
resolve from the directory where you run the command; from outside a checkout,
pass an absolute path to a cookbook config. Logs go to ``.reef/run/``; set
``reef.run-dir`` to move them.

Before any model download or process start, the launcher logs the selected
source and every resolved setting with where its value came from:

.. code:: text

   [reef] config: /srv/reef/stack.yaml
   [reef] resolved settings:
   [reef]   recipe.implementation = recipes.sao.recipe:SAORecipe  (file)
   [reef]   reef.host = 127.0.0.1  (default)
   [reef]   reef.port = 9000  (command line)
   [reef]   reef.tokens = ["****"]  (file)
   [reef]   inference.model-path = /models/demo  (file)
   [reef]   inference.backend = sglang  (automatic)
   [reef]   training.backend = slime  (automatic)

Sources are ``file``, ``command line``, ``environment`` (the ``REEF_*``
variables of a start without a file, or a field's declared fallback variable),
``automatic`` (a choice Reef made because the field was omitted) and
``default``. The report lists a setting at its default only for the recipe and
the HTTP bind. It masks tokens, API keys, passwords and database URLs by key
name at any depth, including inside native ``options`` objects.

``reef serve ... --print-config`` prints the same report on standard output,
defaults included, and exits with status 0, or 2 for an invalid config. It
downloads no model and starts no process. It takes the same ``-c``,
``--model`` and override flags as a real start. A Hugging Face model path is
shown as written; the snapshot is resolved only at startup.

.. code:: bash

   reef serve -c stack.yaml --reef.port 9000 --print-config

**Value rules.** YAML and CLI values use the same parser, and ``reef serve
--help`` lists the public options.

- **Precedence.** CLI, then YAML, then a declared environment fallback, then
  the default. The last occurrence of a flag wins.
- **Strings** keep their text: ``--inference.upstream-model 00123`` stays
  ``00123``, and the literal string ``null`` stays text.
- **Numbers and booleans** are parsed by their declared type. A boolean takes
  an explicit value or a bare flag, and ``--no-...`` disables a YAML setting,
  for example ``--no-reef.allow-implicit-scenario-creation``. False, zero and
  empty values are explicit values. Service fields, resources and timeouts
  must be finite. Recipe floats may be non-finite unless the recipe declares
  ``allow_nonfinite=False``.
- **Lists and objects** take one quoted JSON or YAML value, for example
  ``--reef.tokens '[]'`` or ``--inference.handler-config '{"tool_call_parser":
  "qwen25"}'``. An explicit container, empty or not, replaces the YAML value.
- **Empty string.** An empty service value, such as an unset ``${VAR}``, counts
  as omitted and keeps the default.
- **Null.** An optional field (``str | None`` and similar) resolves to null.
  ``null`` for any other field is an error that names the field. Omit the
  field to use its default.
- **Duplicates.** A file that repeats a YAML key is rejected with both line
  numbers. Two spellings of one field (``model-path`` and ``model_path``) are
  a duplicate field error.
- **Errors** name the field and the expected type without its value. An
  invalid value fails before any model download or process start.
- **Tokens.** ``reef.token`` and ``reef.tokens`` are distinct inputs; Reef
  accepts the credentials of both.

**Interpolation.** A value interpolates from the environment with ``${VAR}``
and from the config itself with ``${dotted.path}``, such as ``${reef.port}``.

- ``${VAR}`` resolves to an empty string when the variable is unset. Use it
  for optional values, such as the API key of a provider without
  authentication.
- ``${VAR:?}`` requires the variable, for example ``upstream_model:
  ${REEF_UPSTREAM_MODEL:?}``. If it is unset, empty or only whitespace, Reef
  reports the missing variable names and their config fields before it
  downloads models or starts processes. Command-line overrides apply first, so
  ``--inference.upstream-model <model-id>`` can supply the value instead.

``REEF_PYTHON`` names the interpreter of the services and defaults to the one
that launched ``reef serve``. A literal ``python`` in a service command keeps
its normal meaning and resolves from that service's ``PATH``.

Component configuration
~~~~~~~~~~~~~~~~~~~~~~~

After Reef selects ``recipe.implementation``, it loads that class's declared
fields without constructing the recipe.

- A recipe's fields are ``--recipe.config.<field>``, for example
  ``--recipe.config.training-mode``. A weight-training recipe also accepts the
  aliases ``--training-mode``, ``--training_mode`` and
  ``--reef.training_mode``. The ``--reef.data.*`` spellings remain aliases of
  ``--recipe.config.*``.
- ``reef serve -c stack.yaml --help`` includes the selected recipe's flags.
  Basic ``reef serve --help`` loads no recipe.
- The selected package must be importable in the launcher and in the services.
- Declared fields follow the value rules above. A declaration that conflicts
  with a public option is rejected, and an unknown field fails validation.
- A recipe can declare opaque sections in ``config_sections``; for example,
  ``HarnessEvolutionRecipe`` owns ``recipe.config.evolution``. A section takes
  object and leaf CLI overrides, and its recipe validates the payload.
- A standalone recipe file that an embedding script loads
  (``load_recipe_config``, ``recipe_config_from_file_layout``) uses the same
  layout.

A dotted recipe that trains no weights can set ``recipe.runtime`` to select a
runtime factory: a bundled kind, or the ``package.module:ClassName`` of a
``RuntimeFactory`` subclass. Its declared fields use names such as
``--recipe.runtime.timeout-s``. A custom factory declares its fields with
``config_type()``; a factory without one receives the section as a mapping.

Executor fields are declared by ``ExecutorSettings`` and ``WorkerResources``
(`Stack execution backends`_):

.. code:: bash

   reef serve -c stack.yaml \
     --execution.evolution.workers 4 \
     --execution.evolution.resources.cpus-per-worker 0.5

A nested override of a named executor profile makes a local copy for that
role; it does not change the shared profile.

**Startup failures.** If a service exits before readiness or exceeds its
readiness deadline, Reef stops the stack and reports the service, the reason
and the log directory; an exited service's message includes its exit code. A
startup failure exits with status 1, and an invalid configuration with status
2. Service output goes to the logs and to the terminal. SIGINT or SIGTERM
during startup also stops the services already started.

The basic and SAO example ``run.sh`` launchers wait for the HTTP health
endpoint, with a five-second timeout for each probe, and stop waiting if Reef
exits. ``reef.ready-timeout`` is the only startup deadline. Startup errors
are recorded in ``work/reef.log``, and exiting the script stops the Reef
process that it started.

Start from a cookbook stack
---------------------------

The source checkout's runnable stacks live under the ``recipes/`` cookbook
and ``tutorials/``.

+-------------------------------------------------------------+----------------------------------------------------------+
| File                                                        | What it starts                                           |
+=============================================================+==========================================================+
| ``recipes/basic/external-provider.yaml``                    | no GPU, no local model: one Reef process proxying an     |
|                                                             | HTTP provider                                            |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/basic/local-sglang.yaml``                         | local inference: an SGLang server plus Reef, no training |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/<method>/examples/<example>/serve.yaml``          | weight training: an inference component and Reef, which  |
|                                                             | trains with Slime                                        |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``tutorials/evolve-your-harness/configs/serve.yaml``        | harness evolution: one Reef process, no GPU; ``run.sh``  |
|                                                             | materializes its recipe preset and starts the stack      |
+-------------------------------------------------------------+----------------------------------------------------------+

``recipes/sao/examples/imo_answerbench/serve.yaml`` is the smallest
weight-training stack: two GPUs for one actor and one rollout engine.
``recipes/tttd/examples/tttd/serve.yaml`` adds LoRA training, and
``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds a PRM engine and a
student model.

Service and storage settings
----------------------------

The HTTP service reads these fields from the ``recipe``, ``reef``,
``inference``, ``harness``, ``storage`` and ``evaluation`` sections.

.. container:: config

   ``recipe.implementation`` : recipe
       the recipe this deployment serves (`Recipe configuration`_)

   ``reef.host`` : 127.0.0.1
       bind address

   ``reef.port`` : 8900
       bind port

   ``reef.run-dir`` : .reef/run
       the directory of the service logs

   ``reef.ready-timeout`` : 30
       HTTP startup deadline in seconds (`Configuration file layout`_)

   ``reef.served-url``
       the URL at which a composite recipe's own evaluation calls (runs and the method's calls to the served model) reach this service; the default is loopback on the bind port, so set it when tasks run on another host. A recipe of one component calls its runtime's endpoint directly and ignores it

   ``reef.console-origins`` : []
       exact browser console origins allowed to access the HTTP service; disabled by default

   ``reef.token``
       the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.

   ``reef.allow-unauthenticated`` : false
       when true, serve without a token on a ``reef.host`` that other machines can reach. When false, Reef does not start on such an address without a token. A loopback address, such as ``127.0.0.1``, ``::1`` or ``localhost``, needs no token

   ``reef.allow-implicit-scenario-creation`` : true
       when false, a report, a record import or a harness install for an unknown scenario is HTTP 404. An inference request never creates a scenario, whatever this setting says

   ``reef.scenario-idle-unload-s`` : unset
       unload a scenario that is idle for this many seconds; the next request loads it again. This key and the three after it are described in `Limit the scenarios a service holds <../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__

   ``reef.max-loaded-scenarios`` : unset
       the most loaded scenarios; at the limit, the idle scenario with the oldest use unloads

   ``reef.max-scenarios`` : unset
       the most registered scenarios

   ``reef.scenario-admission`` : unset
       a ``ScenarioAdmission`` plugin reference that replaces the built-in limits

   ``reef.routing_manifest`` : unset
       the manifest of routing among external targets (`Scenario routing between students and teachers`_)

   ``reef.routing_deployment``
       the settings of the routing deployment (`Managed routing lifecycle`_)

   ``inference.model-path``
       a local HF model directory or a repo id, downloaded on start

   ``inference.upstream-url``
       the OpenAI-compatible provider, with no ``/v1`` suffix

   ``inference.upstream-api-key``
       its credential. Reef is the only party that sees it. It is the only credential that Reef sends to an engine

   ``inference.upstream-model``
       the model to request upstream

   ``inference.upstream-api`` : openai
       the provider dialect: ``openai`` for Chat Completions, ``responses`` for OpenAI Responses, or ``anthropic`` for an Anthropic-style endpoint

   ``inference.url``
       reserved: every built-in deployment refuses it, because Reef starts or names its engines itself (``inference.model-path``, ``inference.upstream-url`` or ``inference.remote``)

   ``harness.inference-url``
       the URL at which clients reach the inference service. A client's harness layer (``GET /reef/harness?form=client``) sends its model calls to ``<url>/reef/scenarios/<scenario>/v1``. The default is the ``service_url`` that the client gave, which is right when one process runs every service

Storage paths default under ``.reef/``, which the basic and sao stacks keep;
the openclawrl stack overrides them to ``/var/lib/reef``. Point them somewhere
persistent.

.. container:: config

   ``storage.artifact-backend`` : git_lfs
       where snapshots are stored: ``git_lfs``, or a ``ConfigurableRepositoryBackendFactory`` plugin reference

   ``storage.artifact-repository`` : .reef/artifacts.git
       the Git LFS repository of the snapshot chain; only for ``git_lfs``. A Hugging Face repository needs the factory's ``refs`` and ``token_env`` options (`Choose the artifact backend <../user-guide/storage.rst#choose-the-artifact-backend>`__)

   ``storage.artifact-work-dir`` : .reef/artifact-work
       materialization scratch

   ``storage.artifact-cache-dir`` : .reef/artifact-cache
       fetched artifact cache; artifact imports run under ``imports/`` in it

   ``storage.artifact-sources`` : []
       ``ArtifactSourceReader`` plugin references that read artifact import URIs, beside the built-in ``hf``, ``file`` and ``s3`` readers

   ``storage.agent-record-dir`` : .reef/agent-record
       the SQLite scenario storage: records, commits and settings

   ``storage.record-backend`` : sqlite
       where scenario records, commits and settings live: ``sqlite``, ``postgres``, or a ``ScenarioStorage`` plugin reference

   ``storage.record-database-url`` : unset
       the PostgreSQL connection URL, for ``postgres``

   ``storage.record-database-schema`` : reef_records
       the PostgreSQL schema of the scenario storage

   ``storage.coordination-backend`` : follows ``record-backend``
       where the services share their work queues, served models and control lease: ``sqlite`` (processes on one host) or ``postgres`` (``record-database-url``); required when ``record-backend`` is a plugin

   ``storage.work-max-attempts`` : 3
       how many times a worker takes one work item (a training job, a record selection, an evaluation rollout, a data task) before the item fails. An attempt that ends because its service stops does not count

   ``storage.work-retention-s`` : 604800
       seconds that control keeps an ended work item, so that its producer reads the result; then control deletes it

   ``storage.task-file-backend`` : local
       where the files of generated tasks live: ``local``, or a ``TaskFileStore`` plugin reference

   ``storage.task-file-dir`` : .reef/task-files
       the files of generated tasks for ``local``, or the local cache of a plugin store

   ``storage.data-store`` : follows ``record-backend``
       where the deployment's data items live: ``sqlite``, ``postgres`` (``record-database-url``, schema ``reef_data``), or a ``DataStore`` plugin reference

   ``storage.segment-backend`` : unset
       where a SQL data store keeps large content as JSON Lines segments: unset (in the database), ``local`` (``<agent-record-dir>/segments``), or a ``SegmentStore`` plugin reference

   ``storage.job-directory-ttl-s`` : 604800
       seconds that the data store keeps a job data directory that no training job used; a job of the same samples uses the directory again (see `The job data directory <../developer-guide/processors.rst#the-job-data-directory>`__)

   ``storage.agent-record-retention-max-bytes`` : 21474836480
       20 GiB shared across all record items of the data store, including unconsumed records

.. warning::

   On ephemeral storage, a restart loses the scenario storage (with SQLite) and
   every version. See `Back up a deployment
   <../user-guide/storage.rst#back-up-a-deployment>`__.

**Record retention.** ``storage.agent-record-retention-max-bytes`` is a budget
for the stored record bodies. Training does not delete the records that it
consumed. The HTTP service runs a cleanup at startup and every 60 seconds.

- **What counts.** The budget measures the stored UTF-8 JSON of payloads,
  references and artifact references, across active and archived scenarios.
  SQLite shares the budget across the record directory, and PostgreSQL across
  the deployment schema. The history and the tools that an agent resends in
  every model call count once. The model calls of evaluation rollouts are
  outside the budget and are never evicted.
- **What is evicted.** Over the budget, storage evicts the oldest records by
  creation time, consumed or not. It keeps the model calls that a stored
  training sample holds. Records do not expire by age. Each eviction logs a
  warning with the scenario, record count, sequence range and freed bytes.
- **After an eviction.** The pipeline status shows ``record_data_incomplete``,
  ``evicted_record_count`` and ``evicted_body_bytes``, and the metrics show
  ``records/evicted_count`` and ``records/data_incomplete``; the totals
  survive a restart. Reef skips, with a warning, a pending report whose inputs
  are no longer stored. Retry hashes and commit metadata survive, so a retried
  upload neither restores evicted records nor trains them again. A consumer
  that reads records several times must tolerate missing records.
- **Limits.** The budget is not a filesystem quota. Concurrent writes can
  exceed it between cleanups, and indexes, retry hashes, commit histories and
  the WAL need more disk. SQLite reuses freed pages without shrinking its
  file. A failed cleanup is logged and retried. A standalone store needs an
  explicit maintenance call; see `Python API <python-api.rst>`__.

The ``evaluation`` section configures the evaluation service:

.. container:: config

   ``evaluation.agent_host`` : host.docker.internal
       the address at which a task's container reaches this host, for an evaluation on ``docker`` whose harness is an
       artifact item. In a remote environment, the agent bridges the container to this host through Harbor's exec
       and needs no address. See `HTTP API <http-api.rst>`__.
   ``evaluation.max_sandboxes`` : 4
       the task sandboxes that one process of the evaluation service holds at once, across all evaluations. Each
       running rollout holds one. Set it to the quota of the sandbox provider, divided by the evaluation processes.
   ``evaluation.on_stop`` : requeue
       how an evaluation worker handles running rollouts when shutdown begins (for example, on SIGTERM).

       - ``requeue`` cancels running rollouts and stops their sandboxes. Another worker can restart them from the
         beginning. The interruption does not count toward the retry limit.
       - ``finish`` stops taking new rollouts and lets running ones finish.
   ``evaluation.finish_timeout_s`` : unset
       how many seconds a worker using ``finish`` waits after shutdown begins. If unset, it waits indefinitely.
       Completed rollouts keep their results. When time runs out, the worker requeues only unfinished rollouts.
       Allow time for sandbox cleanup. If the worker is killed before it can requeue a rollout, that attempt can
       expire and count toward the retry limit.
   ``evaluation.trial_observer`` : unset
       a ``TrialObserver`` plugin (``reef.evaluation.trials``), as ``package.module:ClassName`` or ``module`` and
       ``config``. The evaluation worker gives it each trial: the rollout, the attempt, the trial directory and the
       error. It runs before the worker removes the directory, so it can keep the logs of the agent and the verifier.
       Its failure is logged and does not change the rollout.
   ``evaluation.task_template_builder`` : unset
       a ``TaskTemplateBuilder`` plugin (``reef.evaluation.task_templates``), as ``package.module:ClassName`` or
       ``module`` and ``config``. An evaluation on ``e2b`` calls it once for each task, before it adds the task's
       rollouts, to make the task's sandbox templates ready. Unset, ``DockerfileTemplateBuilder`` builds them from the
       task's Dockerfiles, as a trial would. A deployment that keeps task images in a registry names a builder that
       makes the templates from those images.
   ``evaluation.require_prepared_templates`` : false
       Set to ``true`` to require prepared task and verifier templates for new ``e2b`` evaluations.
       If preparation fails, the affected rollouts report the error and skip the trial.

Recipe configuration
--------------------

A configuration file selects the recipe with ``recipe.implementation``, puts
the recipe's declared fields under ``recipe.config`` and the runtime under
``recipe.runtime``. A field that the selected recipe does not declare is an
error. Harness settings live in ``recipe.config.evolution`` (`Harness
evolution keys`_).

**Select a recipe.** ``recipe.implementation`` takes one of three forms:

- **The core record-only recipe:** ``recipe``.
- **A dotted class:** ``"my_pkg.my_method:MyMethodRecipe"``. The class does
  not have to be pip-installed. ``reef serve`` walks up from the config file's
  directory to the nearest ancestor that holds ``<package>/__init__.py`` and
  appends that directory to ``PYTHONPATH`` for every service. This is how
  ``recipes.sao.recipe:SAORecipe`` resolves from a source checkout. Entries
  already in ``PYTHONPATH`` keep their precedence.
- **A named preset:** ``my-preset``, read from ``my-preset.yaml`` under
  ``REEF_RECIPE_CONFIG_DIR``. The variable has no default, and Reef bundles no
  presets. A bare name other than ``recipe`` is always a preset name; it never
  imports a learning method.

**Training mode.** Every recipe has ``recipe.config.training_mode``:

.. container:: config

   ``recipe.config.training_mode`` : auto
       ``auto`` starts a job when the `next job <../developer-guide/processors.rst#the-next-job>`__ holds ``job_samples`` untrained samples, and refuses ``POST /reef/train``. ``manual`` starts a job only for a ``POST /reef/train`` request

- A ``manual`` job of a weight recipe trains the samples that the request
  names (`Choose the samples that train
  <http-api.rst#choose-the-samples-that-train>`__). The request text goes into
  the commit's ``training_request`` metrics.
- Harness evolution runs a ``manual`` request at once. The method gets the
  request and the traces that it names, possibly none.
- Records arrive and stages make samples in both modes. The samples wait as
  data items until a job trains them, so a switch from ``manual`` to ``auto``
  can train on data already collected.
- A recipe's ``Sample2Job`` declares ``supported_training_modes``; an
  unsupported mode raises ``NotImplementedError``. Harness evolution supports
  both modes. ``manual`` on an inference-only recipe creates no training
  backend.
- The mode is independent of the `activation rule <#the-activation-section>`__.
- ``POST /reef/scenarios/{scenario}/update`` selects another mode at runtime.
  The change applies to later job data; reserved job data completes under its
  original mode. Reef stores the selected mode in the scenario's settings. The
  stored mode survives restarts and takes precedence over the configured one.
  A stored mode that the recipe does not support keeps the configured mode
  and logs a warning.

.. code:: yaml

   recipe:
     config:
       training_mode: manual

**Job parameters.** ``recipe.config.job_parameters`` is an object that a
weight recipe sends with every job, over its default parameters. Reef does not
read it. The training component owns it, and
``GET /reef/scenarios/{scenario}/contract`` serves its JSON Schema. The
component refuses a parameter that it does not take, and the job is not
submitted. A recipe that overrides ``configured_parameters`` decides where the
setting goes.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Slime
     - Tinker
   * - ``objective``
     - required: a registered name or a dotted reference
     - required: a registered name or a dotted reference
   * - ``learning_rate_schedule``
     - optional
     - optional
   * - ``scheduling``
     - optional (`Step scheduling <python-api.rst#training-objective>`__)
     - optional
   * - ``save_interval``
     - optional
     - optional
   * - ``batch_size``, ``learning_rate``, ``kl_coef``
     - no
     - optional
   * - ``dataset``
     - a job without samples (`Dataset jobs on Slime`_)
     - no
   * - ``harbor``
     - required on a task job, refused on another job
     - no
   * - ``rollout``
     - no
     - required on a task job, refused on another job

.. code:: yaml

   recipe:
     config:
       job_parameters:
         scheduling:
           packing_length: 8192
         save_interval: 50

**Presets.** A preset uses the configuration file layout and is read as-is:
``${VAR}`` interpolates in a deployment config, never in a preset.

- A preset carries ``recipe.implementation`` (``recipe`` or a dotted class)
  and ``recipe.config``. ``recipe.config`` holds the training mode and the
  recipe's own sections, and ``pipeline`` holds the stage config.
- An optional ``recipe.runtime`` makes the recipe build its own runtime
  instead of using the deployment's upstream proxy. Such a preset must name
  its model (``inference.upstream-model`` or ``inference.model-path``).
- A preset that uses the deployment's runtime may omit the model and inherit
  the runtime's. An explicit model takes precedence.
- Presets suit recipes whose runtime can be built from the preset or the
  deployment's upstream proxy. Select a weight-training recipe by dotted class
  in the deployment config, so that the service can build its training
  component.

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         method: methods.mine:MyMethod
         evaluation:
           tasks: ["..."]
           scorer: methods.mine:MyScorer
   inference:
     upstream-model: qwen3-8b

**Composite recipes.** A composite recipe serves and evolves several snapshot
components in one scenario, one recipe per recipe component. Its
``components`` object maps each recipe component name to a section with the
shape of the top-level ``recipe`` section: ``implementation`` names the recipe
class, and ``config`` holds the recipe's fields and sections, including its
own ``pipeline`` stage config.

- ``components`` holds at least two recipe components. At most one trains
  weights, at most one is a harness evolution recipe, and none is a composite.
  Reef refuses any other ``components`` object when it loads the
  configuration.
- A recipe component has no ``runtime``: every recipe component uses the
  deployment's model, inference runtime and training component.
- A composite whose recipe component trains weights is deployed the way that
  recipe is: ``training.backend`` selects the backend, and a top-level
  ``pipeline`` or ``evaluation`` section goes to that recipe component.
- A scenario's root snapshot keeps one directory per recipe component, with
  that component's default start. A recipe component whose recipe seeds
  nothing starts empty.
- Each recipe component's trainer commits into the same snapshot chain, so
  every step checkpoints, and the recipe components share one
  ``training_mode``.
- Reef admits a report when the contract of any recipe component accepts it.
  Each trainer keeps the reports that its own contract parses.

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {sample2job: {config: {job_samples: 8, buffer_size: 32}}}
         harness:
           implementation: reef.recipe.harness:HarnessEvolutionRecipe
           config:
             evolution: {adapter: pi, method: methods.mine:MyMethod, evaluation: {tasks: ["..."], scorer: methods.mine:MyScorer}}
   inference:
     model-path: Qwen/Qwen3-8B
   training:
     backend: slime

A composite without a recipe component that trains weights has the same
shape, with ``inference.upstream-model`` in place of ``inference.model-path``
and no ``training`` section.

Stack execution backends
------------------------

The ``execution`` section selects the executor of each role. Reef places its
own service processes itself. Every selector accepts a built-in backend name,
a ``package.module:ClassName`` reference, a profile name under ``executors``,
or an inline object.

.. list-table::
   :header-rows: 1

   * - Selector
     - Role
     - Default
   * - ``execution.training``
     - Slime's training workers
     - ``auto``, which resolves to ``ray``
   * - ``execution.rollout``
     - the serving engines of the inference component, for every training backend
     - ``auto``
   * - ``execution.generator``
     - the generator service that a ``generator`` section adds
     - ``auto``: one local process
   * - ``execution.evolution``
     - harness evaluation workers
     - ``auto``

An inline object or a profile accepts these keys:

.. container:: config

   ``backend`` : auto
       a built-in executor, a ``package.module:ClassName`` reference, or a named profile

   ``options`` : {}
       options that the selected backend owns

   ``workers``
       a positive integer; omitted, the component's own requirement

   ``resources``
       ``cpus_per_worker`` and ``gpus_per_worker``, finite nonnegative numbers; omitted values keep the component's defaults (normally one CPU and no GPUs). A value cannot reduce a component's declared GPU need. Numeric environment interpolation is accepted

How ``auto`` resolves:

- **Harness evaluation workers.** Ordinary harness evolution calls external
  model endpoints and needs no local GPUs. One worker selects ``uni``, and
  several workers select ``mp``. Workers that need local GPUs follow the same
  rule after Reef checks the visible CUDA capacity; with too few GPUs, startup
  fails and asks for an explicit ``ray``. A local allocation needs whole GPUs,
  and a fractional reservation needs an explicit ``ray``. Several workers
  inside a Ray placement group, or declared cluster or actor options, select
  ``ray``. CPU-only local executors reserve no cores and enforce no CPU
  quotas.
- **Services.** ``auto`` selects ``uni`` unless resource or worker options, or
  an existing Ray placement group, call for Ray. Explicit local CUDA
  visibility selects ``uni`` and cannot be combined with cluster resource
  options. A service executor accepts ``resources``, but ``workers`` must be
  one. For a Ray service, ``resources`` supplies actor launch options such as
  ``num_gpus`` and ``num_cpus``.
- **Training and rollout.** These roles reject the generic ``workers`` and
  ``resources`` fields. Their topology comes from Slime's training options.

An explicit selector or Slime CLI flag overrides the role's default. A backend
that fails to start never falls back to another backend. Slime's
``--reef-executor-backend`` option also selects the executor of its training
workers; see `Slime worker executors
<../developer-guide/executors.rst#slime-worker-executors>`__.

Harness evolution keys
~~~~~~~~~~~~~~~~~~~~~~

These keys go under ``recipe.config.evolution``. `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ describes what each one changes,
and `Write a harness method <../developer-guide/write-a-harness-method.rst>`__
describes the method contract. A key may use underscores or hyphens, and
startup refuses an unknown key.

The pipeline stages decide what a job trains on:

- **Scored reports (default).** Every valid scored report is a trace,
  successful outcomes included. A job starts when
  ``pipeline.sample2job.config.job_samples`` traces are held, 1 by default.
- **Recorded traffic alone.** Set ``pipeline.record2sample.module:
  reef.record2sample.traces:InferenceSamples``. A job then starts every
  ``job_samples`` requests, with unscored samples.
- **Imported train tasks (offline).** Set ``pipeline.sample2job.module:
  reef.sample2job.traces:TrainTaskJobData``. Each uploaded train task trains
  once, in upload order. A job holds either traces or train tasks. A job of
  train tasks rolls out up to ``pipeline.sample2job.config.buffer_size`` of
  them (default ``job_samples``) on the head tree as the method's traces.
  Startup refuses this stage without an adapter that plays task directories
  (an npm ``install``) and ``evolution.evaluation.scorer``.

Reef refuses ``batch_size`` and ``batch_policy`` directly under
``recipe.config``.

.. container:: config

   ``evolution.method``
       required: the ``Method`` that evolves the tree, as ``package.module:ClassName`` or a mapping with ``module`` and ``config`` that the class method ``from_config`` receives; its ``run(job)`` makes each job's candidate

   ``evolution.adapter`` : pi
       ``opencode``, ``claude``, ``codex``, ``dsh`` (DeepSeek Harness), ``hermes`` (Hermes Agent), ``native`` (Reef's own agent, whose tools, hook listeners, control flow graphs, delegate agents and loop code are tree files of the ``native_tool``, ``native_hook``, ``native_graph``, ``native_agent`` and ``native_loop`` parts), ``python`` (a Python program that uv runs, whose modules are tree files of the ``program`` part), or an entry-point adapter, such as ``terminus`` from the Meta-Harness Terminal-Bench campaign

   ``evolution.seed``
       the directory of the first tree: every regular file under it, at its relative path as its tree path; a new scenario's root snapshot holds it, and an existing scenario keeps its own root and head. Startup refuses a seed that fails validation for ``evolution.adapter``, holds a symbolic link, or holds a file that is not UTF-8 text. A relative path resolves against the Reef process's working directory. Unset, the empty tree, unless a recipe subclass sets its own (for example Reefine's starter skill)

   ``evolution.binary``
       a path to the harness binary; unset, backend construction installs the adapter's pinned version through the vendor's channel under ``$REEF_HARNESS_PREFIX`` (default ``~/.local/share/reef-harness``)

   ``evolution.models`` : {}
       the method's own models by name, which it calls as ``services.models[name]``: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name; the name ``served`` is refused, since it is the model under test

   ``evolution.model_metadata`` : {}
       optional map of exact model names to ``{context_window: <positive token count>, reasoning: <boolean>}``; Codex uses these values before provider discovery

   ``evolution.state_dir`` : .reef/harness
       the directory of each scenario's harness state; a relative path resolves against the Reef process's working directory. ``<state_dir>/<scenario>/<component>/``, where ``<component>`` is the snapshot component the recipe evolves (``harness`` for ``HarnessEvolutionRecipe``, the component's own name in a composite recipe), holds ``history/``, every candidate, run and decision, which is the score cache; and ``steps/``, one job record per job (the workspace with its ``output/``, and ``calls.json``; ``<step>`` is the scenario step, and a retried job lands in ``<step>-2``). Nothing prunes it. A directory that cannot be created refuses to load the scenario

   ``evolution.max_steps`` : 0
       skip automatic evolve jobs once the scenario's step number passes this; the scenario numbers every commit, activations, train-from operations, imports and syncs included; 0 disables the limit; an instruction from ``POST /reef/train`` still runs past it

   ``evolution.max_failure_streak`` : 0
       stop automatic evolve jobs after this many consecutive rejected jobs, request jobs included; 0 disables the limit; a request from ``POST /reef/train`` still runs while the breaker is open

   ``evolution.max_model_calls_per_step`` : 0
       cap the method's model calls in one job, through ``services.models`` or ``services.calls``; a call past the cap raises ``RuntimeError``; 0 disables the limit

   ``evolution.max_runs_per_step`` : 0
       cap the runs that the method's ``services.evaluate`` runs in one job; reused runs and the evaluation's own runs do not count; 0 disables the limit

   ``evolution.executor`` : local
       ``local`` runs tasks as a plain subprocess; ``sandbox`` uses the host's local sandbox, a bubblewrap jail on Linux or a Seatbelt profile on macOS; ``e2b`` and ``modal`` create a cloud sandbox per run and check provider access, image, binary and namespace support at startup. A cloud sandbox uses remote binaries; Terminus retains its own Harbor execution

   ``evolution.sandbox``
       for ``sandbox``: ``egress_hosts`` and ``limits`` (``cpu_seconds``, ``memory_bytes``, ``processes``, ``file_bytes``; macOS refuses ``memory_bytes``); for ``e2b``: ``e2b_api_key_env`` (default ``E2B_API_KEY``), optional ``e2b_api_key``, ``e2b_template`` and ``forward_ports`` (explicit worker-loopback TCP ports). for ``modal``: optional ``modal_app`` (default ``reef-sandboxes``), ``modal_image``, ``modal_cpu``, ``modal_memory_mb`` and ``forward_ports``, with Modal's credentials from ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``. The cloud sandboxes reject nonempty ``egress_hosts``/``limits``; image resources and outbound internet apply. All accept ``env_from``

   ``execution.evolution.workers`` : 1
       fixed worker-group size (`Stack execution backends`_)

   ``execution.evolution.backend`` : auto
       worker placement, independent of the run isolation that ``evolution.executor`` selects

   ``execution.evolution.resources``
       ``cpus_per_worker`` and ``gpus_per_worker``; GPUs select Ray under ``auto``

   ``evolution.agent``
       the coding agent that a method which runs one, such as ``reef.train.harness_backend:AgentMethod``, runs in its jobs (``services.run_agent``, or ``services.agent`` for a method that starts the agent itself); unset, no agent runs, and such a method is refused at startup unless it also runs without an agent (``runs_without_agent``)

   ``evolution.agent.adapter`` : pi
       the agent's harness: any adapter whose prompt is text

   ``evolution.agent.binary``
       the agent's binary; unset, backend construction installs the adapter's pinned version, as for ``evolution.binary``

   ``evolution.agent.sandbox`` : ``REEF_AGENT_SANDBOX``
       ``bwrap`` jails the agent, with the internet but no host port except its gateway's; ``seatbelt`` does the same on macOS, but the host's LAN addresses stay reachable; ``e2b`` or ``modal`` runs it in a cloud sandbox that reaches the gateway through a tunnel, with that provider's options in the section, as in ``evolution.sandbox``; ``none`` runs it with the service's own privileges and must be chosen. Unset, the agent is jailed in the host's local sandbox (bwrap and pasta on Linux, sandbox-exec on macOS) where the host can run it, and off where it cannot

   ``evolution.agent.timeout_s`` : 1800
       seconds one agent run may take

``evolution.evaluation`` decides how Reef evaluates a candidate and when it
publishes one: the tasks, the scorer, and the publication policy. It takes
one task source: ``tasks``, ``task_manifest`` with ``tasks_root``, or
``imported_tasks``. The task source and ``scorer`` are required unless
``policy`` is ``none``.

.. code:: yaml

   evolution:
     evaluation:
       tasks: ["[fib] Compute fib(90) and reply with the number alone."]
       scorer: harness.scoring:AnswerScorer
       policy: compare
       statistic: wins

.. container:: config

   ``evolution.evaluation.tasks``
       the evaluation tasks: instructions, the text the agent reads, or task directories for an adapter that takes one (``takes_task_directory``)

   ``evolution.evaluation.task_manifest``
       a split manifest written by ``reef.core.tasks``; its eval split names the evaluation's tasks as directories under ``evolution.evaluation.tasks_root``, each passed to the adapter as its path. Each task runs in its own container, and the adapter needs an npm ``install`` (`Task directories <../user-guide/evolve-your-harness.rst#task-directories>`__); startup refuses a manifest that names no eval task

   ``evolution.evaluation.tasks_root``
       the directory the manifest's task names live under; set together with ``task_manifest``

   ``evolution.evaluation.imported_tasks`` : false
       true makes each job evaluate on the task directories of the eval split of the scenario's imported tasks (the ``task-import`` section, or task items whose metadata names the scenario's ``eval`` split) as the job reads them, so tasks that the scenario gets later join the next job. The eval split may be empty when the scenario loads. The same adapter rule as ``task_manifest``. Reef refuses the key ``evolution.evaluation.task_store`` with a message that names this key

   ``evolution.evaluation.scorer``
       a ``TaskScorer`` class, in the same form as ``evolution.method``; ``reef.evaluation.harness:VerifierRewardScorer`` scores a task directory run by its Harbor verifier's reward

   ``evolution.evaluation.policy`` : compare
       the publication policy: ``compare`` scores the candidate and the current tree (the training start) on the same tasks and repeats; ``floor`` scores the candidate alone and publishes it when every task has a run that completed and every run that completed scores at least ``floor_score``; ``none`` runs no task and publishes every candidate that passes validation

   ``evolution.evaluation.statistic`` : wins
       what ``compare`` compares. ``wins`` publishes when the candidate's wins minus losses over the (task, repeat) pairs exceed ``min_margin``. ``mean`` publishes when the candidate's mean score exceeds the current tree's by more than ``min_margin``. A run that gave no answer scores 0. A run that the environment could not run (a failed run) drops its pair under ``wins`` and is left out of the mean under ``mean``

   ``evolution.evaluation.min_margin`` : 0
       the margin a ``compare`` candidate must exceed: in wins minus losses, or in mean score; at least 0

   ``evolution.evaluation.floor_score`` : 1.0
       the score every run that completed must reach under ``floor``; a failed run fails no task, but a task whose runs all failed does not pass; positive

   ``evolution.evaluation.repeats`` : 1
       the runs per task and tree; each repeat is scored, and compared, on its own

   ``evolution.evaluation.task_timeout_s`` : 600
       seconds one evaluation run may take

   ``evolution.evaluation.agent_host`` : host.docker.internal
       the address at which an agent in a task's container reaches Reef; Reef puts it in place of a loopback host (``127.0.0.1``, ``localhost``, ``::1`` or ``0.0.0.0``) of the served model's base URL, and keeps the port and the path; another host stays. Docker Desktop and colima resolve the default. On Linux, name an address of the host that the container reaches, such as the Docker bridge gateway, and make Reef listen there

   ``evolution.evaluation.forbid_residue`` : false
       when true, a run leaving files outside the cleanup whitelist scores 0, as a run that gave no answer

   ``evolution.evaluation.collect_failures`` : false
       when true, prompts of each job's traces become permanent evaluation tasks, so no later candidate can win while it brings a collected failure back; the seed tasks stay the floor. Refused when the evaluation tasks are task directories

   ``evolution.evaluation.collector``
       optional ``TaskCollector`` class, in the same form as ``evolution.method``, only with ``collect_failures``; its ``collect(samples)`` chooses which trace prompts to collect; without it every trace's user prompt is collected, and the caps and the credential and directive screens still apply

   ``evolution.evaluation.max_collected_tasks`` : 50
       the cap on collected tasks; admission stops there so the suite is bounded; 0 collects none

   ``evolution.evaluation.max_collected_per_client`` : 5
       the cap on collected tasks from one tagged client (its ``x-reef-tag-client``, else session, tag); untagged traffic has no identity to count under and meets only ``max_collected_tasks``; 0 disables the cap

The evaluation context is what a run's score depends on besides the
tree and the task: the served model and the named models of
``evolution.models``; the adapter, its version and ``evolution.binary``; the
scorer setting; ``task_timeout_s`` and ``forbid_residue``; and the run
executor, ``evolution.executor`` and ``evolution.sandbox``. The history
reuses a scored run only in the context it ran in, and never reuses a
failed run.

The served model's binding is merged in when a run renders the tree; it
never enters a published tree. The seed is the baseline that the first
candidate is measured against. The job records hold the method's raw model
traffic and the clients' traces, and the history holds every evaluation run's
session log, so treat ``evolution.state_dir`` like the commit history.

The beta Reefine recipe runs its agent on pi with ``evolution.agent``. Its
method reads ``trial_timeout_s`` and ``multimodal`` from
``evolution.method.config``; its `README <../../recipes/beta/reefine/README.md>`__
documents them.

The ``inference`` section
-------------------------

Every serving mode reads ``inference.timeout-s``. The engine and provider keys
are in `Service and storage settings`_, and the serving keys of a
weight-training deployment are in `Serving of a weight-training deployment`_.

.. container:: config

   ``inference.timeout-s`` : 300.0
       maximum time for one inference request, in seconds; an engine that does not answer within it gets a 504

   ``inference.backend`` : sglang
       the managed engine: ``sglang``, ``vllm``, or ``none`` (`Train without an inference engine`_)

An ``inference.remote`` section runs the engines that ``inference.backend``
names on a remote inference component instead of this host. The component's
``--engine`` must name the same backend. Any training backend can serve on a
remote component. `Serve from a remote GPU host
<../user-guide/tinker.rst#serve-from-a-remote-gpu-host>`__ shows the setup.

.. container:: config

   ``inference.remote.url``
       the component's HTTP(S) base URL, without credentials, query or fragment

   ``inference.remote.slot-urls`` : []
       base URLs of further components, in the form of ``url`` and all different; for full-weight artifacts only

   ``inference.remote.control-token-env`` : REEF_REMOTE_CONTROL_TOKEN
       variable with the component's control token; the process that starts the engines reads it

   ``inference.remote.inference-token-env`` : REEF_REMOTE_INFERENCE_TOKEN
       variable with the component's inference token; the HTTP service reads it

   ``inference.remote.startup-timeout-s`` : 3600
       deadline for one engine start on the component, including the model download

   ``inference.remote.model-path`` : ``inference.model-path`` as written
       the model that the component loads: a Hugging Face repository ID or a path on the component's host; this host may replace ``inference.model-path`` with a local snapshot

One component serves one full-weight checkpoint. To let a scenario serve
several checkpoints at the same time, name one more component for each
further checkpoint:

.. code:: yaml

   inference:
     remote:
       url: https://gpu-a.example.com
       slot-urls:
         - https://gpu-b.example.com

The one training scenario then serves as many checkpoints together as there
are components. A further component does not add a training scenario. These
rules apply:

- Start each further component in the same way as the first one, with the
  same ``--engine`` and the same two tokens. Reef starts the same model and
  engine options on each one, and ``inference.num-gpus`` counts the GPUs of
  one component.
- Give each component GPUs that no other component uses: its own host or
  visible GPUs, or one Ray cluster that all of them join
  (``--ray-address``). Different URLs alone do not separate the GPUs.
- Use ``slot-urls`` with full-weight artifacts only. Reef refuses it for
  adapter artifacts and for a trainer's rollout engines.

`Set what a scenario serves
<../user-guide/operate.rst#set-what-a-scenario-serves>`__ describes how a
scenario uses the components, and how to replace a checkpoint when every
component holds one.

The ``training`` section
------------------------

The weight-training stack reads this section. `Configuration file layout`_
lists ``training.backend``, the Ray keys, ``training.ready-timeout``,
``training.max-jobs``, ``training.manager`` and ``training.timeout-s``. See
`Evolve your model <../user-guide/evolve-your-model.rst>`__ for how to size
the stack.

.. container:: config

   ``training.colocate`` : false
       must stay false: Reef's serving never shares GPUs with training, and Slime and Tinker refuse ``true``; Slime's own rollout engines share the training GPUs with ``training.options.colocate``

   ``training.config.num_gpus``
       example-specific GPU count passed to Slime's model topology flags; does not set the Ray cluster's capacity

   ``training.config.global_batch_size``
       samples in one optimizer step, passed to Slime as ``--global-batch-size``. The examples also pass it to ``pipeline.sample2job.config.job_samples``, so one job is one optimizer step. Reef reads ``training.options.global-batch-size`` itself and refuses a Slime deployment without it, even when ``num-steps-per-rollout`` could derive it.

   ``training.config.checkpoint_dir``
       where Megatron and HF checkpoints are written

   ``training.config.megatron_checkpoint_path``
       optional pre-converted torch_dist checkpoint, to skip HF conversion on every start

   ``training.config.checkpoint_retention``
       storage-fraction bounds and the retention policy. When storage cannot hold the next checkpoint, the Slime job fails. A failed job ends, and the records of its data do not train again, because Reef consumed them when it submitted the job.

   ``training.options``
       native training flags: actor GPU layout, optimizer, sequence length, and loss settings (`Native backend options`_)

``training.config`` holds workload variables that the file's other values
reference, such as ``${training.config.global_batch_size}``. Slime fills
architecture flags such as layer counts and hidden sizes from
``inference.model-path``. Do not put them in the config.

The ``activation`` section
--------------------------

Training makes artifacts: every checkpoint that a weight training job saves,
and every tree that a harness trainer publishes (`Harness evolution keys`_).
Training never decides whether an artifact serves. The activation rule that
this section names decides that, for every recipe, a composite included.

- Without the section, the rule is ``always``.
- A scenario can select one of the built-in rules for itself with
  ``POST /reef/scenarios/{scenario}/update`` (`Scenario updates
  <http-api.rst#scenario-updates>`__). The section's rule applies to each
  scenario that selects none.
- An artifact that the rule does not activate stays inactive. A person can
  activate it with the ``served`` setting of the same request.
- Activation does not change where training continues.

`Set what a scenario serves
<../user-guide/operate.rst#set-what-a-scenario-serves>`__ describes
activation, and `Go back to an older artifact
<../user-guide/operate.rst#go-back-to-an-older-artifact>`__ describes how to
make an older artifact serve for good.

.. code:: yaml

   activation:
     module: always
     config:
       review_kinds: [extension]

.. container:: config

   ``activation.module`` : always
       ``always``, ``never``, ``evaluated``, or a ``package.module:ClassName`` reference to an ``ActivationRule`` subclass (``reef.core.activation``)

   ``activation.config``
       the rule's settings, below; a class of your own receives the mapping in ``from_config``

The built-in rules:

- ``always`` activates every new artifact, except a harness tree that changes
  a kind in ``review_kinds`` (a list, default empty). Each name is a kind of
  the ``evolve`` section of the adapter's ``adapter.yaml``, such as ``rules``,
  ``config``, ``skill``, ``command``, ``extension`` or native's ``native_*``
  kinds; the harness recipe refuses a name that the adapter does not evolve.
  A tree that changes a kind that is always reviewed, such as native's
  ``native_loop``, waits for a person whether or not the list names it.
- ``never`` activates no artifact, for fully manual activation. It takes no
  config.
- ``evaluated`` activates an artifact once an evaluation of it finishes with a
  ``mean_score`` of at least ``min_mean_score`` (required). Set the scenario's
  ``artifact_evaluation`` (`Automatic evaluation
  <http-api.rst#automatic-evaluation>`__) so Reef evaluates each new artifact.
  A scenario without ``artifact_evaluation`` under this rule activates no new
  artifact, and Reef logs a warning when it creates or loads the scenario.

A rule of your own subclasses ``ActivationRule``:

- ``activates(made)`` decides when training publishes an artifact. ``made``
  names the component and the kinds of the harness tree that the artifact
  changes.
- ``activates_evaluated(artifact_id, evaluation)`` decides when an evaluation
  of the artifact finishes.

Reef checks the class and builds the rule when it loads the deployment
config, so a bad rule fails at startup. Reef rejects functions, abstract
classes and structural lookalikes.

The ``pipeline`` section
------------------------

A recipe's pipeline has two stages (see `Training pipeline
<../developer-guide/processors.rst>`__). The recipe names a default class for
each stage. This section gives each stage its config, and it can name another
class for a stage. Every stage reads only its own ``config``; the recipe's
``recipe.config`` holds no stage settings.

.. container:: config

   ``pipeline.record2sample``
       the record-to-sample stage, which decides which records train, in which groups, and makes their samples: ``config``, and optionally ``module``, a ``Record2Sample`` subclass

   ``pipeline.sample2job``
       the sample-to-job stage: ``config``, and optionally ``module``, a ``Sample2Job`` subclass

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 8
         rollouts_per_group: 64

Rules of the section:

- A stage without a key is the recipe's class with no config. Any other key
  under ``pipeline`` fails the startup.
- Each recipe's page lists the keys of its stages. Keys are written as the
  stage reads them; a hyphen spelling is also accepted.
- Reef checks each stage when it loads the recipe config, before a scenario
  starts. It imports a named class, rejects functions and abstract classes,
  and parses the stage's ``config``. An unknown key or a value out of range
  fails the startup.
- A stage key under ``recipe.config``, such as ``groups_per_step``, is an
  unknown recipe key and also fails the startup.
- A stage class needs the class method ``from_config``. The sample-to-job
  stage receives the record-to-sample stage and reads the values that the
  record-to-sample stage owns from it. For example, the TTTD step grid is set
  once, in ``pipeline.record2sample.config``.
- A scenario can replace keys of ``pipeline.record2sample.config`` for itself
  with its ``record2sample_config`` setting (`Record-to-sample config
  <http-api.rst#record-to-sample-config>`__).

**Staleness.** Each record-to-sample stage decides which of its samples are
stale; no setting chooses the rule. Most built-in stages are on-policy: only
data of the artifact that the scenario serves trains. An SFT stage and the
harness trace stages never drop a sample. SAO accepts a bounded lag, its own
``max_artifact_lag`` (`SAO <../user-guide/recipes/sao.rst>`__). Startup
refuses a ``pipeline.record2sample.config.max_staleness`` key: remove it. A
training framework's own off-policy limit goes in that backend's
``training.options``; no built-in backend has one now. See `Staleness
<../developer-guide/processors.rst#staleness>`__.

**Job size.** Reef submits one training job for each job's data. Every
built-in sample-to-job stage that trains on records reads these two keys; a
recipe's page lists the other keys of its stage.

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : required for weight recipes; 1 for the harness evolution stages
       the fewest samples of one job, a positive int; at most the stage's grid for TTTD and SDPO. A weight recipe without it fails the recipe build with a ``RecipeConfigError`` that names the key

   ``pipeline.sample2job.config.buffer_size`` : ``job_samples``
       the most samples that one automatic job takes, beside the rest of a group that those samples touch; at least ``job_samples``. Uploaded trajectories count toward it. Samples beyond it wait for later jobs.

The training component cuts each job's data into optimizer batches by its own
batch size: Slime's ``--global-batch-size``
(``training.config.global_batch_size`` in the examples) or Tinker's
``batch_size``. Usually, set ``job_samples`` to that batch size, so that one
job is one optimizer step:

.. code:: yaml

   pipeline:
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}

The two values can differ. Job data takes up to ``buffer_size`` samples and
the rest of each group that those samples touch, so a group of rollouts
trains in one job. A job with more samples than the batch size trains more
than one optimizer step, and a job with fewer trains one partial step. The
``remainder`` field of a job's optional ``scheduling`` parameter decides what
happens to a shorter last step; by default, the component trains it as a
smaller step. See `Job size <../developer-guide/processors.rst#job-size>`__.

Generated tasks
~~~~~~~~~~~~~~~

The built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, generates Harbor tasks in
rounds and trains on rollouts of the saved train tasks. Its stages are
``GeneratedTaskSamples`` and ``GeneratedTaskJobData``.

- The recipe names no objective. The deployment names one in
  ``recipe.config.job_parameters.objective``, as a dotted reference to a
  method that it provides.
- Each job is a task job: the training component rolls the job's tasks out on
  its own sampler and trains on their rollouts. ``job_parameters`` says how:
  ``rollout`` on Tinker, ``harbor`` on Slime, whose objective has the
  ``slime`` loss family (`Task jobs on Slime`_).
- The recipe refuses a config without the objective and the rollouts.
- Without ``pipeline.record2sample.config.rounds``, no round runs, and the
  recipe trains only the train tasks that earlier generations saved in the
  scenario.

.. code:: yaml

   recipe:
     implementation: reef.recipe.generated_tasks:GeneratedTaskRecipe
     config:
       job_parameters:
         objective: my_methods.objectives:MyObjective
         rollout:
           rollouts_per_task: 4
           environment: docker
   pipeline:
     record2sample:
       config:
         jobs_per_round: 2
         generator_url: ${endpoints.generator}
         rounds: 20
         description: shell tasks with hidden state under /var and /etc
         generator:
           module: recipes.beta.spade:SpadeTaskGenerator
           config:
             generator_url: ${endpoints.generator}
             skills: [inspection, repair]
     sample2job:
       config:
         buffer_size: 2

``GeneratedTaskSamples`` paces the rounds, holds their source sessions, runs
the generations, and sends their results to the generator. It trains on no
report: the recipe trains on the rollouts that the training component runs,
which are not records.

.. container:: config

   ``pipeline.record2sample.config.jobs_per_round`` : 1
       trained jobs after which the next round starts; 0 starts it when the previous round ends, so generation runs ahead of training until ``refill_tasks`` tasks wait

   ``pipeline.record2sample.config.source_records_per_round`` : 0
       recorded sessions that each round gives the generator, at most; 0 generates from the description only

   ``pipeline.record2sample.config.source_tag``
       the ``name=value`` tag that marks a source session: each of its model calls carries the header ``x-reef-tag-<name>: <value>``. Required when ``source_records_per_round`` is above 0.

   ``pipeline.record2sample.config.refill_tasks`` : 1
       the next round starts only while fewer saved tasks than this wait to train; a task in a running job does not wait. With ``pool_capacity`` above 0, leave it out: the pool's capacity paces the rounds.

   ``pipeline.record2sample.config.rounds`` : 0
       rounds of task generation; 0 generates nothing. A round counts when it finishes a generation or runs out of its attempts; a round of replayed sessions does not count. After a restart, only complete generations count.

   ``pipeline.record2sample.config.generator``
       the ``TaskGenerator`` class (``module``) and its ``config``. Required with rounds.

   ``pipeline.record2sample.config.description``
       what the tasks are about; every generation request carries it. Required with rounds.

   ``pipeline.record2sample.config.generator_url``
       the generator service, ``${endpoints.generator}``. Required with rounds.

   ``pipeline.record2sample.config.eval_fraction`` : 0.0
       fraction of new source groups held out before their rollouts run; between 0 and 1

   ``pipeline.record2sample.config.split_seed`` : 0
       fixed seed for the task store's train/eval split

   ``pipeline.record2sample.config.generation_attempts`` : 3
       attempts of a generation whose generator or checks fail to run; after them the round is skipped, and the next round finishes the same generation

   ``pipeline.record2sample.config.generation_deadline_seconds`` : 1800
       seconds that all attempts of one generation may take

   ``pipeline.record2sample.config.validation_concurrency`` : 1
       candidates that one generation checks at once. After a check that fails to run, no further check starts, and the attempt fails when the running checks end.

   ``pipeline.record2sample.config.pool_capacity`` : 0
       0 trains each train task once. Above 0, a task trains again after its job ends, and rounds start only while fewer tasks than this are active.

   ``pipeline.record2sample.config.review_interval`` : 0
       task jobs between two pool reviews; 0 reviews nothing

   ``pipeline.record2sample.config.review_window`` : 2
       the most recent rewards of a task that a review reads; at least 2

   ``pipeline.record2sample.config.review_band_low`` : 0
       a review retires a task whose pass rate is below this; between 0 and 1

   ``pipeline.record2sample.config.review_band_high`` : 1
       a review retires a task whose pass rate is above this; between ``review_band_low`` and 1

   ``pipeline.record2sample.config.review_min_reward_std`` : 0
       a review retires a task whose rewards have a smaller population standard deviation; 0 turns this rule off

**The generator.** Reef calls the generator class's
``from_config(config, *, scenario, environ)`` once for each scenario. A
generator that calls a remote service reads its credentials through
environment references in ``config``. The generator keeps its own state in
the scenario's task store, so its ``config`` names no state directory; the
built-in generators refuse a ``state_dir`` key. See `the task generation
contract <../developer-guide/processors.rst#task-generation-contract>`__ for
the generator interface and the round.

A source session is a report, without a training arm, whose inference records
all carry ``source_tag``. The generator service accepts at most 512 KiB of
records and files in one request, so keep ``source_records_per_round`` small
when the sessions are long.

**Saved tasks and splits.** With rounds enabled, each valid task and its
split are saved in the scenario's task store.

- A generation saves tasks and runs no rollouts. The saved train tasks run
  later, and the store keeps their results.
- The task files go in the task file store; see `Choose the task file store
  <../user-guide/storage.rst#choose-the-task-file-store>`__.
- Set ``eval_fraction`` and ``split_seed`` here, not in a generator's
  configuration. The defaults keep every task in train. A published task
  keeps its split; a change of these values applies to later tasks.
- Related source sessions keep the same split across generations. A task that
  joins existing train and eval source groups is rejected.
- Eval tasks are saved without rollouts or Designer rewards.
- The generator's pass-rate band does not filter the store.

**The training pool.** With ``pool_capacity`` above 0, the train tasks form a
training pool. When a task's job ends, the task waits for a job again, after
the tasks that already wait, so the tasks train in turn.

- A new round starts only while fewer tasks than ``pool_capacity`` are
  active. The pool can therefore exceed the capacity by the train tasks of
  one generation.
- The ``review_*`` keys need ``pool_capacity`` above 0, and ``refill_tasks``
  must be left out. Imported train tasks still train once.
- When a deployment sets ``pool_capacity`` on a scenario whose train tasks
  trained once, those tasks join the pool at the start.
- The task store keeps each use of a task with its rewards, each review, and
  each retired task with its reason, so a restart continues the pool in the
  same order.
- The status shows the pool under ``training_pool``: ``capacity``, ``active``,
  ``retired``, ``reviews`` and ``last_review_retired``.

After every ``review_interval`` task jobs, a review reads the last
``review_window`` rewards of each active task. A failed task job counts too.
The pass rate is the mean of those rewards. A task with fewer rewards stays.
The review retires a task for one of these reasons, in this order:

- ``below_band``: the pass rate is below ``review_band_low``.
- ``above_band``: the pass rate is above ``review_band_high``.
- ``no_reward_spread``: the population standard deviation of the rewards is
  below ``review_min_reward_std``.

A pass rate on a bound keeps the task. A retired task does not train again,
except in a job whose tasks the trainer took before the review. The review
reads the rewards that training recorded, so they can be one pool cycle old.
The defaults run no review, and each of their rules keeps every task. The
generator still receives the first rollout of each task only.

Two limits of the training pool:

- When the trainer starts the next job before the stage reads the end of a
  failed task job, the tasks of the next job also get a use with no rewards,
  and their rewards in that job are not kept.
- The tasks of a job that a new training start cancelled return to the pool
  only when a later job ends. When that job held so many tasks that fewer than
  ``job_samples`` wait and the pool is full, no job and no round can start,
  also after a restart. Restart with a larger ``pool_capacity``.

**Task jobs.** The saved train tasks become task items that the scenario
uses, ``<scenario>.generated-tasks.<task name>``, and train like every other
sample. ``GeneratedTaskJobData`` makes one task job of the next job's tasks,
whatever Tinker's batch size (``training.options.batch-size``) is:

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : 1
       the fewest saved tasks of one task job

   ``pipeline.sample2job.config.buffer_size`` : 4
       the most saved tasks of one task job; at least ``job_samples``, and ``job_samples`` when it is more than 4

The Tinker component runs a task job's rollouts on Tinker's sampler at the
job's start weights, as the ``rollout`` job parameter says. The Slime
component runs them inside Slime's own loop, on its rollout engines, as the
``harbor`` job parameter says (`Task jobs on Slime`_). ``harbor`` takes the
keys below except ``rollouts_per_task``: Slime's ``n-samples-per-prompt``
gives the count. These parameters say only how to run Harbor; the
``objective`` parameter says how the samples train.

.. container:: config

   ``rollout.rollouts_per_task`` : required on Tinker
       training rollouts of each task, at least 2; the samples of one task form its group

   ``rollout.agent`` : terminus-2
       the Harbor agent config, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. Its requests must fit the training component's text chat: system, user and assistant text messages, no tools.

   ``rollout.environment`` : docker
       the Harbor environment type of the rollouts, such as ``docker``, ``e2b`` or ``modal``. On ``modal``, a task with ``environment/docker-compose.yaml`` and no GPU runs on Modal's VM runtime, so that its Compose networks work

   ``rollout.agent_host`` : none
       the host at which an agent inside the sandbox reaches the component's host; none for an agent on this host

   ``rollout.concurrency`` : 2
       rollouts in flight

   ``rollout.verifier_timeout_reward`` : 0.0
       the reward of a rollout whose verifier timed out; ``null`` leaves it unscored

   ``rollout.scaffold_tolerance`` : 8
       tokens at the end of a prompt that the next prompt's history may drop, such as the empty think block of a thinking template with thinking off; the sample keeps them, and the response after them trains

**Cloud sandboxes.** Set ``rollout.environment`` to ``e2b`` for an
E2B-compatible service such as AgentENV, or to ``modal`` for Modal.

- For a self-hosted E2B service, set ``E2B_API_URL``, ``E2B_SANDBOX_URL`` and
  ``E2B_API_KEY`` in the process that runs the rollouts. The first two select
  the service's API and sandbox endpoints. The Slime training component
  applies these settings to its workers.
- On AgentENV, also set ``REEF_E2B_DOCKERFILE_BUILD=aenv`` in that process,
  and put AgentENV's ``aenv`` CLI and its ``aenv-buildctl`` on ``PATH``.
  AgentENV's E2B template builds refuse a Dockerfile's COPY steps, so Reef
  then builds each task's Dockerfile with ``aenv build`` and the same
  ``E2B_API_URL`` and ``E2B_API_KEY``. The default, ``e2b``, uses E2B's
  template API.

**Sandbox lifetime.** Harbor deletes the sandbox when the rollout ends. If
the rollout process is killed before cleanup, the sandbox's lifetime limits
how long it remains.

- The lifetime is the sum of the task's build, agent and verifier time
  limits, plus one hour, and at most one day. A rollout that needs more time
  loses its sandbox after one day.
- A nonzero ``override_timeout_sec`` in ``rollout.agent`` replaces the task's
  agent time limit in that sum, and an ``override_setup_timeout_sec`` adds to
  it.
- When neither the task nor a nonzero ``override_timeout_sec`` gives the
  agent a time limit, the sum does not apply. A Modal sandbox then lives for
  two hours, and an E2B sandbox keeps Harbor's default lifetime of one day.
- Reef asks the service to delete an E2B sandbox at the end of its lifetime.
  Without this request, a self-hosted service can pause the sandbox and keep
  it.
- These rules also apply to evaluation and generator rollouts, each with its
  own agent config.
- Standalone ``harbor run -e e2b`` commands and the Meta-Harness
  Terminal-Bench ``terminus`` adapter use Harbor's default timeout behavior.
  That adapter is separate from the ``terminus-2`` agent of Reef's rollouts.

**Task job metrics.**

.. list-table::
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``rollout_rewards``
     - each task's rewards of the rollouts that ran
   * - ``rollout_samples``
     - the samples that train
   * - ``removed_rollouts``
     - the rollouts that give no sample: they fail in the infrastructure, or their model calls do not assemble into one sample
   * - ``rollout_tokens``, ``rollout_response_tokens``, ``rollout_trained_tokens``
     - Tinker only: each rollout's response tokens and the tokens of them that train, and their sums
   * - ``unrenderable_samples``
     - Tinker: samples longer than the model's context window, left out before their group forms; they are in neither ``rollout_samples`` nor the trained tokens

A sample alone in its group trains with advantage 0 on Slime and counts in
``rollout_samples``. On Tinker it does not train, and it is in neither count.

Task jobs on Slime
~~~~~~~~~~~~~~~~~~

The Slime component runs task jobs on Slime's own rollout engines: the SGLang
engines of Slime's serving session. Reef's serving never calls them.

- The first task job starts the session. The session holds the engines, their
  router and the training GPUs until the component closes. Later jobs attach
  to the started engines, and every job trains on the session's GPUs.
- A failed or cancelled job leaves the session running for the next job.
- ``training.options.rollout-num-gpus`` gives the engines GPUs of their own.
  ``training.options.colocate`` puts them on the training GPUs (all of them,
  unless ``rollout-num-gpus`` names fewer), and Slime moves the GPU memory
  between the engines and the workers.
- Slime's ``sglang-*``, ``router-*`` and ``rollout-num-gpus-per-engine``
  options configure the engines.
- A task job whose options give the engines no GPUs fails. A LoRA actor
  cannot have rollout engines.

A task job runs as a dataset job (`Dataset jobs on Slime`_) whose prompts are
Harbor tasks and whose generate step is a Harbor rollout. Slime's own options
set the training method. ``n-samples-per-prompt`` gives the rollouts of each
task, which form a prompt group. Each rollout runs the agent in the task's
Harbor environment, and the agent's model calls reach the engines. The
rollout becomes one Slime sample with the verifier's reward. Slime then
computes the advantages with ``advantage-estimator`` and trains
``policy_loss``.

.. code:: yaml

   training:
     backend: slime
     options:
       colocate: true
       rollout-num-gpus-per-engine: 1
       sglang-mem-fraction-static: 0.5
       advantage-estimator: grpo
       loss-type: policy_loss
       rollout-batch-size: 4
       n-samples-per-prompt: 4
       global-batch-size: 16
       rollout-max-response-len: 2048
   recipe:
     config:
       job_parameters:
         objective: slime_dataset
         save_interval: 10
         harbor:
           environment: modal
           concurrency: 8

Rules of a task job:

- **Loss family.** The recipe's default objective sets the workers' loss
  family, so a deployment that runs task jobs starts with the ``slime``
  family and refuses jobs of trajectories.
- **Rollout count.** The job's rollouts are ``ceil(tasks /
  rollout-batch-size)``, so each task runs at least once. The last batch wraps
  to the first tasks when the tasks do not fill it. To train the tasks again,
  submit another job.
- **Checkpoints.** ``save_interval`` counts Slime rollouts, as on a dataset
  job. A resumed job continues after its last checkpoint and skips the tasks
  of the rollouts that it saved. The rollouts that a stopped driver did not
  train run again.
- **Model calls.** The engines answer OpenAI text chat: system, user and
  assistant text messages, no tools. Each call samples with Slime's
  ``rollout-temperature``, ``rollout-top-p`` and ``rollout-top-k``, whatever
  the agent asks, because the trainer reads that distribution.
  ``rollout-max-response-len`` bounds each call's response.
- **Removed rollouts.** A rollout that fails in the infrastructure, or whose
  model calls do not assemble into one sample, trains nothing. Slime leaves
  its reward out of its group's mean and deviation, so the group's other
  rollouts train against each other. A group with one sample left has
  advantage 0. When no group of a Slime rollout can train, the job fails
  before the workers train it, with the first rollout's error. Slime's own
  metrics, such as ``rollout/raw_reward``, count a removed sample with reward
  0.
- **Score centering.** Slime's ``use-score-centering`` works: each response
  token that the model sampled carries the engines' top
  ``score-centering-top-k`` log-probabilities.
- **Where rollouts run.** The rollouts run in Slime's rollout manager, a Ray
  actor of the serving session. Its node needs the ``harbor`` package and the
  Harbor environment, and reaches the job's directory and the tasks through
  the shared save root. ``agent_host`` names that node. The driver passes
  ``E2B_API_KEY``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` from its
  environment to the job's Ray actors. A driver that stops during a rollout
  leaves its Harbor rollouts running, and the next driver waits for them.

The driver refuses a task job, when it is submitted, in these cases:

- The deployment's loss family, or the objective's, is not ``slime``. The built-in
  objective of that family is ``slime_dataset``.
- The job sets ``learning_rate_schedule`` or ``scheduling``. Slime's options set
  the learning rate and the optimizer steps.
- An option replaces a part of this loop or adds data that the rollouts do not
  record: ``group-rm``, ``reward-key``, ``custom-generate-function-path``,
  ``data-source-path``, another ``rollout-function-path``, ``partial-rollout``, an
  ``over-sampling-batch-size`` other than ``rollout-batch-size``,
  ``dynamic-sampling-filter-path``, a ``rollout-top-p`` other than 1 or
  ``use-rollout-routing-replay``.
- ``n-samples-per-prompt`` is 1 with a group baseline: ``advantage-estimator``
  ``grpo``, ``gspo``, ``cispo`` or ``reinforce_plus_plus_baseline`` with reward
  normalization, which would center each reward to 0.

Dataset jobs on Slime
~~~~~~~~~~~~~~~~~~~~~

A dataset job trains a prompt dataset the way Slime trains on its own. Slime
rolls the prompts out on its serving session's engines, scores them with its
reward model, computes the advantages and trains its loss. Reef records each
saved rollout as a candidate.

The deployment selects the built-in recipe
``reef.recipe.dataset:DatasetRecipe`` with ``recipe.config.training_mode:
manual``. Its loss family, ``slime``, leaves the training signal to Slime and
refuses every other job except a task job. Each training request
(``POST /reef/train``) whose ``parameters`` hold ``dataset`` is one job:

.. code:: json

   {"text": "GRPO on the prompts", "session": "s", "artifact_id": "artifact-1",
    "parameters": {"dataset": {"path": "/data/prompts.jsonl", "epochs": 1}, "save_interval": 10}}

.. container:: config

   ``dataset.path``
       an absolute path, on the Ray cluster, to a JSONL file in Slime's prompt-data format

   ``dataset.epochs`` : 1
       the passes over the dataset. The job trains ``len(dataset) // rollout-batch-size`` rollouts per epoch

   ``save_interval`` : one checkpoint at the end
       rollouts between checkpoints. A checkpoint's step is the job's rollouts up to it, and its reward is the rollout's mean raw reward

A resumed job continues after its last checkpoint and skips the prompts of
the rollouts that it saved. The recipe's ``job_parameters`` can set a default
``dataset``. Slime's own options in ``training.options`` configure the rest:

.. code:: yaml

   training:
     backend: slime
     options:
       colocate: true
       input-key: messages
       label-key: label
       apply-chat-template: true
       rollout-batch-size: 32
       n-samples-per-prompt: 8
       global-batch-size: 256
       rollout-max-response-len: 1024
       rollout-temperature: 1.0
       advantage-estimator: grpo
       custom-rm-path: my_package.reward.reward
       loss-type: policy_loss

A rollout function (``rollout-function-path``,
``custom-generate-function-path``), a reward model (``custom-rm-path`` or
``rm-type``) and a hook on the rollout data after the advantages
(``rollout-data-postprocess-path``) are Slime's plugins and run in Slime's
processes. Reef sets ``data-source-path`` itself, so the option is refused.
``disable-compute-advantages-and-returns`` is refused too.

The ``generator`` section
-------------------------

Absent by default. When present, ``reef serve`` starts the generator service
(``python -m reef.service.deploy.generator_process``) before the HTTP service
and publishes its address as ``${endpoints.generator}``. The generator writes
and checks Harbor tasks for the generated-task recipe, and runs the plays of a
task generator such as SPADE. It runs under the same interpreter as the Reef
service. ``execution.generator`` selects its executor, so a deployment can
place it on another host.

The generator's host needs the ``harbor`` command line and the Harbor
environment that ``generator.environment`` selects; the Reef service itself
needs neither.

.. list-table::
   :header-rows: 1

   * - ``generator.environment``
     - Needs
   * - ``docker`` (default)
     - a Docker daemon on the host
   * - ``e2b``
     - ``E2B_API_KEY`` exported where ``reef serve`` starts, and ``reef-infra[e2b]`` in the generator's interpreter. The readiness check looks there, so ``generator.harbor`` must come from the same Python environment
   * - ``modal``
     - ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` exported where ``reef serve`` starts, and ``reef-infra[modal]`` in the generator's interpreter

With ``e2b``, a task runs only its ``environment/Dockerfile`` (no compose
services), an agent inside the sandbox cannot reach a
``generator.agent-host`` on a private host, and E2B builds a template for each
new task image. The E2B rules of ``rollout.environment`` (`Generated tasks`_)
apply to the checks and to the rollouts: the settings of a self-hosted
service, the sandbox lifetime, and the request to delete the sandbox at its
end. A rollout takes the agent's time limits from ``generator.agent``. A check
runs Harbor's oracle or nop agent, which sets no time limit, so only the
task's time limits count for a check.

The generator runs its proposals, checks and rollouts at the same time.
``generator.concurrency`` limits only the rollouts in flight, so a proposal or
a check never waits behind rollouts.

.. container:: config

   ``generator.tasks-root``
       the directory generated tasks and Harbor job files live under. Required.

   ``generator.host`` : 127.0.0.1
       bind address

   ``generator.port`` : 8910
       bind port

   ``generator.work-dir`` : ``<tasks-root>/.rollouts``
       where the rollout runner keeps trials

   ``generator.agent`` : terminus-2
       the Harbor agent each rollout runs, with ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders. ``import_path: reef.record2sample.generator.terminus:VerifiedTerminus2`` in place of ``name: terminus-2`` also scores a rollout whose agent filled its context or lost its terminal, which Terminus 2 leaves without a reward

   ``generator.agent-host``
       an address of the host the task container can reach, for an agent that runs inside the container

   ``generator.harbor`` : ``harbor`` on PATH
       the harbor command line for the oracle check

   ``generator.environment`` : docker
       the Harbor environment of the oracle and nop checks and of the rollouts: ``docker``, ``e2b`` or ``modal``. The readiness check waits for that environment's requirements

   ``generator.concurrency`` : 2
       rollouts in flight across all rollout requests of the generator service

   ``generator.designer-url``
       a Reef service the designer calls go to instead of this deployment's

   ``generator.designer-token``
       the token for ``designer-url``

   ``generator.designer-model``
       the served model the designer asks for; the deployment's by default

   ``generator.designer-timeout-s`` : 1800
       seconds one designer call may take; ``inference.timeout-s`` must allow it too

   ``generator.designer-api``
       the API the designer asks through: ``openai`` (Chat Completions) or ``responses`` (the Responses API). The default is ``responses`` when designer calls go to this deployment and its ``inference.upstream-api`` is ``responses``, and ``openai`` otherwise

   ``generator.designer-options``
       extra fields of the designer's request, e.g. ``{"reasoning_effort": "none"}``; with ``designer-api: responses``, Responses fields such as ``{"reasoning": {"effort": "low"}}``

   ``generator.ready-timeout`` : 60
       seconds ``reef serve`` waits for the generator to answer

.. code:: yaml

   generator:
     tasks-root: ${REEF_SPADE_STATE_DIR}/tasks
   execution:
     generator: auto
   pipeline:
     record2sample:
       config:
         generator_url: ${endpoints.generator}

The generator reads the resolved deployment: the Reef address from
``reef.host`` and ``reef.port``, the token from ``reef.token``, and the served
model from ``inference.model-path`` or ``inference.upstream-model``. Without
``reef serve``, ``python -m reef.service.deploy.generator_process -c
serve.yaml`` runs it from the same file.

The ``data`` section
--------------------

Settings of the data service, which answers the data routes (`Data items
<http-api.rst#data-items>`__).

.. container:: config

   ``data.max-task-archive-bytes`` : 536870912
       the most bytes of the body of a task archive (512 MiB), for ``POST`` and ``PUT`` on ``/reef/namespaces/{owner}/data/tasks``. A larger body gets HTTP 413. The data service writes the body to a local file before it unpacks it, so the setting limits the disk that one upload uses. A larger task uses the reference form, which sends no files

The ``task-import`` section
---------------------------

Absent by default. When present, each scenario imports the Harbor tasks of a
split manifest when it opens, as its imported tasks. See
`Import Harbor tasks <../user-guide/operate.rst#import-harbor-tasks>`__.

.. container:: config

   ``task-import.manifest``
       a split manifest: a JSON file with ``version``, ``seed``, ``eval_fraction``, and the task names of ``train`` and ``eval``. Required with ``tasks-root``.

   ``task-import.tasks-root``
       the directory that holds a Harbor task directory for each task name of the manifest. Required with ``manifest``.

Experiment tracking
-------------------

Tracking is optional and off by default. It belongs to a Reef scenario, not to
one training backend. Install ``reef[wandb]`` when the training extra does not
already provide it.

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
writes data below ``directory`` for a later ``wandb sync``. ``disabled`` makes
no calls even when ``enabled`` is true. Import, initialization, logging,
summary and upload failures go to the service log and never fail a training
step or its commit.

**Runs.**

- Each scenario maps to a group named after the scenario, or
  ``<group_prefix>/<scenario>``.
- Reef opens one run when the scenario binds and another after each
  train-from. The run id is deterministic, so a restart resumes the same run
  with ``resume=allow``.
- A train-from finishes the current run, records each component's
  training-start artifact ID in ``reef/train_from_artifact_ids``, and resets
  ``train/step`` to zero. An activation does not end a run.
- ``reef/commit_number`` increases across runs, so it joins a run back to the
  commit history.
- A scenario with several trainers shares the run. Each step's metrics carry
  the component name as a prefix (``harness/train/loss``) on the same
  ``train/step`` axis, with ``reef/component`` on the row. Its optimizer step
  rows land under ``<component>/step/*`` on their own counter. The run config
  lists each component's backend under ``reef.components`` and
  ``backend.<component>``, with ``reef.backend`` null.
- The names ``train``, ``step``, ``reef`` and ``operations`` are these
  prefixes, so a composite recipe refuses them as component names.

Recipe and processor code logs through the same logger without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own
``<namespace>/event`` axis. Only finite numeric values are sent.

Operational metrics
~~~~~~~~~~~~~~~~~~~

With W&B enabled, the dispatcher samples each loaded scenario, inference-only
scenarios included, every 10 seconds and once during graceful shutdown. The
samples go to the scenario's run under ``operations/*``, with Unix time in
``operations/time_seconds`` as their horizontal axis. They do not advance
``train/step``. Offline mode records the same samples locally; disabled
tracking starts no sampling.

The following names are relative to ``operations/``:

.. list-table:: Operational measurements
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Meaning
   * - ``ingest/accepted_total``, ``ingest/duplicates_total``
     - New records appended and identical retries acknowledged by the dispatcher.
       Accepted records include incomplete-stream diagnostics; acceptance does
       not imply a trainable record or successful training.
   * - ``ingest/rejected_report_total``, ``ingest/rejected_request_total``
     - Report schema/reference violations and invalid training requests
       detected during dispatcher record validation.
   * - ``ingest/rejected_conflict_total``
     - Record identities reused with different content.
   * - ``ingest/write/*``
     - Record-store append calls, including identical retries that reach the
       store. ``failed_total`` counts append exceptions, including conflicts;
       validation rejections before append do not count as write failures.
   * - ``records/unread_count``
     - Training-visible records after the processor's read cursor. These may
       still need feedback or filtering; this is not a ready job data count.
   * - ``records/oldest_unread_age_seconds``
     - Age of the first unread record, measured from its recorded creation
       time; zero when none remain.
   * - ``processor/waiting_reports``
     - Reported-feedback reports that wait for their group or for the sample
       store. An incomplete group still counts as waiting.
   * - ``processor/oldest_report_wait_seconds``
     - Age of the oldest waiting report; zero when none remain.
   * - ``processor/tracked_records``, ``processor/judging_records``
     - Computed-feedback records waiting for more traffic or a judgment.
   * - ``processor/waiting_candidates``
     - Computed-feedback candidates that wait to become samples.
   * - ``processor/ready_samples``, ``processor/job_samples``, ``processor/buffer_size``
     - Samples that the next automatic job can take now, samples that
       automatic job data needs, and the most samples that one automatic job
       takes.
   * - ``training/reserved_job_data``
     - Zero or one. A reservation can be executing or waiting for settlement;
       it is not necessarily queued job data.
   * - ``training/auto_enabled``
     - One for auto training, zero for manual training. Waiting data in
       manual mode does not by itself indicate a stalled worker.
   * - ``training/error``, ``training/failed_attempts_total``
     - Current recorded training error (zero/one) and cumulative failures
       recorded by the dispatcher, including retries and worker failures.
       Recovery clears the current error but retains the counter. The counter
       resets on dispatcher restart or scenario deletion.
   * - ``training/execution/*``
     - The trainer's execution calls: a weight trainer's job submissions and
       polls, including checkpoint evaluations, or a harness trainer's
       candidate preparation.

Each operation family (``ingest/write``, ``training/execution`` and the
request families) has these fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``started_total``, ``completed_total``, ``failed_total``
     - calls started, completed and failed
   * - ``active``
     - calls in flight; zero or one for execution calls, which are serial
   * - ``elapsed_seconds``
     - age of the oldest active call; zero when idle
   * - ``duration_seconds_total``
     - sum of the durations of finished calls; overlapping calls can add up faster than wall time
   * - ``last_duration_seconds``
     - duration of the last finished call

To get the mean latency over an interval, divide the change in
``duration_seconds_total`` by the change in completed plus failed calls.

How to read the samples:

- A completed execution call is not necessarily a commit: a poll of a running
  job that saved no new checkpoint commits nothing.
- The measurements are not persisted training history. Execution measurements
  reset when their trainer is rebuilt, including recovery;
  ``training/started_at_seconds`` identifies that reset. Request and ingestion
  measurements reset when the scenario is rebuilt;
  ``operations/started_at_seconds`` identifies that reset.
- Malformed HTTP payloads and headers, and failures before Reef resolves the
  scenario, are outside these measurements.
- Additional finite numeric processor status fields appear under
  ``operations/processor/``. Sampling does not advance the processor or
  consume its queue. If ingestion or a commit holds the trainer lock, that
  sample omits the queue gauges and still reports the execution measurements.
- Samples contain no request bodies, record IDs, scenario names or error
  messages. Uploading them configures no alert notifications. The inference
  engines keep their own high-frequency monitoring.

Commit correlation
~~~~~~~~~~~~~~~~~~

Commit metrics carry ``experiment/provider``, ``experiment/project``,
``experiment/group`` and ``experiment/run_id``. Use them to open the run from
a Reef version, and use the run's ``reef/training_job_id`` to go the other way.

The job listing, ``GET /reef/scenarios/{scenario}/jobs``, gives the run of
each training job as ``experiment``. Its ``url`` is the URL that the open W&B
run gives. An offline run gives none: the URL is then
``https://wandb.ai/<entity>/<project>/runs/<run_id>``, where the run is once
it is synced, with the configured ``entity``. Without a configured ``entity``,
the URL of an offline run is ``null``.

With ``upload_checkpoints: true``, Reef reads the files of each committed
checkpoint from the artifact backend and uploads them as a W&B artifact.
Otherwise a run names its checkpoints by their artifact IDs only.

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

.. container:: config

   ``endpoint``
       the complete traces URL

   ``authorization`` : ``REEF_TRACING_AUTHORIZATION``
       the credential that the backend expects in its ``Authorization`` header: ``Basic <base64 public:secret>`` for Langfuse, ``Bearer <token>`` for most others. Write it as an environment reference, or omit it and export the variable. The startup report masks it, and it never appears in a span or a log line

   ``headers``
       a mapping for a backend that expects its credential under another header name, such as ``x-api-key``. The startup report masks every header value. ``Authorization`` is rejected here, so the credential has one place

   ``service_name``
       optional resource ``service.name``

When no endpoint, credential or header is configured, the exporter reads the
standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` and ``OTEL_EXPORTER_OTLP_HEADERS``
environment variables instead.

Each accepted inference record becomes the root span of its own trace, named
``chat <model>``. Its trace and span ids derive from the scenario and record
id, so a restart or a second Reef host produces the same ids. The span carries:

* ``session.id`` and ``reef.scenario``: the scenario.
* ``reef.agent_record_id`` and ``reef.request_type``: the receipt and record kind.
* ``gen_ai.request.model``, ``gen_ai.response.model``, ``gen_ai.response.id``,
  ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens`` and
  ``gen_ai.response.finish_reasons`` from the provider request and response,
  following the OpenTelemetry GenAI semantic conventions.
* ``reef.artifact_id``: the artifact that answered the request.
* ``reef.tags``: the ``x-reef-tag-*`` request tags.

Other records become these spans:

* A feedback record becomes a ``feedback`` span inside the trace of the first
  inference that it references, with ``reef.score`` and ``reef.references``;
  further references are span links.
* A training commit adds a ``training commit N`` span with the commit number
  as ``reef.commit_number``, component artifact IDs, job id, consumed record
  counts, its component as ``reef.component`` when the scenario runs several
  trainers, and its scalar metrics as ``reef.metrics.*``.
* Every record that the commit consumed gets one ``trained in commit N`` child
  span, linked back to the commit span.

Records carry one timestamp, so their spans have zero duration and start at
the record's creation time.

For streamed provider responses, Reef reads token usage from the captured SSE
events, including Chat Completions usage chunks, Responses terminal events
and Anthropic message usage. If the provider sends no usage, token counts
remain absent; for Chat Completions, request ``stream_options:
{include_usage: true}`` when supported. The stored response and the forwarded
stream remain unchanged.

Spans carry the exchange itself: the request messages and the reply as JSON
in ``gen_ai.input.messages`` and ``gen_ai.output.messages``, and feedback text
in ``reef.feedback``. The backend therefore sees the scenario's traffic; point
tracing only at a backend that you trust with it. Export failures go to the
service log and never fail record acceptance or a commit. The exporter batches
spans in a background thread and flushes them during graceful shutdown.
