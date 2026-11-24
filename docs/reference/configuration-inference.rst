The ``inference`` section
=========================

The keys of the ``inference`` section: the model or provider that answers requests, the serving engines of a weight-training deployment, their native options, the request handlers, and the chat template arguments.

.. container:: config

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

   ``inference.timeout-s`` : 300.0
       maximum time for one inference request, in seconds; an engine that does not answer within it gets a 504

   ``inference.backend`` : sglang
       the managed engine: ``sglang``, ``vllm``, or ``none`` (`Train without an inference engine <configuration-training.rst#train-without-an-inference-engine>`__)

The retry keys of routed requests, ``inference.retry-*``, are in `Scenario routing between students and teachers <configuration-routing.rst#scenario-routing-between-students-and-teachers>`__.

Serving of a weight-training deployment
---------------------------------------

The ``inference`` section alone decides how a weight-training deployment serves its snapshots. The training backend builds only its training component. That component declares the format of its snapshots: full weights, or a PEFT adapter with its rank and target modules.

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

The engine settings are ``inference.backend``, ``inference.num-gpus``, ``inference.tensor-parallel-size``, ``inference.options``, ``inference.remote``, ``inference.handler-factory``, ``inference.handler-config``, ``inference.max-loaded-adapters`` and ``inference.prefix-sharing``. Another serving mode refuses them.

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

**What the engines serve.** The HTTP service uploads each published snapshot to the inference component as files:

- Full-weight training publishes the checkpoint's Hugging Face export. The component reloads it in place of the served weights. Each load moves the whole model, for example about 16 GB for an 8B bf16 model.
- LoRA training (``training.options.megatron-lora-rank``) publishes the scenario's PEFT adapter. The component loads it as the scenario's adapter beside the base model. Only ``inference.backend: sglang`` serves adapters.
- Snapshots without weight files, such as Tinker's, are served by engines only when the training backend names how to fetch their weights (``TrainingDeployment.engine_weights``); Tinker's does. Otherwise the HTTP service refuses the pairing when it starts.

The training service stores each checkpoint in the artifact backend from the checkpoint directory, and every other service reads it from the artifact backend. With multi-node training, put the checkpoint directory on storage that the training service's host can read.

**Request pinning.** With ``engines``, the inference component keeps each production request on the weights that the request selected (`Request pinning <../developer-guide/serving-runtimes.rst#request-pinning>`__). No setting turns this off. It needs ``inference.backend: sglang`` or ``vllm`` with the engine's own request handler, so Reef refuses another backend or ``inference.handler-factory`` here. Another engine needs its own serving runtime (``inference.serving: module:Factory``). With ``trainer``, Tinker's sampler serves each snapshot as an immutable revision and needs no pinning.

**Prefix sharing.** With ``inference.prefix-sharing: true``, the engines clear their prefix cache before each change of weights or adapters, after the admitted requests finish. The clear covers the engine's own cache, which includes SGLang's host-memory tier (``enable_hierarchical_cache``). Engines that do not confirm the clear stay unavailable until the control plane replaces them. For adapter artifacts, SGLang keeps the cache only when it keys the entries by adapter; otherwise the engines run without a prefix cache.

The clear does not cover KV that an engine keeps outside itself. Engine serving therefore refuses these ``inference.options``, with or without prefix sharing: SGLang's ``hicache_storage_backend`` and ``enable_lmcache``, and vLLM's ``kv_transfer_config``, ``kv_offloading_size`` and ``kv_offloading_backend``. Reef checks these option names only. Do not give the engines another KV store through a plugin or an environment variable.

**Engine failures.** The control plane checks the inference component every 30 seconds. When the component has restarted, or an engine has failed, the control plane pauses generation, replaces the component's engine stack, loads each scenario's served artifact again, and resumes (`Engine supervision <../developer-guide/serving-runtimes.rst#engine-supervision>`__).

**SGLang engines.** Native engine options use SGLang's names without a ``sglang-`` prefix, for example ``--inference.options.mem-fraction-static 0.7``. ``router-port`` binds the router's HTTP port, and other router flags keep their native ``router-*`` names. The standalone engine launcher includes no router. `Independent SGLang backend <../developer-guide/serving-runtimes.rst#independent-sglang-backend>`__ describes the integration's boundary.

Each engine and the router probe free ports on the host at startup; the router's metrics port is probed from 4000 up. If a start fails because another process took a probed port, Reef starts the engine or the router again on new ports, at most three attempts each. Other startup failures stop the launch. A configured ``router-port`` does not change, so a conflict on it stops the launch. Another stack's metrics port can take a ``router-port`` near 4000, so use a port outside that range.

**vLLM engines.** ``inference.backend: vllm`` launches one vLLM server per engine with ``VLLM_SERVER_DEV_MODE=1`` and ``--logprobs-mode processed_logprobs``. It loads each published snapshot through vLLM's control routes: ``/pause?mode=keep``, ``/update_weight_version``, ``/collective_rpc reload_weights`` and, with ``inference.prefix-sharing: true``, ``/reset_prefix_cache?reset_running_requests=true``.

- Engine options use vLLM's engine-argument names. Reef sets ``model``, ``host``, ``port`` and ``tensor_parallel_size`` itself and refuses ``enable_sleep_mode``.
- Reef defaults ``generation_config`` to ``vllm``, so the model's own generation defaults cannot add truncation or a temperature that the trainer never sees.
- Reef enables prefix caching only with ``inference.prefix-sharing: true``.
- vLLM engines serve full-weight artifacts only. The HTTP service refuses adapter artifacts, such as Slime's with ``training.options.megatron-lora-rank`` or Tinker's, when it starts.
- Reef starts no vLLM router, so one single-node engine serves the stack: ``inference.num-gpus`` must equal ``inference.tensor-parallel-size``.
- Each engine takes one port, probed upward from ``engine_port_base`` (default 15000). Reef restarts an engine that loses its port to another process, at most three attempts.

Remote inference components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``inference.remote`` section runs the engines that ``inference.backend`` names on a remote inference component instead of this host. The component's ``--engine`` must name the same backend. Any training backend can serve on a remote component. `Serve from a remote GPU host <../user-guide/tinker.rst#serve-from-a-remote-gpu-host>`__ shows the setup.

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

One component serves one full-weight checkpoint. To let a scenario serve several checkpoints at the same time, name one more component for each further checkpoint:

.. code:: yaml

   inference:
     remote:
       url: https://gpu-a.example.com
       slot-urls:
         - https://gpu-b.example.com

The one training scenario then serves as many checkpoints together as there are components. A further component does not add a training scenario. These rules apply:

- Start each further component in the same way as the first one, with the same ``--engine`` and the same two tokens. Reef starts the same model and engine options on each one, and ``inference.num-gpus`` counts the GPUs of one component.
- Give each component GPUs that no other component uses: its own host or visible GPUs, or one Ray cluster that all of them join (``--ray-address``). Different URLs alone do not separate the GPUs.
- Use ``slot-urls`` with full-weight artifacts only. Reef refuses it for adapter artifacts and for a trainer's rollout engines.

`Set what a scenario serves <../user-guide/operate.rst#set-what-a-scenario-serves>`__ describes how a scenario uses the components, and how to replace a checkpoint when every component holds one.

Native backend options
----------------------

``inference.options`` and ``training.options`` hold the selected backend's native flags, without the leading ``--``. The backend owns their types, defaults and validation; Reef does not copy that schema.

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8 \
     --inference.options.trust-remote-code true

These become ``--mem-fraction-static=0.8`` and ``--trust-remote-code`` arguments of ``python -m sglang.launch_server``.

- For engines that take command-line arguments, such as SGLang and Slime, ``true`` emits a switch and ``false`` or ``null`` omits it. To disable a feature that the engine enables by default, use the engine's own disabling flag. A list supplies several argument values, and an object is passed as JSON.
- In-process training passes the values to its runtime parser: ``false`` stays false, and null follows the declared field type.
- A field override keeps its YAML siblings. An explicit whole object, such as ``--inference.options '{}'``, replaces the entire object.
- In managed serving, native options cannot override the model, the served model name, the tensor parallel size, the bind address, authentication or multi-node launch controls. Use the public fields for those.

For Slime, ``training.options`` holds flags such as ``lr: 0.000001`` and ``use-critic: true``; override one with ``--training.options.lr 0.000002``. The training component parses the options once at startup, so a malformed option fails before the first job. The driver takes out Reef's own options, such as ``--reef-checkpoint-policy``, and the loss family's options, and passes the rest to Slime's parser.

**Request handlers.** ``inference.handler-factory`` names an ``InferenceHandler`` subclass with a ``from_config`` class method, as ``package.module:ClassName``; handler functions are not supported. ``inference.handler-config`` holds that handler's options, for example its tool parser. It does not configure the engine process.

Two token-native handlers render the prompt with the model's chat template and call the engine's native generate route:

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

Both handlers answer with the provider response, without training tensors. A client gets token log probabilities only when it sends the OpenAI ``logprobs: true``, and token IDs only when it sends vLLM's ``return_token_ids: true``. Per-request sampling extras pass through ``sglang_sampling_params`` or ``vllm_sampling_params``. Both handlers accept these options; any other option, such as ``capture_topk``, fails the handler's start.

.. container:: config

   ``tool_call_parser``
       the engine's parser name

   ``sampling_defaults``
       sampling fields for a request that does not set them. A scenario can set ``temperature``, ``top_p`` and ``top_k`` over them with its `serving config <http-api-scenarios.rst#serving-config>`__; a scenario's value is also a default, and it applies alike to OpenAI chat and Anthropic messages requests

   ``chat_template_kwargs``
       chat template arguments for a request that omits them, for example ``{enable_thinking: false}``; a request's own ``chat_template_kwargs`` win key by key

   ``force_reasoning`` : unset
       ``true`` if the chat template pre-opens ``<think>``, ``false`` if it does not. Unset, the handler checks whether each request's rendered prompt ends with ``<think>``, so the check follows the request's ``chat_template_kwargs``

Both handlers, and the Tinker sampler, split each reply into ``content`` and ``reasoning_content`` with the rules of SGLang's Qwen3 reasoning parser: ``<think>`` starts reasoning, and the first ``</think>`` or ``<tool_call>`` ends it. A reply that ends inside reasoning has an empty ``content``. The Tinker sampler takes no ``force_reasoning`` and always checks the prompt.

Chat template arguments
-----------------------

Serving records the provider exchange only. The training component renders each recorded call with the model's chat template, and must render it as the serving engine did. Give the training component the same chat template arguments that the engine renders with: the handler's ``chat_template_kwargs``, or the arguments that the clients send.

- Slime renders with the tokenizer of ``--hf-checkpoint`` and ``training.options.apply-chat-template-kwargs``, a JSON object.
- Tinker renders with the base model's tokenizer and ``training.options.chat_template_kwargs``, a mapping.

.. code:: yaml

   inference:
     handler-config:
       chat_template_kwargs: {enable_thinking: false}
   training:
     backend: slime
     options:
       apply-chat-template-kwargs: '{"enable_thinking": false}'

A sample that cannot render, or that is longer than a limit of its stage, does not train. The job metric ``unrenderable_samples`` counts these samples. A chat template that drops the reasoning of earlier turns, as Qwen3's does, renders a multi-turn sample as several segments. Each segment trains as one sample (see `Processors <../developer-guide/write-a-recipe.rst>`__).
