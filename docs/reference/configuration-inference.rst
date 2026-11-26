Inference serving
=================

Configure serving engines, remote hosts and request handlers in ``inference``.
See the `Settings reference <configuration-settings.rst#inference>`__ for key types and defaults.

Serving of a weight-training deployment
---------------------------------------

With ``inference.serving: engines``, Reef starts SGLang or vLLM to serve
published weights. The ``inference`` section controls these engines.
See `Serving runtimes <../developer-guide/serving-runtimes.rst>`__ for loading,
request pinning and supervision.

- **Engine settings:** ``backend``, ``num-gpus``, ``tensor-parallel-size``, ``options``,
  ``remote``, ``handler-factory``, ``handler-config``, ``max-loaded-adapters``
  and ``prefix-sharing`` under ``inference`` select ``engines`` when ``serving``
  is unset. Other serving modes reject them.
- **Prefix cache:** SGLang can retain cache entries for adapter snapshots only
  if the cache keys include the adapter. Otherwise Reef disables the cache.
- **External KV stores:** Reef rejects SGLang's ``hicache_storage_backend``
  and ``enable_lmcache``. It also rejects vLLM's ``kv_transfer_config``,
  ``kv_offloading_size`` and ``kv_offloading_backend``.
  These restrictions apply with or without prefix sharing.
  Reef checks only these option names. Do not configure another KV store through plugins or environment variables.
- **SGLang:** use native option names without a ``sglang-`` prefix.
  Router options retain ``router-*`` names; ``router-port`` sets its HTTP port.
  Standalone engines have no router. Reef selects a router metrics port from
  4000 upward, so choose an HTTP port outside that range.
  A configured port conflict stops startup.
- **vLLM:** use native engine-argument names.
  Reef controls ``model``, ``host``, ``port`` and ``tensor_parallel_size``;
  it rejects ``enable_sleep_mode``.
  ``generation_config`` defaults to ``vllm`` to avoid model-specific generation
  defaults that training does not see.
  There is no router, so ``inference.num-gpus`` must equal ``inference.tensor-parallel-size``.

Remote inference components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``inference.remote`` to run engines on another host.
Any training backend can use a remote component.
Follow `Serve from a remote GPU host
<../user-guide/tinker.rst#serve-from-a-remote-gpu-host>`__ for setup.

Each component serves one full-weight checkpoint.
Add ``inference.remote.slot-urls`` to serve more checkpoints
(`Set what a scenario serves <../user-guide/operate.rst#set-what-a-scenario-serves>`__):

- Start each component with the same ``--engine`` and both tokens.
  Reef uses the same model and options on each component.
  ``inference.num-gpus`` counts GPUs per component.
- Allocate separate hosts or visible GPUs, or join one Ray cluster with ``--ray-address``.
  Different URLs alone do not isolate GPUs.
- ``slot-urls`` does not support adapter snapshots or trainer rollout engines.

Native backend options
----------------------

``inference.options`` and ``training.options`` accept native backend flags
without the leading ``--``. The backend defines their types, defaults and validation.

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8

- A leaf override preserves other options. A whole object, such as
  ``--inference.options '{}'``, replaces all options.
- Reef rejects options it controls: model, parallel sizes, bind address and API key.
  Engine serving also rejects ``trust-remote-code``, ``enable-lora``,
  ``max-lora-rank``, ``max-loaded-loras``, ``max-loras-per-batch`` and ``lora-target-modules``.
- **Standalone engines and Slime:** ``true`` becomes a switch;
  ``false`` and ``null`` add no argument. Lists become multiple values; objects become JSON.
  To disable a default feature, use the backend's disabling flag.
- **SGLang engines for weight training:** options become ``ServerArgs`` fields.
  ``false`` remains a value. Reef silently drops names absent from ``ServerArgs``.
- **vLLM engines for weight training:** ``false`` becomes ``--no-<flag>``.
- **In-process training backends**, including Tinker, parse the types they declare
  (`Settings reference <configuration-settings.rst#training>`__).

Request handler options
-----------------------

Each engine uses its Reef request handler.
Set its options in ``inference.handler-config`` with underscores as shown below.
Unknown options fail startup. Built-in deployments reject a
``handler-factory`` other than the engine's own handler.

.. container:: config

   ``tool_call_parser``
       Name of the engine's tool-call parser. A request with tools fails without it.

   ``sampling_defaults`` : {}
       Defaults for omitted request fields, in the engine's names.
       A scenario's `serving config <http-api-scenarios.rst#serving-config>`__
       overrides ``temperature``, ``top_p`` and ``top_k``.

   ``chat_template_kwargs`` : {}
       Default template arguments, such as ``{enable_thinking: false}``.
       Request-level ``chat_template_kwargs`` override individual keys.

   ``force_reasoning``
       Set ``true`` if the template ends with an open ``<think>``, or ``false`` otherwise.
       If unset, the handler inspects each rendered prompt.

Response behavior:

- **Streaming:** SGLang relays incremental output. vLLM sends the completed
  turn as one burst of frames.
- **Sampling:** use ``sglang_sampling_params`` or ``vllm_sampling_params``
  for engine-specific request fields. vLLM rejects ``top_p < 1``,
  ``top_k > 0`` and ``min_p > 0`` because training cannot reproduce their log probabilities.
- **Reasoning:** ``<think>`` starts ``reasoning_content``.
  The first ``</think>`` or ``<tool_call>`` ends it.
  If the reply ends during reasoning, ``content`` is empty.

Chat template arguments
-----------------------

Reef records the provider exchange. Training renders it again with the model's
chat template, so its template arguments must match serving.
Use the handler's ``chat_template_kwargs`` or the values sent by clients:

- Slime reads ``training.options.apply-chat-template-kwargs``, a JSON object.
- Tinker reads ``training.options.chat-template-kwargs``, a mapping.
