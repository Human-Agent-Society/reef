Inference serving
=================

Configure serving engines, remote hosts and request handlers in ``inference``.
See the `Settings reference <configuration-settings.rst#inference>`__ for key types and defaults.

Serving of a weight-training deployment
---------------------------------------

Set ``inference.serving: engines`` to serve published weights with SGLang or vLLM.
If ``serving`` is unset, specifying an engine setting selects this mode. Other modes reject engine settings.

- **SGLang.** Use native option names without ``sglang-``. Router options retain ``router-*``. Keep ``router-port`` clear of the metrics port chosen from 4000 upward. Standalone engines have no router.
- **vLLM.** GPU count must equal the tensor parallel size. Reef sets ``model``, ``host``, ``port`` and ``tensor_parallel_size``, rejects ``enable_sleep_mode``, and defaults ``generation_config`` to ``vllm``.
- **Caches.** SGLang adapter snapshots require adapter-specific cache keys for prefix caching.
  External KV stores are unsupported. Reef rejects ``hicache_storage_backend``, ``enable_lmcache``, ``kv_transfer_config``, ``kv_offloading_size`` and ``kv_offloading_backend``. This restriction also applies to plugins and environment settings.

Remote inference components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``inference.remote`` with any training backend. Each component serves one full-weight checkpoint.
Add ``inference.remote.slot-urls`` for additional checkpoints. Slots exclude adapters and trainer rollout engines.

Use the same engine and tokens on each component. ``inference.num-gpus`` counts GPUs per component.
Allocate separate hosts or visible GPUs, or provide a shared ``--ray-address``.
Follow the `remote GPU setup <../user-guide/tinker.rst#serve-from-a-remote-gpu-host>`__.

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

Put these keys under ``inference.handler-config``. Use underscores. Unknown keys or a different engine's handler fail startup.

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
