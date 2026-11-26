Inference serving
=================

The rules of the ``inference`` section that its keys do not show: the serving engines of a weight-training deployment, remote inference components, the engines' native options, the options of the request handler, and the chat template arguments. `Settings reference <configuration-settings.rst#inference>`__ lists every key with its type and default.

Serving of a weight-training deployment
---------------------------------------

The ``inference`` section alone decides how a weight-training deployment serves its snapshots. With ``inference.serving: engines``, Reef starts SGLang or vLLM engines on an inference component, and the engines load each published snapshot. `Serving runtimes <../developer-guide/serving-runtimes.rst>`__ describes what the engines load, request pinning and engine supervision.

- **Engine settings.** ``inference.backend``, ``inference.num-gpus``, ``inference.tensor-parallel-size``, ``inference.options``, ``inference.remote``, ``inference.handler-factory``, ``inference.handler-config``, ``inference.max-loaded-adapters`` and ``inference.prefix-sharing`` configure the engines. One of them selects ``engines`` when ``inference.serving`` is unset. Another serving mode refuses them.
- **Prefix sharing.** With adapter snapshots, SGLang keeps its prefix cache only when it keys the entries by adapter. Otherwise the engines run without a prefix cache.
- **KV outside the engine.** Engine serving refuses the ``inference.options`` that keep KV outside the engine, with or without prefix sharing: SGLang's ``hicache_storage_backend`` and ``enable_lmcache``, and vLLM's ``kv_transfer_config``, ``kv_offloading_size`` and ``kv_offloading_backend``. Reef checks these option names only. Do not give the engines another KV store through a plugin or an environment variable.
- **SGLang engines.** Options use SGLang's names without a ``sglang-`` prefix. ``router-port`` binds the router's HTTP port, and other router flags keep their ``router-*`` names. A standalone engine has no router. Reef probes the router's metrics port from 4000 up, so use a ``router-port`` outside that range. A conflict on a configured ``router-port`` stops the launch.
- **vLLM engines.** Options use vLLM's engine-argument names. Reef sets ``model``, ``host``, ``port`` and ``tensor_parallel_size`` itself and refuses ``enable_sleep_mode``. Reef defaults ``generation_config`` to ``vllm``, so the model's own generation defaults add no truncation or temperature that the trainer does not see. Reef starts no vLLM router, so ``inference.num-gpus`` must equal ``inference.tensor-parallel-size``.

Remote inference components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``inference.remote`` section runs the engines on an inference component on another host, in place of the local component that ``reef serve`` starts. Any training backend can serve on a remote component. `Serve from a remote GPU host <../user-guide/tinker.rst#serve-from-a-remote-gpu-host>`__ shows the setup.

One component serves one full-weight checkpoint. ``inference.remote.slot-urls`` names further components, so that a scenario serves one more checkpoint on each (`Set what a scenario serves <../user-guide/operate.rst#set-what-a-scenario-serves>`__):

- Start each further component in the same way as the first one, with the same ``--engine`` and the same two tokens. Reef starts the same model and engine options on each one, and ``inference.num-gpus`` counts the GPUs of one component.
- Give each component GPUs that no other component uses: its own host or visible GPUs, or one Ray cluster that all of them join (``--ray-address``). Different URLs alone do not separate the GPUs.
- Reef refuses ``slot-urls`` for adapter snapshots and for a trainer's rollout engines.

Native backend options
----------------------

``inference.options`` and ``training.options`` hold the native options of the selected backend, by flag name without the leading ``--``. The backend owns their types, defaults and validation. Reef copies no schema.

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8

- An override of one option keeps the other options of the file. An explicit whole object, such as ``--inference.options '{}'``, replaces all of them.
- Reef refuses an option that it sets itself, such as the model, the parallel sizes, the bind address and the API key. Engine serving also refuses ``trust-remote-code`` and the adapter options (``enable-lora``, ``max-lora-rank``, ``max-loaded-loras``, ``max-loras-per-batch``, ``lora-target-modules``). The error names the option.
- **A standalone engine and Slime** get the options as command-line arguments. ``true`` gives a switch, ``false`` or ``null`` gives nothing, a list gives several values, and an object is passed as JSON. To turn off a feature that the engine enables by default, use the engine's own disabling flag.
- **The serving engines of a weight-training deployment** differ. SGLang engines get the options as fields of SGLang's ``ServerArgs``: ``false`` is a value, and Reef drops a name that ``ServerArgs`` does not have, without an error. vLLM engines get ``false`` as ``--no-<flag>``.
- **An in-process training backend**, such as Tinker, parses ``training.options`` with the types that it declares (`Settings reference <configuration-settings.rst#training>`__).

Request handler options
-----------------------

The serving engines answer through Reef's own request handler for ``inference.backend``. ``inference.handler-config`` holds the handler's options, with underscores as shown; another option fails the handler's start. ``inference.handler-factory`` selects no other handler: every built-in deployment refuses a class that is not the engine's own.

.. container:: config

   ``tool_call_parser``
       Name of the engine's tool-call parser. A request with tools fails without it.

   ``sampling_defaults`` : {}
       Sampling fields, in the engine's names, for a request that does not set them. A scenario's `serving config <http-api-scenarios.rst#serving-config>`__ sets ``temperature``, ``top_p`` and ``top_k`` over them.

   ``chat_template_kwargs`` : {}
       Chat template arguments for a request that omits them, such as ``{enable_thinking: false}``. A request's own ``chat_template_kwargs`` win key by key.

   ``force_reasoning``
       ``true`` if the chat template ends the prompt with an open ``<think>``, ``false`` if not. Unset, the handler checks the end of each rendered prompt.

What a client of these engines gets:

- **Streaming.** SGLang engines relay their incremental stream. vLLM engines answer a streaming request with the completed turn as one burst of frames.
- **Sampling extras.** Engine-specific sampling fields of one request go in ``sglang_sampling_params`` or ``vllm_sampling_params``. vLLM engines refuse a ``top_p`` below 1, a ``top_k`` above 0 and a ``min_p`` above 0, because the trainer cannot reproduce the log probabilities.
- **Reasoning.** The reply is split into ``content`` and ``reasoning_content``: ``<think>`` starts the reasoning, and the first ``</think>`` or ``<tool_call>`` ends it. A reply that ends inside the reasoning has an empty ``content``.

Chat template arguments
-----------------------

Serving records the provider exchange only, except for a call with ``x-reef-record-training`` (see `Headers <http-api.rst#headers>`__), whose record also keeps the engine's token IDs. The training component renders each other recorded call again with the model's chat template, and it must render the call as the serving engine did. Give it the same chat template arguments that the engine uses: the handler's ``chat_template_kwargs``, or the arguments that the clients send.

- Slime reads ``training.options.apply-chat-template-kwargs``, a JSON object.
- Tinker reads ``training.options.chat-template-kwargs``, a mapping.
