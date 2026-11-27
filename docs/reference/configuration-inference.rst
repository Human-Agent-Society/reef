Inference serving
=================

Set ``inference.serving: engines`` to serve published weights with SGLang or vLLM.
Set the engine's GPU allocation independently of training. For vLLM, GPU count must equal the tensor parallel size.
See the `inference settings <configuration-settings.rst#inference>`__ for defaults.

Use ``inference.remote`` to serve from another GPU host. Each component serves one full-weight checkpoint.
Use ``inference.remote.slot-urls`` for extra slots. Slots do not accept adapters or training rollout engines.
Follow the `remote GPU setup <../user-guide/tinker.rst>`__.

Native backend options
----------------------

Use native flag names without ``--`` under ``inference.options``. For SGLang, omit the ``sglang-`` prefix but keep ``router-``.

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct \
     --inference.options.mem-fraction-static 0.8

Use Reef settings for the model, parallel sizes, bind address and API key. An option override changes one key.
Passing a whole object, such as ``--inference.options '{}'``, replaces all options.
External KV stores are unsupported. SGLang adapter snapshots need adapter-specific cache keys for prefix caching.

Chat template arguments
-----------------------

Configure the handler under ``inference.handler-config``. Use underscores in key names.

- Set ``tool_call_parser`` for requests that use tools.
- Set ``sampling_defaults`` for omitted fields. Scenario serving settings can override ``temperature``, ``top_p`` and ``top_k``.
- Set ``chat_template_kwargs`` for the model's chat template. Request values override individual keys.
  Use ``force_reasoning`` to override the handler's detection of an open ``<think>`` tag.

Training renders the exchange again. Match its chat template arguments to serving:
set ``training.options.apply-chat-template-kwargs`` for Slime or ``training.options.chat-template-kwargs`` for Tinker.
The record of a call with `x-reef-record-training <http-api.rst#headers>`__ also keeps the engine's tokens; a delayed sample trains them as they are.
