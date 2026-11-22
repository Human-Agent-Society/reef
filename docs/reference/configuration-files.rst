Configuration file layout
=========================

How ``reef serve`` reads its settings: the layout of a configuration file, the command-line flags that match it, the value rules, and the starts that need no file.

The sections of a configuration file match the command-line paths. `Configure Reef serving and training <configuration.rst>`__ lists the sections and the page of each one.

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

- An explicit command-line value overrides the YAML value, and an omitted field uses its declared default.
- Section names and declared fields accept hyphens or underscores. Both spellings of one field in one file are an error.
- Reef rejects unknown sections and undeclared fields.
- A configuration file contains no process definitions: ``service``, ``services`` and ``execution.services`` are rejected. The selected recipe and the inference and training backends determine the processes.

Start without a file
--------------------

Local inference needs one flag:

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct

Reef starts SGLang on a free loopback port, waits for its health endpoint, and then starts its HTTP service on ``127.0.0.1:8900``.

- A local path or a Hugging Face ID names the model. Reef resolves a downloaded snapshot once and keeps the identifier that you wrote as the served model name.
- ``--inference.backend vllm`` starts ``vllm.entrypoints.openai.api_server`` instead of SGLang. ``--inference.tensor-parallel-size 2`` uses two visible GPUs; the default is one. The managed path supports a single GPU node.
- The service interpreter (``REEF_PYTHON``, otherwise the launcher's) must have the engine and GPU-enabled PyTorch installed. See the `SGLang installation guide <https://docs.sglang.io/docs/get-started/install>`__ or the `vLLM installation guide <https://docs.vllm.ai/en/latest/getting_started/installation/>`__. The base Reef installation stays CPU-only.
- The engine has a one-hour readiness deadline and Reef has 30 seconds. On failure or interruption, Reef stops both processes. Logs, including ``sglang.log`` or ``vllm.log``, are under ``.reef/run/``.
- ``--inference.model-path`` cannot be combined with an upstream URL or model. Native engine options go in ``inference.options`` (`Native backend options <configuration-inference.rst#native-backend-options>`__).

An external provider needs two flags:

.. code:: bash

   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

Reef starts its core record-only recipe on ``127.0.0.1:8900`` and stores state under ``.reef/`` in the launch directory. It records inference and feedback and trains no weights. It does not start or stop the provider, and readiness does not check the provider's credentials or models.

``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and ``REEF_TOKEN`` are optional environment fallbacks for this mode; explicit command-line settings win. ``--model`` is a shorthand for the provider and the model (`Selecting the model <cli.rst#selecting-the-model>`__).

Weight training without a file uses the same path: pass ``--recipe.implementation package.module:WeightRecipe``, ``--inference.model-path`` and the ``--training.options.*`` flags that the recipe needs.

Explicit file selection
-----------------------

``reef serve -c reef.yaml`` selects a configuration file. Without ``-c``, Reef uses the CLI settings alone: it does not read ``REEF_CONFIG`` or ``./reef.yaml``, and it does not search the Reef installation. A selected file that is missing or invalid is an error. Relative config and state paths resolve from the directory where you run the command; from outside a checkout, pass an absolute path to a cookbook config. Logs go to ``.reef/run/``; set ``reef.run-dir`` to move them.

Before any model download or process start, the launcher logs the selected source and every resolved setting with where its value came from:

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

Sources are ``file``, ``command line``, ``environment`` (the ``REEF_*`` variables of a start without a file, or a field's declared fallback variable), ``automatic`` (a choice Reef made because the field was omitted) and ``default``. The report lists a setting at its default only for the recipe and the HTTP bind. It masks tokens, API keys, passwords and database URLs by key name at any depth, including inside native ``options`` objects.

``reef serve ... --print-config`` prints the same report on standard output, defaults included, and exits with status 0, or 2 for an invalid config. It downloads no model and starts no process. It takes the same ``-c``, ``--model`` and override flags as a real start. A Hugging Face model path is shown as written; the snapshot is resolved only at startup.

.. code:: bash

   reef serve -c stack.yaml --reef.port 9000 --print-config

**Value rules.** YAML and CLI values use the same parser, and ``reef serve --help`` lists the public options.

- **Precedence.** CLI, then YAML, then a declared environment fallback, then the default. The last occurrence of a flag wins.
- **Strings** keep their text: ``--inference.upstream-model 00123`` stays ``00123``, and the literal string ``null`` stays text.
- **Numbers and booleans** are parsed by their declared type. A boolean takes an explicit value or a bare flag, and ``--no-...`` disables a YAML setting, for example ``--no-reef.allow-implicit-scenario-creation``. False, zero and empty values are explicit values. Service fields, resources and timeouts must be finite. Recipe floats may be non-finite unless the recipe declares ``allow_nonfinite=False``.
- **Lists and objects** take one quoted JSON or YAML value, for example ``--reef.tokens '[]'`` or ``--inference.handler-config '{"tool_call_parser": "qwen25"}'``. An explicit container, empty or not, replaces the YAML value.
- **Empty string.** An empty service value, such as an unset ``${VAR}``, counts as omitted and keeps the default.
- **Null.** An optional field (``str | None`` and similar) resolves to null. ``null`` for any other field is an error that names the field. Omit the field to use its default.
- **Duplicates.** A file that repeats a YAML key is rejected with both line numbers. Two spellings of one field (``model-path`` and ``model_path``) are a duplicate field error.
- **Errors** name the field and the expected type without its value. An invalid value fails before any model download or process start.
- **Tokens.** ``reef.token`` and ``reef.tokens`` are distinct inputs; Reef accepts the credentials of both.

**Interpolation.** A value interpolates from the environment with ``${VAR}`` and from the config itself with ``${dotted.path}``, such as ``${reef.port}``.

- ``${VAR}`` resolves to an empty string when the variable is unset. Use it for optional values, such as the API key of a provider without authentication.
- ``${VAR:?}`` requires the variable, for example ``upstream_model: ${REEF_UPSTREAM_MODEL:?}``. If it is unset, empty or only whitespace, Reef reports the missing variable names and their config fields before it downloads models or starts processes. Command-line overrides apply first, so ``--inference.upstream-model <model-id>`` can supply the value instead.

``REEF_PYTHON`` names the interpreter of the services and defaults to the one that launched ``reef serve``. A literal ``python`` in a service command keeps its normal meaning and resolves from that service's ``PATH``.

Component configuration
-----------------------

After Reef selects ``recipe.implementation``, it loads that class's declared fields without constructing the recipe.

- A recipe's fields are ``--recipe.config.<field>``, for example ``--recipe.config.training-mode``. A weight-training recipe also accepts the aliases ``--training-mode``, ``--training_mode`` and ``--reef.training_mode``. The ``--reef.data.*`` spellings remain aliases of ``--recipe.config.*``.
- ``reef serve -c stack.yaml --help`` includes the selected recipe's flags. Basic ``reef serve --help`` loads no recipe.
- The selected package must be importable in the launcher and in the services.
- Declared fields follow the value rules above. A declaration that conflicts with a public option is rejected, and an unknown field fails validation.
- A recipe can declare opaque sections in ``config_sections``; for example, ``HarnessEvolutionRecipe`` owns ``recipe.config.evolution``. A section takes object and leaf CLI overrides, and its recipe validates the payload.
- A standalone recipe file that an embedding script loads (``load_recipe_config``, ``recipe_config_from_file_layout``) uses the same layout.

A dotted recipe that trains no weights can set ``recipe.runtime`` to select a runtime factory: a bundled kind, or the ``package.module:ClassName`` of a ``RuntimeFactory`` subclass. Its declared fields use names such as ``--recipe.runtime.timeout-s``. A custom factory declares its fields with ``config_type()``; a factory without one receives the section as a mapping.

Executor fields are declared by ``ExecutorSettings`` and ``WorkerResources`` (`Stack execution backends <configuration-execution.rst#stack-execution-backends>`__):

.. code:: bash

   reef serve -c stack.yaml \
     --execution.evolution.workers 4 \
     --execution.evolution.resources.cpus-per-worker 0.5

A nested override of a named executor profile makes a local copy for that role; it does not change the shared profile.

**Startup failures.** If a service exits before readiness or exceeds its readiness deadline, Reef stops the stack and reports the service, the reason and the log directory; an exited service's message includes its exit code. A startup failure exits with status 1, and an invalid configuration with status
2. Service output goes to the logs and to the terminal. SIGINT, SIGTERM or,
on POSIX, SIGHUP during startup also stops the services already started.

Start from a cookbook stack
---------------------------

The source checkout's runnable stacks live under the ``recipes/`` cookbook and ``tutorials/``.

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

``recipes/sao/examples/imo_answerbench/serve.yaml`` is the smallest weight-training stack: two GPUs for one actor and one rollout engine. ``recipes/tttd/examples/tttd/serve.yaml`` adds LoRA training, and ``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds a PRM engine and a student model.
