Reef CLI: serve
===============

``reef serve`` reads a deployment config and starts every
process the config declares, in dependency order. (The wheel also installs
``reef-native`` and ``reef-terminus``, the loop runners those two harness
adapters launch per rollout; nothing calls them by hand.)

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c recipes/basic/external-provider.yaml

The config binds every address, so Reef starts only with a token.

``reef serve`` runs in the foreground and holds the terminal until Ctrl-C. Each
service's ``ready`` probe must pass before the next one starts; when they are
all up, Reef blocks, and a watchdog tears the stack down if any process exits
unexpectedly.

Once the stack is running, Ctrl-C requests graceful shutdown. Pressing it
again during shutdown skips the remaining 30-second grace period and proceeds
to forced cleanup of the managed service processes.

.. config::

   -c, --config | optional config file. No file is loaded unless explicitly selected.
   --model [PROVIDER/]MODEL | the upstream model. An ``ollama/`` or ``openai/`` prefix fills the endpoint and the key; any other spelling is the model ID as is.
   --print-config | print every resolved setting with its source (file, command line, environment, automatic, default) and exit without downloading models or starting services. Credentials are masked.
   --help | the command list
   -V, --version | the installed reef version. Takes no command: ``reef --version``.

Selecting the model
-------------------

Select a file with ``-c``. Without it, Reef uses CLI inference settings and
does not discover ``$REEF_CONFIG`` or ``reef.yaml``. Configuration values use
their full public namespaces as shown above.

The optional ``--model`` shorthand remains supported: ``ollama/`` fills
``http://127.0.0.1:11434`` and a placeholder key; ``openai/`` fills
``https://api.openai.com`` and reads ``REEF_UPSTREAM_API_KEY``. Other prefixes
and bare values are model IDs and need an upstream URL.
An explicit ``--inference.upstream-url`` or ``--inference.upstream-api-key`` override wins over
the prefix.

.. code:: bash

   reef serve --model ollama/gemma4:26b

Reef bundles no deployment profiles. ``reef serve --recipe <name>`` exits
with ``--recipe <name>: built in profiles were removed; pass -c <config file>``.
Pass a recipe's deployment config with ``-c`` instead.

Overriding config values
------------------------

Canonical CLI paths match the sections of a configuration file. Declared fields use the
same type parser for both inputs; explicit CLI values override YAML. Opaque
component objects support leaf overrides, validated by their owning component.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct \
     --training.config.checkpoint_dir /tmp/ckpt

Use public namespaces to move a stack's state without editing its config.
Bare and ``reef.*`` aliases remain accepted:

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --storage.agent-record-dir .reef/agent-record \
     --storage.artifact-work-dir .reef/artifact-work \
     --storage.artifact-cache-dir .reef/artifact-cache

Where it writes
---------------

Each service gets a log and a PID file under ``reef.run-dir``, which defaults
to ``.reef/run``. Reef writes its own state, including records, commit histories,
and the Git-backed release chain, to the ``storage.*`` paths in the config.
