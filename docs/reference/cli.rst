Reef CLI: serve
===============

``reef serve`` reads a deployment config and starts every
process the config declares, in dependency order. (The wheel also installs
``reef-native`` and ``reef-terminus``, the loop runners those two harness
adapters launch per episode; nothing calls them by hand.)

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
   --recipe NAME | start a built in recipe's profile instead of a config file. Today: ``reefine``.
   --model [PROVIDER/]MODEL | the upstream model. An ``ollama/`` or ``openai/`` prefix fills the endpoint and the key; any other spelling is the model ID as is.
   --print-config | print every resolved setting with its source (file, command line, environment, automatic, default) and exit without downloading models or starting services. Credentials are masked.
   --help | the command list
   -V, --version | the installed reef version. Takes no command: ``reef --version``.

Starting a recipe's profile
---------------------------

A built in recipe can carry a profile: one deployment config that is the same
for every deployment of that recipe except the model. ``--recipe`` starts it
without a file of your own:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b

Select a file with ``-c`` or a built-in profile with ``--recipe``. With neither,
Reef uses CLI inference settings and does not discover ``$REEF_CONFIG`` or
``reef.yaml``. The profile selector is a launcher option; configuration values
use their full public namespaces as shown above.

The optional ``--model`` shorthand remains supported: ``ollama/`` fills
``http://127.0.0.1:11434`` and a placeholder key; ``openai/`` fills
``https://api.openai.com`` and reads ``REEF_UPSTREAM_API_KEY``. Other prefixes
and bare values are model IDs and need an upstream URL.
An explicit ``--inference.upstream-url`` or ``--inference.upstream-api-key`` override wins over
the prefix. The ``reefine`` profile ships its proposer and evaluator in the
wheel, so it runs from an installed package; it listens on
``127.0.0.1:8901`` without authentication unless ``REEF_TOKEN`` is set, and keeps its state under
``.reef/reefine/``. To change anything else, copy
``reef/service/profiles/reefine.yaml`` and pass the copy with ``-c``.

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
to ``.reef/run``. Reef writes its own state, including records, commit logs,
and the Git-backed release chain, to the ``storage.*`` paths in the config.
