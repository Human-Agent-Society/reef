Reef CLI: serve
===============

``reef serve`` reads a deployment config and starts every process that the
config declares, in dependency order.

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c recipes/basic/external-provider.yaml

This config binds every address, so Reef starts only with a token.

``reef serve`` runs in the foreground until Ctrl-C. Each service's ``ready``
probe must pass before the next service starts. A watchdog then stops the
stack if any process exits unexpectedly. Ctrl-C requests a graceful shutdown;
a second Ctrl-C skips the remaining 30-second grace period and forces cleanup
of the managed processes.

The wheel also installs ``reef-native``, the loop runner that the ``native``
adapter starts for each rollout. Nothing calls it by hand.

Options
-------

.. container:: config

   ``-c, --config``
       optional config file, relative to the working directory. Reef loads no file unless you select one: it does not discover ``$REEF_CONFIG`` or ``reef.yaml``.

   ``--model [PROVIDER/]MODEL``
       the upstream model. See `Selecting the model`_.

   ``--service NAME``
       run one service of the deployment in this process: ``control``, ``inference``, ``samples``, ``training``, ``evaluation``, ``data`` or ``harness``. See `Running one service`_. ``--component`` is the older name of this option and gives the same result.

   ``--print-config``
       print every resolved setting with its source (file, command line, environment, automatic, default) and exit without downloading models or starting services. Credentials are masked.

   ``--<section>.<key> VALUE``
       override one config value. See `Overriding config values`_.

   ``--help``
       the command list

   ``-V, --version``
       the installed reef version. Takes no command: ``reef --version``.

Selecting the model
-------------------

Without ``-c``, Reef uses the inference settings from the command line. Reef
bundles no deployment profiles; pass a recipe's deployment config with ``-c``.

``--model`` is a shorthand:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Spelling
     - Effect
   * - ``ollama/MODEL``
     - Fills ``http://127.0.0.1:11434`` and a placeholder key.
   * - ``openai/MODEL``
     - Fills ``https://api.openai.com`` and reads ``REEF_UPSTREAM_API_KEY``.
   * - any other value
     - The model ID as is. It needs an upstream URL.

An explicit ``--inference.upstream-url`` or ``--inference.upstream-api-key``
wins over the prefix.

.. code:: bash

   reef serve --model ollama/gemma4:26b

Overriding config values
------------------------

Command-line paths match the sections of a configuration file, and a
command-line value overrides the file. Declared fields use the same type parser
for both inputs. Opaque component objects accept leaf overrides, which the
owning component validates.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct \
     --training.config.checkpoint_dir /tmp/ckpt

Use the full public namespace, for example to move a stack's state without
editing its config. Bare and ``reef.*`` aliases remain accepted.

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --storage.agent-record-dir .reef/agent-record \
     --storage.artifact-work-dir .reef/artifact-work \
     --storage.artifact-cache-dir .reef/artifact-cache

Running one service
-------------------

Without ``--service``, one process runs all seven services. With it, the
process runs one service and starts no other child. All processes of a
deployment take the same config, so they find the same shared store:

.. code:: bash

   reef serve -c deploy.yaml --service control --reef.port 8901
   reef serve -c deploy.yaml --service inference --reef.port 8902
   reef serve -c deploy.yaml --service samples --reef.port 8903

`Run the services in separate processes
<../user-guide/split-services.rst>`__ describes
what each service does and which routes it answers.

Where it writes
---------------

Each service gets a log and a PID file under ``reef.run-dir``, which defaults
to ``.reef/run``. Reef writes its own state, including records, commit
histories and the Git-backed snapshot chain, to the ``storage.*`` paths in the
config.
