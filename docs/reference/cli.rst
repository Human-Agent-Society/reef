CLI
===

Reef has one command. It reads a deployment config and starts every process the
config declares, in dependency order.

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml

``reef serve`` runs in the foreground and holds the terminal until Ctrl-C. Each
service's ``ready`` probe must pass before the next one starts; when they are
all up, Reef blocks, and a watchdog tears the stack down if any process exits
unexpectedly.

.. config::

   -c, --config | the config file. Defaults to ``reef.yaml``, or ``$REEF_CONFIG``.
   --help | the command list

Overriding config values
------------------------

Any ``--key value`` pair the parser does not recognize is applied as a config
override, so a stack can be retargeted without editing its file. Values are
YAML-coerced, so ints and bools arrive as ints and bools.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --model_path ~/models/Qwen2.5-1.5B-Instruct \
     --training.checkpoint_dir /tmp/ckpt

A bare key targets the ``reef`` section. A dotted key targets any other section.
The quickstart uses this to keep state out of ``/var/lib/reef``:

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --agent_record_dir .reef/agent-record \
     --artifact_work_dir .reef/artifact-work \
     --artifact_cache_dir .reef/artifact-cache

Where it writes
---------------

Each service gets a log and a PID file under ``run_dir``, which defaults to
``/tmp/reef-stack``. Reef writes its own state, including records, commit logs,
and the Git-backed version chain, to the ``reef.*_dir`` paths in the config.
