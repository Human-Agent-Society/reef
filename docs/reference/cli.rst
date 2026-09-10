Reef CLI: serve and connect
==========================

``reef serve`` reads a deployment config and starts every
process the config declares, in dependency order. ``reef connect`` optionally
links an existing runtime to your API platform account. (The wheel also installs
``reef-native`` and ``reef-terminus``, the loop runners those two harness
adapters launch per episode; nothing calls them by hand.)

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml

``reef serve`` runs in the foreground and holds the terminal until Ctrl-C. Each
service's ``ready`` probe must pass before the next one starts; when they are
all up, Reef blocks, and a watchdog tears the stack down if any process exits
unexpectedly.

.. config::

   -c, --config | the config file. Defaults to ``reef.yaml``, or ``$REEF_CONFIG``.
   --help | the command list
   -V, --version | the installed reef version. Takes no command: ``reef --version``.

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
Use it to move a stack's state without editing its config:

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --agent_record_dir .reef/agent-record \
     --artifact_work_dir .reef/artifact-work \
     --artifact_cache_dir .reef/artifact-cache

Where it writes
---------------

Each service gets a log and a PID file under ``run_dir``, which defaults to
``/tmp/reef-stack``. Reef writes its own state, including records, commit logs,
and the Git-backed release chain, to the ``reef.*_dir`` paths in the config.

Connect to the API platform
--------------------------

With Reef already serving, open another terminal in your Reef project directory.
This example connects the runtime on port 9000 to a local API platform on port 3000:

.. code:: bash

   uv run reef connect \
     --url http://127.0.0.1:9000 \
     --platform http://localhost:3000 \
     --name workstation \
     --no-browser --foreground

Set ``--url`` to your running Reef service's address and ``--platform`` to
your API platform's address. Replace both example addresses to match your setup.
``--name`` sets the label shown in the console.

Open the printed sign-in link, sign in, and paste the device code from your
terminal into the page. The code is required and expires after ten minutes.
The link does not contain the code, and the page never fills it in for you.
With ``--no-browser``, open the link yourself; ``--foreground`` keeps the
connector in this terminal, which must remain open. Omit ``--foreground``
to run it in the background after approval. Open **Local Reef** in
the API platform to view the runtime from another device. No inbound port,
public endpoint, browser access to localhost, or ``console_origins`` setting
is required for this connection.

If omitted, ``--url`` defaults to ``http://127.0.0.1:8900`` and
``--platform`` defaults to ``https://api.reefinfra.ai``. Each invocation
uses its own arguments; it does not inherit addresses from a previous command.

URLs must use HTTPS except for loopback HTTP. If the existing service requires
a token, set ``REEF_TOKEN`` in the connector's environment; use
``--reef-token-env VARIABLE`` to select a different environment variable.
Do not put tokens in URLs or command-line arguments.

The platform receives scenario names, serving release identifiers, training
modes and selected numeric evaluation results. It can create a scenario,
request training, change training mode, promote or roll back a release.
Local provider credentials, artifact files, and recorded prompts are not
uploaded. Instructions you submit through the dashboard are stored on the
platform as commands. Inference continues to use your runtime URL directly.

Lifecycle and local state
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same ``--url`` and ``--platform`` options with ``--status`` to
inspect the connector or ``--stop`` to stop it. Rerun the connection command
above to restart it. Stopping the connector leaves Reef
serving and retains authorization. **Revoke connection** in the console
disables the credential; a revoked connector exits and requires a new login.

By default, each platform/runtime URL pair has a private directory under
``~/.reef/connections/``. It stores an instance UUID, service and connector
credentials, a SQLite command record, a process lock, and a background log.
The directory is mode 0700 and credential files are mode 0600 on POSIX systems.
``--state-dir PATH`` selects an explicit directory. Preserve it to keep the
same identity; do not share or copy it between running machines.

The connector reconnects after network failures. It does not install an OS
startup service; use ``--foreground`` with your process supervisor for restart
after a machine reboot. It reports heartbeats every three seconds and scenario
summaries about every fifteen seconds. The console marks it offline after
45 seconds without a heartbeat and retains its last scenario summary.

Commands are delivered once. An interrupted operation is marked **unknown**,
and is not replayed automatically. Check Reef before submitting it again.
Completed results are saved locally until the platform acknowledges them;
cloud command history is retained for 30 days. Closing a browser or revoking a
connection cannot undo an operation already dispatched to Reef.
