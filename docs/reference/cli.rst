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

Once the stack is running, Ctrl-C requests graceful shutdown. Pressing it
again during shutdown skips the remaining 30-second grace period and proceeds
to forced cleanup of the managed service processes.

.. config::

   -c, --config | optional config file. No file is loaded unless explicitly selected.
   --recipe NAME | start a built in recipe's profile instead of a config file. Today: ``reefine``; ``harness-evolve``, its former name, starts the same profile.
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
``--recipe harness-evolve``, the former name of the profile folded into it,
starts the same profile and says so on stderr.

Overriding config values
------------------------

Canonical CLI paths match ``schema-version: 2`` YAML. Declared fields use the
same type parser for both inputs; explicit CLI values override YAML. Opaque
component objects support leaf overrides, validated by their owning component.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct \
     --training.config.checkpoint_dir /tmp/ckpt

Use public namespaces to move a stack's state without editing its config.
Legacy bare and ``reef.*`` aliases remain accepted for compatibility:

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --storage.agent-record-dir .reef/agent-record \
     --storage.artifact-work-dir .reef/artifact-work \
     --storage.artifact-cache-dir .reef/artifact-cache

Where it writes
---------------

Each service gets a log and a PID file under ``run_dir``, which defaults to
``/tmp/reef-stack``. Reef writes its own state, including records, commit logs,
and the Git-backed release chain, to the ``reef.*_dir`` paths in the config.

Connect to the API platform
--------------------------

The connector can start Reef for you. Run it from your Reef project directory
and put ``--serve`` last, followed by the ``reef serve`` options. This example
starts the Reefine profile on port 9000 and connects it to a local API platform
on port 3000:

.. code:: bash

   uv run reef connect \
     --url http://127.0.0.1:9000 \
     --platform http://localhost:3000 \
     --name workstation \
     --no-browser \
     --serve --recipe reefine --model ollama/qwen3

``--serve`` passes the host and port from ``--url`` to ``reef serve``, so the
two cannot disagree; ``--url`` must be a loopback ``http`` address with a port,
and the options after ``--serve`` must not set ``--reef.host`` or
``--reef.port``. The connector refuses to start if another service already
answers at that address. Reef starts after you approve the connection, in the
directory where you ran the command, with the same environment. Its output goes
to ``serve.log`` in the connection's state directory. If ``REEF_TOKEN`` (or
the ``--reef-token-env`` variable) is set, Reef receives it as its
``REEF_TOKEN``. The console shows Reef as starting until it first answers,
and reports its exit code if it stops; the connector does not restart it.
Stopping the connector stops this Reef service too.

To connect a Reef service that is already running, for example one managed by
systemd or Docker, omit ``--serve``. This example connects the runtime on port
9000 to a local API platform on port 3000:

.. code:: bash

   uv run reef connect \
     --url http://127.0.0.1:9000 \
     --platform http://localhost:3000 \
     --name workstation \
     --no-browser --foreground

Set ``--url`` to your running Reef service's address and ``--platform`` to
your API platform's address. Replace both example addresses to match your setup.
``--name`` sets the label shown in the console. Before pairing, the connector
checks the address once and prints a warning if Reef does not answer; pairing
still completes, and the console shows the runtime once Reef answers.

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

The platform receives scenario names, each scenario's harness adapter,
serving release identifiers, training modes, selected numeric evaluation
results, the Reef URL the connector checks and, with ``--serve``, whether Reef
is starting, running or exited. For each release step it also receives the
step's result (such as ``selected``, ``rejected``, ``skipped`` or ``failed``),
the selection outcome, policy, evaluator and pass and fail counts, the ID of
the request the step answered with the names and kinds of what it requires,
and each change's operation, entry ID and kind. For each harness request it
receives the ID, the state and the time it was filed. It can create a scenario,
request training, change training mode, promote or roll back a release.
Local provider credentials, artifact files, entry contents, request text and
recorded prompts are not uploaded. Instructions you submit through the
dashboard are stored on the platform as commands. Inference continues to use
your runtime URL directly.

Lifecycle and local state
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same ``--url`` and ``--platform`` options with ``--status`` to
inspect the connector or ``--stop`` to stop it. Rerun the connection command
above to restart it. Stopping the connector retains authorization. It leaves
Reef serving, unless the connector started Reef with ``--serve``. **Revoke connection** in the console
disables the credential; a revoked connector exits and requires a new login.

By default, each platform/runtime URL pair has a private directory under
``~/.reef/connections/``. It stores an instance UUID, service and connector
credentials, a SQLite command record, a process lock, and a background log.
With ``--serve``, it also stores the ``reef serve`` options and directory, and
``serve.log``.
The directory is mode 0700 and credential files are mode 0600 on POSIX systems.
``--state-dir PATH`` selects an explicit directory. Preserve it to keep the
same identity; do not share or copy it between running machines.

Connected runtime cards in the console support deletion after confirmation.
The connector sends ``DELETE /reef/scenarios/{scenario}`` to Reef, which
removes that scenario and archives its own saved state. Other scenarios and
the connection remain available. Update and restart older connectors before
using this action; they reject the new ``delete_scenario`` command.

The console lists the harness requests of a connected runtime, including the
ones filed on the machine, through the ``requests`` command. The connector
reads the scenario's training instructions from
``GET /reef/scenarios/{scenario}/records?request_type=train`` and each state
from ``GET /reef/harness/requests/{record_id}/progress``, and sends the newest
100. Older connectors reject this command and send no adapter, so the console
cannot list their requests or name their adapter until you update Reef and
restart the connector.

The connector reconnects after network failures. It does not install an OS
startup service; use ``--foreground`` with your process supervisor for restart
after a machine reboot. It polls every three seconds while idle or running a
slow operation. Completed commands are reported immediately, so queued commands
do not each wait for another heartbeat. For thirty seconds after activity, it
checks for follow-up commands every second. It reports scenario summaries about
every fifteen seconds, or every five seconds while Reef does not answer. When Reef does not answer, the summary names the address it
checked and the reason: nothing listening, a rejected service token, a
timeout, or a service that is not Reef. The console marks it offline after
45 seconds without a heartbeat and retains its last scenario summary.

Commands are delivered once. An interrupted operation is marked **unknown**,
and is not replayed automatically. Check Reef before submitting it again.
Completed results are saved locally until the platform acknowledges them;
cloud command history is retained for 30 days. Closing a browser or revoking a
connection cannot undo an operation already dispatched to Reef.
