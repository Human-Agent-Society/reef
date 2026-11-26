Harness evolution keys
======================

Configure the coding agent, sandbox and named models under ``recipe.config.evolution``.
Start with `Evolve your harness <../user-guide/evolve-your-harness.rst>`__ for the workflow.
See `Settings reference <configuration-settings.rst#harness-evolution>`__ for all keys, types and defaults.

This example uses tutorial classes. Run Reef from the repository root:

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         method: tutorials.evolve-your-harness.harness.plugins:RulesMethod
         adapter: pi
         seed: tutorials/evolve-your-harness/seeds/pi
         state-dir: work/harness
         max-failure-streak: 5
         models:
           critic:
             url: http://127.0.0.1:8001
             model: qwen3-32b
             api_key_env: CRITIC_API_KEY
             timeout_s: 120
         evaluation:
           tasks:
             - "[fib] With fib(1) = 1 and fib(2) = 1, compute fib(90) exactly. Reply with the value as a plain integer alone on the last line."
           scorer: tutorials.evolve-your-harness.harness.plugins:AnswerScorer
           policy: compare
           repeats: 2

Rules
-----

- Use underscores for the object members listed below. In ``agent``, ``models.<name>`` and the local ``sandbox`` object, Reef silently ignores unknown or hyphenated members.
- For ``executor: e2b`` or ``modal``, unknown ``sandbox`` members fail startup.
- Each ``model_metadata`` entry must contain exactly ``context_window`` and ``reasoning``.
- ``seed`` must name an existing directory and cannot be empty. Changing it does not change an existing scenario's root or head.
- If ``binary`` or ``agent.binary`` is unset, Reef installs the adapter's pinned binary under ``$REEF_HARNESS_PREFIX/<adapter>``. The prefix defaults to ``~/.local/share/reef-harness``.
- Set one evaluation task source: ``tasks``, ``task-manifest`` with ``tasks-root``, or ``imported-tasks``. Multiple sources fail startup.

The coding agent
----------------

.. container:: config

   ``evolution.agent.adapter`` : pi
       Harness adapter for the coding agent. It must accept text prompts. The first job validates this choice.

   ``evolution.agent.binary``
       The path of the agent's binary.

   ``evolution.agent.sandbox`` : ``$REEF_AGENT_SANDBOX``
       ``bwrap`` (Linux) or ``seatbelt`` (macOS) isolates the agent, with internet access and access to its gateway's host port. ``e2b`` and ``modal`` read provider options here. ``none`` disables isolation.

   ``evolution.agent.timeout_s`` : 1800
       Maximum run duration in seconds. Must be positive.

If both sandbox settings are unset, Reef uses the host's local sandbox.
The agent cannot start if that sandbox is unavailable. Under ``seatbelt``, the host's LAN addresses remain reachable.

The run sandbox
---------------

Reef reads ``evolution.sandbox`` only when ``evolution.executor`` is not ``local``.

.. container:: config

   ``evolution.sandbox.egress_hosts`` : []
       For ``sandbox``: required endpoints. Any nonempty list opens the whole network; Reef does not filter individual hosts.

   ``evolution.sandbox.limits`` : {}
       For ``sandbox``: resource limits as nonnegative integers. Keys: ``cpu_seconds``, ``memory_bytes``, ``processes`` and ``file_bytes``. Zero means unlimited. macOS rejects ``memory_bytes``.

   ``evolution.sandbox.env_from`` : []
       Host environment variables to pass to a run. Unset or empty variables fail startup.

   ``evolution.sandbox.forward_ports`` : []
       For ``e2b`` and ``modal``: worker loopback TCP ports to forward into the sandbox. Range: 1–65535.

   ``evolution.sandbox.e2b_api_key_env`` : E2B_API_KEY
       The variable that holds the E2B key. A literal ``e2b_api_key`` takes priority.

   ``evolution.sandbox.e2b_template``
       The E2B template of the sandbox image.

   ``evolution.sandbox.modal_app`` : reef-sandboxes
       The Modal app of the sandboxes. Modal reads its credentials from ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, or ``~/.modal.toml``.

   ``evolution.sandbox.modal_image``
       The registry image of the Modal sandbox.

   ``evolution.sandbox.modal_cpu``
       The CPU cores of the Modal sandbox. Unset takes Modal's default.

   ``evolution.sandbox.modal_memory_mb``
       The memory of the Modal sandbox, in MB. Unset takes Modal's default.

``e2b`` and ``modal`` reject nonempty ``egress_hosts`` and ``limits``.
Cloud sandboxes use their image's resources and have outbound internet access.

Named models
------------

.. container:: config

   ``evolution.models.<name>.url``
       The base URL of the model, without ``/v1``.

   ``evolution.models.<name>.model``
       The model name that Reef requests.

   ``evolution.models.<name>.api`` : openai
       The API: ``openai``, ``responses`` or ``anthropic``.

   ``evolution.models.<name>.timeout_s`` : 600
       Maximum call duration in seconds.

   ``evolution.models.<name>.api_key``
       The literal API key. It takes priority over ``api_key_env``.

   ``evolution.models.<name>.api_key_env``
       The environment variable that holds the API key.

   ``evolution.model_metadata.<name>.context_window``
       Required context window in tokens. Must be a positive integer.

   ``evolution.model_metadata.<name>.reasoning``
       True when the endpoint accepts a reasoning effort. Required.
