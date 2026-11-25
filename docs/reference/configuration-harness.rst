Harness evolution keys
======================

The members of the object keys of ``recipe.config.evolution``, and the rules of the section that its keys do not show. `Settings reference <configuration-settings.rst#harness-evolution>`__ lists every key of the section with its type and default. `Evolve your harness <../user-guide/evolve-your-harness.rst>`__ describes what the keys change.

A complete ``recipe`` section, with the classes of the tutorial; start Reef from the repository root to use it:

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

- The members on this page accept only the underscore spelling. Reef ignores a hyphen spelling and an unknown name without an error in ``evolution.agent``, ``evolution.models.<name>`` and ``evolution.sandbox`` with ``executor: sandbox``.
- Startup refuses an unknown name in ``evolution.sandbox`` with ``executor: e2b`` or ``modal``. Startup refuses a ``model_metadata`` entry that does not have exactly ``context_window`` and ``reasoning``.
- Startup refuses an ``evolution.seed`` that is an empty string or a directory that does not exist. A scenario that exists keeps its own root and head when the seed changes.
- When ``evolution.binary`` or ``evolution.agent.binary`` is unset, Reef installs the adapter's pinned binary under ``$REEF_HARNESS_PREFIX/<adapter>``. The default of ``REEF_HARNESS_PREFIX`` is ``~/.local/share/reef-harness``.
- ``evolution.evaluation`` takes one task source: ``tasks``, ``task-manifest`` with ``tasks-root``, or ``imported-tasks``. Startup refuses two sources.

The coding agent
----------------

.. container:: config

   ``evolution.agent.adapter`` : pi
       The agent's harness: an adapter whose prompt is text. The first job checks it, not startup.

   ``evolution.agent.binary``
       The path of the agent's binary.

   ``evolution.agent.sandbox`` : ``$REEF_AGENT_SANDBOX``
       ``bwrap`` (Linux) or ``seatbelt`` (macOS) jails the agent with the internet and only its gateway's host port. ``e2b`` or ``modal`` takes that provider's options from this section; ``none`` runs unisolated.

   ``evolution.agent.timeout_s`` : 1800
       The seconds one agent run may take. It must be positive.

With ``sandbox`` and ``REEF_AGENT_SANDBOX`` unset, the agent runs in the host's local sandbox. If the host cannot run that sandbox, no agent runs. Under ``seatbelt``, the host's LAN addresses stay reachable.

The run sandbox
---------------

Reef reads ``evolution.sandbox`` only when ``evolution.executor`` is not ``local``.

.. container:: config

   ``evolution.sandbox.egress_hosts`` : []
       For ``sandbox``: the endpoints a run must reach. A list that is not empty opens the whole network; Reef does not filter by host.

   ``evolution.sandbox.limits`` : {}
       For ``sandbox``: ``cpu_seconds``, ``memory_bytes``, ``processes`` and ``file_bytes``, nonnegative integers; 0 sets no limit. macOS refuses ``memory_bytes``.

   ``evolution.sandbox.env_from`` : []
       The host environment variables a run gets. Startup refuses a name that is unset or empty.

   ``evolution.sandbox.forward_ports`` : []
       For ``e2b`` and ``modal``: the worker-loopback TCP ports, 1 to 65535, that the sandbox reaches.

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

``e2b`` and ``modal`` refuse ``egress_hosts`` and ``limits`` that are not empty. A cloud sandbox has its image's resources and the outbound internet.

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
       The seconds one call may take.

   ``evolution.models.<name>.api_key``
       The literal API key. It takes priority over ``api_key_env``.

   ``evolution.models.<name>.api_key_env``
       The environment variable that holds the API key.

   ``evolution.model_metadata.<name>.context_window``
       The context window of the model, in tokens: a positive integer. Required.

   ``evolution.model_metadata.<name>.reasoning``
       True when the endpoint accepts a reasoning effort. Required.
