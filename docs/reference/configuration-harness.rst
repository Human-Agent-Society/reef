Harness evolution keys
======================

Set harness options under ``recipe.config.evolution``. Start with the
`tutorial configuration <../../tutorials/evolve-your-harness/configs/deployment.yaml>`__.
The `settings reference <configuration-settings.rst#harness-evolution>`__ lists top-level keys and defaults.

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

Under ``evolution.agent``, choose a text-prompt ``adapter`` (default ``pi``) and optional ``binary``.
The first job validates the adapter. ``timeout_s`` limits one run (default 1800, positive).

``sandbox`` defaults to ``REEF_AGENT_SANDBOX``, then the host's local sandbox.
If local isolation is unavailable, the agent cannot start.
``bwrap`` on Linux and ``seatbelt`` on macOS allow internet and gateway access. Seatbelt also permits access to host LAN addresses.
For a hosted sandbox select ``e2b`` or ``modal``. Use ``none`` to disable isolation.

The run sandbox
---------------

Reef reads ``evolution.sandbox`` only when ``evolution.executor`` is not ``local``.
Use ``env_from`` to pass named host variables to a local or cloud sandbox. Each must have a nonempty value.

For the local sandbox executor:

- ``egress_hosts`` defaults to an empty list. Any entry opens the entire network.
- ``limits`` accepts nonnegative ``cpu_seconds``, ``memory_bytes``, ``processes`` and ``file_bytes``. Zero means unlimited. macOS rejects memory limits.

E2B and Modal allow outbound internet and reject nonempty ``egress_hosts`` and ``limits``.
Set ``forward_ports`` to the worker's loopback ports (1–65535) that the sandbox needs.

- E2B reads ``e2b_api_key_env`` (default ``E2B_API_KEY``) or an explicit ``e2b_api_key``, which takes priority. Select the image with ``e2b_template``.
- Modal accepts ``modal_app`` (default ``reef-sandboxes``), ``modal_image``, ``modal_cpu`` and ``modal_memory_mb``. It reads ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``.

Named models
------------

Give each ``evolution.models.<name>`` a ``url`` without ``/v1`` and a ``model`` ID.
``api`` accepts ``openai`` (default), ``responses`` or ``anthropic``. ``timeout_s`` defaults to 600.
Use ``api_key_env`` for credentials. A literal ``api_key`` takes priority.

Under ``evolution.model_metadata.<name>``, set ``context_window`` to a positive token count and ``reasoning`` to whether the endpoint accepts reasoning effort.
