Harness evolution keys
======================

Use the `tutorial configuration <../../tutorials/evolve-your-harness/configs/deployment.yaml>`__ to configure ``recipe.config.evolution``.
Select a seed directory, method, adapter and evaluation task source. A seed change applies only to new scenarios.

Choose one task source: ``tasks``, ``task-manifest`` with ``tasks-root``, or ``imported-tasks``.
The `settings reference <configuration-settings.rst#harness-evolution>`__ lists the keys and defaults.

The coding agent
----------------

Set ``evolution.agent.adapter`` to a text-prompt adapter (default ``pi``). Reef installs its pinned binary unless you provide one.
Use underscore keys in ``agent`` and ``models.<name>``. Unknown keys here can be ignored.

For a named model, supply ``url`` without ``/v1`` and a ``model`` ID. Put credentials in ``api_key_env``.
Each ``model_metadata`` entry requires both ``context_window`` (positive token count) and ``reasoning`` (boolean).

The run sandbox
---------------

The coding agent uses the local sandbox unless you select another through ``agent.sandbox`` or ``REEF_AGENT_SANDBOX``.
The separate ``evolution.sandbox`` object configures the run when ``evolution.executor`` differs from ``local``.

Use ``env_from`` to pass specific nonempty host variables. On the local sandbox executor, any ``egress_hosts`` entry opens the entire network.
E2B, Modal and boxd allow outbound internet and reject nonempty ``egress_hosts`` or ``limits``.
A boxd machine is always isolated: it has no boxd credentials and joins no private network.
Secrets scoped to the machine still reach commands outside Reef's jail, so do not scope a secret there that a task may not read.
See `Sandboxes <../developer-guide/sandboxes.rst>`__ for setup.
