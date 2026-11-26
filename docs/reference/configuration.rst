Configure a deployment
======================

Save deployment settings in YAML. Flags use the same paths, such as ``--inference.model-path``.
See the `settings reference <configuration-settings.rst>`__ for all keys, types and defaults.

A typical file
--------------

Save this as ``reef.yaml`` to connect a local service to a hosted model.

.. code:: yaml

   inference:
     upstream-url: https://api.openai.com
     upstream-api-key: ${REEF_UPSTREAM_API_KEY:?}
     upstream-model: gpt-4o

Set your provider key and run ``reef serve -c reef.yaml``. Use ``--print-config`` to inspect settings before startup.
For local inference, use ``inference.model-path`` with the engine and GPU-enabled PyTorch installed. Remove upstream URL and model settings.

How Reef reads values
---------------------

Command-line flags override file values, then environment fallbacks, then defaults. The last repeated flag wins.
Without a file, Reef reads ``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and ``REEF_TOKEN``.

- In YAML, ``${VAR}`` expands to an environment variable or an empty string. Use ``${VAR:?}`` to require a value.
- Use ``${dotted.path}`` for an explicit file or flag value. References to defaults remain literal.
- Pass lists and objects as quoted JSON or YAML on the command line. Use ``--no-<path>`` to disable a boolean option.
- Use the spelling in the settings reference. Some nested sections require underscores.

Reef masks secret fields in ``--print-config``. Credentials embedded in ordinary URLs can still appear.
An invalid configuration exits with status 2. A failed startup exits with status 1 and identifies the service and log directory.
