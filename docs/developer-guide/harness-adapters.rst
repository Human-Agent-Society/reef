Harness adapters
================

An adapter is how a rendered harness tree becomes a running agent. The tree
never names a file path; the adapter does. Reef bundles two, one per third-party
coding-agent CLI.

+--------------+-------------------------------------------+-----------------------------------------+
| Adapter      | Config targets                            | Install pin                             |
+==============+===========================================+=========================================+
| ``pi``       | ``primary`` → ``pi-agent/settings.json``, | npm ``@earendil-works/pi-coding-agent`` |
|              | ``models`` → ``pi-agent/models.json``     | 0.84.2                                  |
+--------------+-------------------------------------------+-----------------------------------------+
| ``opencode`` | ``primary`` → ``opencode/opencode.json``  | npm ``opencode-ai`` 1.18.18             |
+--------------+-------------------------------------------+-----------------------------------------+

The descriptor
--------------

One ``descriptor.yaml`` declares how a tree becomes a running agent.

.. config::

   name | the adapter's id
   binary | the executable an episode runs
   argv | the argument list for one headless prompt; ``{prompt}`` is substituted
   files | where each node kind renders, like ``skills/{name}/SKILL.md``
   trajectory | the format and path of the session log Reef reads back
   env | variables pointing the agent's state under the episode root
   install | the npm package and version the one-command install pins
   model_binding | per provider dialect, the config that points the agent at the served model
   cleanup_whitelist | files the agent's own boot creates, tolerated instead of read as drift
   quirks | an optional module for adapter-specific render checks and boot mutations

Connect a new agent
-------------------

To connect an agent that has no adapter yet:

.. steps::

   #. The file it reads configuration from becomes a ``files.config`` target.
   #. The command line that runs one prompt headless becomes ``binary`` and ``argv``.
   #. The path and format of its session log become ``trajectory``. A new format
      subclasses ``TrajectoryReader``
      (`reef/harness/trajectory.py <../../reef/harness/trajectory.py>`__) and
      registers with ``@register_trajectory_reader``.
   #. The files its first boot creates go in ``cleanup_whitelist``, so a fresh
      episode root is treated as clean.

`reef/harness/descriptor.py <../../reef/harness/descriptor.py>`__ validates every
descriptor at load, and the two bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ are complete references. A third-party adapter
registers on the ``reef.harness_adapters`` entry-point group. ``version_check:
true`` writes an update notice into the tree and ships for ``pi`` only, so an
``opencode`` recipe that sets it refuses to boot. An evolved tree is
adapter-specific: ``config`` node contents follow each adapter's schema.

See also
--------

- `Evolve your harness <../user-guide/evolve-your-harness.rst>`__ — the loop that produces the tree.
- `harness_evolve <../user-guide/recipes/harness-evolve.rst>`__ — the recipe's configuration.
