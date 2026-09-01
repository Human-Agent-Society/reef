Harness adapters
================

An adapter maps a harness tree into the files expected by a third-party
coding-agent CLI and binds that harness to the served model. The harness and
model together form the running agent. The tree never names a file path; the
adapter does. Reef bundles two, one per third-party coding-agent CLI.

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

One ``descriptor.yaml`` declares how a tree configures and starts a running
agent.

.. config::

   name | the adapter's id
   binary | the executable an episode runs
   argv | the argument list for one headless prompt; ``{prompt}`` is substituted
   files | where each node kind renders, like ``skills/{name}/SKILL.md``
   trajectory | the format and path of the session log Reef reads back
   env | variables pointing the agent's state under the episode root; ``{root}`` is substituted
   install | the one-command install pin: ``kind`` (``npm`` only), ``package``, ``version``, and ``binary_path`` under the install prefix
   model_binding | per API dialect (``openai``, ``anthropic``), the config nodes Reef appends at evaluation time; ``{base_url}``, ``{api_key}``, and ``{model}`` substitute into string values
   cleanup_whitelist | files the agent itself writes at boot or during the run, tolerated instead of read as drift
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
      episode root is treated as clean. ``dir/**`` tolerates a whole subtree
      (session storage, ``node_modules``); any other entry is a glob against
      the root-relative path, so anchor a single file with its full path, like
      ``pi-agent/auth.json``. A bare directory name matches nothing under it.

`reef/harness/descriptor.py <../../reef/harness/descriptor.py>`__ validates every
descriptor at load, and the two bundled adapters under `reef/harness/adapters/
<../../reef/harness/adapters>`__ are complete references. A third-party adapter
registers on the ``reef.harness_adapters`` entry-point group.
``evolution.version_check: true`` in the recipe config writes an update
prompt into the tree and ships for ``pi`` only. The
prompt offers to run the update or skip in interactive mode and prints the
instructions in headless mode. An ``opencode`` recipe that sets it refuses to
boot. An evolved tree is adapter-specific: ``config`` node contents follow each
adapter's schema.
