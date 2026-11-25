"""Process execution and isolation for harnesses and Harbor trials.

Harnesses run programs through the executor selected in ``config``. Harbor trials use
``task_environment`` to select an environment. Both use the cloud adapters in ``providers``,
which share session, file-transfer and tunnel support in ``remote``.
"""
