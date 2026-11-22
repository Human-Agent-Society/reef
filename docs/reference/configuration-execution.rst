Stack execution backends
========================

The ``execution`` and ``executors`` sections select the executor of each role: Slime's training workers, the serving engines, the generator service and the harness evaluation workers.

Reef places its own service processes itself. Every selector accepts a built-in backend name, a ``package.module:ClassName`` reference, a profile name under ``executors``, or an inline object. Any other key under ``execution`` fails the startup.

.. list-table::
   :header-rows: 1

   * - Selector
     - Role
     - Default
   * - ``execution.training``
     - Slime's training workers
     - ``auto``, which resolves to ``ray``
   * - ``execution.rollout``
     - the serving engines of the inference component, for every training backend
     - ``auto``
   * - ``execution.generator``
     - the generator service that a ``generator`` section adds
     - ``auto``: one local process
   * - ``execution.evolution``
     - harness evaluation workers
     - ``auto``

An inline object or a profile accepts these keys:

.. container:: config

   ``backend`` : auto
       a built-in executor, a ``package.module:ClassName`` reference, or a named profile

   ``options`` : {}
       options that the selected backend owns

   ``workers``
       a positive integer; omitted, the component's own requirement

   ``resources``
       ``cpus_per_worker`` and ``gpus_per_worker``, finite nonnegative numbers; omitted values keep the component's defaults (normally one CPU and no GPUs). A value cannot reduce a component's declared GPU need. Numeric environment interpolation is accepted

How ``auto`` resolves:

- **Harness evaluation workers.** Ordinary harness evolution calls external model endpoints and needs no local GPUs. One worker selects ``uni``, and several workers select ``mp``. Workers that need local GPUs follow the same rule after Reef checks the visible CUDA capacity; with too few GPUs, startup fails and asks for an explicit ``ray``. A local allocation needs whole GPUs, and a fractional reservation needs an explicit ``ray``. Several workers inside a Ray placement group, or declared cluster or actor options, select ``ray``. CPU-only local executors reserve no cores and enforce no CPU quotas.
- **Services.** ``auto`` selects ``uni`` unless resource or worker options, or an existing Ray placement group, call for Ray. Explicit local CUDA visibility selects ``uni`` and cannot be combined with cluster resource options. A service executor accepts ``resources``, but ``workers`` must be one. For a Ray service, ``resources`` supplies actor launch options such as ``num_gpus`` and ``num_cpus``.
- **Training and rollout.** These roles reject the generic ``workers`` and ``resources`` fields. Their topology comes from Slime's training options.

An explicit selector or Slime CLI flag overrides the role's default. A backend that fails to start never falls back to another backend. A nested override of a named profile, such as ``--execution.evolution.workers 4``, makes a local copy for that role and does not change the shared profile. `Whole-stack deployment configuration <../developer-guide/executors.rst#whole-stack-deployment-configuration>`__ describes the selection and the Ray runtime. Slime's ``--reef-executor-backend`` option also selects the executor of its training workers; see `Slime worker executors <../developer-guide/executors.rst#slime-worker-executors>`__.
