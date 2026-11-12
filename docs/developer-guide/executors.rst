Training components, serving and executors
==========================================

Reef's runtime has three boundaries. Each one has its own contract:

- A **training component** (``reef.train.component.TrainingComponent``) runs
  whole training jobs with its framework's own training loop and saves
  checkpoints. Reef submits jobs and reads their status. It does not drive
  optimizer steps.
- An **inference runtime** (``reef.inference.interfaces.InferenceRuntime``)
  admits requests and serves the weights that Reef published. It never
  trains.
- An **executor** (``reef.execution.executor.Executor``) launches workers and
  carries ordered control RPC, health checks and shutdown. Configuration
  selects a concrete executor; callers use the same methods.

The runtime modules are ``jobs`` (training components), ``interfaces``
(inference contracts), ``serving`` and ``residency`` (saved-weight serving
and adapter residency), ``deployment`` (runtime factories and inference
services), ``recovery`` (durable JSON files and engine supervision) and
``executor/``. The namespace package has no ``__init__.py``; import classes
from their owning modules.

.. code:: mermaid

   flowchart TD
       T[Weight trainer / JobLoop] -- submit, status, cancel, import_start --> C[TrainingComponent]
       C --> W[Framework workers: Slime, Tinker]
       C -- saved checkpoint --> T
       T -- evaluate, select --> E[Candidate evaluation]
       T -- publish release --> R[Artifact repository]
       R -- activate, restore --> I[InferenceRuntime]
       I --> S[Serving engines or provider]

Training components
-------------------

A ``TrainingJob`` is one bounded job. It has a job ID, the backend name, the
start, the job data directory that it trains (``data_dir``, and its URI
``data_uri``; see `The job data directory
<processors.rst#the-job-data-directory>`__), the backend's parameters and the
scenario whose trainer submits it. The start is the
directory of the weights that the job starts from, or ``None`` for the base
model. Reef reads none of the parameters. The component contract is:

``submit(job)``
   Record the job durably and return; the job runs in the background.
   Submitting a job ID again does nothing.
``status(job_id)``
   Report ``running``, ``finished`` or ``failed``, the checkpoints saved so
   far and the backend's metrics. Raise ``KeyError`` for an unknown job. A
   restarted component resumes each unfinished job or reports it failed.
``cancel(job_id)``
   Stop the job; it then reports ``failed``.
``changed_parts(start, checkpoint)``
   The parts of the scenario component that the checkpoint changed from
   ``start``, which the scenario's activation rule reads. The default is
   none; the harness component diffs the trees.
``import_start(release)``
   Copy a published snapshot into the component's own storage and return it
   as a start. Reef calls this for a ``train-from``. Raise ``ValueError``
   when the component cannot start from those weights.
``start_jobs()``
   Start running jobs in this process and resume the jobs that have not
   ended. Only the training service's job worker calls it, also in a
   one-process deployment. Control and the other services use the component
   to describe jobs and import starts, and never run a job. A component that
   keeps its job state on disk, such as Tinker's ``state_dir``, locks it
   here, so one process at a time runs its jobs.
``name``
   The backend name that jobs carry.
``shares_scenarios``
   Whether jobs of several scenarios can run on the component. The default
   is false.

Each ``SavedCheckpoint`` has a ``path`` (what a later job starts from) and a
``artifact_path`` (the weights that serving loads, without training state).

A weight trainer has one training component, and it drives the component
through ``reef.train.jobs.JobLoop``:

#. When no job runs, the trainer reserves ready job data. The pipeline's
   sample-to-job stage gives the job's parameters, by the kind of its samples. The job's backend is the
   component's name. The job ID is a hash of the scenario, the backend, the
   start, the records of the job's data and the parameters. A resubmission after a
   crash therefore names the same job. Reef consumes those records when
   it submits the job.
#. The scenario's training worker reads the job's status every 2 seconds.
   Each new checkpoint is a candidate. The recipe's candidate evaluation
   decides whether Reef publishes it; the default selects every checkpoint.
#. A job continues the trainer's training start unless its request names
   another start with ``train_from``. A job that continues the training start
   begins there and moves it to each checkpoint that it saves, whether or not
   evaluation selected the checkpoint. The trainer runs one such job at a
   time. Evaluation decides only what is served.
#. A job with its own start begins from that artifact and never moves the
   training start. The trainer submits it while other jobs of its scenario
   run when the component has ``concurrent_jobs``: a job then depends only on
   its own start, data and parameters. The Tinker component has it. Without
   it, the trainer runs one job at a time.
#. A failed job ends like a finished one. Reef consumed the records of its data
   when it submitted the job, so they do not train again.
#. For a ``train-from``, Reef first imports the snapshot with
   ``import_start`` and cancels the running job that continues the start
   before. It then writes the ``train_from`` commit. The next job that names
   no start begins from the imported snapshot. A component that cannot load a
   snapshot refuses it, and Reef answers the request with HTTP 400. A
   request's ``train_from`` takes its start with ``import_start`` too. Activating an artifact changes only what serves; the
   component never sees it.

The trainer stores its state, a ``JobLoopState``, in the scenario's commit
record. The state holds ``start``, ``jobs``, ``finished_jobs`` and
``last_job``. A component never learns the scenario. Per-scenario state, such
as one adapter per scenario, belongs to serving.

Two scenarios can share a training component only if the component allows it
(``shares_scenarios``). The Tinker component allows it, because each job
carries its own start. The Slime component does not: its workers hold one set
of weights, so a Slime deployment trains one scenario. The inference runtime
must also serve each scenario its own weights.

``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in
components. It keeps each job in a directory under the component's job root:
``job.json``, ``checkpoints.json`` and, once the job ended, ``result.json``.
It runs jobs one at a time in a thread; a subclass with ``concurrent_jobs``
runs every submitted job in its own thread. A subclass implements ``check_job``
and ``run``. It must call ``start_jobs()`` once it can train. Until then no
job runs, including the jobs that a restart left. After a restart,
``start_jobs()`` resumes the unfinished jobs in submission order.

`Tinker integration <../reference/python-api.rst#tinker-integration>`__ in
the Python API describes the Tinker component.

Decide what a scenario trains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``auto`` mode Reef starts a job by its own rule: a trainer trains when
``job_samples`` untrained samples wait. A deployment can replace the rule
with a training manager, a ``TrainingManager`` plugin that
``training.manager`` names. An agent that plans experiments is one use.

Control asks the manager every 2 seconds, once for each trainer of each
loaded scenario in ``auto`` mode. ``actions(view)`` reads a ``ManagerView``
and returns the actions that Reef runs, in order:

- ``Train(samples, reason, parameters, train_from)`` files a training
  request, as ``POST /reef/train`` does. With ``train_from`` the job starts
  from that artifact and runs beside the other jobs. Without it the job
  continues the training start.
- ``TrainFrom(artifact_id)`` sets the training start.
- ``Cancel(job_id)`` cancels a job.

The view holds the training start, whether the selected artifact serves, the
untrained samples that an automatic job would train next, the job size, the
number of waiting requests, the jobs that have not ended, and the artifact
listing with each artifact's metrics.

Reef limits a manager in three ways. It checks each action as it checks a
person's request, because the view can be old when the action runs. It drops
a ``Train`` action while ``job_limit`` jobs of the manager's requests wait or
run; the default limit is 1. And a manager has no action that changes what a
scenario serves: the activation rule decides that. A request of a manager has
the session ``training-manager``, and the commit of its job keeps the
action's ``reason`` as the request's text.

The manager runs on one thread of control, outside every scenario lock, so a
slow decision delays only the other managed trainers. An error in
``actions``, or a refused action, is logged, and the next round reads the
trainer again.

Serving saved weights
---------------------

Serving loads only the weights that a job saved and Reef published. It loads
them when a scenario starts, when a committed snapshot is activated, and when
a person activates a snapshot. No trainer holds a serving engine.

``reef.inference.serving`` has two runtimes over an ``InferenceBackend``:

- ``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
  base model. Requests name the adapter per scenario. At most
  ``adapter_capacity`` adapters are resident.
  ``reef.inference.residency.AdapterResidencyManager`` evicts a leaked
  adapter first, then the oldest-activated adapter that is no scenario's
  current one.
- ``FullWeightServingRuntime`` serves one scenario's full Hugging Face
  weights. The engines start with the base model. Activating the base
  artifact after another snapshot loads its files from the artifact
  repository.

Each load gets a new runtime load ID, and every engine reports it afterwards.
The IDs of one runtime belong to a fresh incarnation, so a restart never
reuses an ID for other weights. Loading content that the engines already serve
does nothing.

The ``inference`` section selects the serving, not the training backend
(``reef.service.deploy.serving``). With ``inference.serving: engines``,
``reef serve`` starts a local **inference component** (``python -m
reef.inference.remote``) as the ``inference`` service, unless
``inference.remote`` names a component that runs elsewhere. The control plane
reaches the component through ``RemoteInferenceBackend`` and the runtime kind
``remote_serving``. The training component's ``artifact_format`` chooses the
runtime (``adapter`` or ``full``) and the engines' adapter settings. The local
component joins the deployment's Ray cluster, so its engines and the training
workers reserve separate GPUs.

The serving runtime checks the component's health every 30 seconds. The
component names its process in each health report, so the control plane sees
a component that restarted and runs no engines. After a failed check, the
runtime pauses generation, starts the lost engines again (or replaces the
dead ones), loads each scenario's active selection into them, and resumes.

Engine supervision
------------------

``reef.inference.recovery.InferenceControl`` pauses, resumes and recovers the
engines of one inference service. The engine integration implements its hook,
``InferenceEngineGroup``, which pauses, resumes, replaces dead engines and
retires engines.

``InferenceControl`` records the pause intent before the engine barrier, so a
partial pause failure still fences replacement engines. Recovery replaces dead
engines. It pauses them again when generation was paused. A failed recovery
keeps generation paused. Only ``resume`` lets the engines serve again. Each
engine actor implements ``reef.inference.interfaces.InferenceEngine``, the
per-engine vocabulary that the group and the backend call by name.

Independent SGLang backend
--------------------------

``reef/inference/sglang/`` owns the chat backend, SGLang plugin, native
engine process, router, Ray engine groups and inference lifecycle. Native
``ServerArgs`` and ``launch_server`` come directly from SGLang, not from
Slime. The runtime-load-ID value type lives in ``reef.inference.interfaces``.

Install ``reef-infra[sglang]`` for Python-side inference dependencies and install
native SGLang in the selected GPU environment. This extra does not install
Slime or Megatron.

Engines load published weights from files. The engine control worker's
``load_weights_from_disk`` reloads full weights through SGLang's
``update_weights_from_disk``. Its ``load_adapter_from_disk`` loads an adapter
through ``load_lora_adapter_from_disk``. The plugin name ``reef`` is
preserved.

CPU tests cover native launch bindings, shared placement, response parsing,
LoRA requirements and recovery.
Actual GPU execution and backend combinations still require acceptance testing.

Slime integration
-----------------

``reef.train.slime_backend.component.SlimeTrainingComponent`` runs in the
HTTP service's process and holds no GPU. It parses Slime's arguments once at
startup and, for each job, starts a driver process
(``reef.train.slime_backend.driver``) with the job's directory. The driver owns
one Ray session and runs ``SlimeJobRunner``: Slime's actor (and critic) workers
and, for a dataset job or a task job, Slime's own rollout engines
(``reef.train.slime_backend.rollout_group``). The engines belong to Slime's
serving session, a detached ``ServingCluster`` named after the save root, which
the first such job's driver creates and the next drivers attach to through
Slime's ``create_rollout_manager``. The session holds the engines, their router
and the placement group of the training GPUs; while it runs, every driver
trains on its placements, and a driver detaches from it when it exits, also
after a failure. Without a session, a driver places its job with Slime's own
placement and starts no engines. A driver records each saved checkpoint in the
job's directory, where the component reports it, and releases its workers (and
the GPUs it placed) when it exits. The session's actors run under a process
lease of their own, recorded with the session's name in the save root's
``reef-serving.json``, so a driver's process cleanup leaves them running; the
component disposes the session and retires that lease when it closes
(``driver --dispose``). Serving never shares GPUs with training; with
``--colocate`` the rollout engines share the training GPUs, and Slime moves the
memory between them. A cancel or a close stops the driver; a restarted
component starts the job's driver again.

A job trains its data as Slime rollouts (a ``rollout_id`` is Slime's unit of
training). Without the job parameter ``save_interval``, the job is one
rollout. With it, each run of ``save_interval`` optimizer steps is one
rollout. The workers train the rollout's optimizer steps, and the component
then saves the paired Hugging Face and Megatron checkpoints. A checkpoint's
step is the job's optimizer steps up to it. The snapshot is the Hugging Face export, or with LoRA
the PEFT adapter. A job's ``objective`` parameter selects the loss family, and its
``learning_rate_schedule`` parameter the learning-rate schedule. The actor workers switch to it when they receive the
job's data.

The job's last checkpoint records the objective's state after the job's
data. An earlier checkpoint records the state of the job's start, because
the objective computes its state for the whole data. When the objective
skips the job's data, the job saves no checkpoint, so the next job continues
the start's weights and objective state. The Tinker component does the same.

Every driver loads Slime's latest checkpoint. A job therefore starts from
the component's latest checkpoint; the start of the startup weights names the
untrained weights, until the first rollout is saved. A scenario trains on Slime
only when its root snapshot holds the default start of the configured inputs
(``root_start``). The component checks the start
when Reef submits the job, and the driver again when the job runs: a queued
job whose start an earlier job moved past fails. ``import_start`` refuses
every snapshot, so a Slime scenario refuses ``train-from``. Because the jobs
continue one save root, a Slime deployment trains one scenario. A restarted
job continues after its last recorded checkpoint. A restart can also stop a job after Megatron saved a
rollout but before the component recorded it. The job then records that
checkpoint and does not train the rollout's data twice.

The optimizer state carries from one job to the next in the Megatron
checkpoint, which every driver loads.

`Slime worker executors <#slime-worker-executors>`__ describes how the
component launches its workers.

Executors
---------

An executor launches workers and carries their control RPC. Reef uses the
same executor factory for model workers, harness evaluation workers and
service processes.

Built-in executors
~~~~~~~~~~~~~~~~~~

``uni``
   One worker in the caller's process. RPC may use one local thread. The
   executor rejects more than one worker. A GPU scorer uses the caller's CUDA
   visibility; ``uni`` changes no global visibility mask or cluster resource
   reservation.
   For services, one ``ProcessWorker`` starts the actual service subprocess.

``mp``
   Independent worker ranks launched with Python ``spawn``. Each rank has
   ordered RPC, constructor readiness, health checks and bounded shutdown.
   Worker classes, scorers and arguments must be spawn-pickleable (normally
   module-level classes/functions), not closures or objects holding locks.
   No Ray installation is needed. Declared GPU scorer requirements produce
   disjoint per-rank CUDA masks before worker imports/construction. This does
   not implement model-parallel collectives. Timeouts stop waiting, not the
   work, and never retry RPC.

``ray``
   Launches Ray actors from ordered ``WorkerSpec`` objects. Worker options
   include Ray resources, placement strategies and runtime environments.
   New actors disable automatic task retries. ``RayExecutor.from_workers``
   attaches existing handles; it borrows them unless ``owned=True`` is explicit.
   Borrowed shutdown closes the executor without terminating its actors.

``auto`` (default)
   Resolves to ``uni`` for one worker or ``mp`` for multiple local workers.
   Existing multi-worker Ray placement context or explicit cluster/Ray actor
   options select ``ray``.

A Python class or the reference ``package.module:ExecutorSubclass``
selects a custom executor. The dot form ``package.module.Class`` is not accepted. CPU-only execution does not import Ray, Slime,
Torch or SGLang. Only declared local GPU requirements probe CUDA capacity.

.. code:: python

   from reef.execution.executor import Executor, ExecutorConfig, WorkerSpec, resolve

   class Counter:
       def __init__(self, rank):
           self.rank = rank
           self.value = 0

       def increment(self, amount):
           self.value += amount
           return self.rank, self.value

   if __name__ == "__main__":  # required for spawn
       executor = Executor.create(ExecutorConfig(
           workers=tuple(WorkerSpec(Counter, args=(rank,)) for rank in range(2)),
       ))  # auto -> mp
       try:
           pending = executor.collective_rpc("increment", args=(3,), non_block=True)
           results = resolve(pending, timeout=10)  # [(0, 3), (1, 3)]
       finally:
           executor.shutdown()

RPC and lifetime contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~

``collective_rpc`` dispatches to every worker before waiting, returning results
in worker/rank order. ``rpc(rank, ...)`` addresses one rank. ``non_block=True``
returns an ``ExecutorFuture``. ``resolve`` waits on futures and on nested
lists or tuples of them, under one shared deadline. Ordinary results remain
ordinary values. Backend object references never cross this control-RPC
interface as futures.

A timeout bounds the wait. It does not cancel GPU work or repeat a mutating
operation. Training-job recovery must consult the durable checkpoint/commit
state before deciding what can be resumed. ``shutdown`` terminates only owned
workers. A failed constructor cleans up the workers that the executor already
acquired. Backend-specific launchers must track every allocation, including
partially initialized workers.

Executors expose a terminal ``failure`` (``ExecutorFailure``) and
``register_failure_listener(listener)``. An observer implements
``on_executor_failure(failure)``. The executor calls it once, also when the
observer registers after the failure. Listeners must be nonblocking and must
not call shutdown from the callback. The executor logs observer exceptions
and still reports the worker failure. Normal shutdown and ordinary
worker-method or scorer errors are not terminal infrastructure failures. A
wait timeout does not mean that a worker died.

``mp`` monitors process sentinels, including idle ranks. Owned ``ray`` groups
monitor actor readiness in the background; a queued probe on a busy actor is
not considered a failure. For borrowed/serialized Ray groups, registering a
listener starts process-local monitoring; listeners are not serialized with
actor handles. Local workers share the caller's process and have no separate
process-death detector. Custom executors report terminal failure via ``_fail``.
When the executor detects a failure, outstanding waits fail promptly with
``ExecutorFailedError``. The executor then rejects new submissions and
retires its owned worker peers. Borrowed actors stay owned by their original
launcher. There
is no automatic recreation, request replay, or distributed-group recovery.

Implement ``_init_executor``, ``rpc``, ``collective_rpc``, ``check_health`` and
``shutdown`` in an ``Executor`` subclass. A transport that submits one call and
gets a future back can subclass ``SubmittingExecutor`` instead. It then
implements only ``_submit`` and ``_submit_all``. The bundled ``uni``, ``mp``
and ``ray`` executors and ``DelegatingExecutor`` are built that way.
Backend-specific launch configuration belongs in ``ExecutorConfig.options``.
GPU tensors, optimizer state, NCCL collectives and KV transfers remain the
model backend's responsibility. Changing executors does not resize a live
topology or reshard checkpoints.

Custom service and inference executors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A service executor receives one ``WorkerSpec`` for a ``ProcessWorker`` (the
Ray backend uses ``RayProcessWorker``). Its constructor arguments are the
config snapshot, a one-element service list, run directory, readiness
timeout and config path. A backend may execute this worker on another node.
It may also implement the equivalent RPC protocol: ``describe``,
``prepare(config)``, ``start``, ``probe(name, timeout)``, ``status``,
``read_log(name, offset)``, ``request_stop(force=False)``, ``tree_alive`` and
``shutdown(grace=...)``. ``describe`` returns ``host``, ``pids`` and
``run_dir``. ``status`` maps service names to ``None`` while running or an
exit code; a missing service is not alive.

Custom inference executors expose one control rank. Its RPC vocabulary is
``inference_url``, ``get_runtime_load_ids``, ``updatable_engines``,
``load_adapter_from_disk``, ``load_weights_from_disk``,
``pause_generation_for_update``, ``continue_generation_after_update``,
``terminate_updatable_engines``, ``recover_updatable_engines`` and
``check_health``. Executor shutdown owns serving resource teardown. The
manager does not manipulate backend server groups or actor handles directly.

Slime worker executors
~~~~~~~~~~~~~~~~~~~~~~

``SlimeTrainGroup`` launches the Slime workers through its default
``SlimeRayExecutor``. That executor uses Slime's allocator for GPU
placement, rank-zero rendezvous and the memory-saver environment. RPC goes
through the shared ``RayExecutor``.

``--reef-executor-backend`` in ``training.options`` selects another worker
executor. The executor receives these values in ``ExecutorConfig.options``:

- ``args``, the parsed Slime arguments.
- The node and GPU counts, and the placement group ``pg``.
- The GPU allocation of each actor.
- ``role``, and the reference and teacher flags.
- ``actor_cls``, the worker class.

The executor must preserve rank order and Slime's worker methods.

Reef refuses ``uni`` and ``mp`` for Slime GPU worker roles.

Harness evolution: worker executor selection
--------------------------------------------

Harness evolution evaluates external model endpoints, so it does not require
a local GPU. ``TaskScorer.execution_requirements()`` declares local worker
needs using ``ExecutionRequirements``. Its default is CPU-only. The recipe
supplies the worker count and resources from ``execution.evolution``. The
selector chooses ``uni`` for one local worker and ``mp`` for multiple local
workers. An existing multi-worker Ray placement context or declared
cluster/actor options selects ``ray``. Installed GPUs or ``RAY_ADDRESS`` alone
do not select Ray. Like vLLM's local-topology default, insufficient local CUDA
capacity is an error, not an implicit switch to Ray. Select ``ray``
explicitly for a cluster.

.. code:: yaml

   execution:
     evolution:
       workers: 8
   evolution:
     # The other evolution keys (adapter, method, evaluation, models, seed) stay as they are.
     executor: sandbox      # independent isolation policy; requires bubblewrap

``execution.evolution`` accepts a backend, named profile, or inline object.
Its backend defaults to ``auto``. Omitted resources keep the scorer's
declared requirements (by default one CPU and zero GPUs per worker). ``workers`` is the
fixed worker-group size. It is not a rollout count or a separate
business-level concurrency limit. For ``uni/mp``, CPU requests describe
capacity but do not reserve cores or enforce a CPU quota. Ray maps them to
``num_cpus``/``num_gpus`` per actor. These are scheduling reservations, not
hard CPU limits.

Set workers and resources under ``execution.evolution``. In Python, pass
``worker_executor=ExecutorSettings(workers=..., resources=WorkerResources(...))``.
``evolution.executor`` still means ``local`` or ``sandbox`` rollout isolation;
it is NOT renamed to ``uni``/``mp``. Sandbox preflight runs on the execution
node too. A custom rollout executor must preserve its own isolation and
owner-loss cleanup. Built-in local rollouts in remote/process workers use a
POSIX owner lease, so loss of a worker also kills the rollout process group.
Commands must not detach. Worker loss can leave temporary rollout
directories.

To retain an in-process callback, choose ``uni`` with exactly one worker.
With ``mp`` each rank gets its own copy of the scorer. Mutable scorer state is
not shared or merged back into the recipe. The parent keeps ownership of
the method's step, composition, selection and publication. Only rollout execution and
scoring cross the worker boundary. Candidate and current results keep their
pairing order.

Each backend/scenario starts one fixed pool of
``execution.evolution.workers`` ranks on its first nonempty evaluation. It
reuses the pool across evaluations. Workers and lazy scorer/model
initialization persist. Each rollout still gets a fresh harness subprocess
and temporary directory.

Reef keeps the upstream step records and per-rollout stage paths. Uni/MP
workers write records on the same host. Ray/custom workers return a
compressed record with each successful RPC, so the parent process owns the
durable path without shared storage. Reef extracts archives with Python's
``data`` filter, and an archive cannot replace an existing record directory.
A failed RPC or lost remote worker may leave no returned trajectory. This is
not a durable artifact-transfer protocol, and large records add RPC memory
and transfer overhead.

RPC is ordered within each rank. A scorer error drains the submitted batch
before the next evaluation, while worker death terminally fails the pool.
Reef does not rebuild failed pools automatically and never replays
evaluations.

Trainer/scenario close (including scenario replacement) closes the pool.
Direct Python callers must call ``backend.close()`` in ``finally``; garbage
collection is only a fallback. Ray GPU reservations remain held until close,
even between evaluations. This is worker reuse, not elastic resizing or
scale-to-zero; no YAML change is needed to enable it.
Python launchers must use the usual ``if __name__ == "__main__":`` guard.

A GPU scorer can override ``execution_requirements()`` to return
``ExecutionRequirements(gpus_per_worker=1)``; ``auto`` selects ``uni/mp`` when
local CUDA capacity is sufficient. Initialize the scorer's GPU model lazily
inside the worker, not in its constructor in the recipe process. For a
plain evaluator function without that interface, declare its needs explicitly:

.. code:: yaml

   execution:
     evolution:
       workers: 2
       resources:
         cpus_per_worker: 2
         gpus_per_worker: 1  # per worker: two local workers need two visible GPUs

This describes worker/scorer GPUs, not the external inference server's GPUs.
Local GPU allocations require whole GPUs. MP assigns disjoint CUDA masks
within its worker group; uni leaves the caller's visibility unchanged. These
are not exclusive reservations against other programs on the host. Choose
``backend: ray`` for cluster scheduling or fractional GPU reservations.
Sandbox device access is unchanged; declaring a GPU for the scorer does not
expose it inside a sandboxed harness. CPU workers don't reserve GPUs, but the
selector is not a hardware-access security boundary. Explicit settings cannot
reduce a scorer's declared GPU needs; ``uni/mp`` cannot fulfill cluster
reservations. Ray nodes need the same modules, binaries, models and
reachable model endpoints. Reef does not inspect arbitrary Python or commands
to guess GPU needs. Slime training still requires its specialized Ray
launcher.

Whole-stack deployment configuration
------------------------------------

A configuration file has no process list. Reef assembles the native
inference, training and HTTP processes with the same executor factory as the
model workers. Method services such as a PRM run independently, and recipes
consume their endpoints. The one auxiliary child that Reef assembles
is the generator service of a ``generator`` section, under the ``generator``
executor role. The orchestrator only handles dependencies, readiness,
endpoint publication, log tailing, failure detection and reverse-order
shutdown. It does not contain local process or Ray placement operations.

Omitted selectors (or YAML ``null``) use ``auto``. Existing YAML without
resource reservations or a Ray placement-group context continues to launch
services locally and Slime workers through Ray. A training stack that runs
Slime starts Ray before the HTTP service, which hosts the training component
(see `Slime integration <#slime-integration>`__).

Selection happens before the Executor factory imports the chosen class.
Service startup logs the selected backend and reason.
Explicit backend/profile selections take precedence over automatic decisions.
For ``auto``, the policy is:

* Services with ``cuda`` or declared ``env.CUDA_VISIBLE_DEVICES`` use ``uni``.
  Combining that visibility pin with resource/worker options is an error.
* Services with nonempty ``resources`` or executor ``options`` use ``ray``.
* Otherwise, services already running inside a Ray placement group use ``ray``;
  remaining services use ``uni``. This selects the backend, not a new placement
  strategy; Ray's normal placement/capture rules and explicit worker options apply.
* Slime training uses its specialized Ray executor, even with one GPU. Reef
  has no built-in ``mp``/``uni`` Slime GPU launcher.

Installing Ray, setting ``RAY_ADDRESS``, or initializing Ray outside a
placement group does not by itself change a service to Ray. The selector
does not import or initialize Ray to probe it. It does not count GPUs, change
TP/PP, or retry a failed backend with another backend. Resource requests must
describe the intended scheduling. Standalone SGLang commands still control
their own model parallelism. ``auto`` is also the default
``ExecutorConfig.backend`` for the low-level worker factory. A service
controller is not a model rank. Slime's specialized Ray requirement and
service resource reservations stay specific to each component; they do not
implement vLLM's TP/PP launcher.

Each selector accepts a built-in name, a ``package.module:ClassName`` reference, an inline ``backend/options``
object, or a named profile under ``executors``. Explicit Slime command-line
flags override the corresponding role selection; they also accept profile
names. Profiles for training and rollout carry domain-launcher options, not
necessarily Ray actor options. The reserved Slime launch arguments (``args``,
``pg``, rank/GPU layout, role) come from the model configuration. Profile
options cannot override them.

OpenClawRL's PRM and user model are outside this lifecycle. The example's
Docker Compose file holds their deployment commands, readiness checks and
resource allocation. Reef receives only recipe client parameters such as
``recipe.config.prm-url``, and its stack gets no auxiliary process
definitions.

Ray executors share a process-wide runtime, initialized only when needed.
Without an external address, Reef starts a local cluster using the deployment's
visible GPUs. Set ``RAY_ADDRESS`` (or the deployment's ``reef.ray_address``) to
connect to an external cluster; connection errors never fall back to local.
Reef borrows an already initialized Ray connection and leaves it intact.
Otherwise, the last owner disconnects. Reef stops only a local cluster that
it started.
Attached/serialized executor handles borrow their owner's runtime.

An explicitly declared ``execution.training`` or ``execution.rollout`` role
that resolves to ``ray`` also requests this runtime, even when all service
controllers run locally. It reserves no GPUs; Slime's model topology still
determines its GPU placement. With an external address and only local
controllers, Reef passes the address through. The HTTP service and the
inference component then connect after their readiness dependencies are
satisfied. Stacks without Ray services or declared Ray training or rollout
roles do not start Ray.

The service stack keeps the runtime alive through service shutdown. It
publishes the actual address as ``reef.ray_address`` in runtime snapshots. It
also supplies ``RAY_ADDRESS`` to later services, including the HTTP service
and the inference component. A managed local stack needs no ``ray-head``
service or fixed port. Explicitly managed heads remain supported: connect to
their address and keep the head on ``executor: uni`` with the appropriate
dependencies. Reef does not create cloud machines or install software on Ray
nodes. The command, working directory, Python environment, models and recipe
modules must exist on the selected nodes. Use shared storage or images, or
Ray runtime environments. Standalone SGLang ``num_gpus`` must cover its
complete local TP group. This launcher does not synthesize multi-node SGLang
CLI arguments.

``resources`` contains worker options (for Ray: ``num_gpus``, ``num_cpus``,
``resources``, scheduling options, etc.). Ray owns CUDA visibility. Reef
rejects Ray execution combined with ``cuda`` or a ``CUDA_VISIBLE_DEVICES``
override.
Local execution continues to use ``cuda`` and does not reserve cluster GPUs.
Slime's GPU-layout flags describe model parallelism, not service-worker
placement.

The readiness command runs on the execution node with the service's
environment and working directory, under a bounded timeout. ``{host}`` in
``endpoint`` is the Ray node address, or ``127.0.0.1`` for local execution. A
local service that must be reachable from remote consumers must declare a
routable ``advertise_host``. Reef prepares eligible services together:
executors acquire placement and publish ``${endpoints.SERVICE_NAME}`` before
their service processes start. Each service receives a private node-local
config snapshot through ``REEF_CONFIG``. The snapshot holds every endpoint
published so far, including those of independent peers prepared in the same
batch. A reference to a producer behind dependencies still requires that
producer to be prepared first. Reef does not rewrite ordinary ``127.0.0.1``
literals elsewhere.

Process launch and readiness waits run concurrently for independent services.
``depends_on`` remains a readiness prerequisite. A dependent starts as soon
as all of its declared dependencies are ready, without waiting for unrelated
services. Each service has its own readiness deadline, measured from its
start RPC, not from the beginning of the deployment. The stack is ready only
when every service is ready. Reef keeps checking services that passed
readiness for process exits while their peers initialize. Explicit Ray-head
services can remain dependencies of Ray-executed services. Reef does not
place their dependents before the head is ready.

Logs are kept on the worker and tailed into ``run_dir/SERVICE.log``. Worker
metadata (host, PIDs and worker log directory) is written to
``run_dir/SERVICE.worker.json``; a remote PID must never be signalled locally.
Each service has a separate config directory. Resolved config
snapshots are private files and may contain credentials; protect the run
directory as deployment state. Ray process workers use a POSIX owner-lease
guard so loss of the owning actor terminates their process group. Commands
must remain foreground processes and must not detach into another session.

Failure policy is fail-stop, not automatic restart/replay. A startup failure
cancels peer readiness waits and joins launch tasks before Reef closes the
executors it already created. A late launch therefore cannot escape cleanup.
Shutdown attempts every dependent before its dependencies, even if an RPC
fails. An individual executor does not delete shared Slime placement groups. The HTTP service closes its training component
and inference runtime after the dispatcher stops every scenario worker.

Verification
------------

CPU tests cover custom service and inference executors, resource
selection, dependency ordering, endpoint propagation, rollback, process-tree
cleanup, bounded probes and owner-lease loss. To run the real Ray/HTTP tests:

.. code:: bash

   REEF_TEST_RAY=1 python -m pytest tests/reef_service/test_service_executor_ray.py

These are control-plane tests, not a GPU SGLang/Megatron validation or a
multi-machine networking benchmark.
