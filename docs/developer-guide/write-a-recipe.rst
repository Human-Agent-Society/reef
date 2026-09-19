Write a recipe
==============

A recipe is the method a deployment runs: how recorded traffic is processed,
how it becomes a batch, what signal that batch carries, and whether the
candidate it produces replaces the served version.

`Quickstart <../getting-started/quickstart.rst>`__ defines the term.

Accepting records, replaying them after a restart, holding a batch until it is
acknowledged, committing algorithm state, running the backend, and publishing
the next version are Reef's job, not the method's.

This page covers a **weight** recipe. For a harness-evolution method with
``propose``, ``evaluate``, and a selection policy against a fixed model, see
`Evolve your harness <../user-guide/evolve-your-harness.rst#write-a-method>`__.

.. code:: mermaid

   flowchart TB
       accTitle: How a recipe collects records and publishes an update
       subgraph COLLECT["1. Collect"]
           direction LR
           REQ["Inference request"]
           REP["Optional report"]
           STORE[("Records")]
           REQ --> STORE
           REP --> STORE
       end
       subgraph RESOLVE["2. Resolve and select"]
           direction LR
           ENG["Resolve reported<br/>or computed feedback"]
           SAMPLE["Make sample"]
           BATCH["Make batch"]
           ENG -->|"resolved feedback"| SAMPLE
           SAMPLE --> BATCH
       end
       subgraph EVOLVE["3. Update and publish"]
           direction LR
           PREP["Prepare step"]
           BACK["Run backend<br/>driver or local"]
           EVAL["Evaluate candidate"]
           SELECT["Select or reject"]
           VER["New release<br/>serves later requests ↻"]
           PREP -->|"step signal"| BACK
           BACK -->|"unpublished checkpoint"| EVAL
           EVAL --> SELECT
           SELECT -->|"select"| VER
       end
       COLLECT -->|"stored records"| RESOLVE
       RESOLVE -->|"typed batch"| EVOLVE
       class SAMPLE,BATCH,PREP,EVAL,SELECT user-owned

The shaded steps are the method's. A reported-feedback processor assembles
valid reports and their already stored inference records into samples, then
batches complete units. Required training fields are validated by the backend;
contract failures raise instead of silently dropping reports. After training,
the evaluator measures the candidate and the selector decides whether to publish.

Before you write one
--------------------

If an existing recipe's processor, objective, gate, and surface
already match your method, change its config instead. Re-read `Choosing a recipe
<../user-guide/recipes.rst>`__.

Build one
---------

A weight recipe is four pieces plus the class that binds them.

.. config::

   training objective | a ``TrainingObjective`` declaring a loss family and implementing ``prepare(batch, state)`` to return advantages, metrics, and proposed state in a ``StepSignal``. No torch, Ray, or Slime import.
   step schedule | a ``StepScheduling`` the recipe binds beside its objective: rollout unit, optimizer step size, epochs, shuffle, and remainder handling. The objective declares only whether its loss tolerates more than one pass.
   processor | assembles valid reports and referenced records into samples and typed batches
   report type | the ``ReportBase`` subclass Reef validates at ingress, so a malformed report is HTTP 400 rather than a training-time surprise
   candidate evaluation | measures the checkpoint the backend exported and decides select or reject. Every recipe carries one; the default, ``BackendAlwaysSelectPlugin``, selects whatever the backend produced
   recipe class | a frozen dataclass whose ``training_spec()`` names the processor, the objective (by registered name or dotted class/instance path), and the step schedule

`Python API <../reference/python-api.rst>`__ is the contract for each.
``recipes/sao/`` is the smallest cookbook implementation and the one to read
alongside this page. Its four files total fewer than 200 lines: ``recipe.py``,
``processor.py``, ``objective.py``, and the ``slime/`` loss family.

Configure it
~~~~~~~~~~~~

Keep your module in a package installed in the environment used by *both* the
Reef service and the training driver, and verify the import they will perform:

.. code:: bash

   python -c "from my_pkg.my_method import MyMethodRecipe"

Copy a weight-training config as described in `Evolve your model
<../user-guide/evolve-your-model.rst>`__ and select the class by dotted path:

.. code:: yaml

   schema-version: 2
   recipe:
     implementation: "my_pkg.my_method:MyMethodRecipe"
     config:
       batch-size: 4

This fragment shows only the new keys; keep the inference, storage and training
settings from the config you copied. Reef assembles the driver and HTTP process. Set
``training.config.global_batch_size`` to the same value, and add the driver flags your
loss family requires (`the mapping
<loss-families.rst#family-to-driver-flags>`__).
The driver reads the same ``recipe.implementation`` value from the deployment config and
gets the loss family from the class's ``training_spec()``. Do not repeat either
value in the driver environment. Reef has no global recipe-implementation
registry.

.. code:: bash

   reef serve -c path/to/my-method.yaml

Declare each method setting once with ``config_field``:

.. code:: python

   from dataclasses import dataclass
   from reef.recipe import WeightTrainingRecipe, config_field

   @dataclass(frozen=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       batch_size: int = config_field(4, env="MY_BATCH_SIZE", help="Samples in one update.")
       temperature: float = config_field(0.5, allow_nonfinite=False)
       tags: tuple[str, ...] = config_field(())

The shared parser reads annotations, defaults, help and environment fallback
from these declarations. Supported types are ``str``, ``int``, ``float``,
``bool``, ``tuple[str, ...]``, ``Mapping[str, Any]`` / ``dict[str, Any]``, and
optional forms. Object fields can use ``config_field(default_factory=dict)``.
Keep range and cross-field checks in ``__post_init__``. Do not repeat scalar
conversion in the service layer or recipe hooks.

``WeightTrainingRecipe.service_config`` translates the flat deployment layout
and rejects unknown fields; it no longer converts scalar values. The service
resolves those fields before connecting the runtime and passes them to
``from_resolved_config``. ``from_environment`` uses that same resolution path
for standalone recipe construction. Custom construction hooks should consume
the resolved values; ``_recipe_kwargs`` continues to own non-field sections.

Gate a candidate
~~~~~~~~~~~~~~~~

A runtime finishes training by exporting a candidate. The recipe's
``candidate_evaluation`` decides what happens to it. A weight recipe declares
its evaluator in the deployment config; the top-level ``evaluation`` section
serves weight recipes only, and a harness recipe builds its evaluator in code
instead:

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:EvaluationFactory
     config:
       benchmark: gsm8k
       threshold: 0.8

The reference names a ``CandidateEvaluationPluginFactory`` subclass with a
no-argument constructor, or an instance of that class. Its constructor must
not allocate model resources: Reef validates it when loading recipe config.
Reef calls ``build`` once per scenario with the opaque ``config``,
``runtime``, ``training_runtime``, ``scenario``, and ``environ``:

.. code:: python

   from reef import CandidateEvaluationPluginFactory
   from my_pkg.benchmarks import BenchmarkGate


   class EvaluationFactory(CandidateEvaluationPluginFactory):
       def build(self, config, *, runtime, training_runtime, scenario, environ):
           return BenchmarkGate(
               benchmark=config["benchmark"], threshold=config["threshold"],
               runtime=runtime, scenario=scenario,
           )

``BenchmarkGate`` must inherit ``CandidateEvaluationPlugin`` and implement
both ``evaluate`` and ``decide``. Plain factory functions and objects that only
have matching method names are rejected. The trainer runs the plugin between the backend
step and publication, calling ``evaluate`` before ``decide``. A rejection
leaves the previous version serving. `Python API
<../reference/python-api.rst#candidate-evaluation>`__ documents the plugin
contract: ``evaluate``, ``decide``, the fail-closed rule,
and idempotency by ``candidate.candidate_id``. The section fields are in
`Configuration <../reference/configuration.rst#the-evaluation-section>`__.

Where the feedback comes from
-----------------------------

Reef never invents feedback. Use whatever already judges your agent; for the
numeric ``score`` field, a consistent scale where higher is better. If you have
no number, `Choosing a recipe <../user-guide/recipes.rst>`__ lists the methods that need
none.

External method services
------------------------

Version 2 configuration describes components and their parameters. It does not
accept ``service``, ``services`` or ``execution.services``. HTTP settings belong
in ``reef``. Reef assembles its HTTP process and supported inference/training
backends. Additional method services run independently of Reef's orchestrator.

Declare a recipe field for the external endpoint and implement the client in
the method package. For example, OpenClawRL uses ``recipe.config.prm-url`` and
``recipe.config.prm-tokenizer-path`` to call its independently served PRM.
CLI overrides such as ``--recipe.config.prm-url http://localhost:23001`` use the
same field declarations and validation as YAML. Request timeouts and scoring
failure behavior belong to that recipe's client.

The method's deployment tools own external service startup, readiness, resources
and shutdown. Reef does not register their processes, probe their health or
reserve their GPUs. The OpenClawRL example's ``docker-compose.yaml`` starts PRM
and user-model containers on devices separate from Reef/Slime. A remote endpoint
can be substituted without changing Reef's process topology. Reef shutdown or a
training startup failure leaves independently deployed services running.

Backend environment defaults belong to their integration; Slime's defaults
live in ``reef/train/slime_backend/launch.py`` and honor explicit environment
overrides.


Training backend deployment
----------------------------


Managed model components use Reef's shared process entrypoint,
``python -m reef.service.training_driver``. A training integration implements
``TrainingDeployment.create_training_plan(config, *, loss_family)`` and returns
``reef.train.deployment.TrainingDeploymentPlan``: an unstarted trainer, resource
reservations, native inference launch data and coordinator options. It must not
construct an inference service or allocate model resources.

The service assembler selects inference independently through
``reef.inference.deployment.inference_service_for``. A factory takes the launch
mapping and returns an unstarted ``InferenceService``. Built-ins resolve lazily;
external integrations can expose a ``module:factory`` reference or register a
``reef.inference_backends`` entry point. The concrete factory owns parsing its
native configuration. Reef combines both definitions into ``ModelDeploymentPlan``
and rejects incompatible weight-transfer protocols before allocating anything.
A new backend still needs an implementation of a transport supported by its peer;
factory discovery does not make arbitrary backend pairs compatible.

``reef.runtime.deployment.ModelDeployment`` owns startup and shutdown:

1. Start ``DeploymentResources`` and the selected ``InferenceService``.
2. Verify the receiver and prepare weight transfer. Colocated inference releases
   its initial device allocations before training workers initialize.
3. Call ``TrainingService.start(resources)`` without an inference connection.
4. Attach a ``WeightTransferSession`` to the trainer's native sender. Its protocol,
   borrowed receiver executor and fresh session identity describe one attachment.
5. Start Reef's ``TrainingCoordinator`` with separate ``TrainingBackend`` and
   ``InferenceBackend``, then verify readiness after durable recovery.

Shutdown closes the coordinator's active operations, training, inference and
reservations in reverse order. Every close operation must handle partial startup
and repeated calls. Serialized training operations must release any workers they
replace; the training service also cleans up initial or partially started groups.
Reef never closes borrowed external inference engines. In-process integrations
may retain their own entrypoint instead of using ``create_training_plan``.

The coordination code belongs in ``reef/runtime/``. Concrete inference code
belongs in ``reef/inference/<integration>/`` and training code in
``reef/train/<integration>/``. Neither integration imports the other. For the
current native pair, Slime converts legacy argument values into plain launch
options; the SGLang factory constructs ``SGLangConfig`` and checks receiver
capabilities. The ``slime-sglang-control-v2`` attachment remains an explicitly
scoped native protocol, not a universal tensor format.

At the recipe boundary, ``RuntimeCandidateBackend`` maps batches and delegates
candidate activation, rejection and durable acknowledgement to
``reef.runtime.scheduler.RuntimeScheduler``. The scheduler receives separate
``TrainingRuntime`` and ``InferenceRuntime`` objects; neither inherits the other
and there is no aggregate ``ModelRuntime``.

Both interfaces live in ``reef.runtime.interfaces``. Concrete scheduling
connections live in the integration's ``runtime.py``: for example,
``reef.train.slime_backend.runtime.SlimeTrainingRuntime`` and
``reef.inference.sglang.runtime.SGLangInferenceRuntime``. They can reuse the
generic coordinator clients in ``reef.train.runtime`` and ``reef.inference.runtime``;
adding another backend must not require importing Slime or SGLang.

For a native trainer such as Slime or MLX, the backend integration interface is
``TrainingBackend`` in ``reef/runtime/interfaces.py``. It exposes
batch preparation, training, checkpoints and separate weight preparation/sending.
Slime implements it with ``SlimeTrainingBackend`` in
``reef/train/slime_backend/reef_adapters/bridge.py``. This is distinct from
``CandidateBackend`` in ``reef/train/backend.py``: that recipe-facing interface
prepares, evaluates and settles candidates, including harness updates that need
no model trainer. ``RuntimeCandidateBackend`` connects that recipe interface to
the runtime scheduler.

Alongside ``TrainingBackend``, ``InferenceBackend`` exposes pause,
resume, recovery, version verification, adapter unload and memory operations.
``TrainingCoordinator`` owns staleness admission, LoRA residency and colocated
handoffs. The native sender transfers weights directly to receiver workers;
weight tensors never pass through the HTTP service or the scheduler.

The coordinator composes ``TrainingExecution`` for durable job ordering and
``TrainingPublication`` for commit gating. These live in ``reef.runtime.scheduler``
and ``reef.runtime.publication`` respectively. ``reef.runtime.recovery`` owns the
marker store and startup reconstruction; publication depends on the shared
``TrainingJobStore`` interface rather than the file implementation. Training adapters return a
``PreparedTrainingJob`` whose ``train`` returns ``TrainingMetrics`` and whose
``save_checkpoint`` synchronously persists model/optimizer state and recovery
metadata. Reef writes the durable markers. Reef selects and persists a target
runtime load ID before sending a checkpoint, verifies the sender and every
receiver, and keeps generation paused until the matching artifact commit is
acknowledged. Retrying a partial transfer reuses its target; republication of
unchanged weights preserves the committed identity. A failed restore or
ambiguous optimizer step cannot reopen serving.

Inference backends can compose ``reef.runtime.recovery.InferenceControl``
with concrete engine, monitoring and update-connection adapters. Serialize calls
in the owning actor, and route legacy monitoring controls through the same pause
state. Its ``resume`` is an internal operation authorized by the training commit
gate, not a public serving action. Backend handles and weight transport remain
inside adapters; an HTTP URL alone is not an update connection.
After the deployment owner retires a prior trainer, use
``InferenceControl.prepare_training_connection`` to require a fresh attachment
and keep inference paused even when the engines are already healthy. Reef's
deployment owner invokes this handshake before replacement workers are created.
Monitoring must drain active probes and retirement before engine mutation.
``EngineHealthMonitor`` provides this barrier using backend ``EngineHealthChecks``
snapshots. Each ``EngineHealthTarget`` must bound its probe/retirement operations
by the supplied timeout and retire only its captured engine handles. A failed
drain must block replacement or cleanup until draining succeeds.
See `commit-gated weight publication <executors.rst#commit-gated-weight-publication>`__
for retry and startup-recovery requirements.

``training.backend`` selects one definition for both process preparation and
HTTP runtime construction. Definitions implement ``TrainingDeployment`` from
``reef.train.deployment`` and live under the owning integration:

* ``prepare(config, settings)`` receives the resolved deployment plus parsed
  service settings as a mapping. It validates backend combinations, binds
  derived values and returns process definitions that HTTP must wait for.
  It must not download models, allocate devices or construct a runtime.
* ``runtime_config(settings, *, max_staleness, connector=None)`` returns the
  configuration consumed by ``RuntimeRegistry`` in the HTTP process. The
  result must construct a ``(TrainingRuntime, InferenceRuntime)`` pair. ``connector`` is an optional
  legacy connection injection; an in-process backend rejects it.

The Slime implementation in ``reef/train/slime_backend/launch.py`` owns the
Ray roles, driver command, native checkpoint binding and bridge connection.
An in-process integration can reuse the supplied implementation:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment

   class Deployment(InProcessTrainingDeployment):
       runtime_type = "my_backend.runtime:factory"

Here ``factory`` is a lightweight ``RuntimeFactory`` instance. Its
``config_type()`` declarations validate ``training.options`` using the shared
parser; older factories can retain their owning parser. Import execution
libraries only when the factory constructs the runtime, and provide any
backend-owned defaults there. No Ray, standalone inference process or driver
is added by this definition.

Select ``--training.backend my_backend.launch:Deployment`` directly, or register
an installed name in the integration distribution:

.. code:: toml

   [project.entry-points."reef.training_backends"]
   my-backend = "my_backend.launch:Deployment"

This enables ``--training.backend my-backend`` in both the launcher and HTTP
child. Only the selected definition is imported. Missing or ambiguous names
fail explicitly, without falling back to Slime. Do not introduce a separate
user-facing runtime selector for weight training: runtime wiring belongs to
the selected backend. Method dependencies remain the Recipe's responsibility.

Weight recipes pass their separate ``training_runtime`` and inference ``runtime``
to ``RuntimeCandidateBackend``, which adapts the shared candidate lifecycle to
Reef's runtime scheduler. The former candidate-layer ``SlimeTrainingBackend``
alias is removed; the name now belongs to Slime's native trainer implementation.
Experiment metadata reports ``RuntimeCandidateBackend`` and the
actual runtime class instead of labeling all weight training as Slime.
