Write a recipe
==============

A recipe is the method a deployment runs: which recorded traffic is eligible,
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
           JUDGE["Judge"]
           BATCH["Make batch"]
           ENG -->|"resolved unit"| JUDGE
           JUDGE -->|TRAIN| BATCH
           JUDGE -.->|"WAIT or NEVER"| ENG
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
       class JUDGE,BATCH,PREP,EVAL,SELECT user-owned

The shaded steps are the method's. A processor *judges* each resolved unit. A
unit consists of one record plus the reports that reference it. ``TRAIN``
batches it, ``WAIT`` holds
it until its remaining references land, ``NEVER`` drops it. After the backend
runs, the method's evaluator measures the candidate and its selector decides
whether it is published.

Before you write one
--------------------

If an existing recipe's processor, preparer, loss family, gate, and surface
already match your method, change its config instead. Re-read `Choosing a recipe
<../user-guide/recipes.rst>`__.

Build one
---------

A weight recipe is four pieces plus the class that binds them.

.. config::

   step preparer | a plain function turning a typed batch into a ``StepSignal``: the loss family, the per-sample advantages, and the next algorithm state. No torch, Ray, or Slime import.
   processor | decides which reports are eligible and shapes the accepted ones into one typed batch
   report type | the ``ReportBase`` subclass Reef validates at ingress, so a malformed report is HTTP 400 rather than a training-time surprise
   candidate evaluation | measures the checkpoint the backend exported and decides select or reject. Every recipe carries one; the default, ``BackendAlwaysSelectPlugin``, selects whatever the backend produced
   recipe class | a frozen dataclass whose ``training_spec()`` names the processor, the preparer (by dotted path), and the loss family

`Python API <../reference/python-api.rst>`__ is the contract for each.
``recipes/sao/`` is the smallest cookbook implementation and the one to read
alongside this page. Its four files total fewer than 200 lines: ``recipe.py``,
``processor.py``, ``preparer.py``, and the ``slime/`` loss family.

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
     module: my_pkg.evaluation:build_evaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

Reef calls the factory once per scenario with that opaque ``config`` and the
scenario's training runtime. The trainer runs the plugin between the backend
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

``training.backend`` selects one definition for both process preparation and
HTTP runtime construction. Definitions implement ``TrainingDeployment`` from
``reef.train.deployment`` and live under the owning integration:

* ``prepare(config, settings)`` receives the resolved deployment plus parsed
  service settings as a mapping. It validates backend combinations, binds
  derived values and returns process definitions that HTTP must wait for.
  It must not download models, allocate devices or construct a runtime.
* ``runtime_config(settings, *, max_staleness, connector=None)`` returns the
  configuration consumed by ``RuntimeRegistry`` in the HTTP process. The
  result must construct a ``TrainingRuntime``. ``connector`` is an optional
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

Weight recipes use ``RuntimeTrainingBackend`` to adapt any ``TrainingRuntime``
to the shared training lifecycle. The old ``SlimeTrainingBackend`` import remains
an alias. Experiment metadata now reports ``RuntimeTrainingBackend`` and the
actual runtime class instead of labeling all weight training as Slime.
