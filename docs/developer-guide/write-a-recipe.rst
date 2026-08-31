Write a recipe
==============

A recipe is the method a deployment runs: which recorded traffic is eligible,
how it becomes a batch, what signal that batch carries, and whether the
candidate it produces replaces the served version.

`Quickstart <../getting-started/quickstart.rst#recipe-and-artifact>`__ defines the term.

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
           VER["New artifact version<br/>serves later requests ↻"]
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
   candidate evaluation | measures the checkpoint the backend exported and decides select or reject. Every recipe carries one; the default selects whatever the backend produced
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

   reef:
     recipe: "my_pkg.my_method:MyMethodRecipe"
     batch_size: 4

This fragment shows only the new keys; keep the model, storage, runtime, and
``services`` settings from the config you copied. Set
``training.global_batch_size`` to the same value, and add the driver flags your
loss family requires (`the mapping
<loss-families.rst#family-to-driver-flags>`__).
The driver reads the same ``reef.recipe`` value from the deployment config and
gets the loss family from the class's ``training_spec()``. Do not repeat either
value in the driver environment. Reef has no global recipe-implementation
registry.

.. code:: bash

   reef serve -c path/to/my-method.yaml

Gate a candidate
~~~~~~~~~~~~~~~~

A runtime finishes training by exporting a candidate. The recipe's
``candidate_evaluation`` decides what happens to it. A harness recipe builds
its evaluator in code:

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
