Configuration
   evolution.seed | entry options loaded into the tree on first boot, or a dotted ``module:attribute`` naming a sequence of them (``reef.harness.native.seed:SEED_NODES`` is the native harness's shipped tools and hook); recovered state takes precedence
   evolution.models | auxiliary models for the method: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name
   evolution.version_check | appends the adapter's update notice; an interactive pulled tree offers to run the update or skip when behind

The served model's binding is appended at render time; it never enters the
published files. The seed defines the baseline the first mutation is measured
against.

The ``services`` list
---------------------

Each entry is one process. ``command`` can be a command-line string or an argv
list. Prefer the list form when exact argument boundaries matter; existing
string commands retain their current ``shlex`` parsing.

.. config::

   services[].name | the service's id, used by ``depends_on``; unique within one stack
   services[].command | the command line string or argv list to run
   services[].ready | a shell command that succeeds once the service is up
   services[].ready_timeout | seconds to wait for ``ready`` before giving up; the top-level ``ready_timeout`` sets the default
   services[].depends_on | services that must be ready first
   services[].cuda | the value of ``CUDA_VISIBLE_DEVICES`` for this process
   services[].env | extra environment variables

The ``training`` section
------------------------

Read by the weight-training stack. See `Evolve your model
<../user-guide/evolve-your-model.rst>`__ for how to size it.

.. config::

   training.num_gpus | GPUs handed to the Ray head
   training.cuda_visible_devices | the devices Ray and Slime may use
   training.global_batch_size | samples in one optimizer step. Must equal the recipe's ``batch_size``.
   training.checkpoint_dir | where Megatron and HF checkpoints are written
   training.megatron_checkpoint_path | optional pre-converted torch_dist checkpoint, to skip HF conversion on every start
   training.checkpoint_retention | storage-fraction bounds and the retention policy
   training.slime_flags | GPU layout, optimizer, sequence length, and loss settings, as one literal string

Slime fills architecture flags such as layer counts and hidden sizes from
``reef.model_path``. Do not put them in the config.

The ``evaluation`` section
--------------------------

Only weight-training recipes read this section; a deployment that pairs it
with any other recipe fails at startup, because a harness recipe builds its
evaluator in code. Absent by default, in which case a successful
weight-training step publishes without a gate. When present, Reef calls the
named factory once per scenario and hands the plugin the exported but
unpublished checkpoint.

.. config::

   evaluation.module | a ``package.module:factory`` reference to the plugin factory. Required.
   evaluation.config | opaque mapping handed to the factory; Reef never reads it

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:build_evaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

The plugin interface is in `Write a recipe
<../developer-guide/write-a-recipe.rst#gate-a-candidate>`__.

Experiment tracking
-------------------

Tracking is optional, off by default, and belongs to a Reef *scenario* rather
than to one training backend. The same provider-neutral logger is shared by the
recipe, the processor, backend results, and the commit lifecycle. Install
``reef[wandb]`` when the training extra does not already provide it.

.. code:: yaml

   observability:
     wandb:
       enabled: true
       project: reef
       entity: your-team             # optional
       group_prefix: prod-us-east    # optional scenario-group namespace
       name_prefix: baseline         # optional run-name prefix
       tags: [openclawrl, qwen]
       mode: online                  # online, offline, or disabled
       directory: /var/lib/reef/wandb
       upload_checkpoints: false

Export ``WANDB_API_KEY`` before starting, or log in once with the credential
store on the cluster.

.. warning::

   There is no API-key field here. Reef rejects Slime's ``--wandb-key`` flag and
   never writes a credential into metrics or run config. Do not put one in the
   YAML, in ``slime_flags``, in a tag, or in a run name.

``online`` sends data to the project. ``offline`` makes no network calls and
writes syncable data below ``directory`` for a later ``wandb sync``.
``disabled`` makes no calls even when ``enabled`` is true.

Each scenario maps to a group named after the scenario or
``<group_prefix>/<scenario>``. Within it, Reef opens one run when the scenario
binds and another after each rollback. The deterministic run id includes those
identities, so restarting resumes the same run with ``resume=allow``. A rollback
finishes the current run, marks its summary with the source and target, and
resets ``train/step`` to zero; the globally monotonic ``reef/step`` stays
attached for joining a run back to the commit log.

Recipe and processor code logs through the same object without importing W&B:

.. code:: python

   experiment_logger.log({"temperature": 0.6}, namespace="recipe")
   self.experiment_logger.log({"accepted": 12}, namespace="processor")

Those become ``recipe/*`` and ``processor/*``, each namespace on its own
``<namespace>/event`` axis. Only finite numeric values are sent.

Durable commit metrics carry ``experiment/provider``, ``experiment/project``,
``experiment/group``, and ``experiment/run_id``. Use them to open the run from
a Reef version, and use the run's ``reef/training_job_id`` to go the other way.
Checkpoint paths are metadata only unless ``upload_checkpoints: true``.

Import, initialization, logging, summary, and upload failures are reported in
the service log and never fail a training step or its commit.
