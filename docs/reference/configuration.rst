Configuration
=============

A deployment config is one YAML file. ``reef serve -c <file>`` reads it, starts
every process in its ``services`` list in dependency order, and hands the
``reef`` section to the HTTP service.

.. code:: yaml

   reef:
     host: 0.0.0.0
     port: 8900
     recipe: recipe
     token: ${REEF_TOKEN}
     upstream_url: ${REEF_UPSTREAM_URL}
     upstream_api_key: ${REEF_UPSTREAM_API_KEY}

   services:
     - name: reef
       command: python -m reef.service
       ready: curl -sf http://127.0.0.1:${reef.port}/healthz

Values interpolate from the environment with ``${VAR}`` and from the config
itself with ``${dotted.path}``. Any value can be overridden on the command line:
a bare ``--model_path /models/demo`` targets the ``reef`` section, and a dotted
``--training.checkpoint_dir /tmp/ckpt`` targets any other. Each process writes a
log under ``/tmp/reef-stack/``; set ``run_dir`` to move it.

Start from a cookbook stack
---------------------------

The source checkout's runnable stacks live under the ``recipes/`` cookbook
and ``tutorials/``: the learn-nothing ones use the core ``recipe``
implementation in ``recipes/basic/``, each weight-training method owns its
examples, and harness evolution ships as a tutorial.

+-------------------------------------------------------------+----------------------------------------------------------+
| File                                                        | What it starts                                           |
+=============================================================+==========================================================+
| ``recipes/basic/external-provider.yaml``                    | no GPU, no local model: one Reef process proxying an     |
|                                                             | HTTP provider                                            |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/basic/local-sglang.yaml``                         | local inference: an SGLang server plus Reef, no training |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``recipes/<method>/examples/<example>/serve.yaml``          | weight training: Ray head, Slime driver, Reef, and the   |
|                                                             | method's own services                                    |
+-------------------------------------------------------------+----------------------------------------------------------+
| ``tutorials/harness_evolve/serve.yaml``                     | harness evolution: one Reef process, no GPU; ``run.sh``  |
|                                                             | materializes its recipe preset and starts the stack      |
+-------------------------------------------------------------+----------------------------------------------------------+

Each weight-training example ships its stack as ``serve.yaml``.
``recipes/sao/examples/sao/serve.yaml`` is the smallest, two GPUs for one
actor and one rollout engine; ``recipes/tttd/examples/tttd/serve.yaml`` adds
LoRA training, and ``recipes/openclawrl/examples/openclawrl/serve.yaml`` adds
a PRM engine and a student model.

The ``reef`` section
--------------------

.. config::

   reef.recipe | the recipe this deployment serves. Required.
   reef.host | 0.0.0.0 | bind address
   reef.port | 8900 | bind port
   reef.token | the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.
   reef.model_path | a local HF model directory or a repo id, downloaded on start
   reef.upstream_url | the OpenAI-compatible provider, with no ``/v1`` suffix
   reef.upstream_api_key | its credential. Reef is the only party that sees it.
   reef.upstream_model | the model to request upstream
   reef.upstream_api | openai | the provider dialect; ``anthropic`` for an Anthropic-style endpoint
   reef.inference_url | the address the training backend reports | the local engine; set only to front the engines with something else
   reef.inference_timeout_s | 300.0 | per-request timeout
   reef.allow_implicit_scenario_creation | true | when false, an unknown scenario is HTTP 404
   reef.checkpoint_every_n_versions | 1 | how often a version becomes durable

Storage paths default under ``.reef/``, which the basic and sao stacks keep;
the openclawrl stack overrides them to ``/var/lib/reef``. Point them somewhere
persistent.

.. config::

   reef.artifact_repository | .reef/artifacts.git | the Git-backed release chain
   reef.artifact_work_dir | .reef/artifact-work | materialization scratch
   reef.artifact_cache_dir | .reef/artifact-cache | fetched artifact cache
   reef.agent_record_dir | .reef/agent-record | the record store

.. warning::

   On ephemeral storage, a restart loses the record store, the commit logs, and
   every version.

Recipe settings such as ``batch_size`` sit beside these in the same section,
along with any others the recipe declares with ``config_field``. When
``reef.recipe`` is a dotted weight-training class, keys the service does not
recognize are handed to the recipe, and the recipe rejects any key it does
not declare. With the core ``recipe`` or a named preset, the recipe reads its
configuration from the preset, and unrecognized keys here are silently
ignored.

Recipe configuration
--------------------

A recipe is selected three ways:

- **The core record-only recipe:** ``recipe: recipe``
- **A dotted class:** ``recipe: "my_pkg.my_method:MyMethodRecipe"``
- **A named preset:** ``recipe: my-preset``, resolved to ``my-preset.yaml``
  under ``REEF_RECIPE_CONFIG_DIR``

There is no recipe-implementation registry. A bare name other than ``recipe``
is always a preset name; it never imports a learning method implicitly.

``REEF_RECIPE_CONFIG_DIR`` is the directory preset YAML is read from, and it has
**no default**: a bare recipe name resolves to a preset only when it is set.

A preset is read as-is. ``${VAR}`` interpolates in a deployment config, never
in a preset. A preset carries its own ``implementation``, ``model``, and
``data`` sections, plus an optional ``runtime`` section when the recipe
builds its own runtime instead of using the deployment's upstream proxy.
Harness-evolution presets also carry an ``evolution`` section:

.. code:: yaml

   implementation: reef.train.cordis_backend.recipe:CordisRecipe
   model:
     path: qwen3-8b
   data:
     batch_size: 1
     max_score: 0.0
   evolution:
     adapter: pi
     propose: methods.mine:propose
     evaluate: methods.mine:evaluate
     tasks: ["..."]

The preset's ``implementation`` is ``recipe`` or a dotted recipe class. Weight-training
recipes are selected directly by dotted class in the deployment config, so the
service can assemble their Ray training runtime; their fields are flat
``reef.<name>`` keys. Presets suit recipes whose runtime can be built from the
preset or the deployment's upstream proxy. There, ``data`` holds batching
fields and a recipe-specific section holds the rest.

Harness evolution keys
~~~~~~~~~~~~~~~~~~~~~~

``batch_size`` and ``max_score`` go under ``data:``; the rest goes under
``evolution:``. `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ describes what each one changes.

.. config::

   data.batch_size | 1 | traces per mutation attempt
   data.max_score | 0.0 | upper bound of the score window that batches
   data.batch_policy | reports | ``records`` batches recorded traffic alone, every ``batch_size`` requests, with unscored samples

The window has no lower bound, so the default keeps only traces at or below
zero.

.. config::

   evolution.propose | a ``Proposer``, a plain callable, or a dotted ``module:attribute``
   evolution.evaluate | an ``EpisodeScorer``, likewise
   evolution.selection | score_comparison | ``always``, or a dotted reference to an object with ``decide``
   evolution.tasks | non-empty list of episode prompts, scored once per tree per step
   evolution.adapter | pi | ``opencode``, or an entry-point adapter
   evolution.binary | overrides the adapter's binary name
   evolution.episode_timeout_s | 600 | seconds one evaluation episode may run
   evolution.episode_repeats | 1 | episode pairings per task per step; each repeat tallies on its own
   evolution.forbid_residue | false | when true, an episode leaving files outside the cleanup whitelist scores as one that could not run
   evolution.seed | entry options loaded into the tree on first boot; recovered state takes precedence
   evolution.models | auxiliary models for the method: ``url``, ``model``, optional ``api`` (default ``openai``) and ``timeout_s``, with the credential as a literal ``api_key`` or an ``api_key_env`` variable name
   evolution.version_check | appends the adapter's update notice; an interactive pulled tree offers to run the update or skip when behind

The served model's binding is appended at render time; it never enters the
published files. The seed defines the baseline the first mutation is measured
against.

The ``services`` list
---------------------

Each entry is one process.

.. config::

   services[].name | the service's id, used by ``depends_on``
   services[].command | the command line to run
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
